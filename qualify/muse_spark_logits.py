"""Read full-vocabulary teacher-forced logits from captured Muse Spark text graphs.

This diagnostic prepares the complete TP8 catalog with 262144-token context,
4096-token chunks, concurrency four and prompt scoring. It replays prefill128
for the first recorded token, then decode1 for each remaining recorded token.
A whole-prompt prefill also records its final full-vocabulary row. The output
contains fenced CPU logits and normalized hidden vectors for every
causal endpoint: logits[j] predicts after token_ids[:j + 1]. It does not
measure native serving performance.
"""

import argparse
from functools import partial
import json
from pathlib import Path


def read_tokens(path):
    value = json.loads(Path(path).read_text())
    tokens = value['token_ids']
    if not tokens or any(type(token) is not int or token < 0 for token in tokens):
        raise ValueError('token_ids must contain nonnegative integer IDs')
    return tuple(tokens)


def prepare_and_read(snapshot, config, *, rank, group, package_storage, progress=None, tokens, output):
    import torch
    import torch.distributed as dist

    from ignition.components.sampling import DeviceRequestState, RequestAdmission, RUNNING
    from ignition.recipes.muse_spark.bootstrap import prepare_rank
    from ignition.recipes.muse_spark.pool import ContinuationPool

    model = prepare_rank(snapshot, config, rank=rank, group=group, package_storage=package_storage,
                         progress=progress)
    if any(token >= model.server_config['model_contract']['vocabulary'] for token in tokens):
        raise ValueError('recorded token lies outside the model vocabulary')
    forwards = next(value for value in model.resources if isinstance(value, dict) and 'decode1' in value)
    prefill, decode = (forwards[name] for name in ('prefill128', 'decode1'))
    pool = next(value for value in model.resources if isinstance(value, ContinuationPool))
    requests = next(value for value in model.resources if isinstance(value, DeviceRequestState))
    if prefill.decoder.pool is not pool or decode.decoder.pool is not pool:
        raise AssertionError('captured text variants do not share the continuation pool')
    if prefill.batch.requests is not requests or decode.batch.requests is not requests:
        raise AssertionError('captured text variants do not share request state')
    if len(tokens) > 4096:
        raise ValueError("recorded prompt must fit one 4096-token prefill chunk")
    if len(decode.decoder.layers) != 62:
        raise AssertionError('diagnostic requires the complete 62-layer decoder')
    if len(tokens) > int(pool.memory_capacity[1]) * 128:
        raise ValueError('recorded context exceeds the physically backed preparation pages')
    buffers, device = model.buffers, pool.backing.device
    def reset():
        pool.clear_for_preparation()
        requests.generation.zero_()
        requests.status.zero_()
        requests.error_code.zero_()
        requests.pending_token.fill_(-1)
        requests.admit(0, RequestAdmission(generation=1, seed=17, max_new_tokens=len(tokens) + 1,
                                          top_k=-1, top_p=1.), tokens)
        buffers['row_slot_ids'].fill_(-1)
        buffers['row_slot_ids'][0] = 0
        buffers['row_generations'].zero_()
        buffers['row_generations'][0] = 1
        buffers['request_state_rows'].zero_()
        buffers['request_state_rows'][0, 0] = 1
        buffers['live_rows'].fill_(1)
        buffers['query_lengths'].zero_()
        buffers['query_lengths'][0] = 1
        buffers['sample_enabled'].fill_(1)
        buffers['page_tables'].fill_(-1)
        pages = (len(tokens) + 127) // 128
        buffers['page_tables'][0, :pages].copy_(torch.arange(pages, device=device, dtype=torch.int32))
        buffers['token_ids'].zero_()
        buffers['token_ids'][0] = tokens[0]
        buffers['frame_index'].zero_()
        buffers['logprob_requested'].zero_()
        buffers['logprob_token_id_counts'].zero_()
    reset()
    capacity = min(signature.token_capacity for signature in model.signatures.values()
                   if signature.kind == 'prefill' and signature.token_capacity >= len(tokens))
    whole_name = f'prefill{capacity}'
    whole = forwards[whole_name]
    buffers['query_lengths'][0] = len(tokens)
    buffers['token_ids'][:len(tokens)].copy_(torch.tensor(tokens, device=device))
    buffers['submission_sequence'].fill_(1)
    torch.cuda.synchronize(device)
    dist.barrier(group=group)
    model.graphs[whole_name].replay()
    torch.cuda.synchronize(device)
    if int(requests.status[0]) != RUNNING or int(requests.consumed_tokens[0]) != len(tokens):
        raise RuntimeError('captured prefill rejected the complete recorded prompt')
    if not torch.equal(whole.batch.metadata.token_ids[:len(tokens)].cpu(), torch.tensor(tokens)):
        raise AssertionError('captured prefill packed a different recorded prompt')
    prefill_logits = whole.decoder.scratch.logits[0].cpu().clone() if rank == 0 else None
    prefill_hidden = whole.decoder.scratch.hidden[0].cpu().clone() if rank == 0 else None
    reset()
    logits, hidden = [], []
    torch.cuda.synchronize(device)
    dist.barrier(group=group)
    for index, token in enumerate(tokens):
        name, forward = ('prefill128', prefill) if index == 0 else ('decode1', decode)
        if index:
            requests.pending_token[0] = token
        buffers['submission_sequence'].fill_(index + 1)
        torch.cuda.synchronize(device)
        dist.barrier(group=group)
        model.graphs[name].replay()
        torch.cuda.synchronize(device)
        if int(requests.status[0]) != RUNNING or int(requests.error_code[0]) != 0:
            raise RuntimeError(f'text graph rejected token endpoint {index + 1}: '
                f'status={int(requests.status[0])}, error={int(requests.error_code[0])}')
        if int(requests.consumed_tokens[0]) != index + 1:
            raise AssertionError('captured text graph consumed a different causal endpoint')
        if int(forward.batch.metadata.token_ids[0]) != token:
            raise AssertionError('captured text graph packed a different recorded token')
        if int(forward.batch.metadata.positions[0]) != index:
            raise AssertionError('captured text graph packed a different token position')
        if rank == 0:
            logits.append(forward.decoder.scratch.logits[0].cpu().clone())
            hidden.append(forward.decoder.scratch.hidden[0].cpu().clone())
            if (index + 1) % 16 == 0 or index + 1 == len(tokens):
                print(f'captured teacher-forced endpoints: {index + 1}/{len(tokens)}', flush=True)
    dist.barrier(group=group)
    if rank == 0:
        torch.save(dict(schema_version=1, prompt_tokens=list(tokens[:1]), teacher_tokens=list(tokens[1:]),
            logits=torch.stack(logits), hidden=torch.stack(hidden),
            prefill_logits=prefill_logits, prefill_hidden=prefill_hidden,
            model_id='meta-models/Muse-Spark-Open-816B-A42B-NVFP4', revision=Path(snapshot).name), output)
    dist.barrier(group=group)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--tokens', type=Path, required=True, help='JSON object containing exact token_ids')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpus', required=True)
    parser.add_argument('--config', type=Path)
    args = parser.parse_args()
    tokens = read_tokens(args.tokens)
    if args.output.exists():
        parser.error('output already exists')
    from ignition.cli import environment
    from ignition.cli.config import Configuration, Setting
    from ignition.cli.preparation import Host, derive, resolve_model
    configuration = Configuration.load(args.config)
    model = resolve_model(configuration, str(args.snapshot.resolve()))
    plan = derive(configuration, model, Host.discover(), gpus=args.gpus,
        concurrency=Setting(4, 'diagnostic'), speculation=None, prefill_chunk=4096, prompt_logprobs=True)
    environment.apply(configuration, plan.gpus, plan.recipe.b12x_environment)
    import torch
    from ignition.preparation.package_storage import PackageStorage
    from ignition.preparation.rank_group import prepare_group
    from ignition.recipes.muse_spark.bootstrap import PreparationConfig
    config = PreparationConfig(collective_tuning=plan.collective_tuning, prompt_logprobs=plan.prompt_logprobs,
        compile_workers=plan.compile_workers, autotune_effort=plan.autotune_effort, tuning_replica=plan.tuning_replica,
        native_dense_activations=plan.native_dense_activations,
        lock_gpu_clocks=plan.lock_gpu_clocks and plan.autotune_effort != 'off',
        devices=tuple(range(plan.world_size)), context=plan.context, max_active=plan.concurrency,
        prefill_chunk_size=plan.prefill_chunk, snapshot_budget_bytes=plan.snapshots * plan.costs.state_row_bytes,
        headroom_bytes=plan.headroom_bytes + plan.reserve_bytes, compile_decoder=True,
        kernel_cache=configuration.directory('kernel_cache'))
    torch.cuda.set_device(0)
    storage = PackageStorage(world=plan.world_size)
    runner = partial(prepare_and_read, tokens=tokens, output=args.output.resolve())
    with storage:
        prepared = prepare_group(args.snapshot.resolve(), config, runner, package_storage=storage,
                                 progress=lambda text: print(text, flush=True))
    prepared.close()
    for region in storage.host_regions:
        region.close()


if __name__ == '__main__':
    main()
