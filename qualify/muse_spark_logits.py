"""Read full-vocabulary teacher-forced logits from captured Muse Spark text graphs.

This diagnostic prepares the complete TP8 catalog with 262144-token context,
4096-token chunks, concurrency four and prompt scoring. It replays prefill128
for the first recorded token, then decode1 for each remaining recorded token.
A whole-prompt prefill also records its final full-vocabulary row. The output
contains fenced CPU logits and normalized hidden vectors for every
causal endpoint: logits[j] predicts after token_ids[:j + 1]. It does not
measure native serving performance. Optional B1/B4 replay measurements preserve
ordinary EOS handling and report unprofiled and profiled rounds separately.
"""

import argparse
from contextlib import contextmanager
from functools import partial
import json
from pathlib import Path


def read_tokens(path):
    value = json.loads(Path(path).read_text())
    tokens = value['token_ids']
    if not tokens or any(type(token) is not int or token < 0 for token in tokens):
        raise ValueError('token_ids must contain nonnegative integer IDs')
    return tuple(tokens)


@contextmanager
def fence_operations(rank, enabled):
    if not enabled:
        yield
        return
    import torch
    from ignition.preparation.compilation import _Invocation

    original = _Invocation.run
    sequence = 0

    def run(operation, reads, writes, integers=()):
        nonlocal sequence
        sequence += 1
        label = f'operation-fence rank={rank} sequence={sequence} target={operation.target}'
        stage = 'capture-query'
        try:
            capturing = torch.cuda.is_current_stream_capturing()
            if not capturing:
                stage = 'before-fence'
                print(f'{label} before-fence begin', flush=True)
                torch.cuda.current_stream().synchronize()
        except Exception as error:
            print(f'{label} before-operation failure stage={stage}: {type(error).__name__}: {error}', flush=True)
            raise
        if capturing:
            return original(operation, reads, writes, integers)
        stage = 'call'
        try:
            result = original(operation, reads, writes, integers)
            stage = 'after-fence'
            torch.cuda.current_stream().synchronize()
        except Exception as error:
            print(f'{label} opaque-operation failure stage={stage}: {type(error).__name__}: {error}', flush=True)
            raise
        print(f'{label} after-fence complete', flush=True)
        return result

    _Invocation.run = run
    try:
        yield
    finally:
        _Invocation.run = original


def measure_replay(model, pool, requests, tokens, group, rank, *, profile):
    import time
    import torch
    import torch.distributed as dist
    from ignition.components.sampling import FINISHED, RequestAdmission, RUNNING

    buffers, device = model.buffers, pool.backing.device
    rows = []
    keys = ('allocation.all.allocated', 'allocation.all.freed',
            'segment.all.allocated', 'segment.all.freed')
    eos = tuple(model.server_config['frontend']['eos_token_ids'])
    captured = False
    try:
        for batch in (1, 4):
            with torch.cuda.nvtx.range(f'muse.setup.b{batch}'):
                pool.clear_for_preparation()
                requests.generation.zero_()
                requests.status.zero_()
                requests.error_code.zero_()
                requests.pending_token.fill_(-1)
                buffers['request_state_rows'].zero_()
                buffers['page_tables'].fill_(-1)
                pages = (len(tokens) + 64 + 127) // 128
                if batch * pages > int(pool.memory_capacity[1]):
                    raise ValueError('measurement requests exceed backed preparation pages')
                for slot in range(batch):
                    requests.admit(slot, RequestAdmission(generation=1, seed=17 + slot, max_new_tokens=64,
                        top_k=-1, top_p=1., eos_token_ids=eos), tokens)
                    buffers['request_state_rows'][slot, 0] = slot + 1
                    buffers['page_tables'][slot, :pages].copy_(
                        torch.arange(slot * pages, (slot + 1) * pages, dtype=torch.int32, device=device))
                capacity = min(s.token_capacity for s in model.signatures.values()
                               if s.kind == 'prefill' and s.token_capacity >= len(tokens))
                for slot in range(batch):
                    buffers['row_slot_ids'].fill_(-1)
                    buffers['row_slot_ids'][0] = slot
                    buffers['row_generations'].zero_()
                    buffers['row_generations'][0] = 1
                    buffers['live_rows'].fill_(1)
                    buffers['query_lengths'].zero_()
                    buffers['query_lengths'][0] = len(tokens)
                    buffers['sample_enabled'].fill_(1)
                    buffers['token_ids'].zero_()
                    buffers['token_ids'][:len(tokens)].copy_(torch.tensor(tokens, device=device))
                    buffers['submission_sequence'].add_(1)
                    torch.cuda.synchronize(device)
                    dist.barrier(group=group)
                    model.graphs[f'prefill{capacity}'].replay()
                    torch.cuda.synchronize(device)
                buffers['row_slot_ids'].fill_(-1)
                buffers['row_slot_ids'][:batch].copy_(torch.arange(batch, device=device, dtype=torch.int32))
                buffers['row_generations'].zero_()
                buffers['row_generations'][:batch] = 1
                buffers['live_rows'].fill_(batch)
                buffers['query_lengths'].zero_()
                buffers['query_lengths'][:batch] = 1
                graph = model.graphs[f'decode{batch}']
                start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            warmups, measurements = 0, []
            for iteration in range(20):
                statuses = requests.status[:batch].cpu().tolist()
                errors = requests.error_code[:batch].cpu().tolist()
                agreed = [None] * dist.get_world_size(group)
                dist.all_gather_object(agreed, (statuses, errors), group=group)
                if any(value != agreed[0] for value in agreed):
                    raise RuntimeError('request status or errors differ across TP ranks')
                if any(errors) or any(status not in (RUNNING, FINISHED) for status in statuses):
                    raise RuntimeError(f'invalid measurement request states: {statuses}, errors={errors}')
                if any(status == FINISHED for status in statuses):
                    break
                before_tokens = requests.consumed_tokens[:batch].cpu().tolist()
                buffers['submission_sequence'].add_(1)
                torch.cuda.synchronize(device)
                dist.barrier(group=group)
                measured = iteration >= 4
                if measured and profile and not captured:
                    if rank == 0:
                        torch.cuda.profiler.start()
                    dist.barrier(group=group)
                    captured = True
                before_allocations = torch.cuda.memory_stats(device)
                host_start = time.monotonic_ns()
                start.record()
                with torch.cuda.nvtx.range(f'muse.{"measured" if measured else "warmup"}.b{batch}.step{iteration}'):
                    graph.replay()
                    end.record()
                    end.synchronize()
                host_end = time.monotonic_ns()
                after_allocations = torch.cuda.memory_stats(device)
                if any(after_allocations[key] != before_allocations[key] for key in keys):
                    raise RuntimeError('captured decode allocated or freed CUDA storage')
                after_tokens = requests.consumed_tokens[:batch].cpu().tolist()
                if after_tokens != [value + 1 for value in before_tokens]:
                    raise RuntimeError('captured decode did not consume every active request')
                if any(requests.error_code[:batch].cpu().tolist()):
                    raise RuntimeError('captured decode reported a request error')
                if measured:
                    local = dict(rank=rank, gpu_ms=start.elapsed_time(end),
                        host_start_ns=host_start, host_end_ns=host_end,
                        consumed_before=before_tokens, consumed_after=after_tokens)
                    gathered = [None] * dist.get_world_size(group)
                    dist.all_gather_object(gathered, local, group=group)
                    if rank == 0:
                        measurements.append(dict(per_rank=gathered,
                            rank_max_gpu_ms=max(value['gpu_ms'] for value in gathered),
                            host_span_ms=(max(value['host_end_ns'] for value in gathered)
                                          - min(value['host_start_ns'] for value in gathered)) / 1e6))
                else:
                    warmups += 1
            if rank == 0:
                rows.append(dict(batch=batch, prompt_tokens=len(tokens), budget=64, warmups=warmups,
                    profiled=profile, world_size=dist.get_world_size(group),
                    sampling=dict(temperature=1., top_p=1., top_k=-1, eos_token_ids=list(eos),
                                  seeds=list(range(17, 17 + batch))),
                    measured_count=len(measurements), final_status=requests.status[:batch].cpu().tolist(),
                    final_consumed=requests.consumed_tokens[:batch].cpu().tolist(), measurements=measurements))
                print(f'captured replay B{batch} profiled={profile}: {warmups} warmups, {len(measurements)} measured rounds', flush=True)
    finally:
        torch.cuda.synchronize(device)
        dist.barrier(group=group)
        if captured and rank == 0:
            torch.cuda.profiler.stop()
    return rows


def prepare_and_read(snapshot, config, *, rank, group, package_storage, progress=None, tokens, output,
                     measure=False, profile=False, fence=False):
    import torch
    import torch.distributed as dist

    from ignition.components.sampling import DeviceRequestState, RequestAdmission, RUNNING
    from ignition.recipes.muse_spark.bootstrap import prepare_rank
    from ignition.recipes.muse_spark.pool import ContinuationPool

    with fence_operations(rank, fence):
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
    if measure:
        measurements = measure_replay(model, pool, requests, tokens, group, rank, profile=False)
        if profile:
            measurements.extend(measure_replay(model, pool, requests, tokens, group, rank, profile=True))
        if rank == 0:
            output.with_suffix('.replay.json').write_text(json.dumps(measurements, indent=2) + '\n')
    dist.barrier(group=group)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--tokens', type=Path, required=True, help='JSON object containing exact token_ids')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpus', required=True)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--measure-replay', action='store_true', help='measure valid B1/B4 captured decode after correctness readback')
    parser.add_argument('--profile-replay', action='store_true', help='control profiler capture around measured B1/B4 regions')
    parser.add_argument('--fence-operations', action='store_true',
                        help='diagnose preparation faults by logging and synchronizing compiled storage operations outside capture')
    args = parser.parse_args()
    if args.profile_replay and not args.measure_replay:
        parser.error('--profile-replay requires --measure-replay')
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
        kernel_cache=configuration.directory('kernel_cache'), retain_graphs=args.measure_replay)
    torch.cuda.set_device(0)
    storage = PackageStorage(world=plan.world_size)
    runner = partial(prepare_and_read, tokens=tokens, output=args.output.resolve(),
                     measure=args.measure_replay, profile=args.profile_replay, fence=args.fence_operations)
    with storage:
        prepared = prepare_group(args.snapshot.resolve(), config, runner, package_storage=storage,
                                 progress=lambda text: print(text, flush=True))
    prepared.close()
    for region in storage.host_regions:
        region.close()


if __name__ == '__main__':
    main()
