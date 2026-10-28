"""Muse Spark text contracts retain device tokens and complete cache row storage."""

from types import SimpleNamespace

from google.protobuf.json_format import ParseDict

from ignition.components.paged_batch import BatchInputs
from ignition.components.parallel import TensorParallelShard
from ignition.preparation import contract as c, runtime_package_pb2 as p
from ignition.preparation.execution import GraphSignature
from ignition.recipes.muse_spark.contract import prepare
from ignition.recipes.muse_spark.pool import ContinuationPool

from .test_muse_spark_attention import Geometry


def test_text_contract_binds_complete_state_and_physical_fp8_page_bytes():
    config = SimpleNamespace(layers=2, attention=lambda i: Geometry(2048 if i == 0 else -1),
                             text={"hidden_size": 128})
    pool = ContinuationPool.allocate(config, TensorParallelShard(0, 1), 3, 2, 'cpu')
    buffers = BatchInputs.allocate(128, 2, 'cpu', capacity=4).buffers()
    signatures = {kind: GraphSignature(kind, 1 if 'prefill' in kind else 4,
        128 if 'prefill' in kind else 1, tuple(c.text_inputs(kind)))
        for kind in ('init', 'restore', 'snapshot', 'prefill', 'scored_prefill', 'decode', 'resume')}
    encoded = ParseDict(prepare(signatures, buffers, pool, vocabulary=201088), p.ModelContract())
    assert encoded.version == 3
    assert encoded.cache.page_tokens == 128
    assert encoded.vision.max_images == 0
    assert encoded.frontend.tools.kind == p.ATEM
    assert encoded.frontend.reasoning.kind == p.RECIPIENT_HEADERS
    histories = [r for r in encoded.cache.regions if r.kind == p.PAGED_KEY_VALUE]
    assert {r.storage.buffer for r in histories} == {'cache.layer1.key', 'cache.layer1.value'}
    assert all(r.storage.stride == 128 * 2 * 64 for r in histories)
    for graph in encoded.graphs:
        assert not graph.HasField('draft_cursor')
        assert any(s.view.buffer == 'state_pool' and s.view.nbytes == pool.layout.row_bytes
                   and s.access == p.READ_WRITE for s in graph.state)
    decode = next(g for g in encoded.graphs if g.kind == p.DECODE)
    assert decode.pending_token.buffer == 'request_pending_token'
    assert all(b.source not in (p.PROMPT_TOKEN, p.PENDING_TOKEN) for b in decode.inputs)
    scored = next(g for g in encoded.graphs if g.kind == p.SCORED_PREFILL)
    assert any(b.source == p.PROMPT_TARGET_TOKEN for b in scored.inputs)


def test_preparation_accepts_shared_64_token_chunk_geometry():
    from ignition.recipes.muse_spark.bootstrap import PreparationConfig
    assert PreparationConfig(prefill_chunk_size=64).prefill_chunk_size == 64
    assert PreparationConfig(prefill_chunk_size=192).prefill_chunk_size == 192
