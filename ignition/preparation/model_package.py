"""Publish a model package containing one native resource bundle per rank."""

from pathlib import Path

from .runtime_package import export_model
from .package_file import DEFAULT_SHARD_SIZE, combine
from .memory_inventory import print_inventory


def export_package(prepared, images, storage, path, *, options, publisher, progress=None, console=None,
                   staging=None, shard_size=DEFAULT_SHARD_SIZE, form='directory'):
    """Export every rank, close preparation resources, and assemble the model file.

    Ranks reuse immutable allocation blobs and stage execution resources under `staging`
    (default: beside `path`). Publication links payload blobs beside the metadata primary.
    """
    if publisher is None:
        raise ValueError("final model publication requires a signing publisher")
    import torch.distributed as dist

    cache_only = path is None
    import flare_heliopause
    from .package_snapshot import _OPTIONS

    settings = _OPTIONS.get()
    if settings is None:
        raise RuntimeError("export requires the configured package cache")
    # Native chooses the workspace; a cache-only export has no destination.
    native_stage = flare_heliopause.NativeExportStage(
        None if cache_only else Path(path), staging, settings.cache)
    stage = Path(native_stage.path)
    token = _OPTIONS.set(settings)
    result = None
    try:
        roots = [Path(native_stage.rank_path(rank)) for rank in range(storage.world)]
        if storage.world > 1:
            for rank, control in enumerate(prepared.controls, 1):
                control.send(('export', roots[rank], options, _OPTIONS.get()))
            dist.barrier()
            storage.peer_storage.reset()
            dist.barrier()
        model = prepared if storage.world == 1 else prepared.model
        package = export_model(model, images, storage, roots[0], options=options, progress=progress,
                               sources=True)
        inventories = [package.memory_inventory]
        memory = [[]] if storage.world == 1 else [list(storage.peer_storage.owned.items())]
        if storage.world > 1:
            for control in prepared.controls:
                reply = control.recv()
                memory.append(reply['peer_memory'])
                inventories.append(reply['memory_inventory'])
        peers = []
        for rank, allocations in enumerate(memory):
            for address, size in allocations:
                peers.append(dict(owner=rank, address=address, nbytes=size, readers=list(range(storage.world))))
        # The stage leases every cache payload the rank sources name, so peer
        # preparation leases can drop before combine reads them.
        native_stage.retain_sources(roots)
        prepared.close()
        for region in storage.host_regions:
            region.close()
        result = combine(None if cache_only else path, roots, peers,
                         publisher=publisher, shard_size=shard_size, form=form, cache_only=cache_only)
        print_inventory(inventories, console=console or getattr(progress, "console", None))
    finally:
        _OPTIONS.reset(token)
        native_stage.close()
    return result
