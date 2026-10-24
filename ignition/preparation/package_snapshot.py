"""Publishing options and adapters for native immutable runtime snapshots."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
import json

from .package_file import ALIGNMENT, DEFAULT_SHARD_SIZE, MAX_HEADER_BYTES, MAX_INDEX_BYTES

MAX_BLOBS = 65536
MAX_PUBLISHED_BLOB_BYTES = 64_000_000_000
BLOCK = 8 << 20
_OPTIONS = ContextVar('package_snapshot_options', default=None)


@dataclass(frozen=True)
class ExportOptions:
    cache: Path
    no_weights: bool = False


@contextmanager
def export_options(cache, *, no_weights=False):
    token = _OPTIONS.set(ExportOptions(Path(cache), no_weights))
    try:
        yield
    finally:
        _OPTIONS.reset(token)


def read_container(source):
    import flare_heliopause
    record = json.loads(flare_heliopause.export_read_container(str(source)) if isinstance(source, (str, Path))
                        else flare_heliopause.export_read_container_fd(source.fileno()))
    return record['header'], record['base'], bytes(record['envelope']), record['layout']



def validate_layout(layout):
    import flare_heliopause
    flare_heliopause.export_validate_layout(json.dumps(layout))


def write_primary(path, envelope, layout):
    import flare_heliopause
    flare_heliopause.export_write_primary(str(path), envelope, json.dumps(layout))


def export_convert_cache(source, cache, *, profiles, shard_size=DEFAULT_SHARD_SIZE):
    """Convert a legacy package into a committed leased shared-cache entry."""
    import flare_heliopause
    from types import SimpleNamespace
    value = flare_heliopause.export_convert_cache(
        str(source), str(cache), json.dumps(profiles), shard_size)
    return SimpleNamespace(**json.loads(value))


def payload_convert_cache(source, cache, destination=None, *, profiles, form='directory'):
    """Install a payload package as a shared cache entry, optionally exporting it.

    An omitted destination leaves the committed leased entry in the shared cache;
    the returned `entry_path` is that entry's actual directory.
    """
    import flare_heliopause
    from types import SimpleNamespace
    if destination is not None and Path(destination).exists():
        raise FileExistsError(destination)
    value = flare_heliopause.export_payload_convert_cache(
        str(source), str(cache), '' if destination is None else str(destination),
        json.dumps(profiles), form)
    return SimpleNamespace(**json.loads(value))


def payload_combine(sources, destination, *, publisher, profiles, accepted, target_accepted,
                    form='directory'):
    """Combine prepared TP packages into one signed payload directory."""
    import flare_heliopause
    from types import SimpleNamespace
    if Path(destination).exists():
        raise FileExistsError(destination)
    value = flare_heliopause.export_payload_combine(
        [str(source) for source in sources], str(destination), json.dumps(publisher.native_request),
        json.dumps(profiles), accepted, target_accepted, form)
    return SimpleNamespace(**json.loads(value))


def payload_promote(source, destination, *, publisher, profiles, accepted, target_accepted, target_publisher):
    """Re-sign one payload package under a fresh identity while preserving compressed executables."""
    import flare_heliopause
    from types import SimpleNamespace
    if Path(destination).exists():
        raise FileExistsError(destination)
    value = flare_heliopause.export_payload_promote(
        str(source), str(destination), json.dumps(publisher.native_request),
        json.dumps(profiles), accepted, target_accepted, json.dumps(target_publisher))
    return SimpleNamespace(**json.loads(value))


def payload_form(source):
    """Classify a package as payload packed, payload directory, or legacy."""
    import flare_heliopause
    return flare_heliopause.export_payload_form(str(source))


def convert(source, destination, *, profiles=(), cache=None, shard_size=DEFAULT_SHARD_SIZE,
            form='directory'):
    """Convert a package to the selected canonical form.

    Legacy blob packages keep shard-size conversion. Payload packages convert
    between the packed file and the directory of deterministic companions using
    the shared Helios container reader/writer; conversion copies signed bytes and
    needs no signing key.
    """
    import flare_heliopause
    from types import SimpleNamespace
    if form not in ('packed', 'directory'):
        raise ValueError(f'unknown payload package form {form!r}')
    if Path(destination).exists():
        raise FileExistsError(destination)
    if payload_form(source) == 'legacy':
        if form == 'packed':
            raise ValueError('--form packed requires a payload-format package')
        value = flare_heliopause.export_convert(str(source), str(destination), json.dumps(profiles),
                                              None if cache is None else str(cache), shard_size)
        return SimpleNamespace(**json.loads(value))
    value = flare_heliopause.export_payload_convert(str(source), str(destination), form)
    return SimpleNamespace(**json.loads(value))
