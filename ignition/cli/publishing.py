"""Publisher key creation and promotion of prepared development packages."""

from ignition.errors import UsageError


def validate_repository(repository):
    """Accept a portable repository namespace or local recipe name."""
    import re
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*(/[A-Za-z0-9][A-Za-z0-9._-]*)?', repository) or '..' in repository:
        raise UsageError('export repository must be a portable namespace/repository identifier')


def convert(args):
    import json
    from pathlib import Path
    from google.protobuf.message import DecodeError
    from ignition.cli.config import Configuration
    from ignition.preparation.package_snapshot import convert as convert_package, payload_form

    configuration = Configuration.load(args.config)
    form = args.form
    output = args.output
    try:
        kind = payload_form(args.package)
        if kind == 'legacy' and form == 'packed':
            raise UsageError('--form packed requires a payload-format package')
        if kind != 'legacy' and args.shard_size is not None:
            raise UsageError('--shard-size applies to legacy blob conversion; payload containers are not sharded')
        embedded = Path(__file__).parents[2] / 'helios/src/runtime_package/publishers.json'
        profiles = json.loads(embedded.read_text()) if embedded.is_file() else []
        paths = list(args.publisher_keys)
        development = configuration.path.parent / 'development-publisher.json'
        if development.is_file():
            paths.append(development)
        for path in paths:
            profiles.extend(json.loads(path.expanduser().read_text()))
        if kind == 'legacy':
            if args.output is None:
                # Legacy no-output conversion is imported into the shared cache as
                # a committed leased entry; there is no second snapshot namespace.
                from ignition.preparation.package_snapshot import export_convert_cache
                store = export_convert_cache(
                    args.package, configuration.directory('package_cache'),
                    profiles=profiles, shard_size=configuration.shard_size(args.shard_size))
                print(f'Converted cache entry: {store.entry_path} ({store.model}/{store.entry})')
                return
            output = args.output
        elif form == 'packed':
            if args.output is None:
                raise UsageError(
                    'packed payload conversion requires --output: it writes one .safetensors file and '
                    'does not use the cache snapshot path')
            output = args.output
        else:
            output = args.output
        if kind != 'legacy' and form == 'directory' and args.output is None:
            from ignition.preparation.package_snapshot import payload_convert_cache
            store = payload_convert_cache(
                args.package, configuration.directory('package_cache'),
                profiles=profiles, form=form)
        else:
            store = convert_package(args.package, output, profiles=profiles,
                                    cache=configuration.directory('package_cache'),
                                    shard_size=configuration.shard_size(args.shard_size), form=form)
    except (ValueError, OSError, KeyError, TypeError, DecodeError) as error:
        raise UsageError(str(error)) from error
    if hasattr(store, 'entry'):
        print(f'Converted cache entry: {store.entry_path} ({store.model}/{store.entry})')
    elif hasattr(store, 'form'):
        print(f'Converted package: {output}')
        print(f'Payload form: {store.form}; bytes written: {store.bytes}')
    else:
        print(f'Converted package: {output}')
        print(f'Payload bytes written: {store.written}; reused: {store.reused}')


def keygen(args):
    from ignition.signing import generate, passphrase

    try:
        password = passphrase('New signing-key passphrase: ')
        if password != passphrase('Confirm signing-key passphrase: '):
            raise ValueError('passphrases do not match')
        fingerprint = generate(args.directory, args.name, args.namespace, password)
    except (ValueError, OSError, EOFError) as error:
        raise UsageError(str(error)) from error
    directory = args.directory.expanduser()
    print(f'Encrypted key: {directory / "signing-key.pem"}')
    print(f'Public profile: {directory / "publisher.json"}')
    print(f'SHA-256 fingerprint: {fingerprint}')


def key_password(args):
    from ignition.signing import change_password

    try:
        fingerprint = change_password(args.signing_key)
    except (ValueError, OSError, EOFError) as error:
        raise UsageError(str(error)) from error
    print(f'Updated encrypted key: {args.signing_key.expanduser()}')
    print(f'SHA-256 fingerprint: {fingerprint}')


def _public_profiles(args, configuration):
    import json
    from pathlib import Path
    embedded = Path(__file__).parents[2] / 'helios/src/runtime_package/publishers.json'
    profiles = json.loads(embedded.read_text()) if embedded.is_file() else []
    paths = list(getattr(args, 'publisher_keys', []) or [])
    development = getattr(args, 'development_publisher', None)
    if development is None:
        development = configuration.path.parent / 'development-publisher.json'
    development = Path(development).expanduser()
    if development.is_file():
        paths.append(development)
    for path in paths:
        profiles.extend(json.loads(Path(path).expanduser().read_text()))
    return profiles


def _source_profile(args, configuration):
    """Load the single public profile that authenticates a payload source."""
    import json
    from pathlib import Path
    path = getattr(args, 'development_publisher', None)
    if path is None:
        path = configuration.path.parent / 'development-publisher.json'
    path = Path(path).expanduser()
    if not path.is_file():
        raise ValueError(
            'payload sign requires the source publisher profile; supply '
            '--development-publisher or place development-publisher.json beside the config')
    profiles = json.loads(path.read_text())
    if not isinstance(profiles, list) or len(profiles) != 1:
        raise ValueError('source publisher profile must contain exactly one profile')
    profile = profiles[0]
    namespace = profile.get('namespace') if isinstance(profile, dict) else None
    if not isinstance(namespace, str) or not namespace:
        raise ValueError('source publisher profile must name a nonempty namespace')
    return profile


def _publisher(args, configuration):
    from ignition.preparation.protection import Publisher
    values = configuration.values.get('publishing', {})
    key = getattr(args, 'signing_key', None) or values.get('signing_key')
    namespace = getattr(args, 'namespace', None) or values.get('namespace')
    source = getattr(args, 'official_source', None) or values.get('official_source')
    if not all((key, namespace, source)):
        raise ValueError('publishing requires signing_key, namespace, and official_source')
    return Publisher.load(key, namespace, source)


def combine(args):
    import json
    from google.protobuf.message import DecodeError
    from ignition.cli.config import Configuration
    from ignition.preparation.package_snapshot import payload_combine, payload_form

    configuration = Configuration.load(args.config)
    try:
        for source in args.sources:
            if payload_form(source) == 'legacy':
                raise UsageError('combine accepts payload-format packages only; convert legacy packages first')
        publisher = _publisher(args, configuration)
        profiles = _public_profiles(args, configuration)
        result = payload_combine(args.sources, args.output, publisher=publisher, profiles=profiles,
                                 accepted=publisher.namespace, target_accepted=publisher.namespace)
    except (ValueError, OSError, KeyError, TypeError, DecodeError) as error:
        raise UsageError(str(error)) from error
    print(f'Combined package: {args.output}')
    print(f'Deployments: {getattr(result, "payloads", 0)} payloads; form: {result.form}')


def sign(args):
    import json
    from google.protobuf.message import DecodeError
    from ignition.cli.config import Configuration
    from ignition.preparation.protection import Publisher
    from ignition.preparation.resign import resign as promote
    from ignition.progress import StartupProgress

    configuration = Configuration.load(args.config)
    values = configuration.values.get('publishing', {})
    try:
        if args.output.exists():
            raise FileExistsError(args.output)
        from ignition.preparation.package_snapshot import payload_form, payload_promote
        if payload_form(args.package) != 'legacy':
            publisher = _publisher(args, configuration)
            source_profile = _source_profile(args, configuration)
            target = {
                'namespace': publisher.namespace,
                'official_source': publisher.official_source,
                'development': publisher.development,
                'license_name': '' if publisher.license is None else publisher.license.name,
                'license_link': '' if publisher.license is None else publisher.license.link,
            }
            result = payload_promote(args.package, args.output, publisher=publisher,
                                     profiles=[source_profile],
                                     accepted=source_profile['namespace'],
                                     target_accepted=publisher.namespace,
                                     target_publisher=target)
            print(f'Signed package: {args.output}')
            print(f'Payload form: {result.form}; payloads: {getattr(result, "payloads", 0)}')
            return
        key = args.signing_key or values.get('signing_key')
        namespace = args.namespace or values.get('namespace')
        source = args.official_source or values.get('official_source')
        if not all((key, namespace, source)):
            raise ValueError('signing requires signing_key, namespace, and official_source')
        publisher = Publisher.load(key, namespace, source)
        profile = args.development_publisher or configuration.path.parent / 'development-publisher.json'
        profiles = json.loads(profile.expanduser().read_text())
        if not isinstance(profiles, list) or len(profiles) != 1:
            raise ValueError('development profile must contain exactly one publisher')
        from ignition.preparation.package_snapshot import export_options
        with export_options(configuration.directory('package_cache')):
            promote(args.package, args.output, publisher=publisher, development_profile=profiles[0],
                    progress=StartupProgress(), shard_size=configuration.shard_size(getattr(args, "shard_size", None)))
    except (ValueError, OSError, EOFError, DecodeError) as error:
        raise UsageError(str(error)) from error
    print(f'Signed package: {args.output}')
    print(f'Publisher: {namespace}; SHA-256 fingerprint: {publisher.fingerprint.hex()}')
