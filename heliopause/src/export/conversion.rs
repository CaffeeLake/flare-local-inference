//! Authenticate and relocate packages without CUDA initialization.
#[cfg(test)]
use super::archive::Container;
use super::{ExportOptions, ExportResult as Result, FilePart, FileSource, invalid, wire};
use super::{
    archive::{self, Layout, PackageSources},
    blob::BlobStore,
    io::IoBudget,
    package::{self, PublishStats},
    protection::{self, Publisher, PublisherRequest},
};
use helios_core::runtime_package::{Publisher as Profile, VerifiedPackage};
use prost::Message;
use rayon::prelude::*;
use ring::digest;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, OpenOptions},
    path::Path,
    sync::{Arc, Mutex, atomic::Ordering},
    time::SystemTime,
};

struct SourceRangeReader<'a> {
    sources: &'a SharedSources,
    budget: &'a IoBudget,
    filename: &'a str,
    offset: u64,
    position: u64,
    remaining: u64,
}

impl<'a> SourceRangeReader<'a> {
    fn new(
        sources: &'a SharedSources,
        budget: &'a IoBudget,
        filename: &'a str,
        offset: u64,
        nbytes: u64,
    ) -> Self {
        Self {
            sources,
            budget,
            filename,
            offset,
            position: 0,
            remaining: nbytes,
        }
    }
}

impl std::io::Read for SourceRangeReader<'_> {
    fn read(&mut self, output: &mut [u8]) -> std::io::Result<usize> {
        if self.remaining == 0 || output.is_empty() {
            return Ok(0);
        }
        let count = output.len().min(self.remaining as usize);
        self.sources
            .read_at(
                self.budget,
                self.filename,
                self.offset + self.position,
                &mut output[..count],
            )
            .map_err(|error| std::io::Error::other(error.to_string()))?;
        self.position += count as u64;
        self.remaining -= count as u64;
        Ok(count)
    }
}

#[derive(Clone)]
struct SharedSources(Arc<Mutex<PackageSources>>);
impl SharedSources {
    fn new(files: PackageSources) -> Self {
        Self(Arc::new(Mutex::new(files)))
    }
    fn length(&self, name: &str) -> Result<u64> {
        self.0
            .lock()
            .map_err(|_| invalid("package source lock poisoned"))?
            .logical_length(name)
    }
    fn sources(&self, name: &str, offset: u64, nbytes: u64) -> Result<Vec<archive::SourcePiece>> {
        self.0
            .lock()
            .map_err(|_| invalid("package source lock poisoned"))?
            .sources(name, offset, nbytes)
    }
    fn read_at(&self, budget: &IoBudget, name: &str, offset: u64, output: &mut [u8]) -> Result<()> {
        let mut cursor = 0;
        for piece in self.sources(name, offset, output.len() as u64)? {
            match piece {
                archive::SourcePiece::Data(source) => {
                    let n = source.length as usize;
                    budget.read_at(&source, 0, &mut output[cursor..cursor + n])?;
                    cursor += n;
                }
                archive::SourcePiece::Zero(n) => {
                    output[cursor..cursor + n as usize].fill(0);
                    cursor += n as usize;
                }
            }
        }
        Ok(())
    }
}

pub(crate) struct Authenticated {
    sources: SharedSources,
    pub index: wire::ModelIndex,
    pub envelope: Vec<u8>,
    pub package_id: Vec<u8>,
    pub publisher: Profile,
    pub namespace: String,
    pub license: Option<wire::License>,
    pub resources: BTreeMap<(u32, Option<usize>, u64), wire::ProtectedResource>,
    pub stored: BTreeMap<Option<usize>, wire::StoredData>,
}
pub(crate) fn authenticate(
    path: &Path,
    profiles: &[Profile],
    development_only: bool,
) -> Result<Authenticated> {
    let files = PackageSources::open(path)?;
    let container = &files.container;
    let payload_length = container
        .layout
        .as_ref()
        .map(|layout| layout.files[""].iter().map(|(n, _)| *n).sum())
        .unwrap_or(container.payload_length);
    let (index, publisher) = helios_core::runtime_package::verify_package_metadata(
        &container.envelope,
        profiles,
        payload_length,
    )?;
    let envelope = wire::ProtectedIndex::decode(container.envelope.as_slice())?;
    let metadata = wire::PublisherMetadata::decode(envelope.groups[0].body.as_slice())?;
    if development_only
        && (!publisher.development
            || publisher.namespace != "local-development"
            || metadata.official_source != "local-development")
    {
        return Err(invalid("only local-development packages can be promoted"));
    }
    let shared = wire::ResourceCatalog::decode(envelope.groups[1].body.as_slice())?;
    let mut resources = BTreeMap::new();
    for resource in shared.resources {
        resources.insert((3, None, resource.offset), resource);
    }
    let mut stored = BTreeMap::from([(
        None,
        wire::StoredData::decode(envelope.groups[2].body.as_slice())?,
    )]);
    for (rank, group) in envelope.ranks.iter().enumerate() {
        let signed = wire::RankProtection::decode(group.body.as_slice())?;
        stored.insert(
            Some(rank),
            signed
                .data
                .ok_or_else(|| invalid("missing rank stored-data description"))?,
        );
        for (kind, catalog) in [(2, signed.executables), (3, signed.support)] {
            for resource in catalog.unwrap_or_default().resources {
                resources.insert((kind, Some(rank), resource.offset), resource);
            }
        }
    }
    package::roles(&index)?;
    Ok(Authenticated {
        index,
        envelope: container.envelope.clone(),
        package_id: envelope.package_id,
        publisher,
        namespace: metadata.namespace,
        license: metadata.license,
        resources,
        stored,
        sources: SharedSources::new(files),
    })
}
fn hash_range(
    files: &SharedSources,
    budget: &IoBudget,
    name: &str,
    offset: u64,
    length: u64,
    context: &mut digest::Context,
    buffer: &mut [u8],
) -> Result<()> {
    let mut position = 0;
    while position < length {
        let count = (length - position).min(buffer.len() as u64) as usize;
        files.read_at(budget, name, offset + position, &mut buffer[..count])?;
        context.update(&buffer[..count]);
        position += count as u64;
    }
    Ok(())
}
fn zero_range(
    files: &SharedSources,
    budget: &IoBudget,
    name: &str,
    offset: u64,
    length: u64,
    buffer: &mut [u8],
) -> Result<()> {
    let mut position = 0;
    while position < length {
        let count = (length - position).min(buffer.len() as u64) as usize;
        files.read_at(budget, name, offset + position, &mut buffer[..count])?;
        if buffer[..count].iter().any(|b| *b != 0) {
            return Err(invalid("package padding is not zero"));
        }
        position += count as u64;
    }
    Ok(())
}
fn verify_with(
    auth: &Authenticated,
    files: &SharedSources,
    budget: &IoBudget,
    check_length: bool,
) -> Result<()> {
    auth.stored
        .par_iter()
        .try_for_each(|(_, data)| -> Result<()> {
            let mut buffer = budget.buffer()?;
            let mut context = digest::Context::new(&digest::SHA256);
            for part in protection::stored_parts(data)? {
                if part.nbytes == 0
                    || part
                        .offset
                        .checked_add(part.nbytes)
                        .is_none_or(|end| end > files.length(&part.filename).unwrap_or(0))
                {
                    return Err(invalid("signed stored range exceeds payload"));
                }
                hash_range(
                    files,
                    budget,
                    &part.filename,
                    part.offset,
                    part.nbytes,
                    &mut context,
                    buffer.bytes_mut(),
                )?;
            }
            if context.finish().as_ref() != data.sha256 {
                return Err(invalid("stored-data digest mismatch"));
            }
            Ok(())
        })?;
    let mut roles = package::roles(&auth.index)?;
    roles.sort_by_key(|r| (r.span.filename.clone(), r.span.offset));
    roles
        .par_iter()
        .filter(|r| r.kind != 4)
        .try_for_each(|role| -> Result<()> {
            let mut buffer = budget.buffer()?;
            let resource = &auth.resources[&(role.kind, role.owner, role.span.offset)];
            for (i, expected) in resource.sha256.iter().enumerate() {
                let pos = if role.kind == 2 {
                    i as u64 * protection::CHUNK
                } else {
                    0
                };
                let count = if role.kind == 2 {
                    protection::CHUNK.min(role.span.nbytes - pos)
                } else {
                    role.span.nbytes
                };
                let mut context = digest::Context::new(&digest::SHA256);
                hash_range(
                    files,
                    budget,
                    &role.span.filename,
                    role.span.offset + pos,
                    count,
                    &mut context,
                    buffer.bytes_mut(),
                )?;
                if context.finish().as_ref() != expected {
                    return Err(invalid("publisher resource content verification failed"));
                }
            }
            Ok(())
        })?;
    let mut buffer = budget.buffer()?;
    let mut cursors = BTreeMap::new();
    for role in roles {
        let cursor = cursors.entry(role.span.filename.clone()).or_insert(0);
        zero_range(
            files,
            budget,
            &role.span.filename,
            *cursor,
            role.span.offset - *cursor,
            buffer.bytes_mut(),
        )?;
        *cursor = role.span.offset + role.span.nbytes;
    }
    if check_length {
        for (name, cursor) in cursors {
            if cursor != files.length(&name)? {
                return Err(invalid("unreferenced source payload bytes"));
            }
        }
    }
    Ok(())
}
pub(crate) fn verify_source(_path: &Path, auth: &Authenticated, budget: &IoBudget) -> Result<()> {
    verify_with(auth, &auth.sources, budget, true)
}
pub(crate) fn verify_layout(
    directory: &Path,
    layout: &Layout,
    auth: &Authenticated,
    budget: &IoBudget,
) -> Result<()> {
    verify_with(
        auth,
        &SharedSources::new(PackageSources::from_layout(directory, layout.clone())?),
        budget,
        true,
    )
}

pub(crate) fn convert(
    source: &Path,
    destination: &Path,
    profiles: &[Profile],
    cache: Option<&Path>,
    shard_size: u64,
) -> Result<PublishStats> {
    if destination.try_exists()? {
        return Err(invalid("destination already exists"));
    }
    let progress = helios_core::startup::StartupProgress::stderr_rows();
    let phase = progress.phase("Authenticate package and verify source bytes", None);
    let auth = authenticate(source, profiles, false)?;
    let planned = package::groups(&auth.index, shard_size)?;
    let budget = IoBudget::for_ranks(1)?;
    verify_source(source, &auth, &budget)?;
    phase.finish(format!("{} ranks", auth.index.ranks.len()));
    let phase = progress.phase("Hash and relocate payload blobs", None);
    let parent = destination.parent().unwrap_or(Path::new("."));
    fs::create_dir_all(parent)?;
    let cache = cache
        .map(Path::to_path_buf)
        .unwrap_or_else(|| parent.join(".flare-cache"));
    let store = BlobStore::new(&cache, budget.clone())?;
    let names: Result<Vec<_>> = planned
        .par_iter()
        .map(|group| {
            let name = store.put(group.nbytes, false, None, |offset, buffer| {
                auth.sources
                    .read_at(&budget, &group.filename, group.offset + offset, buffer)
            })?;
            store.link(&name, parent)?;
            Ok(name)
        })
        .collect();
    let mut files = BTreeMap::from([(String::new(), vec![])]);
    for (group, name) in planned.iter().zip(names?) {
        let parts = files.entry(group.filename.clone()).or_insert_with(Vec::new);
        let cursor: u64 = parts.iter().map(|(n, _)| *n).sum();
        if cursor < group.offset {
            parts.push((group.offset - cursor, String::new()));
        }
        parts.push((group.nbytes, name));
    }
    let layout = Layout { version: 1, files };
    phase.finish(format!(
        "{} bytes written; {} bytes reused",
        store.written.load(Ordering::Relaxed),
        store.reused.load(Ordering::Relaxed)
    ));
    let phase = progress.phase("Verify relocated package and publish metadata", None);
    verify_layout(parent, &layout, &auth, &budget)?;
    store.sync()?;
    archive::install_snapshot(destination, &auth.envelope, &layout)?;
    phase.finish(format!("{} envelope bytes preserved", auth.envelope.len()));
    Ok(PublishStats {
        written: store.written.load(Ordering::Relaxed),
        reused: store.reused.load(Ordering::Relaxed),
    })
}

pub(crate) fn resign(
    source: &Path,
    destination: &Path,
    publisher: &mut Publisher,
    profile: Profile,
    options: &ExportOptions,
    shard_size: u64,
) -> Result<PublishStats> {
    if publisher.development {
        return Err(invalid("re-signing requires a production publisher"));
    }
    if destination.try_exists()? {
        return Err(invalid("destination already exists"));
    }
    let auth = authenticate(source, &[profile], true)?;
    let budget = IoBudget::for_ranks(1)?;
    verify_source(source, &auth, &budget)?;
    if publisher.license.is_none() {
        publisher.license = auth.license.clone();
    }
    let parent = destination.parent().unwrap_or(Path::new("."));
    fs::create_dir_all(parent)?;
    let stage = tempfile::Builder::new()
        .prefix(".flare-resign-")
        .tempdir_in(parent)?;
    let mut index = auth.index.clone();
    let mut roles = package::roles(&index)?;
    roles.sort_by_key(|r| {
        (
            r.kind != 4,
            if r.kind == 4 {
                r.owner.map(|o| o + 1).unwrap_or(0)
            } else {
                0
            },
            if r.kind == 4 {
                protection::stored_parts(&auth.stored[&r.owner])
                    .unwrap()
                    .iter()
                    .position(|p| {
                        p.filename == r.span.filename
                            && p.offset <= r.span.offset
                            && r.span.offset < p.offset + p.nbytes
                    })
                    .unwrap()
            } else {
                0
            },
            r.span.offset,
        )
    });
    let mut plain_files = BTreeMap::new();
    for role in roles.iter().filter(|r| r.kind != 4) {
        if let std::collections::btree_map::Entry::Vacant(entry) = plain_files.entry(role.owner) {
            let path = stage.path().join(format!(
                "plaintext-{}",
                role.owner
                    .map(|r| r.to_string())
                    .unwrap_or_else(|| "shared".into())
            ));
            let file = std::sync::Arc::new(
                OpenOptions::new()
                    .write(true)
                    .read(true)
                    .create_new(true)
                    .open(&path)?,
            );
            entry.insert((path, file));
        }
    }
    let mut staging_offsets = Vec::with_capacity(roles.len());
    let mut staging_ends: BTreeMap<Option<usize>, u64> = BTreeMap::new();
    for role in &roles {
        if role.kind == 4 {
            staging_offsets.push(0);
            continue;
        }
        let resource = &auth.resources[&(role.kind, role.owner, role.span.offset)];
        let offset = staging_ends.get(&role.owner).copied().unwrap_or(0);
        staging_offsets.push(offset);
        staging_ends.insert(
            role.owner,
            offset
                .checked_add(resource.plain_bytes)
                .ok_or_else(|| invalid("decoded executable staging offset overflow"))?,
        );
    }
    let processed: Result<Vec<_>> = roles
        .par_iter()
        .enumerate()
        .filter(|(_, r)| r.kind != 4)
        .map(|(i, role)| {
            let mut buffer = budget.buffer()?;
            let resource = &auth.resources[&(role.kind, role.owner, role.span.offset)];
            let (plain, output) = &plain_files[&role.owner];
            let staging_offset = staging_offsets[i];
            if role.kind == 2 {
                let source = SourceRangeReader::new(
                    &auth.sources,
                    &budget,
                    &role.span.filename,
                    role.span.offset,
                    resource.stored_bytes,
                );
                let mut reader = helios_core::package::protection::open_compressed_reader(
                    source,
                    resource.stored_bytes,
                    resource.plain_bytes,
                    &resource.sha256,
                )?;
                let mut decoded = 0u64;
                let mut buffer = vec![0u8; 1 << 20];
                loop {
                    let count = std::io::Read::read(&mut reader, &mut buffer)
                        .map_err(|error| invalid(format!("decompress executable: {error}")))?;
                    if count == 0 {
                        break;
                    }
                    super::io::write_all_at(
                        output,
                        &buffer[..count],
                        staging_offset + decoded,
                    )?;
                    decoded += count as u64;
                }
                if decoded != resource.plain_bytes {
                    return Err(invalid("decompressed executable length is inconsistent"));
                }
            } else {
                let mut remaining = resource.plain_bytes;
                let mut position = 0;
                let mut hash = digest::Context::new(&digest::SHA256);
                while remaining > 0 {
                    let count = remaining.min(protection::CHUNK);
                    let block = &mut buffer.bytes_mut()[..count as usize];
                    auth.sources.read_at(
                        &budget,
                        &role.span.filename,
                        role.span.offset + position,
                        block,
                    )?;
                    position += block.len() as u64;
                    hash.update(block);
                    super::io::write_all_at(
                        output,
                        block,
                        staging_offset + resource.plain_bytes - remaining,
                    )?;
                    remaining -= count;
                }
                if hash.finish().as_ref() != resource.sha256[0] {
                    return Err(invalid("development supporting-resource digest mismatch"));
                }
            }
            Ok((
                i,
                FileSource {
                    path: plain.clone(),
                    offset: staging_offset,
                    nbytes: resource.plain_bytes,
                },
            ))
        })
        .collect();
    let plain: BTreeMap<_, _> = processed?.into_iter().collect();
    for (_, file) in plain_files.values() {
        file.sync_all()?;
    }
    drop(plain_files);
    let mut payload = Vec::new();
    let mut cursor = 0;
    for (i, role) in roles.iter().enumerate() {
        cursor = package::align(cursor)?;
        let span = package::span_mut(&mut index, &role.key);
        span.offset = cursor;
        span.filename.clear();
        if role.kind == 4 {
            for piece in
                auth.sources
                    .sources(&role.span.filename, role.span.offset, role.span.nbytes)?
            {
                match piece {
                    archive::SourcePiece::Data(data) => {
                        let source_path = data.pinned_path()?;
                        payload.push(FilePart {
                            offset: cursor,
                            source: FileSource {
                                path: source_path,
                                offset: data.offset,
                                nbytes: data.length,
                            },
                        });
                        cursor += data.length;
                    }
                    archive::SourcePiece::Zero(_) => {
                        return Err(invalid("stored resource overlaps layout padding"));
                    }
                }
            }
        } else {
            let source = plain[&i].clone();
            span.nbytes = source.nbytes;
            cursor += source.nbytes;
            payload.push(FilePart {
                offset: span.offset,
                source,
            });
        }
    }
    package::publish_checked(
        destination,
        index,
        payload,
        Some(publisher),
        options,
        shard_size,
        |protected, layout| {
            verify_promoted_stored(&auth, &protected.index, parent, layout, &budget)
        },
    )
}

use pyo3::{exceptions::PyValueError, prelude::*};
#[pyfunction]
fn export_convert(
    py: Python<'_>,
    source: String,
    destination: String,
    profiles: String,
    cache: Option<String>,
    shard_size: u64,
) -> PyResult<String> {
    let profiles = decode_profiles(&profiles).map_err(|e| PyValueError::new_err(e.to_string()))?;
    py.detach(|| {
        convert(
            Path::new(&source),
            Path::new(&destination),
            &profiles,
            cache.as_deref().map(Path::new),
            shard_size,
        )
        .and_then(|value| Ok(serde_json::to_string(&value)?))
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))
}
#[pyfunction]
fn export_resign(
    py: Python<'_>,
    source: String,
    destination: String,
    publisher: String,
    profile: String,
    options: String,
    shard_size: u64,
) -> PyResult<String> {
    let request: PublisherRequest =
        serde_json::from_str(&publisher).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let profile = decode_profiles(&format!("[{profile}]"))
        .map_err(|e| PyValueError::new_err(e.to_string()))?
        .remove(0);
    let options: ExportOptions =
        serde_json::from_str(&options).map_err(|e| PyValueError::new_err(e.to_string()))?;
    py.detach(|| {
        let mut publisher = request.publisher()?;
        let result = resign(
            Path::new(&source),
            Path::new(&destination),
            &mut publisher,
            profile,
            &options,
            shard_size,
        )?;
        Ok::<_, Box<dyn std::error::Error + Send + Sync>>(serde_json::to_string(&result)?)
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))
}
/// Report which canonical form a package uses, or `legacy` for the blob format.
#[pyfunction]
fn export_payload_form(py: Python<'_>, source: String) -> PyResult<String> {
    py.detach(|| {
        let form = helios_core::package::conversion::form(Path::new(&source))?;
        Ok::<_, Box<dyn std::error::Error + Send + Sync>>(
            match form {
                None => "legacy",
                Some(helios_core::package::conversion::Form::Packed) => "packed",
                Some(helios_core::package::conversion::Form::Directory) => "directory",
            }
            .to_string(),
        )
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Convert a payload-format package between its packed and directory forms.
///
/// This is keyless: signed bodies, signatures, compressed executables and payload bytes are
/// copied exactly; only the unsigned location tensor is re-encoded.
#[pyfunction]
fn export_payload_convert(
    py: Python<'_>,
    source: String,
    destination: String,
    form: String,
) -> PyResult<String> {
    py.detach(|| {
        let (source, destination) = (Path::new(&source), Path::new(&destination));
        match form.as_str() {
            "packed" => helios_core::package::conversion::pack(source, destination)?,
            "directory" => helios_core::package::conversion::unpack(source, destination)?,
            other => {
                return Err(super::invalid(format!(
                    "unknown payload package form {other:?}"
                )));
            }
        }
        let bytes = packed_bytes(destination);
        Ok::<_, Box<dyn std::error::Error + Send + Sync>>(
            serde_json::json!({
                "form": form,
                "destination": destination.display().to_string(),
                "bytes": bytes,
            })
            .to_string(),
        )
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))
}
fn packed_bytes(path: &Path) -> u64 {
    let Ok(meta) = std::fs::metadata(path) else {
        return 0;
    };
    if meta.is_file() {
        return meta.len();
    }
    let mut total = 0u64;
    let Ok(entries) = std::fs::read_dir(path) else {
        return 0;
    };
    for entry in entries.flatten() {
        total = total.saturating_add(packed_bytes(&entry.path()));
    }
    total
}

#[derive(serde::Deserialize)]
struct PublicProfile {
    #[serde(default)]
    name: String,
    namespace: String,
    public_key: Vec<u8>,
    #[serde(default)]
    development: bool,
}

fn payload_publishers(profiles: &str) -> Result<Vec<helios_core::runtime_package::Publisher>> {
    let profiles: Vec<PublicProfile> = serde_json::from_str(profiles)?;
    profiles
        .into_iter()
        .map(|profile| {
            Ok(helios_core::runtime_package::Publisher {
                name: if profile.name.is_empty() {
                    profile.namespace.clone()
                } else {
                    profile.name
                },
                namespace: profile.namespace,
                public_key: profile
                    .public_key
                    .as_slice()
                    .try_into()
                    .map_err(|_| super::invalid("publisher public key must be 32 bytes"))?,
                development: profile.development,
            })
        })
        .collect()
}

/// Combine independently prepared TP packages into one signed package.
#[pyfunction]
fn export_payload_combine(
    py: Python<'_>,
    sources: Vec<String>,
    destination: String,
    publisher: String,
    profiles: String,
    accepted: String,
    target_accepted: String,
    form: String,
) -> PyResult<String> {
    let request: PublisherRequest =
        serde_json::from_str(&publisher).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let publishers =
        payload_publishers(&profiles).map_err(|e| PyValueError::new_err(e.to_string()))?;
    if form != "directory" && form != "packed" {
        return Err(PyValueError::new_err(format!(
            "unknown package form {form:?}"
        )));
    }
    py.detach(|| {
        use helios_core::package::conversion;
        let source_paths: Vec<std::path::PathBuf> =
            sources.iter().map(std::path::PathBuf::from).collect();
        let publisher = request.publisher()?;
        let destination = Path::new(&destination);
        // Packed output is published to native staging then converted by the
        // shared keyless container writer; Python never touches the tree.
        let published = if form == "packed" {
            let parent = destination.parent().unwrap_or(Path::new("."));
            std::fs::create_dir_all(parent)?;
            let staging = tempfile::Builder::new()
                .prefix(".flare-packed-")
                .tempdir_in(parent)?;
            let directory = staging.path().join("package");
            let published = conversion::combine(&conversion::CombineRequest {
                sources: &source_paths,
                output: &directory,
                publishers: &publishers,
                accepted: &accepted,
                target_accepted: &target_accepted,
                target_publisher: None,
                key: publisher.signing_key(),
            })?;
            conversion::pack(&published.directory, destination)?;
            published
        } else {
            conversion::combine(&conversion::CombineRequest {
                sources: &source_paths,
                output: destination,
                publishers: &publishers,
                accepted: &accepted,
                target_accepted: &target_accepted,
                target_publisher: None,
                key: publisher.signing_key(),
            })?
        };
        Ok::<_, Box<dyn std::error::Error + Send + Sync>>(
            serde_json::json!({
                "form": form,
                "destination": destination.display().to_string(),
                "payloads": published.payloads.len(),
            })
            .to_string(),
        )
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Re-sign one payload package under a fresh identity, recompressing executables.
#[pyfunction]
fn export_payload_promote(
    py: Python<'_>,
    source: String,
    destination: String,
    publisher: String,
    profiles: String,
    accepted: String,
    target_accepted: String,
    target_publisher: Option<String>,
) -> PyResult<String> {
    let request: PublisherRequest =
        serde_json::from_str(&publisher).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let publishers =
        payload_publishers(&profiles).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let target_publisher: Option<helios_core::package::conversion::TargetPublisher> =
        target_publisher
            .map(|value| serde_json::from_str(&value))
            .transpose()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
    py.detach(|| {
        let publisher = request.publisher()?;
        let published = helios_core::package::conversion::promote(
            Path::new(&source),
            Path::new(&destination),
            &publishers,
            &accepted,
            &target_accepted,
            target_publisher.as_ref(),
            publisher.signing_key(),
        )?;
        Ok::<_, Box<dyn std::error::Error + Send + Sync>>(
            serde_json::json!({
                "form": "directory",
                "destination": published.directory.display().to_string(),
                "payloads": published.payloads.len(),
            })
            .to_string(),
        )
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Install a payload package into the shared cache as a committed leased entry,
/// optionally exporting that entry to an exact destination path.
///
/// Cache entries use the directory canonical form. Packed input and a primary
/// file path are accepted directly; no Python or intermediate tree is created.
#[pyfunction]
fn export_payload_convert_cache(
    py: Python<'_>,
    source: String,
    cache: String,
    destination: String,
    profiles: String,
    form: String,
) -> PyResult<String> {
    py.detach(|| convert_cache(&source, &cache, &destination, &profiles, &form))
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Install one payload package into the shared cache without selecting it until
/// the optional standalone export has succeeded.
fn convert_cache(
    source: &str,
    cache: &str,
    destination: &str,
    profiles: &str,
    form: &str,
) -> Result<String> {
    convert_cache_at(
        source,
        cache,
        destination,
        profiles,
        form,
        SystemTime::now(),
    )
}

fn convert_cache_at(
    source: &str,
    cache: &str,
    destination: &str,
    profiles: &str,
    form: &str,
    at: SystemTime,
) -> Result<String> {
    let publishers = payload_publishers(profiles)?;
    if form != "directory" {
        return Err(invalid("cache entries use the directory canonical form"));
    }
    use helios_core::package::cache::{PackageSource, Root};
    use helios_core::package::protection::Selection;
    let root = Root::open(cache)?;
    let metadata = helios_core::package::reader::Metadata::open(Path::new(source))?;
    let envelope = helios_core::package::protection::envelope(&metadata.envelope)?;
    let mut selections = Vec::with_capacity(envelope.deployments.len());
    for index in 0..envelope.deployments.len() {
        selections.push(Selection::Deployment {
            deployment: index,
            rank: None,
        });
    }
    if selections.is_empty() {
        return Err(invalid("payload package has no deployment"));
    }
    let mut package_source = PackageSource::open(Path::new(source), &publishers, selections[0])?;
    if selections.len() > 1 {
        package_source = package_source.union(&publishers, &selections[1..])?;
    }
    let model = package_source
        .verified()
        .first()
        .map(|verified| verified.package().model_identity.as_str())
        .filter(|identity| !identity.is_empty())
        .ok_or_else(|| invalid("payload package has no model identity"))?;
    let model = helios_core::package::cache::model_key(model)?;
    let candidate = converted_package_candidate(package_source.verified(), at)?;
    // Reserve the readable name before installation. The reservation stays held
    // through commit, so same-minute conversions receive numeric suffixes rather
    // than racing a caller-chosen entry string.
    let staging = root.stage_named(&model, &candidate)?;
    let plan = root.plan_acquisition(
        package_source,
        &helios_core::package::derivation::Limits::default(),
    )?;
    // Commit the entry without selecting it: a standalone export may still
    // fail, and a failed conversion must leave the prior default untouched.
    let committed = root.install_planned_named(plan, staging)?;
    if !destination.is_empty() {
        if let Err(error) = root.export_entry(&committed, Path::new(destination)) {
            let entry_id = committed.id().to_string();
            let model_key = committed.model().to_string();
            drop(committed);
            return match root.remove(&model_key, &entry_id, false) {
                Ok(()) => Err(error.into()),
                Err(cleanup) => Err(invalid(format!(
                    "{error}; the failed conversion left cache entry {model_key}@{entry_id}, which could not be removed: {cleanup}; remove it with `flare cache remove`"
                ))),
            };
        }
    }
    // The entry is complete; this is the only step that selects it. A failure
    // here may already be visible, so no rollback of another caller's ref.
    if let Err(error) = root.publish_entry(&committed) {
        let entry_id = committed.id().to_string();
        let model_key = committed.model().to_string();
        drop(committed);
        return Err(invalid(format!(
            "publishing the default for completed cache entry {model_key}@{entry_id} failed: {error}; the entry is retained and the ref update may already be visible; verify with `flare cache list`"
        )));
    }
    Ok::<_, Box<dyn std::error::Error + Send + Sync>>(
        serde_json::json!({
            "form": "directory",
            "model": committed.model(),
            "entry": committed.id(),
            "entry_path": committed.directory().display().to_string(),
            "destination": destination,
        })
        .to_string(),
    )
}

/// Name a converted package from its authenticated deployment identities.
fn converted_package_candidate(
    verified: &[helios_core::package::protection::VerifiedPackage],
    at: SystemTime,
) -> Result<String> {
    use helios_core::package::naming;
    let mut sizes = BTreeSet::new();
    let mut platform: Option<&str> = None;
    for package in verified {
        let deployment = package
            .deployment()
            .ok_or_else(|| invalid("converted package lacks a deployment"))?;
        if deployment.tensor_parallel == 0 {
            return Err(invalid("converted package has a zero tensor-parallel size"));
        }
        sizes.insert(deployment.tensor_parallel);
        let hardware = package.package().hardware_identity.as_str();
        if naming::PLATFORMS.contains(&hardware) {
            if let Some(previous) = platform {
                if previous != hardware {
                    return Err(invalid(format!(
                        "converted deployments span {previous} and {hardware}; names require one device identity"
                    )));
                }
            } else {
                platform = Some(hardware);
            }
        }
    }
    let sizes: Vec<u32> = sizes.into_iter().collect();
    helios_core::package::naming::deployment_candidate(platform, &sizes, at)
        .ok_or_else(|| invalid("converted package naming requires a tensor-parallel size"))
}

/// Convert a legacy package into a managed shared-cache entry.
///
/// Legacy no-output conversion has no second persistent namespace: the
/// converted primary and companions are imported into `Root` as a committed,
/// leased entry and the caller receives that entry's actual directory.
#[pyfunction]
fn export_convert_cache(
    py: Python<'_>,
    source: String,
    cache: String,
    profiles: String,
    shard_size: u64,
) -> PyResult<String> {
    let profiles = decode_profiles(&profiles).map_err(|e| PyValueError::new_err(e.to_string()))?;
    py.detach(|| convert_legacy_cache(&source, &cache, &profiles, shard_size, SystemTime::now()))
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

fn convert_legacy_cache(
    source: &str,
    cache: &str,
    profiles: &[Profile],
    shard_size: u64,
    at: SystemTime,
) -> Result<String> {
    use helios_core::package::cache::{Root, model_key};
    use helios_core::package::naming;
    let root = Root::open(cache)?;
    let work = tempfile::Builder::new()
        .prefix(".flare-legacy-convert-")
        .tempdir_in(root.path().join("staging"))?;
    let primary = work.path().join("model.safetensors");
    convert(
        Path::new(source),
        &primary,
        profiles,
        Some(Path::new(cache)),
        shard_size,
    )?;
    // The converted target is authenticated before its metadata supplies either
    // the model key or the deployment size used in the readable entry name.
    let verified = VerifiedPackage::open(&primary, profiles, None)?;
    let (identity_model, tensor_parallel, hardware) = match verified.identity() {
        Ok(identity) => (identity.model, identity.tensor_parallel, identity.hardware),
        Err(error) => {
            // Legacy packages written before the serving contract may have no
            // model id. Preserve the historical file-stem key for those packages;
            // the device label still comes only from the unknown-device path.
            let (index, _) = super::package::read_index(&primary)?;
            let model = index
                .contract
                .as_ref()
                .map(|contract| contract.model_id.clone())
                .filter(|id| !id.is_empty());
            if model.is_some() {
                return Err(error.into());
            }
            let model = Path::new(source)
                .file_stem()
                .map(|stem| stem.to_string_lossy().into_owned())
                .unwrap_or_else(|| "model".into());
            let tensor_parallel = match index.cluster.as_ref() {
                Some(cluster) if cluster.world != 0 => cluster.world,
                _ => u32::try_from(index.ranks.len())
                    .map_err(|_| invalid("legacy package rank count overflow"))?,
            };
            (model, tensor_parallel, String::new())
        }
    };
    let platform = naming::PLATFORMS
        .contains(&hardware.as_str())
        .then_some(hardware.as_str());
    let candidate =
        helios_core::package::naming::deployment_candidate(platform, &[tensor_parallel], at)
            .ok_or_else(|| invalid("legacy conversion naming requires a tensor-parallel size"))?;
    let model = model_key(&identity_model)?;
    let staging = root.stage_named(&model, &candidate)?;
    let committed = root.import_legacy_named(&primary, staging)?;
    // `import_legacy_named` reserves and imports without selecting. Legacy
    // conversion has always selected the completed entry; do that explicitly
    // here so the default-ref behavior matches `import_legacy`.
    root.publish_entry(&committed)?;
    Ok::<_, Box<dyn std::error::Error + Send + Sync>>(
        serde_json::json!({
            "format": "legacy-directory",
            "model": committed.model(),
            "entry": committed.id(),
            "entry_path": committed.directory().display().to_string(),
        })
        .to_string(),
    )
}
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(export_convert, module)?)?;
    module.add_function(wrap_pyfunction!(export_resign, module)?)?;
    module.add_function(wrap_pyfunction!(export_payload_form, module)?)?;
    module.add_function(wrap_pyfunction!(export_payload_convert, module)?)?;
    module.add_function(wrap_pyfunction!(export_payload_combine, module)?)?;
    module.add_function(wrap_pyfunction!(export_payload_promote, module)?)?;
    module.add_function(wrap_pyfunction!(export_payload_convert_cache, module)?)?;
    module.add_function(wrap_pyfunction!(export_convert_cache, module)?)?;
    Ok(())
}

fn verify_promoted_stored(
    auth: &Authenticated,
    index: &wire::ModelIndex,
    directory: &Path,
    layout: &Layout,
    budget: &IoBudget,
) -> Result<()> {
    let original = package::roles(&auth.index)?;
    let relocated = package::roles(index)?;
    let files = SharedSources::new(PackageSources::from_layout(directory, layout.clone())?);
    auth.stored
        .par_iter()
        .try_for_each(|(owner, data)| -> Result<()> {
            let mut buffer = budget.buffer()?;
            let mut digest = digest::Context::new(&digest::SHA256);
            for part in protection::stored_parts(data)? {
                let mut resources: Vec<_> = original
                    .iter()
                    .enumerate()
                    .filter(|(_, r)| {
                        r.kind == 4
                            && r.owner == *owner
                            && r.span.filename == part.filename
                            && r.span.offset >= part.offset
                            && r.span.offset + r.span.nbytes <= part.offset + part.nbytes
                    })
                    .collect();
                resources.sort_by_key(|(_, r)| r.span.offset);
                let mut cursor = part.offset;
                for (i, role) in resources {
                    absorb_zeros(&mut digest, role.span.offset - cursor, buffer.bytes_mut());
                    let output = &relocated[i].span;
                    hash_range(
                        &files,
                        budget,
                        &output.filename,
                        output.offset,
                        output.nbytes,
                        &mut digest,
                        buffer.bytes_mut(),
                    )?;
                    cursor = role.span.offset + role.span.nbytes;
                }
                absorb_zeros(
                    &mut digest,
                    part.offset + part.nbytes - cursor,
                    buffer.bytes_mut(),
                );
            }
            if digest.finish().as_ref() != data.sha256 {
                return Err(invalid("stored-data digest mismatch during promotion"));
            }
            Ok(())
        })
}
fn absorb_zeros(digest: &mut digest::Context, mut count: u64, buffer: &mut [u8]) {
    buffer.fill(0);
    while count > 0 {
        let n = count.min(buffer.len() as u64) as usize;
        digest.update(&buffer[..n]);
        count -= n as u64;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::export::{FilePart, FileSource};

    fn publisher(development: bool, seed: u8) -> Publisher {
        let namespace = if development {
            "local-development"
        } else {
            "flare-tests"
        };
        Publisher::from_seed(
            &[seed; 32],
            namespace.into(),
            if development {
                namespace.into()
            } else {
                "https://huggingface.co/flare-tests/fixture".into()
            },
            development,
            None,
        )
        .unwrap()
    }
    fn profile(publisher: &Publisher) -> Profile {
        Profile {
            name: "Native publisher fixture".into(),
            namespace: publisher.namespace.clone(),
            public_key: publisher.public_key().try_into().unwrap(),
            development: publisher.development,
        }
    }
    fn fixture(directory: &Path) -> (wire::ModelIndex, Vec<FilePart>) {
        let mut index = wire::ModelIndex {
            version: 4,
            ..Default::default()
        };
        let mut payload = vec![];
        let mut cursor = 0;
        for rank in 0..2 {
            let mut bundle = wire::BundleIndex::default();
            for (name, bytes) in [
                ("weights", vec![rank as u8 + 1; 8193]),
                ("cubin", vec![9; protection::CHUNK as usize + 3]),
                ("global", vec![6; 13]),
            ] {
                cursor = package::align(cursor).unwrap();
                let path = directory.join(format!("rank-{rank}-{name}"));
                fs::write(&path, &bytes).unwrap();
                bundle.resources.insert(
                    name.into(),
                    wire::Span {
                        offset: cursor,
                        nbytes: bytes.len() as u64,
                        filename: String::new(),
                    },
                );
                payload.push(FilePart {
                    offset: cursor,
                    source: FileSource {
                        path,
                        offset: 0,
                        nbytes: bytes.len() as u64,
                    },
                });
                cursor += bytes.len() as u64;
            }
            bundle.allocations.push(wire::Allocation {
                name: "weights".into(),
                file: Some("weights".into()),
                nbytes: 8193,
                ..Default::default()
            });
            bundle.modules.push(wire::Module {
                name: "module".into(),
                file: "cubin".into(),
                globals: vec![wire::Global {
                    symbol: "constant".into(),
                    file: "global".into(),
                    nbytes: 13,
                    relocations: vec![],
                }],
            });
            index.ranks.push(bundle);
        }
        (index, payload)
    }
    fn multiple_executable_fixture(directory: &Path) -> (wire::ModelIndex, Vec<FilePart>) {
        let (mut index, mut payload) = fixture(directory);
        let mut cursor = payload
            .iter()
            .map(|part| part.offset + part.source.nbytes)
            .max()
            .unwrap();
        for (rank, bundle) in index.ranks.iter_mut().enumerate() {
            for resource in 0..4 {
                let bytes = vec![
                    rank as u8 + 10 + resource as u8;
                    protection::CHUNK as usize * 2 + 17 + resource * 31
                ];
                let name = format!("extra-cubin-{resource}");
                let path = directory.join(format!("rank-{rank}-{name}"));
                fs::write(&path, &bytes).unwrap();
                cursor = package::align(cursor).unwrap();
                bundle.resources.insert(
                    name.clone(),
                    wire::Span {
                        offset: cursor,
                        nbytes: bytes.len() as u64,
                        filename: String::new(),
                    },
                );
                payload.push(FilePart {
                    offset: cursor,
                    source: FileSource {
                        path,
                        offset: 0,
                        nbytes: bytes.len() as u64,
                    },
                });
                cursor += bytes.len() as u64;
                bundle.modules.push(wire::Module {
                    name: format!("extra-module-{resource}"),
                    file: name,
                    globals: vec![],
                });
            }
            let support = vec![0x7e; 4096];
            let path = directory.join(format!("rank-{rank}-support"));
            fs::write(&path, &support).unwrap();
            cursor = package::align(cursor).unwrap();
            bundle.resources.insert(
                "extra-support".into(),
                wire::Span {
                    offset: cursor,
                    nbytes: support.len() as u64,
                    filename: String::new(),
                },
            );
            bundle.modules[0].globals.push(wire::Global {
                symbol: "extra-support".into(),
                file: "extra-support".into(),
                nbytes: support.len() as u64,
                relocations: vec![],
            });
            payload.push(FilePart {
                offset: cursor,
                source: FileSource {
                    path,
                    offset: 0,
                    nbytes: support.len() as u64,
                },
            });
            cursor += support.len() as u64;
        }
        (index, payload)
    }

    fn decoded_resources(path: &Path, profile: &Profile) -> Vec<Vec<u8>> {
        let auth = authenticate(path, std::slice::from_ref(profile), false).unwrap();
        let budget = IoBudget::for_ranks(1).unwrap();
        let mut roles: Vec<_> = package::roles(&auth.index)
            .unwrap()
            .into_iter()
            .filter(|role| role.kind != 4)
            .collect();
        roles.sort_by_key(|role| (role.kind, role.owner, role.span.offset));
        let mut contents = Vec::with_capacity(roles.len());
        for role in roles {
            let resource = &auth.resources[&(role.kind, role.owner, role.span.offset)];
            let mut bytes = Vec::with_capacity(resource.plain_bytes as usize);
            if role.kind == 2 {
                let source = SourceRangeReader::new(
                    &auth.sources,
                    &budget,
                    &role.span.filename,
                    role.span.offset,
                    resource.stored_bytes,
                );
                let mut reader = helios_core::package::protection::open_compressed_reader(
                    source,
                    resource.stored_bytes,
                    resource.plain_bytes,
                    &resource.sha256,
                )
                .unwrap();
                std::io::Read::read_to_end(&mut reader, &mut bytes).unwrap();
            } else {
                bytes.resize(resource.plain_bytes as usize, 0);
                auth.sources
                    .read_at(
                        &budget,
                        &role.span.filename,
                        role.span.offset,
                        &mut bytes,
                    )
                    .unwrap();
            }
            contents.push(bytes);
        }
        contents
    }

    #[test]
    fn resign_preserves_multiple_dense_executables_and_supporting_bytes() {
        let directory = tempfile::tempdir().unwrap();
        let (index, payload) = multiple_executable_fixture(directory.path());
        let options = ExportOptions {
            cache: Some(directory.path().join("cache")),
            no_weights: false,
        };
        let development = publisher(true, 21);
        let source = directory.path().join("development.safetensors");
        package::publish(
            &source,
            index,
            payload,
            Some(&development),
            &options,
            64 << 30,
        )
        .unwrap();
        let original = helios_core::runtime_package::test_support::inspect(
            &source,
            "local-development",
            true,
            &[profile(&development)],
            None,
        )
        .unwrap();
        let original_bytes = decoded_resources(&source, &profile(&development));
        let promoted = directory.path().join("promoted.safetensors");
        let mut production = publisher(false, 43);
        resign(
            &source,
            &promoted,
            &mut production,
            profile(&development),
            &options,
            64 << 30,
        )
        .unwrap();
        let actual = helios_core::runtime_package::test_support::inspect(
            &promoted,
            "flare-tests",
            true,
            &[profile(&production)],
            None,
        )
        .unwrap();
        assert_eq!(original, actual);
        assert_eq!(
            original_bytes,
            decoded_resources(&promoted, &profile(&production))
        );
        assert_eq!(
            original_bytes,
            decoded_resources(&source, &profile(&development))
        );
        assert!(original.len() > 4);
    }

    fn incompressible_bytes(nbytes: usize) -> Vec<u8> {
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut bytes = Vec::with_capacity(nbytes);
        for _ in 0..nbytes {
            state ^= state << 7;
            state ^= state >> 9;
            state ^= state << 8;
            bytes.push((state >> 24) as u8);
        }
        bytes
    }

    #[test]
    fn convert_and_resign_preserve_large_incompressible_executables_and_reject_late_corruption() {
        let directory = tempfile::tempdir().unwrap();
        let (mut index, mut payload) = fixture(directory.path());
        let bytes = incompressible_bytes(protection::CHUNK as usize + 4096);
        let mut cursor = payload
            .iter()
            .map(|part| part.offset + part.source.nbytes)
            .max()
            .unwrap();
        for (rank, bundle) in index.ranks.iter_mut().enumerate() {
            let path = directory.path().join(format!("rank-{rank}-large-incompressible"));
            fs::write(&path, &bytes).unwrap();
            cursor = package::align(cursor).unwrap();
            bundle.resources.insert(
                "large-incompressible".into(),
                wire::Span {
                    offset: cursor,
                    nbytes: bytes.len() as u64,
                    filename: String::new(),
                },
            );
            bundle.modules.push(wire::Module {
                name: "large-incompressible".into(),
                file: "large-incompressible".into(),
                globals: vec![],
            });
            payload.push(FilePart {
                offset: cursor,
                source: FileSource {
                    path,
                    offset: 0,
                    nbytes: bytes.len() as u64,
                },
            });
            cursor += bytes.len() as u64;
        }
        let options = ExportOptions {
            cache: Some(directory.path().join("cache")),
            no_weights: false,
        };
        let development = publisher(true, 31);
        let source = directory.path().join("development-large.safetensors");
        package::publish(&source, index, payload, Some(&development), &options, 64 << 30).unwrap();
        let original = decoded_resources(&source, &profile(&development));
        let converted = directory.path().join("converted-large.safetensors");
        convert(
            &source,
            &converted,
            &[profile(&development)],
            options.cache.as_deref(),
            protection::CHUNK + 16384,
        )
        .unwrap();
        assert_eq!(original, decoded_resources(&converted, &profile(&development)));
        let promoted = directory.path().join("promoted-large.safetensors");
        let mut production = publisher(false, 47);
        resign(
            &converted,
            &promoted,
            &mut production,
            profile(&development),
            &options,
            64 << 30,
        )
        .unwrap();
        assert_eq!(original, decoded_resources(&promoted, &profile(&production)));

        let (mut stored, resource) =
            helios_core::package::protection::compress(&bytes).unwrap();
        assert!(stored.len() > protection::CHUNK as usize);
        stored[protection::CHUNK as usize + 5] ^= 0x5a;
        let mut reader = helios_core::package::protection::open_compressed_reader(
            std::io::Cursor::new(stored),
            resource.stored_nbytes,
            bytes.len() as u64,
            &resource.chunk_sha256,
        )
        .unwrap();
        let mut decoded = Vec::new();
        let error = std::io::Read::read_to_end(&mut reader, &mut decoded)
            .expect_err("corrupted later compressed chunk must reject");
        assert!(!error.to_string().is_empty(), "{error}");
    }

    #[test]
    fn native_publication_is_readable_and_reuses_stored_blobs_without_writes() {
        let directory = tempfile::tempdir().unwrap();
        let (index, payload) = fixture(directory.path());
        let options = ExportOptions {
            cache: Some(directory.path().join("cache")),
            no_weights: false,
        };
        let signer = publisher(false, 42);
        let first = directory.path().join("first.safetensors");
        package::publish(
            &first,
            index.clone(),
            payload.clone(),
            Some(&signer),
            &options,
            64 << 30,
        )
        .unwrap();
        let first_verified = VerifiedPackage::open(&first, &[profile(&signer)], None).unwrap();
        first_verified.verify_stored_data().unwrap();
        let expected = helios_core::runtime_package::test_support::inspect(
            &first,
            "flare-tests",
            true,
            &[profile(&signer)],
            None,
        )
        .unwrap();
        assert_eq!(expected.len(), 4);
        let second = directory.path().join("second.safetensors");
        let strict = ExportOptions {
            no_weights: true,
            ..options.clone()
        };
        let stats =
            package::publish(&second, index, payload, Some(&signer), &strict, 64 << 30).unwrap();
        assert!(stats.reused >= 2 * 8193);
        assert_eq!(
            expected,
            helios_core::runtime_package::test_support::inspect(
                &second,
                "flare-tests",
                true,
                &[profile(&signer)],
                None
            )
            .unwrap()
        );
        let second_index = package::read_index(&second).unwrap().0;
        let expected_reused: u64 = package::groups(&second_index, 64 << 30)
            .unwrap()
            .iter()
            .map(|g| g.nbytes)
            .sum();
        assert_eq!(stats.written, 0);
        assert_eq!(stats.reused, expected_reused);
        VerifiedPackage::open(&second, &[profile(&signer)], None)
            .unwrap()
            .verify_stored_data()
            .unwrap();
    }
    #[test]
    fn publication_preserves_owner_hashes_across_multiple_cold_and_reused_shards() {
        let directory = tempfile::tempdir().unwrap();
        let (mut index, mut payload) = fixture(directory.path());
        let weights = directory.path().join("additional-weights");
        fs::write(&weights, vec![37; 1 << 20]).unwrap();
        let mut cursor = payload
            .iter()
            .map(|p| p.offset + p.source.nbytes)
            .max()
            .unwrap();
        for bundle in &mut index.ranks {
            for i in 0..9 {
                cursor = package::align(cursor).unwrap();
                let name = format!("additional-{i}");
                bundle.resources.insert(
                    name.clone(),
                    wire::Span {
                        offset: cursor,
                        nbytes: 1 << 20,
                        filename: String::new(),
                    },
                );
                bundle.allocations.push(wire::Allocation {
                    name: name.clone(),
                    file: Some(name),
                    nbytes: 1 << 20,
                    ..Default::default()
                });
                payload.push(FilePart {
                    offset: cursor,
                    source: FileSource {
                        path: weights.clone(),
                        offset: 0,
                        nbytes: 1 << 20,
                    },
                });
                cursor += 1 << 20;
            }
        }
        let options = ExportOptions {
            cache: Some(directory.path().join("cache")),
            no_weights: false,
        };
        let signer = publisher(false, 42);
        for no_weights in [false, true] {
            let path = directory.path().join(format!("{no_weights}.safetensors"));
            package::publish(
                &path,
                index.clone(),
                payload.clone(),
                Some(&signer),
                &ExportOptions {
                    no_weights,
                    ..options.clone()
                },
                protection::CHUNK + 16384,
            )
            .unwrap();
            let package = VerifiedPackage::open(&path, &[profile(&signer)], None).unwrap();
            package.verify_stored_data().unwrap();
            let inspected = helios_core::runtime_package::test_support::inspect(
                &path,
                "flare-tests",
                true,
                &[profile(&signer)],
                None,
            )
            .unwrap();
            assert_eq!(inspected.len(), 4);
            assert!(
                package::groups(
                    &package::read_index(&path).unwrap().0,
                    protection::CHUNK + 16384
                )
                .unwrap()
                .iter()
                .filter(|g| g.roles[0].kind == 4)
                .count()
                    >= 6
            );
        }
    }
    #[cfg(unix)]
    #[test]
    fn publication_bounds_handles_for_more_than_4096_distinct_sources() {
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "export::conversion::tests::distinct_source_publication_child",
                "--ignored",
            ])
            .status()
            .unwrap();
        assert!(status.success());
    }
    #[test]
    fn publication_rejects_changed_and_replaced_sources_after_preparation() {
        for replace in [false, true] {
            let directory = tempfile::tempdir().unwrap();
            let (index, payload) = fixture(directory.path());
            let original = payload[0].source.path.clone();
            let signer = publisher(false, 42);
            let budget = IoBudget::for_ranks(1).unwrap();
            let progress = helios_core::startup::StartupProgress::stderr_rows();
            let phase = progress.phase("Prepare fixture resources", None);
            let protected = protection::prepare(
                index,
                payload,
                &signer,
                directory.path(),
                64 << 30,
                "fixture",
                budget.clone(),
                &phase,
            )
            .unwrap();
            if replace {
                fs::rename(&original, directory.path().join("replaced-source")).unwrap();
            }
            fs::write(&original, vec![123; 8193]).unwrap();
            let store = BlobStore::new(&directory.path().join("cache"), budget).unwrap();
            let error =
                package::publish_groups(&store, &protected, false, 64 << 30, &phase).unwrap_err();
            assert!(error.to_string().contains("changed"), "{error}");
        }
    }
    #[cfg(unix)]
    #[test]
    #[ignore]
    fn distinct_source_publication_child() {
        let mut limit = libc::rlimit {
            rlim_cur: 0,
            rlim_max: 0,
        };
        assert_eq!(
            unsafe { libc::getrlimit(libc::RLIMIT_NOFILE, &mut limit) },
            0
        );
        limit.rlim_cur = limit.rlim_cur.min(4096);
        assert_eq!(unsafe { libc::setrlimit(libc::RLIMIT_NOFILE, &limit) }, 0);
        let directory = tempfile::tempdir().unwrap();
        let mut index = wire::ModelIndex {
            version: 4,
            ranks: vec![wire::BundleIndex::default()],
            ..Default::default()
        };
        let mut payload = Vec::new();
        for i in 0..4200u64 {
            let name = format!("weight-{i}");
            let path = directory.path().join(&name);
            fs::write(&path, i.to_le_bytes()).unwrap();
            index.ranks[0].resources.insert(
                name.clone(),
                wire::Span {
                    offset: i * 4096,
                    nbytes: 8,
                    filename: String::new(),
                },
            );
            index.ranks[0].allocations.push(wire::Allocation {
                name: name.clone(),
                file: Some(name),
                nbytes: 8,
                ..Default::default()
            });
            payload.push(FilePart {
                offset: i * 4096,
                source: FileSource {
                    path,
                    offset: 0,
                    nbytes: 8,
                },
            });
        }
        let path = directory.path().join("catalog.safetensors");
        let signer = publisher(false, 42);
        package::publish(
            &path,
            index,
            payload,
            Some(&signer),
            &ExportOptions {
                cache: Some(directory.path().join("cache")),
                no_weights: false,
            },
            64 << 30,
        )
        .unwrap();
        VerifiedPackage::open(&path, &[profile(&signer)], None)
            .unwrap()
            .verify_stored_data()
            .unwrap();
    }
    #[test]
    fn signed_legacy_cluster_custom_primary_converts_sparse_companions() {
        let directory = tempfile::tempdir().unwrap();
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../helios/tests/fixtures/legacy-cluster-custom");
        let source_dir = directory.path().join("source");
        fs::create_dir_all(&source_dir).unwrap();
        for entry in fs::read_dir(fixture).unwrap() {
            let entry = entry.unwrap();
            let name = entry.file_name();
            let target = if name == "model.safetensors" {
                source_dir.join("custom-name.safetensors")
            } else {
                source_dir.join(name)
            };
            fs::copy(entry.path(), target).unwrap();
        }
        let source = source_dir.join("custom-name.safetensors");
        let signer = publisher(false, 42);
        let target = directory.path().join("converted.safetensors");
        convert(
            &source,
            &target,
            &[profile(&signer)],
            Some(directory.path().join("cache").as_path()),
            protection::CHUNK + 16384,
        )
        .unwrap();
        assert!(source.is_file());
        assert_eq!(
            Container::open(&source).unwrap().envelope,
            Container::open(&target).unwrap().envelope
        );
        VerifiedPackage::open(&target, &[profile(&signer)], Some(1))
            .unwrap()
            .verify_stored_data()
            .unwrap();
    }

    #[test]
    fn signed_legacy_custom_primary_converts_without_rewriting_its_container() {
        let directory = tempfile::tempdir().unwrap();
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../helios/tests/fixtures/legacy-custom/model.safetensors");
        let source = directory.path().join("custom-name.safetensors");
        fs::copy(&fixture, &source).unwrap();
        let signer = publisher(false, 42);
        let target = directory.path().join("converted.safetensors");
        convert(
            &source,
            &target,
            &[profile(&signer)],
            Some(directory.path().join("cache").as_path()),
            protection::CHUNK + 16384,
        )
        .unwrap();
        assert!(
            source.is_file(),
            "the fixture's custom primary must remain untouched"
        );
        assert_eq!(
            Container::open(&source).unwrap().envelope,
            Container::open(&target).unwrap().envelope
        );
        VerifiedPackage::open(&target, &[profile(&signer)], None)
            .unwrap()
            .verify_stored_data()
            .unwrap();
    }

    #[test]
    fn conversion_preserves_envelope_and_promotion_preserves_plaintext() {
        let directory = tempfile::tempdir().unwrap();
        let (index, payload) = fixture(directory.path());
        let options = ExportOptions {
            cache: Some(directory.path().join("cache")),
            no_weights: false,
        };
        let development = publisher(true, 17);
        let source = directory.path().join("development.safetensors");
        package::publish(
            &source,
            index,
            payload,
            Some(&development),
            &options,
            64 << 30,
        )
        .unwrap();
        let target = directory.path().join("converted.safetensors");
        convert(
            &source,
            &target,
            &[profile(&development)],
            options.cache.as_deref(),
            protection::CHUNK + 16384,
        )
        .unwrap();
        assert_eq!(
            Container::open(&source).unwrap().envelope,
            Container::open(&target).unwrap().envelope
        );
        let original = helios_core::runtime_package::test_support::inspect(
            &target,
            "local-development",
            true,
            &[profile(&development)],
            None,
        )
        .unwrap();
        let mut production = publisher(false, 42);
        let promoted = directory.path().join("production.safetensors");
        resign(
            &target,
            &promoted,
            &mut production,
            profile(&development),
            &options,
            64 << 30,
        )
        .unwrap();
        let actual = helios_core::runtime_package::test_support::inspect(
            &promoted,
            "flare-tests",
            true,
            &[profile(&production)],
            None,
        )
        .unwrap();
        assert_eq!(
            original.iter().map(|(_, h)| h).collect::<Vec<_>>(),
            actual.iter().map(|(_, h)| h).collect::<Vec<_>>()
        );
    }
    #[test]
    fn absent_required_weights_and_mutated_resources_never_publish_primary() {
        let directory = tempfile::tempdir().unwrap();
        let (index, payload) = fixture(directory.path());
        let options = ExportOptions {
            cache: Some(directory.path().join("cache")),
            no_weights: true,
        };
        let signer = publisher(false, 42);
        let destination = directory.path().join("missing.safetensors");
        let error = package::publish(
            &destination,
            index.clone(),
            payload.clone(),
            Some(&signer),
            &options,
            64 << 30,
        )
        .unwrap_err();
        assert!(error.to_string().contains("no-weights"));
        assert!(!destination.exists());
        let source = directory.path().join("source.safetensors");
        package::publish(
            &source,
            index,
            payload,
            Some(&signer),
            &ExportOptions {
                no_weights: false,
                ..options.clone()
            },
            64 << 30,
        )
        .unwrap();
        let container = Container::open(&source).unwrap();
        let layout = container.layout.unwrap();
        let blob = directory.path().join(&layout.files[""][0].1);
        let mut permissions = fs::metadata(&blob).unwrap().permissions();
        permissions.set_readonly(false);
        fs::set_permissions(&blob, permissions).unwrap();
        let file = OpenOptions::new().write(true).open(&blob).unwrap();
        crate::export::io::write_all_at(&file, b"wrong", 4096).unwrap();
        let output = directory.path().join("invalid.safetensors");
        let error = convert(
            &source,
            &output,
            &[profile(&signer)],
            options.cache.as_deref(),
            64 << 30,
        )
        .unwrap_err();
        assert!(error.to_string().contains("digest"));
        assert!(!output.exists());
    }
    /// Publish one signed payload package whose model identity is `t/model`.
    fn payload_package(root: &std::path::Path) -> (std::path::PathBuf, Publisher) {
        use helios_core::package::{PayloadId, container, export as payload_export, schema::wire};
        let publisher = Publisher::from_seed(
            &[7u8; 32],
            "t".into(),
            "https://huggingface.co/t/model".into(),
            false,
            None,
        )
        .unwrap();
        let directory = root.join("package");
        std::fs::create_dir_all(&directory).unwrap();
        let weights = b"convert cache weights".to_vec();
        let weights_id = PayloadId::of(&weights);
        let weights_path = root.join("weights.bin");
        std::fs::write(&weights_path, &weights).unwrap();
        let graph = root.join("graph.bin");
        std::fs::write(&graph, b"graph-bytes").unwrap();
        let runtime = helios_core::runtime_package::wire::ServingContract {
            model_id: "t/model".into(),
            identity: "t/model".into(),
            max_sequence_tokens: 8,
            prefill_chunk_size: 1,
            pages: 1,
            state_rows: 1,
            page_tokens: 1,
            model_contract: Some(helios_core::runtime_package::wire::ModelContract {
                vocabulary: 8,
                graphs: vec![helios_core::runtime_package::wire::GraphContract {
                    graph: "decode".into(),
                    ..Default::default()
                }],
                ..Default::default()
            }),
            ..Default::default()
        };
        let rank = wire::RankBody {
            rank: 0,
            deployment_id: "sm120-tp1".into(),
            context: Some(Default::default()),
            allocations: vec![wire::Allocation {
                name: "device-0".into(),
                nbytes: weights.len() as u64,
                alignment: 4096,
                ..Default::default()
            }],
            initializers: vec![wire::Initializer {
                allocation: "device-0".into(),
                destination_offset: 0,
                source: Some(wire::PayloadUse {
                    slice: Some(wire::PayloadSlice {
                        payload_id: weights_id.as_bytes().to_vec(),
                        offset: 0,
                        nbytes: weights.len() as u64,
                    }),
                    role: wire::PayloadRole::Weights as i32,
                }),
            }],
            graphs: vec![wire::Graph {
                name: "decode".into(),
                resource: 0,
                node_count: 1,
                ..Default::default()
            }],
            ..Default::default()
        };
        let request = payload_export::PublishRequest {
            package: wire::PackageBody {
                model_identity: "t/model".into(),
                hardware_identity: "sm120".into(),
                frontend: Some(helios_core::runtime_package::wire::FrontendSettings::default()),
                sampling_defaults: Some(
                    helios_core::runtime_package::wire::DefaultSamplingParameters::default(),
                ),
                publisher: Some(wire::PublisherMetadata {
                    namespace: "t".into(),
                    official_source: "https://huggingface.co/t/model".into(),
                    ..Default::default()
                }),
                ..Default::default()
            },
            payloads: vec![],
            deployments: vec![payload_export::PublishDeployment {
                payloads: vec![],
                body: wire::DeploymentBody {
                    deployment_id: "sm120-tp1".into(),
                    architecture: vec![12, 0],
                    tensor_parallel: 1,
                    contract: Some(runtime),
                    ..Default::default()
                },
                ranks: vec![payload_export::PublishRank {
                    body: rank,
                    payloads: vec![payload_export::PublishPayload {
                        role: wire::PayloadRole::Weights,
                        relative: "weights/layer-000.safetensors".into(),
                        tensor: Some(container::COMPANION_TENSOR.into()),
                        source: container::Slice::new(
                            std::sync::Arc::new(std::fs::File::open(&weights_path).unwrap()),
                            0,
                            weights.len() as u64,
                        )
                        .unwrap(),
                        id: weights_id,
                    }],
                    resources: vec![payload_export::PublishResource {
                        kind: wire::ExecutableKind::Graph,
                        path: graph,
                        nbytes: 11,
                    }],
                }],
            }],
            package_id: [51u8; 32],
            public_key: publisher.public_key().try_into().unwrap(),
            accepted: "t".into(),
            no_weights: false,
        };
        payload_export::publish(
            &request,
            publisher.signing_key(),
            payload_export::Destination::Directory(&directory),
        )
        .unwrap();
        (directory.join("model.safetensors"), publisher)
    }

    fn profiles(publisher: &Publisher) -> String {
        serde_json::json!([{
            "name": "t",
            "namespace": "t",
            "public_key": publisher.public_key(),
            "development": false,
        }])
        .to_string()
    }

    fn fixed_time() -> SystemTime {
        std::time::UNIX_EPOCH + std::time::Duration::from_secs(1_792_228_980)
    }

    #[test]
    fn cache_conversion_selects_only_after_a_successful_standalone_export() {
        use helios_core::package::cache::{PackageSource, Root};
        let scratch = tempfile::tempdir().unwrap();
        let (source, publisher) = payload_package(scratch.path());
        let cache = scratch.path().join("cache");
        let model = helios_core::package::cache::model_key("t/model").unwrap();
        let publishers = payload_publishers(&profiles(&publisher)).unwrap();
        let selection = helios_core::package::protection::Selection::Deployment {
            deployment: 0,
            rank: None,
        };
        {
            // The prior default is a verified payload entry.
            let source_package = PackageSource::open(&source, &publishers, selection).unwrap();
            let root = Root::open(&cache).unwrap();
            root.install_package(source_package, &model, "prior")
                .unwrap();
        }

        // A failed standalone export must not select the new entry or void the old one.
        let occupied = scratch.path().join("occupied");
        std::fs::create_dir_all(&occupied).unwrap();
        std::fs::write(occupied.join("keep"), b"keep").unwrap();
        let error = convert_cache_at(
            &source.display().to_string(),
            &cache.display().to_string(),
            &occupied.display().to_string(),
            &profiles(&publisher),
            "directory",
            fixed_time(),
        )
        .unwrap_err();
        assert_eq!(
            error.to_string(),
            format!("export destination {} already exists", occupied.display())
        );
        assert_eq!(std::fs::read(occupied.join("keep")).unwrap(), b"keep");
        let root = Root::open(&cache).unwrap();
        assert_eq!(root.ref_entry(&model).unwrap().as_deref(), Some("prior"));
        assert_eq!(root.versions(&model).unwrap(), vec!["prior".to_string()]);

        // A successful conversion selects the converted entry.
        let destination = scratch.path().join("converted");
        convert_cache_at(
            &source.display().to_string(),
            &cache.display().to_string(),
            &destination.display().to_string(),
            &profiles(&publisher),
            "directory",
            fixed_time(),
        )
        .unwrap();
        // The exported tree verifies as a signed package and still matches the
        // installed entry's payload closure.
        let exported = PackageSource::open(
            &destination.join("model.safetensors"),
            &publishers,
            selection,
        )
        .unwrap();
        let root = Root::open(&cache).unwrap();
        let current = root.ref_entry(&model).unwrap().unwrap();
        assert_ne!(current, "prior");
        assert_eq!(current, "unknown-device-tp1-202610170923");
        let selected = root.entry(&model, &current).unwrap();
        root.verify_ready(&selected, &exported).unwrap();
        assert!(root.versions(&model).unwrap().contains(&current));
    }

    #[test]
    fn payload_cache_conversion_reserves_same_minute_names() {
        let scratch = tempfile::tempdir().unwrap();
        let (source, publisher) = payload_package(scratch.path());
        let cache = scratch.path().join("cache");
        let model = helios_core::package::cache::model_key("t/model").unwrap();
        for _ in 0..2 {
            convert_cache_at(
                &source.display().to_string(),
                &cache.display().to_string(),
                "",
                &profiles(&publisher),
                "directory",
                fixed_time(),
            )
            .unwrap();
        }
        let root = helios_core::package::cache::Root::open(&cache).unwrap();
        let mut versions = root.versions(&model).unwrap();
        versions.sort();
        assert_eq!(
            versions,
            vec![
                "unknown-device-tp1-202610170923".to_string(),
                "unknown-device-tp1-202610170923-2".to_string(),
            ]
        );
        assert_eq!(
            root.ref_entry(&model).unwrap().as_deref(),
            Some("unknown-device-tp1-202610170923-2")
        );
    }

    #[test]
    fn legacy_cache_conversion_names_unknown_device_and_reserves_collisions() {
        let scratch = tempfile::tempdir().unwrap();
        let source = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../helios/tests/fixtures/legacy-cluster-custom/model.safetensors");
        let signer = publisher(false, 42);
        let profiles = [profile(&signer)];
        let cache = scratch.path().join("cache");
        for _ in 0..2 {
            convert_legacy_cache(
                &source.display().to_string(),
                &cache.display().to_string(),
                &profiles,
                protection::CHUNK + 16384,
                fixed_time(),
            )
            .unwrap();
        }
        let root = helios_core::package::cache::Root::open(&cache).unwrap();
        let models = root.models().unwrap();
        assert_eq!(models.len(), 1);
        let mut versions = root.versions(&models[0]).unwrap();
        versions.sort();
        assert_eq!(
            versions,
            vec![
                "unknown-device-tp2-202610170923".to_string(),
                "unknown-device-tp2-202610170923-2".to_string(),
            ]
        );
        assert_eq!(
            root.ref_entry(&models[0]).unwrap().as_deref(),
            Some("unknown-device-tp2-202610170923-2")
        );
    }

    #[test]
    fn failed_legacy_cache_conversion_preserves_the_selected_default() {
        let scratch = tempfile::tempdir().unwrap();
        let source = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../helios/tests/fixtures/legacy-cluster-custom/model.safetensors");
        let signer = publisher(false, 42);
        let cache = scratch.path().join("cache");
        convert_legacy_cache(
            &source.display().to_string(),
            &cache.display().to_string(),
            &[profile(&signer)],
            protection::CHUNK + 16384,
            fixed_time(),
        )
        .unwrap();
        let root = helios_core::package::cache::Root::open(&cache).unwrap();
        let model = root.models().unwrap()[0].clone();
        let selected = root.ref_entry(&model).unwrap();
        let versions = root.versions(&model).unwrap();

        let other = publisher(false, 43);
        assert!(
            convert_legacy_cache(
                &source.display().to_string(),
                &cache.display().to_string(),
                &[profile(&other)],
                protection::CHUNK + 16384,
                fixed_time(),
            )
            .is_err()
        );
        assert_eq!(root.ref_entry(&model).unwrap(), selected);
        assert_eq!(root.versions(&model).unwrap(), versions);
    }
}

fn decode_profiles(encoded: &str) -> Result<Vec<Profile>> {
    let mut values: Vec<serde_json::Value> = serde_json::from_str(encoded)?;
    for value in &mut values {
        if let Some(object) = value.as_object_mut() {
            if !object.contains_key("name") {
                let namespace = object
                    .get("namespace")
                    .cloned()
                    .unwrap_or_else(|| "Package publisher".into());
                object.insert("name".into(), namespace);
            }
        }
    }
    Ok(serde_json::from_value(values.into())?)
}
