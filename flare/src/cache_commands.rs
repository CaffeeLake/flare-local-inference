//! Prepared-model cache commands: list, inspect, select, remove, and gc.

use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

use flare_helios::package::cache::{self, Root};

use crate::cli::{
    CacheArgs, CacheCommand, CacheGc, CacheInspect, CacheList, CacheRemove, CacheSpec,
};

pub fn run(args: CacheArgs) -> Result<(), Box<dyn std::error::Error>> {
    let root = Root::open(
        crate::cache_support::model_root(args.model_cache.as_deref())
            .map_err(|error| error.to_string())?,
    )?;
    match args.command {
        CacheCommand::List(list) => list_entries(&root, list),
        CacheCommand::Inspect(inspect) => inspect_entry(&root, inspect),
        CacheCommand::Select(spec) => select(&root, spec),
        CacheCommand::Remove(remove) => remove_entry(&root, remove),
        CacheCommand::Gc(gc) => collect(&root, gc),
    }
}

fn split(spec: &str) -> Result<(String, String), Box<dyn std::error::Error>> {
    let (model, version) = spec.rsplit_once('@').ok_or("expected MODEL@VERSION")?;
    if model.is_empty() || version.is_empty() {
        return Err("expected MODEL@VERSION".into());
    }
    Ok((model.to_owned(), version.to_owned()))
}

/// Unique-inode accounting for one entry tree.
///
/// Logical bytes are the sizes recorded in the manifest and in the files. Physical bytes are the
/// allocated blocks, so sparse files and hard links are not over-counted as disk usage. On
/// platforms without block accounting the physical figures are reported as unavailable.
#[derive(Default)]
struct Usage {
    logical_bytes: u64,
    physical_bytes: Option<u64>,
    shared_logical_bytes: u64,
    shared_physical_bytes: Option<u64>,
    single_physical_bytes: Option<u64>,
    files: usize,
    shared_objects: usize,
}

#[cfg(unix)]
fn usage(directory: &Path) -> Result<Usage, Box<dyn std::error::Error>> {
    use std::os::unix::fs::MetadataExt;
    let mut seen: BTreeMap<(u64, u64), (u64, u64, u64)> = BTreeMap::new();
    let mut stack = vec![directory.to_path_buf()];
    while let Some(path) = stack.pop() {
        for entry in std::fs::read_dir(&path)? {
            let entry = entry?;
            let metadata = std::fs::symlink_metadata(entry.path())?;
            if metadata.is_dir() {
                stack.push(entry.path());
                continue;
            }
            if !metadata.is_file() {
                continue;
            }
            seen.insert(
                (metadata.dev(), metadata.ino()),
                (metadata.len(), metadata.blocks() * 512, metadata.nlink()),
            );
        }
    }
    let mut usage = Usage {
        physical_bytes: Some(0),
        shared_physical_bytes: Some(0),
        single_physical_bytes: Some(0),
        ..Default::default()
    };
    for (logical, physical, links) in seen.values() {
        usage.logical_bytes += logical;
        usage.physical_bytes = usage.physical_bytes.map(|bytes| bytes + physical);
        usage.files += 1;
        if *links > 1 {
            usage.shared_logical_bytes += logical;
            usage.shared_physical_bytes = usage.shared_physical_bytes.map(|bytes| bytes + physical);
            usage.shared_objects += 1;
        } else {
            usage.single_physical_bytes = usage.single_physical_bytes.map(|bytes| bytes + physical);
        }
    }
    Ok(usage)
}

#[cfg(not(unix))]
fn usage(directory: &Path) -> Result<Usage, Box<dyn std::error::Error>> {
    let mut seen: BTreeMap<PathBuf, u64> = BTreeMap::new();
    let mut stack = vec![directory.to_path_buf()];
    while let Some(path) = stack.pop() {
        for entry in std::fs::read_dir(&path)? {
            let entry = entry?;
            let metadata = std::fs::symlink_metadata(entry.path())?;
            if metadata.is_dir() {
                stack.push(entry.path());
                continue;
            }
            if metadata.is_file() {
                seen.insert(entry.path(), metadata.len());
            }
        }
    }
    let mut usage = Usage::default();
    for (_, logical) in seen {
        usage.logical_bytes += logical;
        usage.files += 1;
    }
    Ok(usage)
}

fn cache_key(input: &str) -> Result<String, Box<dyn std::error::Error>> {
    cache::model_key(input).map_err(|error| error.to_string().into())
}

/// Accepted user syntax is the model identity (for example `owner/name`); a cache key is
/// accepted only when the identity it decodes to names the same entry, so both spellings resolve
/// deterministically.
fn resolve_model(root: &Root, input: &str) -> Result<(String, String), Box<dyn std::error::Error>> {
    let key = cache_key(input)?;
    if !root.versions(&key)?.is_empty() {
        return Ok((key, input.to_owned()));
    }
    if let Ok(identity) = cache::model_identity(input) {
        let decoded = cache_key(&identity)?;
        if decoded != key && !root.versions(&decoded)?.is_empty() {
            return Ok((decoded, identity));
        }
    }
    Ok((key, input.to_owned()))
}

fn list_entries(root: &Root, list: CacheList) -> Result<(), Box<dyn std::error::Error>> {
    let models = match &list.model {
        Some(model) => vec![resolve_model(root, model)?],
        None => root
            .models()?
            .into_iter()
            .map(|key| {
                let identity = cache::model_identity(&key).unwrap_or_else(|_| key.clone());
                (key, identity)
            })
            .collect(),
    };
    let mut documents = Vec::new();
    let mut lines = Vec::new();
    for (model, identity) in models {
        let default = root.ref_entry(&model)?;
        let versions = root.versions(&model)?;
        let mut entries = Vec::new();
        for version in versions {
            let selected = default.as_deref() == Some(version.as_str());
            let report = (|| -> Result<(bool, flare_helios::package::cache::EntrySummary, Usage), Box<dyn std::error::Error>> {
                let leased = root.is_leased(&model, &version)?;
                let entry = root.entry(&model, &version)?;
                let summary = root.summarize_entry(&entry)?;
                let usage = usage(entry.directory())?;
                Ok((leased, summary, usage))
            })();
            let (leased, summary, usage) = match report {
                Ok(values) => values,
                Err(error) => {
                    // One unreadable entry must not hide the rest of the cache.
                    lines.push(format!("{identity}@{version} (unreadable)"));
                    lines.push(format!("  error: {error}"));
                    entries.push(serde_json::json!({
                        "model": identity,
                        "model_key": model,
                        "version": version,
                        "default": selected,
                        "error": error.to_string(),
                    }));
                    continue;
                }
            };
            // Physical reclamation is only attributable when this entry owns the only link to
            // its bytes and nothing leases it. Otherwise report why it is unavailable.
            let physical = usage.physical_bytes;
            let (reclaimable, reclaimable_note) = if leased {
                (None, Some("leased by a running process"))
            } else if selected {
                (None, Some("selected default"))
            } else if physical.is_none() {
                (
                    None,
                    Some("allocated-block accounting is unavailable on this platform"),
                )
            } else if usage.shared_objects > 0 {
                (
                    usage.single_physical_bytes,
                    Some(
                        "allocated bytes of single-linked inodes only; shared or externally hardlinked objects are excluded",
                    ),
                )
            } else {
                (physical, None)
            };
            lines.push(format!(
                "{identity}@{version}{}{} [{form}]",
                if selected { " (default)" } else { "" },
                if leased { " (leased)" } else { "" },
                form = summary.form
            ));
            for deployment in &summary.deployments {
                lines.push(format!(
                    "  deployment {} tp{}{} pages {:?} state_rows {:?} max_tokens {:?} prefill {:?}",
                    deployment.id,
                    deployment.tensor_parallel,
                    if deployment.ready { "" } else { " (metadata only)" },
                    deployment.pages,
                    deployment.state_rows,
                    deployment.max_sequence_tokens,
                    deployment.prefill_chunk_size
                ));
            }
            lines.push(format!(
                "  logical {}, disk logical {} B, disk physical {}, shared {} B, reclaimable {}",
                if summary.form == "legacy" {
                    "unavailable (legacy container)".to_string()
                } else {
                    format!("{} B", summary.logical_bytes)
                },
                usage.logical_bytes,
                physical.map_or("unavailable".to_string(), |bytes| format!("{bytes} B")),
                usage.shared_logical_bytes,
                reclaimable.map_or("unavailable".to_string(), |bytes| format!("{bytes} B"))
            ));
            if let Some(note) = reclaimable_note {
                lines.push(format!("  reclaimable note: {note}"));
            }
            entries.push(serde_json::json!({
                "model": identity,
                "model_key": model,
                "version": version,
                "default": selected,
                "leased": leased,
                "metadata": "unverified",
                "form": summary.form,
                "format_version": summary.format_version,
                "model_identity": summary.model_identity,
                "hardware_identity": summary.hardware_identity,
                "payloads": summary.payloads,
                "logical_bytes": summary.logical_bytes,
                "disk_logical_bytes": usage.logical_bytes,
                "disk_physical_bytes": physical,
                "shared_logical_bytes": usage.shared_logical_bytes,
                "shared_physical_bytes": usage.shared_physical_bytes,
                "files": usage.files,
                "shared_objects": usage.shared_objects,
                "reclaimable_bytes": reclaimable,
                "reclaimable_note": reclaimable_note,
                "deployments": summary.deployments.iter().map(|deployment| serde_json::json!({
                    "id": deployment.id,
                    "tensor_parallel": deployment.tensor_parallel,
                    "ready": deployment.ready,
                    "pages": deployment.pages,
                    "state_rows": deployment.state_rows,
                    "max_sequence_tokens": deployment.max_sequence_tokens,
                    "prefill_chunk_size": deployment.prefill_chunk_size,
                })).collect::<Vec<_>>(),
            }));
        }
        documents.push(serde_json::json!({
            "model": identity,
            "model_key": model,
            "default": default,
            "versions": entries,
        }));
    }
    if list.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "cache": root.path().display().to_string(),
                "models": documents,
            }))?
        );
    } else if lines.is_empty() {
        println!("No cached models in {}", root.path().display());
    } else {
        for line in lines {
            println!("{line}");
        }
    }
    Ok(())
}

fn inspect_entry(root: &Root, inspect: CacheInspect) -> Result<(), Box<dyn std::error::Error>> {
    let (spec, version) = split(&inspect.spec)?;
    let (model, _) = resolve_model(root, &spec)?;
    let entry = root.entry(&model, &version)?;
    // The entry lease stays alive while the shared inspection path reads the package. Inspect
    // loading takes the primary file, so resolve the format's primary inside the entry directory.
    let directory = entry.directory();
    // Resolve the shared canonical primary first: legacy entries may have been imported from a
    // custom primary basename, and payload entries use model.safetensors.
    let primary = root.primary(&entry)?;
    let format = flare_helios::runtime_package::NativeResources::package_format(&primary)?;
    let package = match format {
        flare_helios::runtime_package::PackageFormat::Payload => {
            flare_helios::package::reader::Metadata::open(directory)?.path
        }
        flare_helios::runtime_package::PackageFormat::Legacy => primary.clone(),
    };
    if !package.is_file() {
        return Err(format!("cache entry {model}@{version} has no primary file").into());
    }
    crate::inspect::run(crate::cli::Inspect {
        package,
        json: inspect.json,
        plain: inspect.plain,
        memory_map: inspect.memory_map,
        graphs: inspect.graphs,
        frontend: inspect.frontend,
        all: inspect.all,
        unverified: inspect.unverified,
        rank: inspect.rank,
        tensor_parallel: inspect.tensor_parallel,
        publisher_keys: inspect.publisher_keys,
        trust_store: inspect.trust_store,
    })
}

fn select(root: &Root, spec: CacheSpec) -> Result<(), Box<dyn std::error::Error>> {
    let (spec, version) = split(&spec.spec)?;
    let (model, identity) = resolve_model(root, &spec)?;
    root.publish(&model, &version)?;
    println!("{identity}@{version} is now the default");
    Ok(())
}

fn remove_entry(root: &Root, remove: CacheRemove) -> Result<(), Box<dyn std::error::Error>> {
    let (spec, version) = split(&remove.spec)?;
    let (model, identity) = resolve_model(root, &spec)?;
    root.remove(&model, &version, remove.clear_default)?;
    println!("Removed {identity}@{version}");
    Ok(())
}

fn collect(root: &Root, gc: CacheGc) -> Result<(), Box<dyn std::error::Error>> {
    let collected = root.collect()?;
    if gc.json {
        println!(
            "{}",
            serde_json::json!({
                "staging_trees": collected.staging,
                "payload_objects": collected.objects,
                "legacy_containers": collected.legacy,
            })
        );
    } else {
        println!(
            "Reclaimed {} staging trees, {} payload objects, {} legacy containers",
            collected.staging, collected.objects, collected.legacy
        );
    }
    Ok(())
}

/// Unused on platforms without hard-linked cache entries; keeps the module portable.
#[allow(dead_code)]
fn _path_type(_: PathBuf) -> Option<PathBuf> {
    None
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    #[test]
    fn usage_counts_allocated_blocks_and_shared_inodes() {
        let directory = std::env::temp_dir().join(format!(
            "flare-cache-usage-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir_all(&directory).unwrap();
        let sparse = directory.join("sparse.bin");
        {
            use std::io::Write;
            let mut file = std::fs::File::create(&sparse).unwrap();
            file.set_len(8 << 20).unwrap();
            file.write_all(&[1u8; 4096]).unwrap();
        }
        std::fs::hard_link(&sparse, directory.join("shared.bin")).unwrap();
        let usage = usage(&directory).unwrap();
        assert_eq!(usage.files, 1, "one inode counts once");
        assert_eq!(usage.logical_bytes, 8 << 20);
        let physical = usage.physical_bytes.unwrap();
        assert!(
            physical < usage.logical_bytes,
            "sparse data must not count as disk usage"
        );
        assert_eq!(usage.shared_objects, 1);
        assert_eq!(usage.shared_logical_bytes, 8 << 20);
        assert_eq!(usage.single_physical_bytes, Some(0));
        let _ = std::fs::remove_dir_all(directory);
    }
}
