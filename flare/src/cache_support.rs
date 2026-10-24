//! Cross-platform prepared-model cache selection and deployment choice.
//!
//! Unix-specific rank transport and locking remain in `package_cache`.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

type Failure = Box<dyn std::error::Error + Send + Sync>;

/// Prepared-model cache root: explicit option, then `paths.package_cache`, then the default.
///
/// Callers choose the explicit value: Hub and cache commands use `--model-cache`;
/// distribution peers use `--model-cache` and fall back to `--package-cache`.
pub fn model_root(explicit: Option<&Path>) -> Result<PathBuf, Failure> {
    let home = crate::environment::home_directory()?;
    let config = home.join(".config/flare/config.toml");
    model_root_from(&home, Some(&config), explicit)
}

pub(crate) fn model_root_from(
    home: &Path,
    config: Option<&Path>,
    explicit: Option<&Path>,
) -> Result<PathBuf, Failure> {
    if let Some(path) = explicit {
        return Ok(expand(home, path));
    }
    if let Some(config) = config
        && let Some(configured) = configured_package_cache(config)?
    {
        return Ok(expand(home, &configured));
    }
    Ok(home.join(".cache/flare/packages"))
}

#[derive(serde::Deserialize, Default)]
struct ConfigFile {
    #[serde(default)]
    paths: Paths,
}

#[derive(serde::Deserialize, Default)]
struct Paths {
    package_cache: Option<String>,
}

fn configured_package_cache(path: &Path) -> Result<Option<PathBuf>, Failure> {
    let text = match fs::read_to_string(path) {
        Ok(text) => text,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(format!("cannot read {}: {error}", path.display()).into()),
    };
    let config: ConfigFile = toml::from_str(&text)
        .map_err(|error| format!("cannot parse {}: {error}", path.display()))?;
    Ok(config
        .paths
        .package_cache
        .filter(|value| !value.trim().is_empty())
        .map(PathBuf::from))
}

fn expand(home: &Path, path: &Path) -> PathBuf {
    match path.strip_prefix("~") {
        Ok(rest) => home.join(rest),
        Err(_) => path.to_owned(),
    }
}

/// Normalize requested tensor-parallel sizes: deduplicated, ascending, non-zero.
pub fn requested_tensor_parallel(requested: &[u32], gpu_count: usize) -> Result<Vec<u32>, Failure> {
    let mut sizes: Vec<u32> = if requested.is_empty() {
        vec![u32::try_from(gpu_count.max(1)).map_err(|_| "GPU count is too large")?]
    } else {
        requested.to_vec()
    };
    if sizes.contains(&0) {
        return Err("tensor-parallel sizes must be positive".into());
    }
    sizes.sort_unstable();
    sizes.dedup();
    Ok(sizes)
}

/// Resolve signed deployments for the requested tensor-parallel sizes.
///
/// Topology comes from each authenticated deployment body, never from
/// availability hints: a size that was never exported, appears twice, or has an
/// incomplete payload closure is an error rather than a silently dropped request.
pub fn select_deployments(
    metadata: &flare_helios::package::reader::Metadata,
    publishers: &[flare_helios::runtime_package::Publisher],
    sizes: &[u32],
) -> Result<Vec<flare_helios::package::protection::Selection>, Failure> {
    use flare_helios::package::{protection, reader};

    let catalog = protection::verify(
        &metadata.envelope,
        publishers,
        protection::Selection::Catalog,
    )?;
    let count = catalog.deployment_count();
    let mut found: HashMap<u32, Vec<usize>> = HashMap::new();
    for index in 0..count {
        let selection = protection::Selection::Deployment {
            deployment: index,
            rank: None,
        };
        let verified = protection::verify(&metadata.envelope, publishers, selection)?;
        let deployment = verified
            .deployment()
            .ok_or_else(|| format!("deployment {index} has no authenticated body"))?;
        found
            .entry(deployment.tensor_parallel)
            .or_default()
            .push(index);
    }
    let mut selected = Vec::with_capacity(sizes.len());
    for size in sizes {
        let indices = found.get(size).ok_or_else(|| {
            format!("package exports no deployment with tensor-parallel size {size}")
        })?;
        match indices.as_slice() {
            [index] => selected.push(protection::Selection::Deployment {
                deployment: *index,
                rank: None,
            }),
            _ => {
                return Err(format!(
                    "package exports {} deployments with tensor-parallel size {size}",
                    indices.len()
                )
                .into());
            }
        }
    }
    let mut verified = Vec::with_capacity(selected.len());
    for selection in &selected {
        verified.push(protection::verify(
            &metadata.envelope,
            publishers,
            *selection,
        )?);
    }
    flare_helios::package::cache::Root::validate_ready_closure(
        metadata,
        &verified,
        &flare_helios::package::derivation::Limits::default(),
    )
    .map_err(|error| error.to_string())?;
    Ok(selected)
}
