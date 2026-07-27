//! Canonical file listing, content hashing, and `COMPLETE` manifests.
//!
//! The canonical file list of a source directory is the set of regular files
//! (plus in-root file symlinks) under it, named by `/`-joined relative paths
//! and sorted lexicographically. The top-level entries `.cache` (hf's own
//! download metadata) and `COMPLETE` (our manifest marker) are excluded.
//! Symlinks whose target escapes the source root, and symlinked directories,
//! are rejected.

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use hologram_ai_core::{AiError, AiResult, CancellationToken, ErrorCategory, ProgressSink};

use crate::emit_progress;
use crate::validate::hex_lower;

/// Name of the manifest marker file inside a complete cache entry.
pub(crate) const COMPLETE_MARKER: &str = "COMPLETE";

/// Top-level names excluded from the canonical listing.
const IGNORED_TOP_LEVEL: [&str; 2] = [".cache", COMPLETE_MARKER];

/// One file of the canonical listing: relative path + blake3 content digest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct FileEntry {
    pub relpath: String,
    pub digest: [u8; 32],
}

/// Map an I/O error into an [`AiError`] with context; never panics.
pub(crate) fn io_error(category: ErrorCategory, context: &str, error: &std::io::Error) -> AiError {
    AiError::new(category, format!("{context}: {error}"))
}

/// Cooperative cancellation check.
pub(crate) fn check_cancel(cancellation: &CancellationToken) -> AiResult<()> {
    if cancellation.is_cancelled() {
        Err(AiError::cancelled("operation cancelled"))
    } else {
        Ok(())
    }
}

/// Sorted canonical relative-path listing of `root` (must be canonicalized).
///
/// Rejects non-UTF-8 names, symlinks escaping `root`, and symlinked
/// directories with [`ErrorCategory::InvalidArgument`].
pub(crate) fn list_relpaths(root: &Path) -> AiResult<Vec<String>> {
    let mut out = Vec::new();
    recurse(root, root, "", true, &mut out)?;
    out.sort();
    Ok(out)
}

fn recurse(
    root: &Path,
    dir: &Path,
    prefix: &str,
    top_level: bool,
    out: &mut Vec<String>,
) -> AiResult<()> {
    let entries = fs::read_dir(dir).map_err(|e| {
        io_error(
            ErrorCategory::SourceAcquisition,
            &format!("cannot read directory {}", dir.display()),
            &e,
        )
    })?;
    let mut entries: Vec<_> = entries.collect::<Result<_, _>>().map_err(|e| {
        io_error(
            ErrorCategory::SourceAcquisition,
            &format!("cannot read directory {}", dir.display()),
            &e,
        )
    })?;
    entries.sort_by_key(|e| e.file_name());
    for entry in entries {
        let name = match entry.file_name().to_str() {
            Some(name) => name.to_string(),
            None => {
                return Err(AiError::new(
                    ErrorCategory::InvalidArgument,
                    format!(
                        "non-UTF-8 file name in source directory: {}",
                        entry.path().display()
                    ),
                ));
            }
        };
        if top_level && IGNORED_TOP_LEVEL.contains(&name.as_str()) {
            continue;
        }
        let relpath = if prefix.is_empty() {
            name.clone()
        } else {
            format!("{prefix}/{name}")
        };
        let file_type = entry.file_type().map_err(|e| {
            io_error(
                ErrorCategory::SourceAcquisition,
                &format!("cannot stat {}", entry.path().display()),
                &e,
            )
        })?;
        if file_type.is_dir() {
            recurse(root, &entry.path(), &relpath, false, out)?;
        } else if file_type.is_file() {
            out.push(relpath);
        } else if file_type.is_symlink() {
            check_symlink(root, &entry.path(), &relpath)?;
            out.push(relpath);
        }
    }
    Ok(())
}

fn check_symlink(root: &Path, link: &Path, relpath: &str) -> AiResult<()> {
    let target = fs::canonicalize(link).map_err(|e| {
        io_error(
            ErrorCategory::InvalidArgument,
            &format!("dangling symlink `{relpath}` in source directory"),
            &e,
        )
    })?;
    if !target.starts_with(root) {
        return Err(AiError::new(
            ErrorCategory::InvalidArgument,
            format!("symlink `{relpath}` escapes the source directory"),
        ));
    }
    if target.is_dir() {
        return Err(AiError::new(
            ErrorCategory::InvalidArgument,
            format!("symlinked directory `{relpath}` is not supported"),
        ));
    }
    Ok(())
}

/// Validate the required HF-style file set of a source directory.
///
/// Requires `config.json`, `tokenizer.json`, and weights: either
/// `model.safetensors`, a `model.safetensors.index.json` index, or at least
/// one `model-*.safetensors` shard. Failures are
/// [`ErrorCategory::UnsupportedModel`] with a message stating what is missing.
pub(crate) fn validate_required_files(relpaths: &[String]) -> AiResult<()> {
    let has = |name: &str| relpaths.iter().any(|p| p == name);
    if !has("config.json") {
        return Err(unsupported("missing config.json"));
    }
    if !has("tokenizer.json") {
        return Err(unsupported("missing tokenizer.json"));
    }
    let has_weights = has("model.safetensors")
        || has("model.safetensors.index.json")
        || relpaths
            .iter()
            .any(|p| p.starts_with("model-") && p.ends_with(".safetensors") && !p.contains('/'));
    if !has_weights {
        return Err(unsupported(
            "missing weights (model.safetensors, model.safetensors.index.json, \
             or model-*.safetensors shards)",
        ));
    }
    Ok(())
}

fn unsupported(detail: &str) -> AiError {
    AiError::new(
        ErrorCategory::UnsupportedModel,
        format!("not a supported Hugging Face model source: {detail}"),
    )
}

/// Hash each listed file, emitting `verify` progress and honouring
/// cancellation between files.
pub(crate) fn hash_files(
    root: &Path,
    relpaths: &[String],
    progress: &mut dyn ProgressSink,
    cancellation: &CancellationToken,
) -> AiResult<Vec<FileEntry>> {
    let total = relpaths.len();
    let mut entries = Vec::with_capacity(total);
    for (index, relpath) in relpaths.iter().enumerate() {
        check_cancel(cancellation)?;
        let digest = hash_file(&root.join(relpath)).map_err(|e| {
            io_error(
                ErrorCategory::SourceAcquisition,
                &format!("cannot hash `{relpath}`"),
                &e,
            )
        })?;
        entries.push(FileEntry {
            relpath: relpath.clone(),
            digest,
        });
        let percent = ((index + 1) * 100 / total.max(1)) as u8;
        emit_progress(
            progress,
            "verify",
            Some(percent),
            format!("hashed {relpath}"),
        );
    }
    Ok(entries)
}

fn hash_file(path: &Path) -> std::io::Result<[u8; 32]> {
    let mut file = fs::File::open(path)?;
    let mut hasher = blake3::Hasher::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(*hasher.finalize().as_bytes())
}

/// Source digest: blake3 over the canonical listing, as
/// `relpath || 0x00 || file_digest` per entry in sorted order.
pub(crate) fn source_digest(entries: &[FileEntry]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    for entry in entries {
        hasher.update(entry.relpath.as_bytes());
        hasher.update(&[0]);
        hasher.update(&entry.digest);
    }
    *hasher.finalize().as_bytes()
}

/// Validate a source directory end to end: canonical listing, required
/// files, per-file hashes, and the source digest.
pub(crate) fn validate_source_dir(
    root: &Path,
    progress: &mut dyn ProgressSink,
    cancellation: &CancellationToken,
) -> AiResult<(Vec<FileEntry>, [u8; 32])> {
    let relpaths = list_relpaths(root)?;
    validate_required_files(&relpaths)?;
    let entries = hash_files(root, &relpaths, progress, cancellation)?;
    let digest = source_digest(&entries);
    Ok((entries, digest))
}

/// Write the `COMPLETE` manifest (`"<hex>  <relpath>"` lines, sorted).
pub(crate) fn write_manifest(dir: &Path, entries: &[FileEntry]) -> AiResult<()> {
    let mut content = String::new();
    for entry in entries {
        content.push_str(&hex_lower(&entry.digest));
        content.push_str("  ");
        content.push_str(&entry.relpath);
        content.push('\n');
    }
    fs::write(dir.join(COMPLETE_MARKER), content).map_err(|e| {
        io_error(
            ErrorCategory::Cache,
            &format!("cannot write {} manifest", dir.display()),
            &e,
        )
    })
}

/// Read a `COMPLETE` manifest.
pub(crate) fn read_manifest(dir: &Path) -> AiResult<Vec<FileEntry>> {
    let content = fs::read_to_string(dir.join(COMPLETE_MARKER)).map_err(|e| {
        io_error(
            ErrorCategory::Cache,
            &format!("cannot read {} manifest", dir.display()),
            &e,
        )
    })?;
    let mut entries = Vec::new();
    for line in content.lines() {
        let (hex, relpath) = line.split_once("  ").ok_or_else(|| {
            AiError::new(
                ErrorCategory::Cache,
                format!("malformed manifest line in {}: `{line}`", dir.display()),
            )
        })?;
        let digest = parse_hex_digest(hex).ok_or_else(|| {
            AiError::new(
                ErrorCategory::Cache,
                format!("malformed digest in {}: `{hex}`", dir.display()),
            )
        })?;
        entries.push(FileEntry {
            relpath: relpath.to_string(),
            digest,
        });
    }
    Ok(entries)
}

fn parse_hex_digest(hex: &str) -> Option<[u8; 32]> {
    if hex.len() != 64 {
        return None;
    }
    let mut digest = [0u8; 32];
    for (i, byte) in digest.iter_mut().enumerate() {
        *byte = u8::from_str_radix(hex.get(2 * i..2 * i + 2)?, 16).ok()?;
    }
    Some(digest)
}

/// Verify a complete cache entry against its manifest: required file set,
/// every listed file present, and every digest matching. Re-hashing honours
/// cancellation. Returns the verified entries.
pub(crate) fn verify_cache_entry(
    root: &Path,
    progress: &mut dyn ProgressSink,
    cancellation: &CancellationToken,
) -> AiResult<Vec<FileEntry>> {
    let expected = read_manifest(root)?;
    let relpaths: Vec<String> = expected.iter().map(|e| e.relpath.clone()).collect();
    validate_required_files(&relpaths).map_err(|e| {
        AiError::new(
            ErrorCategory::Cache,
            format!("corrupt cache entry {}: {}", root.display(), e.message()),
        )
    })?;
    let actual = hash_files(root, &relpaths, progress, cancellation).map_err(|e| {
        AiError::new(
            ErrorCategory::Cache,
            format!("corrupt cache entry {}: {}", root.display(), e.message()),
        )
    })?;
    for (actual, expected) in actual.iter().zip(expected.iter()) {
        if actual.digest != expected.digest {
            return Err(AiError::new(
                ErrorCategory::Cache,
                format!(
                    "corrupt cache entry {}: digest mismatch for `{}`",
                    root.display(),
                    actual.relpath
                ),
            ));
        }
    }
    Ok(actual)
}

/// Whether `entry` is a complete cache entry (directory + `COMPLETE` marker).
pub(crate) fn is_complete_entry(entry: &Path) -> bool {
    entry.is_dir() && entry.join(COMPLETE_MARKER).is_file()
}

/// Path of the staging directory for `key` under the cache's `hf` root.
pub(crate) fn staging_dir(hf_root: &Path, key: &str) -> PathBuf {
    hf_root.join(format!(".staging-{key}"))
}
