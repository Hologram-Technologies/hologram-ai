//! Hugging Face acquisition via the official `hf` executable.
//!
//! The provider shells out to `hf download` with an opaque argv (repository,
//! revision, and destination as separate arguments — never a shell, never
//! string interpolation), downloads into a resumable staging directory,
//! verifies the result, and atomically renames it into a content-addressed
//! cache entry. See the crate-level docs for the cache layout and the
//! resolved-revision limitation.

use std::collections::BTreeSet;
use std::ffi::OsString;
use std::fs::{self, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use hologram_ai_core::{AiError, AiResult, CancellationToken, ErrorCategory, ProgressSink};

use crate::emit_progress;
use crate::types::{AcquiredSource, ModelSourceProvider, Revision, SourceKind, SourceRequest};
use crate::validate::{cache_key, is_full_sha, redact, validate_repository_id, validate_revision};
use crate::walk::{
    check_cancel, io_error, is_complete_entry, source_digest, staging_dir, validate_source_dir,
    verify_cache_entry, write_manifest,
};

/// Default age after which a lock file is considered stale (24 hours).
const DEFAULT_LOCK_TTL: Duration = Duration::from_secs(24 * 60 * 60);

/// Default maximum time to wait for another holder's cache lock.
const DEFAULT_LOCK_TIMEOUT: Duration = Duration::from_secs(5 * 60);

/// Poll interval while waiting for the `hf` child or a cache lock.
const POLL_INTERVAL: Duration = Duration::from_millis(20);

/// Optional Hugging Face access token.
///
/// The token is passed to the `hf` child process via the `HF_TOKEN`
/// environment variable only — never via argv, never logged, never persisted
/// — and is scrubbed from all error messages with [`redact`]. Its `Debug`
/// representation is redacted.
#[derive(Clone, PartialEq, Eq)]
pub struct HfCredentials {
    token: String,
}

impl HfCredentials {
    /// Credentials from an explicit token value.
    pub fn new(token: impl Into<String>) -> Self {
        Self {
            token: token.into(),
        }
    }

    /// Read credentials from the environment variable `var_name`.
    ///
    /// Returns `None` when the variable is unset or empty.
    pub fn from_env(var_name: &str) -> Option<Self> {
        std::env::var(var_name)
            .ok()
            .filter(|token| !token.is_empty())
            .map(Self::new)
    }

    /// Read credentials from the conventional `HF_TOKEN` variable.
    pub fn from_default_env() -> Option<Self> {
        Self::from_env("HF_TOKEN")
    }

    fn token(&self) -> &str {
        &self.token
    }
}

impl std::fmt::Debug for HfCredentials {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("HfCredentials([REDACTED])")
    }
}

/// Build the exact opaque argv for `hf download`.
///
/// Returns `["download", repository, "--revision", revision, "--local-dir",
/// destination, "--quiet"]` as separate [`OsString`]s. The arguments are
/// passed to `execvp`-style process creation with no shell in between, so
/// shell metacharacters in any value are inert. `--quiet` keeps the child's
/// stdout to the downloaded path only (progress bars are also disabled via
/// the `HF_HUB_DISABLE_PROGRESS_BARS` environment variable).
pub fn build_download_args(repository: &str, revision: &str, destination: &Path) -> Vec<OsString> {
    vec![
        OsString::from("download"),
        OsString::from(repository),
        OsString::from("--revision"),
        OsString::from(revision),
        OsString::from("--local-dir"),
        destination.as_os_str().to_os_string(),
        OsString::from("--quiet"),
    ]
}

/// Acquires Hugging Face model sources with the official `hf` executable.
///
/// Behaviour highlights (see crate-level docs for the full cache layout):
///
/// - Cache key: `hex(blake3(repository || 0x00 || resolved_revision))`.
/// - Downloads go to `<cache>/hf/.staging-<key>/` (`hf --local-dir` resumes
///   partial files there on retry) and are atomically renamed to
///   `<cache>/hf/<key>/` only after validation and `COMPLETE` manifest
///   writing. An interrupted staging directory is never a cache hit.
/// - A per-entry lock file (`<cache>/hf/.lock-<key>`, created with
///   `create_new`) serializes concurrent acquisitions of the same key; locks
///   older than the TTL are treated as stale. Age-based staleness is a
///   heuristic: if a single download can legitimately outlive the default
///   24h TTL, raise it with [`Self::with_lock_ttl`].
/// - `offline` mode serves verified cache hits and rejects misses with a
///   [`ErrorCategory::Download`] error.
/// - Progress maps to `resolve` / `download` / `verify` stages; cancellation
///   is polled before spawning, while the child runs (the child is killed
///   and staging preserved for resume), and between file validations.
pub struct HuggingFaceProvider {
    cache_dir: PathBuf,
    offline: bool,
    credentials: Option<HfCredentials>,
    hf_executable: OsString,
    lock_ttl: Duration,
    lock_timeout: Duration,
}

impl std::fmt::Debug for HuggingFaceProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HuggingFaceProvider")
            .field("cache_dir", &self.cache_dir)
            .field("offline", &self.offline)
            .field(
                "credentials",
                &self.credentials.as_ref().map(|_| "[REDACTED]"),
            )
            .field("hf_executable", &self.hf_executable)
            .field("lock_ttl", &self.lock_ttl)
            .field("lock_timeout", &self.lock_timeout)
            .finish()
    }
}

impl HuggingFaceProvider {
    /// Provider storing its cache under `cache_dir`.
    pub fn new(cache_dir: impl Into<PathBuf>) -> Self {
        Self {
            cache_dir: cache_dir.into(),
            offline: false,
            credentials: None,
            hf_executable: OsString::from("hf"),
            lock_ttl: DEFAULT_LOCK_TTL,
            lock_timeout: DEFAULT_LOCK_TIMEOUT,
        }
    }

    /// Enable or disable offline mode (cache hits only).
    pub fn with_offline(mut self, offline: bool) -> Self {
        self.offline = offline;
        self
    }

    /// Attach credentials passed to `hf` via the environment.
    pub fn with_credentials(mut self, credentials: HfCredentials) -> Self {
        self.credentials = Some(credentials);
        self
    }

    /// Override the `hf` executable (default: `"hf"`, resolved via `PATH`).
    ///
    /// Primarily useful for hermetic tests and non-standard installations.
    pub fn with_hf_executable(mut self, executable: impl Into<OsString>) -> Self {
        self.hf_executable = executable.into();
        self
    }

    /// Override the stale-lock TTL (default 24 hours).
    pub fn with_lock_ttl(mut self, ttl: Duration) -> Self {
        self.lock_ttl = ttl;
        self
    }

    /// Override how long to wait for another holder's lock (default 5 min).
    pub fn with_lock_timeout(mut self, timeout: Duration) -> Self {
        self.lock_timeout = timeout;
        self
    }

    fn hf_root(&self) -> PathBuf {
        self.cache_dir.join("hf")
    }

    fn acquire_hf(
        &self,
        repository: &str,
        revision: &Revision,
        progress: &mut dyn ProgressSink,
        cancellation: &CancellationToken,
    ) -> AiResult<AcquiredSource> {
        emit_progress(progress, "resolve", Some(0), "validating request");
        validate_repository_id(repository)?;
        validate_revision(revision)?;
        check_cancel(cancellation)?;
        fs::create_dir_all(self.hf_root()).map_err(|e| {
            io_error(
                ErrorCategory::Cache,
                &format!("cannot create cache directory {}", self.hf_root().display()),
                &e,
            )
        })?;
        match revision {
            Revision::Pinned(sha) => self.acquire_pinned(repository, sha, progress, cancellation),
            Revision::Resolve(name) => {
                self.acquire_resolvable(repository, name, progress, cancellation)
            }
        }
    }

    fn acquire_pinned(
        &self,
        repository: &str,
        sha: &str,
        progress: &mut dyn ProgressSink,
        cancellation: &CancellationToken,
    ) -> AiResult<AcquiredSource> {
        let key = cache_key(repository, sha);
        let entry = self.hf_root().join(&key);
        let target = CacheTarget::new(repository, sha, key);
        emit_progress(progress, "resolve", Some(100), format!("pinned {sha}"));
        if is_complete_entry(&entry) {
            return self.finish_from_cache(&entry, &target, progress, cancellation);
        }
        if self.offline {
            return Err(AiError::new(
                ErrorCategory::Download,
                format!("offline mode: {repository}@{sha} is not present in the cache"),
            ));
        }
        let _lock = self.acquire_lock(&target.key, cancellation)?;
        // Re-check under the lock: another acquirer may have finished.
        if is_complete_entry(&entry) {
            return self.finish_from_cache(&entry, &target, progress, cancellation);
        }
        let staging = staging_dir(&self.hf_root(), &target.key);
        fs::create_dir_all(&staging).map_err(|e| {
            io_error(
                ErrorCategory::Cache,
                &format!("cannot create staging directory {}", staging.display()),
                &e,
            )
        })?;
        self.run_hf_download(repository, sha, &staging, progress, cancellation)?;
        self.commit_staging(&staging, &entry, &target, progress, cancellation)
    }

    fn acquire_resolvable(
        &self,
        repository: &str,
        name: &str,
        progress: &mut dyn ProgressSink,
        cancellation: &CancellationToken,
    ) -> AiResult<AcquiredSource> {
        if self.offline {
            return Err(AiError::new(
                ErrorCategory::Download,
                format!(
                    "offline mode cannot resolve the mutable revision `{name}`; \
                     acquire once online or use Revision::Pinned"
                ),
            ));
        }
        // Provisional identity for staging + locking until the sha is known.
        let provisional = cache_key(repository, &format!("refs:{name}"));
        let staging = staging_dir(&self.hf_root(), &provisional);
        let _lock = self.acquire_lock(&provisional, cancellation)?;
        fs::create_dir_all(&staging).map_err(|e| {
            io_error(
                ErrorCategory::Cache,
                &format!("cannot create staging directory {}", staging.display()),
                &e,
            )
        })?;
        self.run_hf_download(repository, name, &staging, progress, cancellation)?;
        emit_progress(
            progress,
            "resolve",
            None,
            format!("resolving `{name}` to a commit sha"),
        );
        let sha = extract_resolved_sha(&staging).ok_or_else(|| {
            AiError::new(
                ErrorCategory::SourceAcquisition,
                format!(
                    "could not determine the resolved commit for {repository}@{name} from \
                     hf download metadata; re-run with Revision::Pinned for a full \
                     immutability guarantee"
                ),
            )
        })?;
        let target = CacheTarget::new(repository, &sha, cache_key(repository, &sha));
        let entry = self.hf_root().join(&target.key);
        if is_complete_entry(&entry) {
            let _ = fs::remove_dir_all(&staging);
            return self.finish_from_cache(&entry, &target, progress, cancellation);
        }
        self.commit_staging(&staging, &entry, &target, progress, cancellation)
    }

    /// Validate staging, write the manifest, and atomically promote it to
    /// the final entry. Falls back to a cache hit if a concurrent acquirer
    /// completed the same entry first.
    fn commit_staging(
        &self,
        staging: &Path,
        entry: &Path,
        target: &CacheTarget,
        progress: &mut dyn ProgressSink,
        cancellation: &CancellationToken,
    ) -> AiResult<AcquiredSource> {
        let (entries, digest) = validate_source_dir(staging, progress, cancellation)?;
        write_manifest(staging, &entries)?;
        match fs::rename(staging, entry) {
            Ok(()) => Ok(target.acquired(entry.to_path_buf(), digest, false)),
            Err(error) => {
                if is_complete_entry(entry) {
                    // Another acquirer won the race; our staging is redundant.
                    let _ = fs::remove_dir_all(staging);
                    self.finish_from_cache(entry, target, progress, cancellation)
                } else {
                    Err(io_error(
                        ErrorCategory::Cache,
                        &format!(
                            "cannot move {} into cache entry {}",
                            staging.display(),
                            entry.display()
                        ),
                        &error,
                    ))
                }
            }
        }
    }

    /// Verify an existing complete entry and build the hit result.
    fn finish_from_cache(
        &self,
        entry: &Path,
        target: &CacheTarget,
        progress: &mut dyn ProgressSink,
        cancellation: &CancellationToken,
    ) -> AiResult<AcquiredSource> {
        emit_progress(progress, "verify", Some(0), "verifying cache entry");
        let entries = verify_cache_entry(entry, progress, cancellation)?;
        Ok(target.acquired(entry.to_path_buf(), source_digest(&entries), true))
    }

    /// Spawn `hf download` and wait, honouring cancellation. On cancel the
    /// child is killed and staging is left in place so a retry resumes.
    fn run_hf_download(
        &self,
        repository: &str,
        revision: &str,
        staging: &Path,
        progress: &mut dyn ProgressSink,
        cancellation: &CancellationToken,
    ) -> AiResult<()> {
        let argv = build_download_args(repository, revision, staging);
        let mut command = Command::new(&self.hf_executable);
        command
            .args(&argv)
            .env("HF_HUB_DISABLE_PROGRESS_BARS", "1")
            .stdout(Stdio::null())
            .stderr(Stdio::piped());
        if let Some(credentials) = &self.credentials {
            command.env("HF_TOKEN", credentials.token());
        }
        emit_progress(
            progress,
            "download",
            Some(0),
            format!("hf download {repository}@{revision}"),
        );
        let mut child = command.spawn().map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                AiError::new(
                    ErrorCategory::SourceAcquisition,
                    format!(
                        "`{}` executable not found; install the Hugging Face Hub CLI \
                         (`pip install huggingface_hub[cli]`) or use with_hf_executable",
                        self.hf_executable.to_string_lossy()
                    ),
                )
            } else {
                io_error(ErrorCategory::SourceAcquisition, "cannot spawn hf", &e)
            }
        })?;
        let status = loop {
            match child.try_wait() {
                Ok(Some(status)) => break status,
                Ok(None) => {
                    if cancellation.is_cancelled() {
                        let _ = child.kill();
                        let _ = child.wait();
                        return Err(AiError::cancelled(
                            "cancelled during download; partial files remain in the \
                             staging directory so a retry resumes",
                        ));
                    }
                    thread::sleep(POLL_INTERVAL);
                }
                Err(e) => {
                    return Err(io_error(
                        ErrorCategory::SourceAcquisition,
                        "cannot wait for hf",
                        &e,
                    ));
                }
            }
        };
        if status.success() {
            emit_progress(progress, "download", Some(100), "download complete");
            return Ok(());
        }
        let mut stderr = String::new();
        if let Some(mut pipe) = child.stderr.take() {
            let _ = pipe.read_to_string(&mut stderr);
        }
        let mut message = format!("hf download failed with {status}");
        if !stderr.trim().is_empty() {
            message.push_str(": ");
            message.push_str(stderr.trim());
        }
        if let Some(credentials) = &self.credentials {
            message = redact(credentials.token(), &message);
        }
        let lower = message.to_lowercase();
        let category = if lower.contains("401")
            || lower.contains("403")
            || lower.contains("unauthorized")
            || lower.contains("authentication")
        {
            ErrorCategory::Authentication
        } else {
            ErrorCategory::Download
        };
        Err(AiError::new(category, message))
    }

    /// Take the per-key lock: create_new spin with age-based stale breaking.
    fn acquire_lock(&self, key: &str, cancellation: &CancellationToken) -> AiResult<EntryLock> {
        let path = self.hf_root().join(format!(".lock-{key}"));
        let deadline = Instant::now() + self.lock_timeout;
        loop {
            match OpenOptions::new().write(true).create_new(true).open(&path) {
                Ok(mut file) => {
                    let pid = std::process::id();
                    let created = SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .map(|d| d.as_secs())
                        .unwrap_or(0);
                    let _ = writeln!(file, "pid={pid} created={created}");
                    return Ok(EntryLock { path });
                }
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                    check_cancel(cancellation)?;
                    if self.lock_is_stale(&path) {
                        // Best effort: the next create_new attempt arbitrates.
                        let _ = fs::remove_file(&path);
                        continue;
                    }
                    if Instant::now() >= deadline {
                        return Err(AiError::new(
                            ErrorCategory::Cache,
                            format!("timed out waiting for cache lock {}", path.display()),
                        ));
                    }
                    thread::sleep(POLL_INTERVAL);
                }
                Err(e) => {
                    return Err(io_error(
                        ErrorCategory::Cache,
                        &format!("cannot create cache lock {}", path.display()),
                        &e,
                    ));
                }
            }
        }
    }

    fn lock_is_stale(&self, path: &Path) -> bool {
        match fs::metadata(path).and_then(|m| m.modified()) {
            Ok(modified) => modified
                .elapsed()
                .map(|age| age > self.lock_ttl)
                .unwrap_or(false),
            Err(_) => false,
        }
    }
}

impl ModelSourceProvider for HuggingFaceProvider {
    fn acquire(
        &self,
        request: &SourceRequest,
        progress: &mut dyn ProgressSink,
        cancellation: &CancellationToken,
    ) -> AiResult<AcquiredSource> {
        match &request.kind {
            SourceKind::HfRepo {
                repository,
                revision,
            } => self.acquire_hf(repository, revision, progress, cancellation),
            SourceKind::LocalDir { .. } => Err(AiError::new(
                ErrorCategory::InvalidArgument,
                "HuggingFaceProvider requires an HfRepo request",
            )),
        }
    }
}

/// Identity of a cache entry being acquired: repository, resolved sha, key.
struct CacheTarget {
    repository: String,
    sha: String,
    key: String,
}

impl CacheTarget {
    fn new(repository: &str, sha: &str, key: String) -> Self {
        Self {
            repository: repository.to_string(),
            sha: sha.to_string(),
            key,
        }
    }

    fn acquired(&self, source_dir: PathBuf, digest: [u8; 32], from_cache: bool) -> AcquiredSource {
        AcquiredSource {
            source_dir,
            repository: Some(self.repository.clone()),
            resolved_revision: Some(self.sha.clone()),
            source_digest: digest,
            cache_key: Some(self.key.clone()),
            from_cache,
        }
    }
}

/// Removes the lock file when the guard drops.
struct EntryLock {
    path: PathBuf,
}

impl Drop for EntryLock {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

/// Recover the resolved commit sha from the metadata `hf` writes into
/// `<local-dir>/.cache/huggingface/*.metadata`.
///
/// The exact metadata line format varies across `huggingface_hub` versions,
/// so every whitespace-separated token of every `.metadata` file is scanned
/// for full 40-hex shas. Returns the sha iff exactly one distinct value is
/// found; otherwise `None` (caller falls back to the documented
/// pinned-revision requirement).
fn extract_resolved_sha(staging: &Path) -> Option<String> {
    let metadata_dir = staging.join(".cache").join("huggingface");
    let mut shas = BTreeSet::new();
    let entries = fs::read_dir(metadata_dir).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("metadata") {
            continue;
        }
        if let Ok(content) = fs::read_to_string(&path) {
            for token in content.split_whitespace() {
                if is_full_sha(token) {
                    shas.insert(token.to_string());
                }
            }
        }
    }
    if shas.len() == 1 {
        shas.into_iter().next()
    } else {
        None
    }
}
