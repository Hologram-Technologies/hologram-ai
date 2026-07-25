//! Public request/response types for model-source acquisition.

use std::path::PathBuf;

use hologram_ai_core::{AiResult, CancellationToken, ProgressSink};

/// Kind of model source to acquire: exactly one variant per request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SourceKind {
    /// A Hugging Face repository at an explicit revision.
    HfRepo {
        /// Repository id in `owner/repo` form (validated conservatively).
        repository: String,
        /// Pinned commit sha, or a mutable name when explicitly requested.
        revision: Revision,
    },
    /// A local directory containing HF-style model files.
    LocalDir {
        /// Directory holding `config.json`, `tokenizer.json`, and weights.
        path: PathBuf,
    },
}

/// A request to acquire a verified local model source.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceRequest {
    /// The source to acquire.
    pub kind: SourceKind,
}

impl SourceRequest {
    /// Request a Hugging Face repository at `revision`.
    pub fn hf_repo(repository: impl Into<String>, revision: Revision) -> Self {
        Self {
            kind: SourceKind::HfRepo {
                repository: repository.into(),
                revision,
            },
        }
    }

    /// Request a local HF-style directory.
    pub fn local_dir(path: impl Into<PathBuf>) -> Self {
        Self {
            kind: SourceKind::LocalDir { path: path.into() },
        }
    }
}

/// Revision of a Hugging Face repository.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Revision {
    /// Full 40-character lowercase hex commit sha. Immutable; the only
    /// revision kind with a complete immutability guarantee.
    Pinned(String),
    /// Mutable branch or tag name, resolved to a commit sha at acquisition
    /// time. Used only when the caller explicitly requests it; see the
    /// crate-level "Resolved-revision limitation" note.
    Resolve(String),
}

impl Revision {
    /// A pinned full commit sha.
    pub fn pinned(sha: impl Into<String>) -> Self {
        Self::Pinned(sha.into())
    }

    /// A mutable branch/tag name to resolve at acquisition time.
    pub fn resolve(name: impl Into<String>) -> Self {
        Self::Resolve(name.into())
    }

    /// The raw revision string passed to `hf`.
    pub fn as_str(&self) -> &str {
        match self {
            Self::Pinned(sha) => sha,
            Self::Resolve(name) => name,
        }
    }
}

/// A verified local model source produced by a [`ModelSourceProvider`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AcquiredSource {
    /// Verified local directory containing the HF-style model files.
    pub source_dir: PathBuf,
    /// Hugging Face repository id, when the source came from the Hub.
    pub repository: Option<String>,
    /// Resolved full commit sha, when the source came from the Hub.
    pub resolved_revision: Option<String>,
    /// blake3 digest over the canonical file list (sorted relative paths,
    /// each followed by its per-file blake3 digest).
    pub source_digest: [u8; 32],
    /// Content-addressed cache key, when the source is cache-backed.
    pub cache_key: Option<String>,
    /// Whether the source was served from an existing verified cache entry.
    pub from_cache: bool,
}

/// Provider of verified local model sources.
///
/// Implementations validate the request, make the source available as a
/// local directory, verify it, and report progress / honour cancellation.
pub trait ModelSourceProvider {
    /// Acquire and verify the source described by `request`.
    fn acquire(
        &self,
        request: &SourceRequest,
        progress: &mut dyn ProgressSink,
        cancellation: &CancellationToken,
    ) -> AiResult<AcquiredSource>;
}
