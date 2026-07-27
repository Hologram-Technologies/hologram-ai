//! Local-directory model source provider.

use hologram_ai_core::{AiError, AiResult, CancellationToken, ErrorCategory, ProgressSink};

use crate::types::{AcquiredSource, ModelSourceProvider, SourceKind, SourceRequest};
use crate::walk::{check_cancel, io_error, validate_source_dir};

/// Acquires a model source from a local HF-style directory.
///
/// The directory must contain the required file set (`config.json`,
/// `tokenizer.json`, and safetensors weights — see
/// [`crate::LocalDirectoryProvider`] docs on errors). The provider
/// canonicalizes the path, rejects symlink escapes and path traversal,
/// verifies the required files, and computes a deterministic
/// [`AcquiredSource::source_digest`] over the canonical file list. It never
/// mutates the source directory and never caches: `cache_key` is `None` and
/// `from_cache` is `false`.
#[derive(Debug, Default, Clone, Copy)]
pub struct LocalDirectoryProvider;

impl LocalDirectoryProvider {
    /// Create a new provider.
    pub fn new() -> Self {
        Self
    }
}

impl ModelSourceProvider for LocalDirectoryProvider {
    fn acquire(
        &self,
        request: &SourceRequest,
        progress: &mut dyn ProgressSink,
        cancellation: &CancellationToken,
    ) -> AiResult<AcquiredSource> {
        let path = match &request.kind {
            SourceKind::LocalDir { path } => path,
            SourceKind::HfRepo { .. } => {
                return Err(AiError::new(
                    ErrorCategory::InvalidArgument,
                    "LocalDirectoryProvider requires a LocalDir request",
                ));
            }
        };
        check_cancel(cancellation)?;
        let canonical = std::fs::canonicalize(path).map_err(|e| {
            io_error(
                ErrorCategory::InvalidArgument,
                &format!("source directory {} is not accessible", path.display()),
                &e,
            )
        })?;
        if !canonical.is_dir() {
            return Err(AiError::new(
                ErrorCategory::InvalidArgument,
                format!("{} is not a directory", path.display()),
            ));
        }
        let (_entries, digest) = validate_source_dir(&canonical, progress, cancellation)?;
        Ok(AcquiredSource {
            source_dir: canonical,
            repository: None,
            resolved_revision: None,
            source_digest: digest,
            cache_key: None,
            from_cache: false,
        })
    }
}
