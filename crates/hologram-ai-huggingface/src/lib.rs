//! Model-source acquisition for hologram-ai.
//!
//! This crate owns **source acquisition only** in the hologram-ai flow
//! (`HF repo@<full-sha>` | local dir → verified immutable local source →
//! uor-r4 compile → deterministic bundle → `.holo` InferenceModel layer).
//! Networking and process execution live here and only here — never in the
//! runtime path. It has no uor-r4 dependency.
//!
//! Providers:
//!
//! - [`LocalDirectoryProvider`] validates a local HF-style directory
//!   (config + tokenizer + safetensors weights), rejects path traversal and
//!   symlink escapes, and computes a deterministic `source_digest` over the
//!   canonical file list. It never mutates the source.
//! - [`HuggingFaceProvider`] downloads a pinned Hugging Face repository with
//!   the official `hf` executable (opaque argv, no shell) into a
//!   content-addressed, locked, resumable on-disk cache, then verifies the
//!   result against a manifest before exposing it.
//!
//! # Cache layout
//!
//! ```text
//! <cache_dir>/hf/<key>/            complete, verified entry
//! <cache_dir>/hf/<key>/COMPLETE    manifest: "<blake3-hex>  <relpath>" lines
//! <cache_dir>/hf/.staging-<key>/   in-progress download (resumable, never a hit)
//! <cache_dir>/hf/.lock-<key>       per-entry lock file (pid + creation time)
//! ```
//!
//! where `key = hex(blake3(repository || 0x00 || resolved_revision))`.
//! Downloads land in staging (`hf --local-dir` resumes partial files there on
//! retry) and are moved into place with a single atomic rename only after the
//! required-file set is present, all files are hashed, and the `COMPLETE`
//! manifest is written. A staging directory — even a complete-looking one —
//! is never treated as a cache hit.
//!
//! # Resolved-revision limitation
//!
//! [`Revision::Pinned`] (a full 40-hex commit sha) is the only fully
//! immutable source reference. [`Revision::Resolve`] (branch/tag) is accepted
//! only when the caller explicitly chooses it: after download the provider
//! recovers the resolved commit sha from the metadata `hf` writes into
//! `<local-dir>/.cache/huggingface/*.metadata`. If that metadata is absent or
//! inconsistent (older/newer `hf` versions, non-standard transports), the
//! acquisition fails with a [`hologram_ai_core::ErrorCategory::SourceAcquisition`]
//! error asking for a pinned revision — this crate deliberately contains no
//! HTTP client to query the Hub resolve endpoint directly. In `offline` mode
//! a mutable revision cannot be resolved at all and is rejected up front.

#![forbid(unsafe_code)]

mod hf;
mod local;
mod types;
mod validate;
mod walk;

pub use hf::{build_download_args, HfCredentials, HuggingFaceProvider};
pub use local::LocalDirectoryProvider;
pub use types::{AcquiredSource, ModelSourceProvider, Revision, SourceKind, SourceRequest};
pub use validate::{cache_key, redact, validate_repository_id, validate_revision};

use hologram_ai_core::ProgressSink;

/// Emit one progress event; internal helper so stage strings stay consistent.
pub(crate) fn emit_progress(
    progress: &mut dyn ProgressSink,
    stage: &str,
    percent: Option<u8>,
    detail: impl Into<String>,
) {
    progress.on_progress(hologram_ai_core::ProgressEvent {
        stage: stage.to_string(),
        percent,
        detail: detail.into(),
    });
}
