//! Compile orchestration: source → uor-r4 → bundle → `.holo`.
//!
//! The only public compile product is a `.holo` archive. Intermediate
//! R4G1 files live in a private work directory and are never user-facing.

use std::path::{Path, PathBuf};

use hologram_ai_core::{
    AiError, AiResult, CancellationToken, ErrorCategory, NullProgressSink, ProgressSink,
};
use hologram_ai_huggingface::{
    HuggingFaceProvider, LocalDirectoryProvider, ModelSourceProvider, SourceRequest,
};

use crate::package::{build_model_archive, ModelLayer};

/// The default callable service name for a compiled model.
pub const DEFAULT_ENTRY: &str = "ai.default";

/// A Hugging Face model source.
#[derive(Debug, Clone)]
pub struct HuggingFaceSource {
    repository: String,
    revision: String,
    resolve_mutable: bool,
}

impl HuggingFaceSource {
    /// Pin to an immutable full commit SHA (the default, recommended form).
    pub fn pinned(repository: impl Into<String>, revision: impl Into<String>) -> Self {
        Self {
            repository: repository.into(),
            revision: revision.into(),
            resolve_mutable: false,
        }
    }

    /// Explicitly resolve a branch or tag. The resolved immutable commit is
    /// recorded in provenance; prefer [`HuggingFaceSource::pinned`].
    pub fn resolving(repository: impl Into<String>, branch_or_tag: impl Into<String>) -> Self {
        Self {
            repository: repository.into(),
            revision: branch_or_tag.into(),
            resolve_mutable: true,
        }
    }
}

/// A local model source directory (HF-style layout).
#[derive(Debug, Clone)]
pub struct LocalSource {
    dir: PathBuf,
}

impl LocalSource {
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self { dir: dir.into() }
    }
}

/// The model source for a compilation.
#[derive(Debug, Clone)]
pub enum Source {
    HuggingFace(HuggingFaceSource),
    Local(LocalSource),
}

impl From<HuggingFaceSource> for Source {
    fn from(s: HuggingFaceSource) -> Self {
        Self::HuggingFace(s)
    }
}

impl From<LocalSource> for Source {
    fn from(s: LocalSource) -> Self {
        Self::Local(s)
    }
}

/// The result of a successful compilation.
#[derive(Debug)]
pub struct CompiledModel {
    /// The `.holo` bytes (also written to the output path when one was
    /// given).
    pub archive: Vec<u8>,
    /// The callable service name of the compiled model layer.
    pub entry: String,
    /// BLAKE3 fingerprint of the archive.
    pub archive_fingerprint: [u8; 32],
    /// The resolved source identity used for provenance.
    pub source_repository: Option<String>,
    /// The immutable source revision, when known.
    pub source_revision: Option<String>,
}

/// Compilation knobs surfaced by the facade. Everything not set uses the
/// uor-r4 stage defaults.
#[derive(Debug, Clone, Default)]
pub struct CompileOptions {
    /// Teacher-observation time budget in seconds (fixture compiles use a
    /// small value).
    pub seconds: Option<u64>,
    /// Sequence length override.
    pub sequence_length: Option<u32>,
}

/// Builder for [`Compiler`].
#[derive(Default)]
pub struct CompilerBuilder {
    source: Option<Source>,
    entry: Option<String>,
    cache_dir: Option<PathBuf>,
    work_dir: Option<PathBuf>,
    options: CompileOptions,
    offline: bool,
}

impl CompilerBuilder {
    /// The model source (HF pinned/resolving, or local directory).
    pub fn source(mut self, source: impl Into<Source>) -> Self {
        self.source = Some(source.into());
        self
    }

    /// Callable service name of the model layer (default `ai.default`).
    pub fn entry(mut self, entry: impl Into<String>) -> Self {
        self.entry = Some(entry.into());
        self
    }

    /// Content-addressed source cache directory (HF sources).
    pub fn cache_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.cache_dir = Some(dir.into());
        self
    }

    /// Private resumable compile work directory.
    pub fn work_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.work_dir = Some(dir.into());
        self
    }

    /// Cache-only acquisition: never touch the network.
    pub fn offline(mut self, offline: bool) -> Self {
        self.offline = offline;
        self
    }

    /// Compilation knobs (time budget, sequence length).
    pub fn options(mut self, options: CompileOptions) -> Self {
        self.options = options;
        self
    }

    pub fn build(self) -> AiResult<Compiler> {
        let source = self
            .source
            .ok_or_else(|| AiError::invalid_argument("no model source configured"))?;
        Ok(Compiler {
            source,
            entry: self.entry.unwrap_or_else(|| DEFAULT_ENTRY.into()),
            cache_dir: self.cache_dir,
            work_dir: self.work_dir,
            options: self.options,
            offline: self.offline,
        })
    }
}

/// The compile pipeline. Construct via [`Compiler::builder`].
pub struct Compiler {
    source: Source,
    entry: String,
    cache_dir: Option<PathBuf>,
    work_dir: Option<PathBuf>,
    options: CompileOptions,
    offline: bool,
}

impl Compiler {
    pub fn builder() -> CompilerBuilder {
        CompilerBuilder::default()
    }

    /// Compile and write exactly one `.holo` to `output` (atomic:
    /// staged write + rename). Returns the compiled model summary.
    pub fn compile_to_path(self, output: impl AsRef<Path>) -> AiResult<CompiledModel> {
        self.compile_inner(
            &mut NullProgressSink,
            &CancellationToken::new(),
            Some(output.as_ref()),
        )
    }

    /// Like [`Compiler::compile_to_path`] with progress and cancellation.
    pub fn compile_to_path_with(
        self,
        output: impl AsRef<Path>,
        progress: &mut dyn ProgressSink,
        cancellation: &CancellationToken,
    ) -> AiResult<CompiledModel> {
        self.compile_inner(progress, cancellation, Some(output.as_ref()))
    }

    /// Compile and return the `.holo` bytes without writing a file.
    pub fn compile_to_bytes(self) -> AiResult<CompiledModel> {
        self.compile_inner(&mut NullProgressSink, &CancellationToken::new(), None)
    }

    fn compile_inner(
        self,
        progress: &mut dyn ProgressSink,
        cancellation: &CancellationToken,
        output: Option<&Path>,
    ) -> AiResult<CompiledModel> {
        let acquired = self.acquire(progress, cancellation)?;
        let work_dir = self.work_dir.unwrap_or_else(|| {
            std::env::temp_dir().join(format!("hologram-ai-compile-{}", std::process::id()))
        });
        let r4_options = hologram_ai_r4::CompileOptions {
            seconds: self.options.seconds,
            sequence_length: self.options.sequence_length,
            ..hologram_ai_r4::CompileOptions::default()
        };
        let bundle = hologram_ai_r4::compile_source_to_bundle(
            acquired.source_dir.as_path(),
            &hologram_ai_r4::SourceIdentity {
                repository: acquired.repository.clone(),
                revision: acquired.resolved_revision.clone(),
                source_digest: acquired.source_digest,
            },
            &work_dir,
            &r4_options,
            progress,
            cancellation,
        )?;
        let archive = build_model_archive(&[ModelLayer::uor_r4(&self.entry, bundle)])?;
        if let Some(path) = output {
            write_atomic(path, &archive)?;
        }
        // Verify the final artifact through the public load path.
        let app = crate::Application::open_bytes(&archive)?;
        let fingerprint = app.archive_fingerprint();
        Ok(CompiledModel {
            archive,
            entry: self.entry,
            archive_fingerprint: fingerprint,
            source_repository: acquired.repository,
            source_revision: acquired.resolved_revision,
        })
    }

    fn acquire(
        &self,
        progress: &mut dyn ProgressSink,
        cancellation: &CancellationToken,
    ) -> AiResult<hologram_ai_huggingface::AcquiredSource> {
        use hologram_ai_huggingface::Revision;
        match &self.source {
            Source::Local(local) => LocalDirectoryProvider.acquire(
                &SourceRequest::local_dir(&local.dir),
                progress,
                cancellation,
            ),
            Source::HuggingFace(hf) => {
                let cache_dir = self.cache_dir.clone().ok_or_else(|| {
                    AiError::invalid_argument("a cache directory is required for HF sources")
                })?;
                let provider = HuggingFaceProvider::new(cache_dir).with_offline(self.offline);
                let revision = if hf.resolve_mutable {
                    Revision::Resolve(hf.revision.clone())
                } else {
                    Revision::Pinned(hf.revision.clone())
                };
                let request = SourceRequest::hf_repo(&hf.repository, revision);
                provider.acquire(&request, progress, cancellation)
            }
        }
    }
}

/// Stage the full archive next to the destination, then atomically rename.
fn write_atomic(path: &Path, bytes: &[u8]) -> AiResult<()> {
    let tmp = path.with_extension("holo.tmp");
    std::fs::write(&tmp, bytes).map_err(|e| {
        AiError::new(
            ErrorCategory::ArchiveEncode,
            format!("write {}: {e}", tmp.display()),
        )
    })?;
    std::fs::rename(&tmp, path).map_err(|e| {
        AiError::new(
            ErrorCategory::ArchiveEncode,
            format!("rename to {}: {e}", path.display()),
        )
    })
}
