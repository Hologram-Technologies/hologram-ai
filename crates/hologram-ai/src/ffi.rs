//! Binding-oriented surface consumed by Hologram's FFI (`hologram-ffi`,
//! feature `ai`). Opaque app/session handles plus JSON envelopes; errors
//! carry codes in the Hologram AI band (100–111) mapped from the stable
//! `hologram-ai-core` categories (1–20).
//!
//! This module is the exact contract `crates/hologram-ffi/src/ai.rs` in the
//! Hologram repo is written against (see
//! `docs/upstream/hologram-contract.md`).

use std::fmt::Write as _;
use std::path::PathBuf;

use hologram_ai_core::{AiError as CoreError, ErrorCategory, FinishReason, InferenceRequest};

use crate::json::{parse_flat_object, push_json_string, JsonValue};
use crate::{Application, Compiler, HuggingFaceSource, LocalSource, Session, DEFAULT_ENTRY};

/// An FFI error: Hologram AI-band code plus message.
#[derive(Debug)]
pub struct AiError {
    pub code: i32,
    pub message: String,
}

impl From<CoreError> for AiError {
    fn from(e: CoreError) -> Self {
        Self {
            code: band_code(e.category()),
            message: e.message().to_string(),
        }
    }
}

/// Map stable core categories onto the Hologram FFI AI band (100–111).
fn band_code(c: ErrorCategory) -> i32 {
    use ErrorCategory as E;
    match c {
        E::InvalidArgument | E::SourceAcquisition | E::Download | E::Cache => 100, // AI_SOURCE
        E::Authentication => 101,
        E::UnsupportedModel | E::UnsupportedCapability => 102,
        E::Compile | E::QualityGate => 103,
        E::BundleEncode | E::BundleDecode => 104,
        E::ArchiveEncode | E::ArchiveDecode => 105,
        E::IntegrityMismatch => 106,
        E::ModelSelection => 107,
        E::ProcessorMismatch | E::EngineInit => 108,
        E::Inference => 109,
        E::Cancelled => 110,
        E::AbiMismatch => 111,
    }
}

/// An opaque loaded `.holo` AI application.
pub struct AiApp {
    app: Application,
}

impl AiApp {
    /// The callable service entries declared by the application.
    pub fn model_entries(&self) -> Vec<String> {
        self.app
            .models()
            .iter()
            .map(|m| m.entry().to_string())
            .collect()
    }
}

/// An opaque invocation session over one service entry.
pub struct AiSession {
    session: Session,
}

fn cache_dir() -> PathBuf {
    crate::cli::default_cache_dir().join("hf")
}

/// Compile a pinned Hugging Face revision into one `.holo` archive.
pub fn compile_huggingface(
    repository: &str,
    revision: &str,
    entry: &str,
    output_path: &str,
) -> Result<(), AiError> {
    Compiler::builder()
        .source(HuggingFaceSource::pinned(repository, revision))
        .entry(entry)
        .cache_dir(cache_dir())
        .build()
        .map_err(AiError::from)?
        .compile_to_path(output_path)
        .map_err(AiError::from)?;
    Ok(())
}

/// Compile a local HF-style source directory into one `.holo` archive.
pub fn compile_source(source_dir: &str, entry: &str, output_path: &str) -> Result<(), AiError> {
    Compiler::builder()
        .source(LocalSource::new(source_dir))
        .entry(entry)
        .build()
        .map_err(AiError::from)?
        .compile_to_path(output_path)
        .map_err(AiError::from)?;
    Ok(())
}

/// Download and validate a pinned Hugging Face revision (never compiles).
pub fn download(repository: &str, revision: &str, offline: bool) -> Result<(), AiError> {
    use hologram_ai_huggingface::{
        HuggingFaceProvider, ModelSourceProvider, Revision, SourceRequest,
    };
    let provider = HuggingFaceProvider::new(cache_dir()).with_offline(offline);
    provider
        .acquire(
            &SourceRequest::hf_repo(repository, Revision::Pinned(revision.to_string())),
            &mut hologram_ai_core::NullProgressSink,
            &hologram_ai_core::CancellationToken::new(),
        )
        .map_err(AiError::from)?;
    Ok(())
}

/// Load an application from a `.holo` path.
pub fn app_load_path(path: &str) -> Result<AiApp, AiError> {
    Ok(AiApp {
        app: Application::open_path(path).map_err(AiError::from)?,
    })
}

/// Load an application from archive bytes.
pub fn app_load_bytes(bytes: &[u8]) -> Result<AiApp, AiError> {
    Ok(AiApp {
        app: Application::open_bytes(bytes).map_err(AiError::from)?,
    })
}

/// Open a session on a service entry.
pub fn session_open(app: &AiApp, entry: &str) -> Result<AiSession, AiError> {
    let model = if entry.is_empty() {
        app.app.default_model().map_err(AiError::from)?
    } else {
        app.app.model(entry).map_err(AiError::from)?
    };
    Ok(AiSession {
        session: model.session().map_err(AiError::from)?,
    })
}

/// Invoke an operation with a JSON request envelope:
///
/// ```json
/// { "operation": "generate", "prompt": "…", "maxOutputTokens": 128 }
/// ```
///
/// Returns a JSON response envelope:
///
/// ```json
/// { "text": "…", "finishReason": "…", "status": "…", "widened": false }
/// ```
pub fn session_invoke_json(session: &mut AiSession, request_json: &str) -> Result<String, AiError> {
    let fields = parse_flat_object(request_json).map_err(AiError::from)?;
    let mut operation = "generate".to_string();
    let mut prompt: Option<String> = None;
    let mut max_output_tokens: Option<u32> = None;
    for (key, value) in fields {
        match (key.as_str(), value) {
            ("operation", JsonValue::Str(s)) => operation = s,
            ("prompt", JsonValue::Str(s)) => prompt = Some(s),
            ("maxOutputTokens", JsonValue::Num(n)) => max_output_tokens = Some(n as u32),
            (other, _) => {
                return Err(AiError {
                    code: 100,
                    message: format!("unknown request field '{other}'"),
                })
            }
        }
    }
    let mut builder = InferenceRequest::builder();
    if let Some(prompt) = &prompt {
        builder = builder.text("prompt", prompt);
    }
    if let Some(max) = max_output_tokens {
        builder = builder.max_output_tokens(max);
    }
    let completion = session
        .session
        .invoke(&operation, builder.build())
        .map_err(AiError::from)?;

    let mut out = String::from("{");
    out.push_str("\"text\":");
    push_json_string(&mut out, completion.output.text("text").unwrap_or_default());
    let _ = write!(
        out,
        ",\"finishReason\":\"{}\"",
        finish_name(completion.finish_reason)
    );
    if let Some(status) = completion.status {
        let _ = write!(out, ",\"status\":\"{}\"", status_name(status));
    }
    let _ = write!(out, ",\"widened\":{}", completion.widened);
    out.push('}');
    Ok(out)
}

fn finish_name(r: FinishReason) -> &'static str {
    match r {
        FinishReason::EndOfSequence => "end-of-sequence",
        FinishReason::OutputLimit => "output-limit",
        FinishReason::Abstained => "abstained",
        FinishReason::Cancelled => "cancelled",
    }
}

fn status_name(s: hologram_ai_core::ResolutionStatus) -> &'static str {
    use hologram_ai_core::ResolutionStatus as R;
    match s {
        R::Exact => "exact",
        R::Graph => "graph",
        R::Novel => "novel",
    }
}

/// The default entry used when compile callers pass an empty entry string.
pub fn default_entry() -> &'static str {
    DEFAULT_ENTRY
}
