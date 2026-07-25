//! Compile side: verified local source → uor-r4 compile → deterministic
//! schema v1 inference bundle.
//!
//! The bundle carries exactly what the deployed R4G1 runtime needs
//! (ADR-0002/ADR-0005): scored graph, signature artifact, tokenizer when
//! the compile produced one, and the score report. `tless_store.bin` is
//! never packaged (it is not even returned by `uor-r4-api`), and the
//! cover stage's `compile_report.json` stays in the private work
//! directory — it is a build diagnostic, not a runtime component.

use std::path::Path;

use hologram_ai_bundle::BundleBuilder;
use hologram_ai_core::{
    canon::CanonWriter, AbiVersions, AiError, AiResult, ArtifactRole, CancellationToken,
    ErrorCategory, ModelProvenance, OperationDescriptor, ProgressEvent, ProgressSink, StatusPolicy,
    ValueDescriptor, ValueKind, ARTIFACT_FORMAT_R4G1, ENGINE_UOR_R4,
};

/// Pinned uor-r4 compiler base revision (rewrite-plan.md) plus the
/// `feature/typed-integration-facade` patch set (`uor-r4-api`). Recorded
/// in bundle provenance as `compiler_revision`; bump deliberately when
/// the pin moves.
pub const COMPILER_REVISION: &str = "f1b4859e65363eda9aa7dbeb0db467d93c8f4b02+facade";

/// `.holo` container format version the bundle is packaged with.
const HOLO_FORMAT: u16 = 4;

/// What the compile was built from, for bundle provenance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceIdentity {
    /// Source repository (e.g. `HuggingFaceTB/SmolLM2-135M-Instruct`) or
    /// `None` for an anonymous local source.
    pub repository: Option<String>,
    /// Immutable source revision (full commit SHA) when known.
    pub revision: Option<String>,
    /// BLAKE3 digest over the canonical source file list (paths +
    /// digests), computed by the acquisition layer.
    pub source_digest: [u8; 32],
}

/// Compile knobs for the uor-r4 pipeline. Every field is an optional
/// override over the upstream stage defaults (`None` = upstream default:
/// 300 seconds, sequence length 128, target 20 000 records, no R4
/// attention, upstream cover induction knobs).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CompileOptions {
    /// Teacher-corpus generation budget override (wall-clock seconds per
    /// run).
    pub seconds: Option<u64>,
    /// Oracle sequence length override.
    pub sequence_length: Option<u32>,
    /// Teacher-corpus target size override (records).
    pub target: Option<usize>,
    /// Enable the R4 attention variant in the teacher oracle.
    pub r4_attention: Option<bool>,
    /// Cover depth count override.
    pub depths: Option<usize>,
    /// Cover branching factor override.
    pub k0: Option<usize>,
    /// Cover region budget override.
    pub regions_budget: Option<usize>,
    /// Cover induction memory budget override (MiB).
    pub memory_budget_mb: Option<u64>,
}

impl CompileOptions {
    /// Convert to the upstream facade options: upstream defaults, patched
    /// by every override that is set.
    fn to_upstream(self) -> uor_r4_api::CompileOptions {
        let mut upstream = uor_r4_api::CompileOptions::default();
        if let Some(seconds) = self.seconds {
            upstream.seconds = seconds;
        }
        if let Some(sequence_length) = self.sequence_length {
            upstream.sequence_length = sequence_length as usize;
        }
        if let Some(target) = self.target {
            upstream.target = target;
        }
        if let Some(r4_attention) = self.r4_attention {
            upstream.r4_attention = r4_attention;
        }
        if let Some(depths) = self.depths {
            upstream.depths = depths;
        }
        if let Some(k0) = self.k0 {
            upstream.k0 = k0;
        }
        if let Some(regions_budget) = self.regions_budget {
            upstream.regions_budget = regions_budget;
        }
        if let Some(memory_budget_mb) = self.memory_budget_mb {
            upstream.memory_budget_mb = memory_budget_mb;
        }
        upstream
    }

    /// BLAKE3 digest over the canonical (deterministic) encoding of these
    /// options, recorded in bundle provenance. Same options ⇒ same digest,
    /// on every platform; `None` (upstream default) and `Some(<default
    /// value>)` are distinct rows by construction.
    pub fn canonical_digest(&self) -> [u8; 32] {
        let mut w = CanonWriter::new();
        let opt_u64 = |w: &mut CanonWriter, value: Option<u64>| {
            w.bool(value.is_some());
            if let Some(v) = value {
                w.u64(v);
            }
        };
        opt_u64(&mut w, self.seconds);
        opt_u64(&mut w, self.sequence_length.map(u64::from));
        opt_u64(&mut w, self.target.map(|v| v as u64));
        opt_u64(&mut w, self.r4_attention.map(u64::from));
        opt_u64(&mut w, self.depths.map(|v| v as u64));
        opt_u64(&mut w, self.k0.map(|v| v as u64));
        opt_u64(&mut w, self.regions_budget.map(|v| v as u64));
        opt_u64(&mut w, self.memory_budget_mb);
        *blake3::hash(&w.finish()).as_bytes()
    }
}

/// Compile a verified local HF-style source directory into deterministic
/// schema v1 inference bundle bytes.
///
/// - Validates the source layout (`config.json`, `tokenizer.json`, at
///   least one `*.safetensors`) before any compiler stage runs;
///   violations are [`ErrorCategory::UnsupportedModel`].
/// - Honors `cancellation` before the compile starts and at every
///   progress observation (the upstream facade has no cancellation hook,
///   so an in-flight stage runs to its next progress report; a cancelled
///   run is then reported as [`ErrorCategory::Cancelled`] and its
///   outputs discarded).
/// - An incomplete teacher corpus is [`ErrorCategory::Compile`] with the
///   upstream resume hint in the message: re-run with the same
///   `work_dir` to resume from the checkpoint.
/// - On completion, assembles the deterministic bundle: uor-r4/R4G1
///   mandatory components, normalized R4 default status policy, ABI
///   versions from the facade, and provenance from `identity`,
///   [`COMPILER_REVISION`], the canonical options digest, and the score
///   report digest.
pub fn compile_source_to_bundle(
    source_dir: &Path,
    identity: &SourceIdentity,
    work_dir: &Path,
    options: &CompileOptions,
    progress: &mut dyn ProgressSink,
    cancellation: &CancellationToken,
) -> AiResult<Vec<u8>> {
    validate_source(source_dir)?;
    if cancellation.is_cancelled() {
        return Err(AiError::cancelled("compile cancelled before it started"));
    }

    let model_name = read_model_name(source_dir);

    let mut cancelled = false;
    let request = uor_r4_api::CompileRequest {
        source_dir: source_dir.to_path_buf(),
        work_dir: work_dir.to_path_buf(),
        options: options.to_upstream(),
    };
    let outcome = uor_r4_api::compile(&request, &mut |event: uor_r4_api::ProgressEvent| {
        // Progress events bracket every stage boundary, so this is the
        // adapter's cooperative cancellation checkpoint mid-compile.
        if cancellation.is_cancelled() {
            cancelled = true;
        }
        progress.on_progress(ProgressEvent {
            stage: stage_name(event.stage).to_owned(),
            percent: Some(event.percent),
            detail: event.label.to_owned(),
        });
    })
    .map_err(map_compile_error)?;

    if cancelled || cancellation.is_cancelled() {
        return Err(AiError::cancelled(
            "compile cancelled; partial outputs remain in the work directory",
        ));
    }

    let model = match outcome {
        uor_r4_api::CompileOutcome::Complete(model) => model,
        uor_r4_api::CompileOutcome::Incomplete { resume_hint } => {
            return Err(AiError::new(
                ErrorCategory::Compile,
                format!(
                    "compile incomplete: {} (work dir: {})",
                    resume_hint.detail,
                    resume_hint.work_dir.display()
                ),
            ));
        }
    };

    build_bundle(model_name, &model, identity, options)
}

/// Assemble the deterministic schema v1 bundle from a completed compile.
fn build_bundle(
    model_name: String,
    model: &uor_r4_api::CompiledModel,
    identity: &SourceIdentity,
    options: &CompileOptions,
) -> AiResult<Vec<u8>> {
    let has_tokenizer = model.tokenizer.is_some();
    let mut builder = BundleBuilder::new(ENGINE_UOR_R4, ARTIFACT_FORMAT_R4G1, &model_name)
        .set_operations(operations(has_tokenizer))
        .set_status_policy(StatusPolicy::R4_DEFAULT)
        .set_abi(abi_versions())
        .set_provenance(ModelProvenance {
            source_repository: identity.repository.clone(),
            source_revision: identity.revision.clone(),
            source_digest: identity.source_digest,
            compiler_revision: COMPILER_REVISION.to_owned(),
            compile_options_digest: options.canonical_digest(),
            quality_report_digest: *blake3::hash(&model.score_report).as_bytes(),
        });
    builder.add_component(ArtifactRole::Graph, &model.graph)?;
    builder.add_component(ArtifactRole::SignatureArtifact, &model.signature_artifact)?;
    if let Some(tokenizer) = &model.tokenizer {
        builder.add_component(ArtifactRole::Tokenizer, tokenizer)?;
    }
    builder.add_component(ArtifactRole::ScoreReport, &model.score_report)?;
    // Deliberately omitted: `compile_report` (cover-stage diagnostic, not
    // a runtime component — ADR-0005) and `tless_store.bin` (never
    // packaged — ADR-0002; `uor-r4-api` does not even return it).
    builder.finish()
}

/// The advertised operations. The manifest is authoritative for
/// invocation: `generate` (streaming) and `predict`. Text in/out is
/// advertised iff the bundle carries a tokenizer (the bundle validator
/// enforces capability⇔processor consistency).
fn operations(has_tokenizer: bool) -> Vec<OperationDescriptor> {
    let (prompt, generated) = if has_tokenizer {
        (ValueKind::Text, ValueKind::Text)
    } else {
        (ValueKind::Tokens, ValueKind::Tokens)
    };
    let descriptor = |name: &str, kind: ValueKind, required: bool| ValueDescriptor {
        name: name.to_owned(),
        kind,
        media_type: None,
        required,
    };
    vec![
        OperationDescriptor {
            name: "generate".to_owned(),
            inputs: vec![descriptor("prompt", prompt, true)],
            outputs: vec![descriptor(
                if prompt == ValueKind::Text {
                    "text"
                } else {
                    "tokens"
                },
                generated,
                true,
            )],
            streaming: true,
        },
        OperationDescriptor {
            name: "predict".to_owned(),
            inputs: vec![descriptor("context", ValueKind::Tokens, true)],
            outputs: vec![descriptor("token", ValueKind::Tokens, true)],
            streaming: false,
        },
    ]
}

/// ABI versions of the compiling facade, recorded in the bundle and
/// surfaced by `engine::Engine::abi_version`. The version gate runs at
/// load time (see `engine::Engine::load`).
pub(crate) fn abi_versions() -> AbiVersions {
    let abi = uor_r4_api::AbiVersion::current();
    AbiVersions {
        compiler_abi: parse_semver(abi.api_crate_version),
        r4g1_format: (u16::from(abi.format_major), u16::from(abi.format_minor)),
        contract: (abi.contract.major, abi.contract.minor, abi.contract.patch),
        holo_format: HOLO_FORMAT,
    }
}

/// Parse a `major.minor.patch` crate version into `(u16, u16, u16)`;
/// malformed components parse as 0 (the ABI gate keys off the R4G1 format
/// and contract versions, not this informational triple).
fn parse_semver(version: &str) -> (u16, u16, u16) {
    let mut parts = version.split('.');
    let next = |parts: &mut std::str::Split<'_, char>| {
        parts
            .next()
            .and_then(|p| p.parse::<u16>().ok())
            .unwrap_or(0)
    };
    (next(&mut parts), next(&mut parts), next(&mut parts))
}

/// Map a uor-r4 stage onto the coarse hologram-ai progress vocabulary.
fn stage_name(stage: uor_r4_api::Stage) -> &'static str {
    match stage {
        uor_r4_api::Stage::TeacherBundle => "observe",
        uor_r4_api::Stage::GraphCover => "cover",
        uor_r4_api::Stage::Scoring => "score",
    }
}

fn map_compile_error(error: uor_r4_api::CompileError) -> AiError {
    use uor_r4_api::CompileError as E;
    match error {
        E::SourceInvalid { message } => AiError::new(ErrorCategory::UnsupportedModel, message),
        E::NonUtf8Path { stage, path } => AiError::invalid_argument(format!(
            "{stage} stage: path is not UTF-8: {}",
            path.display()
        )),
        other => AiError::new(ErrorCategory::Compile, other.to_string()),
    }
}

/// A verified local HF-style source: `config.json`, `tokenizer.json`,
/// and at least one `*.safetensors` weight file.
fn validate_source(source_dir: &Path) -> AiResult<()> {
    let unsupported = |message: String| AiError::new(ErrorCategory::UnsupportedModel, message);
    let metadata = source_dir
        .metadata()
        .map_err(|e| unsupported(format!("{}: {e}", source_dir.display())))?;
    if !metadata.is_dir() {
        return Err(unsupported(format!(
            "{} is not a directory",
            source_dir.display()
        )));
    }
    for required in ["config.json", "tokenizer.json"] {
        if !source_dir.join(required).is_file() {
            return Err(unsupported(format!(
                "{} is missing {required}",
                source_dir.display()
            )));
        }
    }
    let has_weights = std::fs::read_dir(source_dir)
        .map_err(|e| unsupported(format!("{}: {e}", source_dir.display())))?
        .filter_map(Result::ok)
        .any(|entry| {
            entry
                .path()
                .extension()
                .is_some_and(|ext| ext == "safetensors")
        });
    if !has_weights {
        return Err(unsupported(format!(
            "{} carries no *.safetensors weights",
            source_dir.display()
        )));
    }
    Ok(())
}

/// Derive the model name from the source `config.json`: `_name_or_path`
/// when it names a real model, then `model_type`, else a fallback.
fn read_model_name(source_dir: &Path) -> String {
    let fallback = || "uor-r4-model".to_owned();
    let bytes = match std::fs::read(source_dir.join("config.json")) {
        Ok(bytes) => bytes,
        Err(_) => return fallback(),
    };
    let config: serde_json::Value = match serde_json::from_slice(&bytes) {
        Ok(config) => config,
        Err(_) => return fallback(),
    };
    for key in ["_name_or_path", "model_type"] {
        if let Some(name) = config.get(key).and_then(serde_json::Value::as_str) {
            let name = name.trim();
            if !name.is_empty() {
                return name.to_owned();
            }
        }
    }
    fallback()
}

#[cfg(test)]
mod tests {
    use super::*;
    use hologram_ai_core::NullProgressSink;

    /// A unique scratch directory under the system temp dir.
    fn scratch_dir(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "hologram-ai-r4-test-{}-{}",
            std::process::id(),
            tag
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn options_digest_is_deterministic() {
        let a = CompileOptions::default();
        let b = CompileOptions::default();
        assert_eq!(a.canonical_digest(), b.canonical_digest());

        let mut c = a;
        c.seconds = Some(300);
        assert_ne!(a.canonical_digest(), c.canonical_digest());

        let mut d = a;
        d.depths = Some(7);
        assert_ne!(a.canonical_digest(), d.canonical_digest());

        // Distinguishing None from Some(default) matters: an explicit
        // override must not collide with the unset row.
        let mut e = a;
        e.memory_budget_mb = Some(0);
        assert_ne!(a.canonical_digest(), e.canonical_digest());

        // Different fields with equal payloads must not collide either.
        let mut f = a;
        f.k0 = Some(7);
        assert_ne!(d.canonical_digest(), f.canonical_digest());
    }

    #[test]
    fn empty_source_dir_is_unsupported_model() {
        let dir = scratch_dir("empty");
        let identity = SourceIdentity {
            repository: None,
            revision: None,
            source_digest: [0u8; 32],
        };
        let err = compile_source_to_bundle(
            &dir,
            &identity,
            &dir.join("work"),
            &CompileOptions::default(),
            &mut NullProgressSink,
            &CancellationToken::new(),
        )
        .unwrap_err();
        assert_eq!(err.category(), ErrorCategory::UnsupportedModel);
        assert!(err.message().contains("config.json"), "{}", err.message());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn missing_tokenizer_json_is_unsupported_model() {
        let dir = scratch_dir("no-tokenizer");
        std::fs::write(dir.join("config.json"), b"{}").unwrap();
        std::fs::write(dir.join("model.safetensors"), b"garbage").unwrap();
        let identity = SourceIdentity {
            repository: Some("org/model".to_owned()),
            revision: Some("abc".to_owned()),
            source_digest: [1u8; 32],
        };
        let err = compile_source_to_bundle(
            &dir,
            &identity,
            &dir.join("work"),
            &CompileOptions::default(),
            &mut NullProgressSink,
            &CancellationToken::new(),
        )
        .unwrap_err();
        assert_eq!(err.category(), ErrorCategory::UnsupportedModel);
        assert!(
            err.message().contains("tokenizer.json"),
            "{}",
            err.message()
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn missing_weights_is_unsupported_model() {
        let dir = scratch_dir("no-weights");
        std::fs::write(dir.join("config.json"), b"{}").unwrap();
        std::fs::write(dir.join("tokenizer.json"), b"{}").unwrap();
        let identity = SourceIdentity {
            repository: None,
            revision: None,
            source_digest: [0u8; 32],
        };
        let err = compile_source_to_bundle(
            &dir,
            &identity,
            &dir.join("work"),
            &CompileOptions::default(),
            &mut NullProgressSink,
            &CancellationToken::new(),
        )
        .unwrap_err();
        assert_eq!(err.category(), ErrorCategory::UnsupportedModel);
        assert!(err.message().contains("safetensors"), "{}", err.message());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn pre_cancelled_token_cancels_before_any_work() {
        let dir = scratch_dir("cancelled");
        std::fs::write(dir.join("config.json"), b"{}").unwrap();
        std::fs::write(dir.join("tokenizer.json"), b"{}").unwrap();
        std::fs::write(dir.join("model.safetensors"), b"garbage").unwrap();
        let token = CancellationToken::new();
        token.cancel();
        let identity = SourceIdentity {
            repository: None,
            revision: None,
            source_digest: [0u8; 32],
        };
        let err = compile_source_to_bundle(
            &dir,
            &identity,
            &dir.join("work"),
            &CompileOptions::default(),
            &mut NullProgressSink,
            &token,
        )
        .unwrap_err();
        assert_eq!(err.category(), ErrorCategory::Cancelled);
        // The work directory must not have been created.
        assert!(!dir.join("work").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn model_name_falls_back_sensibly() {
        let dir = scratch_dir("name");
        assert_eq!(read_model_name(&dir), "uor-r4-model");
        std::fs::write(dir.join("config.json"), b"{\"model_type\":\"llama\"}").unwrap();
        assert_eq!(read_model_name(&dir), "llama");
        std::fs::write(
            dir.join("config.json"),
            b"{\"_name_or_path\":\"HuggingFaceTB/SmolLM2-135M-Instruct\",\"model_type\":\"llama\"}",
        )
        .unwrap();
        assert_eq!(read_model_name(&dir), "HuggingFaceTB/SmolLM2-135M-Instruct");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The bundle must carry only the deployed components: graph,
    /// signature artifact, tokenizer, score report. The cover stage's
    /// compile report stays in the work dir, and `tless_store.bin` is
    /// never packaged (ADR-0002) — `uor-r4-api` does not even return it,
    /// so the assertion is over the bundle's role set and payloads
    /// against a stand-in store payload.
    #[test]
    fn bundle_excludes_compile_report_and_tless_store() {
        let store_stand_in = b"tless-store-bytes-never-packaged".to_vec();
        let compile_report = b"compile-report-diagnostic".to_vec();
        let model = uor_r4_api::CompiledModel {
            graph: b"graph".to_vec(),
            signature_artifact: b"sig".to_vec(),
            tokenizer: Some(b"tokenizer".to_vec()),
            score_report: b"{}".to_vec(),
            compile_report: compile_report.clone(),
            provenance: uor_r4_api::CompileProvenance {
                options: uor_r4_api::CompileOptions::default(),
                format_version: (0, 0),
                contract_version: uor_r4_api::AbiVersion::current().contract,
                digests: uor_r4_api::ComponentDigests {
                    graph: String::new(),
                    signature_artifact: String::new(),
                    tokenizer: None,
                    score_report: String::new(),
                    compile_report: String::new(),
                },
            },
        };
        let identity = SourceIdentity {
            repository: None,
            revision: None,
            source_digest: [0u8; 32],
        };
        let bytes = build_bundle(
            "test-model".to_owned(),
            &model,
            &identity,
            &CompileOptions::default(),
        )
        .unwrap();
        let bundle = hologram_ai_bundle::Bundle::parse_verified(&bytes).unwrap();
        let roles: Vec<_> = bundle.components().map(|(role, _)| role).collect();
        assert_eq!(
            roles,
            vec![
                ArtifactRole::Graph,
                ArtifactRole::SignatureArtifact,
                ArtifactRole::Tokenizer,
                ArtifactRole::ScoreReport,
            ]
        );
        for (_, payload) in bundle.components() {
            assert_ne!(payload, store_stand_in.as_slice());
            assert_ne!(payload, compile_report.as_slice());
        }
    }
}
