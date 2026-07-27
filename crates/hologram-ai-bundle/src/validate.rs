//! Semantic validation rules shared by the builder and the parser.
//!
//! Both paths report through the same stable error categories so downstream
//! handlers see one mapping regardless of where the invalid bundle was
//! caught: missing mandatory roles are [`ErrorCategory::BundleDecode`] and
//! capability⇔processor inconsistencies are
//! [`ErrorCategory::ProcessorMismatch`].

use alloc::format;

use hologram_ai_core::{
    AiError, AiResult, ArtifactRole, ErrorCategory, InferenceModelManifest, ValueKind,
    ARTIFACT_FORMAT_R4G1, ENGINE_UOR_R4,
};

/// Mandatory components for uor-r4 R4G1 models.
const R4G1_MANDATORY_ROLES: [ArtifactRole; 3] = [
    ArtifactRole::Graph,
    ArtifactRole::SignatureArtifact,
    ArtifactRole::ScoreReport,
];

fn has_role(m: &InferenceModelManifest, role: ArtifactRole) -> bool {
    m.artifacts.iter().any(|a| a.role == role)
}

/// Validate a manifest against the bundle schema v1 semantic rules.
///
/// - When `engine == uor-r4` and `artifact_format == R4G1`, the mandatory
///   roles `graph`, `signature-artifact`, and `score-report` must be
///   present. Unknown engine/format strings skip this rule (forward
///   compatibility).
/// - Capability⇔processor consistency (engine-agnostic): any operation
///   input/output of kind `Text` requires a `tokenizer` component.
///   Token-only models — every text-ish value is kind `Tokens`, with no
///   `Text` anywhere — do not. Kind `Image` requires an `image-processor`,
///   kind `Audio` requires an `audio-processor`.
pub(crate) fn validate_manifest(m: &InferenceModelManifest) -> AiResult<()> {
    if m.engine == ENGINE_UOR_R4 && m.artifact_format == ARTIFACT_FORMAT_R4G1 {
        for role in R4G1_MANDATORY_ROLES {
            if !has_role(m, role) {
                return Err(AiError::new(
                    ErrorCategory::BundleDecode,
                    format!(
                        "mandatory component `{}` is required for engine `{ENGINE_UOR_R4}` / format `{ARTIFACT_FORMAT_R4G1}`",
                        role.name()
                    ),
                ));
            }
        }
    }

    let mut has_text = false;
    let mut has_image = false;
    let mut has_audio = false;
    for op in &m.operations {
        for v in op.inputs.iter().chain(op.outputs.iter()) {
            match v.kind {
                ValueKind::Text => has_text = true,
                ValueKind::Image => has_image = true,
                ValueKind::Audio => has_audio = true,
                _ => {}
            }
        }
    }

    let tokenizer = ArtifactRole::Tokenizer;
    if has_text && !has_role(m, tokenizer) {
        return Err(processor_mismatch(tokenizer, "text"));
    }
    let image_processor = ArtifactRole::ImageProcessor;
    if has_image && !has_role(m, image_processor) {
        return Err(processor_mismatch(image_processor, "image"));
    }
    let audio_processor = ArtifactRole::AudioProcessor;
    if has_audio && !has_role(m, audio_processor) {
        return Err(processor_mismatch(audio_processor, "audio"));
    }
    Ok(())
}

fn processor_mismatch(role: ArtifactRole, capability: &str) -> AiError {
    AiError::new(
        ErrorCategory::ProcessorMismatch,
        format!(
            "model advertises {capability} capability but has no `{}` component",
            role.name()
        ),
    )
}
