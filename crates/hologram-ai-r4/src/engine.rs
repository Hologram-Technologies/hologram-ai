//! Engine side: the loaded R4G1 inference engine behind hologram-ai's
//! typed status/abstention vocabulary.
//!
//! # Ownership
//!
//! `uor_r4_api::R4Engine` parses the borrowed bundle component slices
//! into fully owned state at load time (it carries no lifetime), so
//! [`Engine`] owns the upstream engine directly — no self-referential
//! buffer pattern, no unsafe. The `Bundle` is borrowed only for the
//! duration of [`Engine::load`].
//!
//! # Generation semantics (deliberate delegation)
//!
//! [`Engine::generate_into`] delegates the per-step loop to
//! `uor_r4_api::R4Engine::generate_into`. The EOS decision (which token
//! ids end a run) and the sliding-window policy are upstream contract
//! details not exported for reuse; re-implementing the loop over
//! `predict_decision` here would have to duplicate that policy, and a
//! divergent copy is worse than a missing feature. Accepted
//! consequences:
//!
//! - [`StreamEvent`]s are emitted as a batch when the run completes, not
//!   per step mid-run (sinks are still invoked from the engine thread;
//!   `NullEventSink` costs nothing).
//! - A cancellation token installed via [`Engine::set_cancellation`] is
//!   honored at run entry only — not between steps — because the
//!   upstream loop has no cancellation checkpoint.
//!
//! # Allocation contract (ADR-0009)
//!
//! [`Engine::predict_next_into`] performs zero heap allocations: it
//! writes into the caller-owned [`Prediction`] slot through the upstream
//! `PredictOutput` slot, both plain stack data. The steady-state loop of
//! [`Engine::generate_into`] is the upstream allocation-free loop; the
//! post-run event batch may allocate (event payloads are owned), which
//! the census test tolerates by counting only the loop region when a
//! fixture bundle is available.

use hologram_ai_bundle::Bundle;
use hologram_ai_core::{
    AbiVersions, AiError, AiResult, ArtifactRole, CancellationToken, ErrorCategory, EventSink,
    FinishReason, ResolutionStatus, StreamEvent, ARTIFACT_FORMAT_R4G1, ENGINE_UOR_R4,
};

/// Upper bound on the encode/decode scratch growth: text helpers sit at
/// the allocating boundary, but a tokenizer that never fits signals a
/// real failure, not an unbounded doubling loop.
const MAX_ENCODE_TOKENS: usize = 1 << 20;
/// Upper bound on decoded byte scratch (see above).
const MAX_DECODE_BYTES: usize = 1 << 22;

/// Caller-owned prediction result slot for [`Engine::predict_next_into`].
/// `Default` is the abstaining form, so a slot that was never written by
/// a successful call cannot be misread as a served token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Prediction {
    pub outcome: PredictOutcome,
}

impl Default for Prediction {
    fn default() -> Self {
        Self {
            outcome: PredictOutcome::Abstain { widened: false },
        }
    }
}

/// The status-aware prediction outcome: either the policy serves a token
/// or it abstains (typed abstention — no token is fabricated).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PredictOutcome {
    /// A token was served, with its resolution status and whether a
    /// widened re-probe ran for this prediction.
    Serve {
        token: u32,
        status: ResolutionStatus,
        widened: bool,
    },
    /// No token was emitted and none is guessed.
    Abstain { widened: bool },
}

/// Summary of a finished [`Engine::generate_into`] run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FinishInfo {
    pub finish_reason: FinishReason,
    /// Status of the final step; `None` when nothing was produced.
    pub status: Option<ResolutionStatus>,
    /// Whether any step widened its membership set.
    pub widened: bool,
    /// Number of tokens written into the caller's output buffer.
    pub produced: usize,
    /// Witness payload, when the engine produced one. The current
    /// upstream facade exposes no per-run witness bytes, so this is
    /// always `None` today.
    pub witness: Option<(String, Vec<u8>)>,
}

/// The loaded R4G1 inference engine. Construct via [`Engine::load`].
pub struct Engine {
    inner: uor_r4_api::R4Engine,
    /// Whether the bundle carried a tokenizer (text helpers fail with
    /// [`ErrorCategory::ProcessorMismatch`] otherwise; the upstream
    /// `_into` APIs fold "no tokenizer" and "encode failed" into one
    /// `None`, so the distinction is tracked here).
    has_tokenizer: bool,
    cancellation: Option<CancellationToken>,
}

impl std::fmt::Debug for Engine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Engine")
            .field("has_tokenizer", &self.has_tokenizer)
            .finish_non_exhaustive()
    }
}

impl Engine {
    /// Load an engine from a parsed bundle. The bundle is re-verified
    /// here (digest verification + semantic validation are cheap) so the
    /// engine never consumes unverified component bytes regardless of how
    /// the caller obtained the `Bundle`.
    ///
    /// Errors: [`ErrorCategory::IntegrityMismatch`] on digest failure,
    /// [`ErrorCategory::BundleDecode`] on structural/semantic violations,
    /// [`ErrorCategory::AbiMismatch`] when the bundle was compiled
    /// against an incompatible R4G1 format or contract major version,
    /// [`ErrorCategory::EngineInit`] when the components fail upstream
    /// validation, [`ErrorCategory::ProcessorMismatch`] for malformed
    /// tokenizer bytes, and [`ErrorCategory::QualityGate`] when the
    /// score report digresses from the pinned quality floor.
    pub fn load(bundle: &Bundle<'_>) -> AiResult<Self> {
        bundle.verify_digests()?;
        bundle.validate()?;

        let manifest = bundle.manifest();
        if manifest.engine != ENGINE_UOR_R4 || manifest.artifact_format != ARTIFACT_FORMAT_R4G1 {
            return Err(AiError::new(
                ErrorCategory::EngineInit,
                format!(
                    "not a uor-r4/R4G1 bundle (engine `{}`, format `{}`)",
                    manifest.engine, manifest.artifact_format
                ),
            ));
        }
        check_abi(&manifest.abi)?;

        let required = |role: ArtifactRole| {
            bundle.component(role).ok_or_else(|| {
                AiError::new(
                    ErrorCategory::BundleDecode,
                    format!("mandatory component `{}` is missing", role.name()),
                )
            })
        };
        let graph = required(ArtifactRole::Graph)?;
        let signature_artifact = required(ArtifactRole::SignatureArtifact)?;
        let score_report = required(ArtifactRole::ScoreReport)?;
        let tokenizer = bundle.component(ArtifactRole::Tokenizer);
        let has_tokenizer = tokenizer.is_some();

        let inner = uor_r4_api::R4Engine::load(uor_r4_api::EngineParts {
            graph,
            signature_artifact,
            tokenizer,
            score_report: Some(score_report),
        })
        .map_err(map_load_error)?;

        Ok(Self {
            inner,
            has_tokenizer,
            cancellation: None,
        })
    }

    /// Install (or clear) a cooperative cancellation token. Checked at
    /// [`Engine::generate_into`] entry; see the module docs for why it is
    /// not polled between steps.
    pub fn set_cancellation(&mut self, cancellation: Option<CancellationToken>) {
        self.cancellation = cancellation;
    }

    /// Reset the session state (status counters, widen-once memory,
    /// scoring scratch). No reallocation.
    pub fn reset(&mut self) {
        self.inner.reset();
    }

    /// The ABI surface this engine runs against, in hologram-ai terms.
    pub fn abi_version(&self) -> AbiVersions {
        crate::compile::abi_versions()
    }

    /// Allocation-free single-step prediction into a caller-owned slot
    /// (ADR-0009 steady state).
    pub fn predict_next_into(&mut self, window: &[u32], out: &mut Prediction) -> AiResult<()> {
        let mut raw = uor_r4_api::PredictOutput::default();
        self.inner
            .predict_next_into(window, &mut raw)
            .map_err(map_inference_error)?;
        out.outcome = if raw.abstained {
            PredictOutcome::Abstain {
                widened: raw.widened,
            }
        } else {
            PredictOutcome::Serve {
                token: raw.token,
                // A served token without a status is defensive-only (the
                // upstream engine always sets one); resolve it as Exact,
                // the strongest evidence class.
                status: map_status(raw.status.unwrap_or(uor_r4_api::ScoreStatus::ExactContext)),
                widened: raw.widened,
            }
        };
        Ok(())
    }

    /// Deterministic status-aware generation into a caller-owned buffer.
    /// Stops at the first abstention, at EOS, or when the buffer fills;
    /// never fabricates a token. Stream events are emitted as a batch
    /// when the run completes (see the module docs).
    pub fn generate_into(
        &mut self,
        seed: &[u32],
        output_tokens: &mut [u32],
        events: &mut dyn EventSink,
    ) -> AiResult<FinishInfo> {
        if self
            .cancellation
            .as_ref()
            .is_some_and(CancellationToken::is_cancelled)
        {
            return Err(AiError::cancelled("generation cancelled before it started"));
        }

        let status = self
            .inner
            .generate_into(seed, output_tokens)
            .map_err(map_inference_error)?;

        let finish_reason = if status.abstained {
            FinishReason::Abstained
        } else if status.count == output_tokens.len() {
            FinishReason::OutputLimit
        } else {
            FinishReason::EndOfSequence
        };
        let mapped_status = status.status.map(map_status);

        // Post-run event batch (may allocate inside the sink or in the
        // Text payload; the steady-state loop above is the counted path).
        for &token in &output_tokens[..status.count] {
            events.on_event(StreamEvent::Token(token));
        }
        if self.has_tokenizer && status.count > 0 {
            if let Ok(text) = self.decode_tokens(&output_tokens[..status.count]) {
                events.on_event(StreamEvent::Text(text));
            }
        }
        if let Some(status) = mapped_status {
            events.on_event(StreamEvent::Status(status));
        }
        if status.widened {
            events.on_event(StreamEvent::Widened);
        }
        events.on_event(StreamEvent::Finished(finish_reason));

        Ok(FinishInfo {
            finish_reason,
            status: mapped_status,
            widened: status.widened,
            produced: status.count,
            witness: None,
        })
    }

    /// Encode text with the bundle-matched tokenizer. Allocating
    /// convenience wrapper over the upstream `_into` API (boundary, not
    /// steady state). Without a tokenizer component this is
    /// [`ErrorCategory::ProcessorMismatch`].
    pub fn encode_text(&self, text: &str) -> AiResult<Vec<u32>> {
        encode_text_with(self.has_tokenizer, text, |text, out| {
            self.inner.encode_text_into(text, out)
        })
    }

    /// Decode tokens with the bundle-matched tokenizer. Allocating
    /// convenience wrapper over the upstream `_into` API. Without a
    /// tokenizer component this is [`ErrorCategory::ProcessorMismatch`].
    pub fn decode_tokens(&self, tokens: &[u32]) -> AiResult<String> {
        decode_tokens_with(self.has_tokenizer, tokens, |tokens, out| {
            self.inner.decode_tokens_into(tokens, out)
        })
    }
}

/// Map the upstream score status onto the engine-agnostic resolution
/// status.
fn map_status(status: uor_r4_api::ScoreStatus) -> ResolutionStatus {
    match status {
        uor_r4_api::ScoreStatus::ExactContext => ResolutionStatus::Exact,
        uor_r4_api::ScoreStatus::Graph => ResolutionStatus::Graph,
        uor_r4_api::ScoreStatus::Novel => ResolutionStatus::Novel,
    }
}

/// Gate the bundle's recorded ABI against this build: R4G1 format and
/// contract major versions must match exactly (minor bumps are
/// forward-compatible per the format RFC).
fn check_abi(abi: &AbiVersions) -> AiResult<()> {
    let current = uor_r4_api::AbiVersion::current();
    if abi.r4g1_format.0 != u16::from(current.format_major) {
        return Err(AiError::new(
            ErrorCategory::AbiMismatch,
            format!(
                "bundle R4G1 format major {} is incompatible with this build ({})",
                abi.r4g1_format.0, current.format_major
            ),
        ));
    }
    if abi.contract.0 != current.contract.major {
        return Err(AiError::new(
            ErrorCategory::AbiMismatch,
            format!(
                "bundle contract major {} is incompatible with this build ({})",
                abi.contract.0, current.contract.major
            ),
        ));
    }
    Ok(())
}

fn map_load_error(error: uor_r4_api::LoadError) -> AiError {
    use uor_r4_api::LoadError as E;
    match error {
        E::InvalidTokenizer(e) => AiError::new(
            ErrorCategory::ProcessorMismatch,
            format!("invalid tokenizer bytes: {e}"),
        ),
        E::QualityGate(message) => AiError::new(ErrorCategory::QualityGate, message),
        other => AiError::new(ErrorCategory::EngineInit, other.to_string()),
    }
}

fn map_inference_error(error: uor_r4_api::InferenceError) -> AiError {
    AiError::new(ErrorCategory::Inference, error.to_string())
}

fn processor_mismatch(message: &str) -> AiError {
    AiError::new(ErrorCategory::ProcessorMismatch, message)
}

/// Shared encode helper: grow the caller-side scratch until the
/// tokenizer's `_into` API fits (or a real failure surfaces). Factored
/// out of `Engine` so the boundary policy — including the missing-
/// tokenizer error — is testable without a loadable graph.
fn encode_text_with(
    has_tokenizer: bool,
    text: &str,
    mut encode: impl FnMut(&str, &mut [u32]) -> Option<usize>,
) -> AiResult<Vec<u32>> {
    if !has_tokenizer {
        return Err(processor_mismatch(
            "model has no tokenizer component; text input is unavailable",
        ));
    }
    let mut capacity = text.len().saturating_add(2).max(16);
    loop {
        let mut buf = vec![0u32; capacity];
        if let Some(n) = encode(text, &mut buf) {
            buf.truncate(n);
            return Ok(buf);
        }
        capacity = capacity.saturating_mul(2);
        if capacity > MAX_ENCODE_TOKENS {
            return Err(AiError::new(
                ErrorCategory::Inference,
                "tokenizer failed to encode the input text",
            ));
        }
    }
}

/// Shared decode helper (see [`encode_text_with`]).
fn decode_tokens_with(
    has_tokenizer: bool,
    tokens: &[u32],
    mut decode: impl FnMut(&[u32], &mut [u8]) -> Option<usize>,
) -> AiResult<String> {
    if !has_tokenizer {
        return Err(processor_mismatch(
            "model has no tokenizer component; text output is unavailable",
        ));
    }
    let mut capacity = tokens.len().saturating_mul(4).saturating_add(16).max(64);
    loop {
        let mut buf = vec![0u8; capacity];
        if let Some(n) = decode(tokens, &mut buf) {
            buf.truncate(n);
            return String::from_utf8(buf).map_err(|e| {
                AiError::new(
                    ErrorCategory::Inference,
                    format!("tokenizer produced invalid UTF-8: {e}"),
                )
            });
        }
        capacity = capacity.saturating_mul(2);
        if capacity > MAX_DECODE_BYTES {
            return Err(AiError::new(
                ErrorCategory::Inference,
                "tokenizer failed to decode the token sequence",
            ));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hologram_ai_bundle::BundleBuilder;
    use hologram_ai_core::OperationDescriptor;

    /// A structurally valid bundle whose components are garbage. The
    /// token-only operation set keeps the capability⇔processor rule out
    /// of the way (no Text values ⇒ no tokenizer required).
    fn garbage_bundle() -> Vec<u8> {
        let mut builder = BundleBuilder::new(ENGINE_UOR_R4, ARTIFACT_FORMAT_R4G1, "garbage")
            .set_operations(vec![OperationDescriptor {
                name: "predict".to_owned(),
                inputs: vec![],
                outputs: vec![],
                streaming: false,
            }]);
        builder
            .add_component(ArtifactRole::Graph, b"not an R4G1 graph")
            .unwrap();
        builder
            .add_component(ArtifactRole::SignatureArtifact, b"not a teacher artifact")
            .unwrap();
        builder
            .add_component(ArtifactRole::ScoreReport, b"{}")
            .unwrap();
        builder.finish().unwrap()
    }

    #[test]
    fn garbage_components_fail_with_engine_init() {
        let bytes = garbage_bundle();
        let bundle = Bundle::parse_verified(&bytes).unwrap();
        let err = Engine::load(&bundle).unwrap_err();
        assert_eq!(err.category(), ErrorCategory::EngineInit);
    }

    #[test]
    fn corrupted_component_bytes_fail_with_integrity_mismatch() {
        let mut bytes = garbage_bundle();
        // Corrupt the final payload byte (the score report) without
        // touching lengths: parse still succeeds, digest verification
        // inside Engine::load must not.
        let last = bytes.last_mut().unwrap();
        *last = last.wrapping_add(1);
        let bundle = Bundle::parse(&bytes).unwrap();
        let err = Engine::load(&bundle).unwrap_err();
        assert_eq!(err.category(), ErrorCategory::IntegrityMismatch);
    }

    #[test]
    fn non_r4_bundle_is_rejected() {
        let builder = BundleBuilder::new("other-engine", "OTHER", "x");
        let bytes = builder.finish().unwrap();
        let bundle = Bundle::parse_verified(&bytes).unwrap();
        let err = Engine::load(&bundle).unwrap_err();
        assert_eq!(err.category(), ErrorCategory::EngineInit);
    }

    #[test]
    fn missing_mandatory_role_is_bundle_decode() {
        let mut builder = BundleBuilder::new(ENGINE_UOR_R4, ARTIFACT_FORMAT_R4G1, "x");
        builder
            .add_component(ArtifactRole::Graph, b"not an R4G1 graph")
            .unwrap();
        // No signature-artifact / score-report: the builder's own
        // semantic validation must reject at finish().
        let err = builder.finish().unwrap_err();
        assert_eq!(err.category(), ErrorCategory::BundleDecode);
    }

    #[test]
    fn text_helpers_require_a_tokenizer() {
        let err = encode_text_with(false, "hello", |_, _| None).unwrap_err();
        assert_eq!(err.category(), ErrorCategory::ProcessorMismatch);
        let err = decode_tokens_with(false, &[1, 2], |_, _| None).unwrap_err();
        assert_eq!(err.category(), ErrorCategory::ProcessorMismatch);
    }

    #[test]
    fn encode_grows_scratch_until_it_fits() {
        // Simulates a tokenizer that needs one token per byte and
        // reports failure while the scratch is too small.
        let tokens = encode_text_with(true, "abcd", |text, out| {
            if out.len() < text.len() {
                return None;
            }
            for (slot, _) in out.iter_mut().zip(text.bytes()) {
                *slot = 7;
            }
            Some(text.len())
        })
        .unwrap();
        assert_eq!(tokens, vec![7u32; 4]);
    }

    #[test]
    fn decode_grows_scratch_until_it_fits() {
        let text = decode_tokens_with(true, &[1, 2, 3], |tokens, out| {
            let needed = tokens.len() * 8;
            if out.len() < needed {
                return None;
            }
            out[..needed].copy_from_slice(&vec![b'a'; needed]);
            Some(needed)
        })
        .unwrap();
        assert_eq!(text, "a".repeat(24));
    }

    #[test]
    fn persistently_failing_tokenizer_is_inference_error() {
        let err = encode_text_with(true, "hello", |_, _| None).unwrap_err();
        assert_eq!(err.category(), ErrorCategory::Inference);
        let err = decode_tokens_with(true, &[1], |_, _| None).unwrap_err();
        assert_eq!(err.category(), ErrorCategory::Inference);
    }
}
