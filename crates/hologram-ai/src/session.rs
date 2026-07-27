//! Inference sessions: generic invocation, prediction, generation.

use hologram_ai_bundle::Bundle;
use hologram_ai_core::{
    AiError, AiResult, ErrorCategory, EventSink, FinishReason, InferenceCompletion,
    InferenceOutput, InferenceRequest, NullEventSink, Value, Witness,
};
use hologram_ai_r4::{Engine, FinishInfo, PredictOutcome, Prediction};

use crate::model::Model;

/// A live inference session. Engine state is initialized and all
/// fixed-capacity buffers are allocated; the steady-state path is
/// allocation-free (ADR-0009).
pub struct Session {
    engine: Engine,
}

impl Session {
    pub(crate) fn new(model: &Model) -> AiResult<Self> {
        let bundle = Bundle::parse(model.bundle_bytes())?;
        let engine = Engine::load(&bundle)?;
        Ok(Self { engine })
    }

    /// Reset the session to its initial state (no reallocation).
    pub fn reset(&mut self) {
        self.engine.reset();
    }

    /// Generic manifest-driven invocation.
    pub fn invoke(
        &mut self,
        operation: &str,
        request: InferenceRequest,
    ) -> AiResult<InferenceCompletion> {
        match operation {
            "generate" => self.invoke_generate(&request, &mut NullEventSink),
            "predict" => self.invoke_predict(&request),
            other => Err(AiError::new(
                ErrorCategory::UnsupportedCapability,
                format!("operation '{other}' is not advertised by this model"),
            )),
        }
    }

    /// Generic invocation with a streaming event sink.
    pub fn invoke_streaming(
        &mut self,
        operation: &str,
        request: InferenceRequest,
        events: &mut dyn EventSink,
    ) -> AiResult<InferenceCompletion> {
        match operation {
            "generate" => self.invoke_generate(&request, events),
            "predict" => self.invoke_predict(&request),
            other => Err(AiError::new(
                ErrorCategory::UnsupportedCapability,
                format!("operation '{other}' does not support streaming"),
            )),
        }
    }

    fn invoke_generate(
        &mut self,
        request: &InferenceRequest,
        events: &mut dyn EventSink,
    ) -> AiResult<InferenceCompletion> {
        let seed = self.request_tokens(request, "prompt")?;
        let max = request.max_output_tokens.unwrap_or(64) as usize;
        let mut buf = vec![0u32; max];
        let finish = self.engine.generate_into(&seed, &mut buf, events)?;
        Ok(self.completion_from_finish(&buf[..finish.produced], &finish))
    }

    fn invoke_predict(&mut self, request: &InferenceRequest) -> AiResult<InferenceCompletion> {
        let window = self.request_tokens(request, "context")?;
        let mut prediction = Prediction::default();
        self.predict_next_into(&window, &mut prediction)?;
        Ok(match prediction.outcome {
            PredictOutcome::Serve {
                token,
                status,
                widened,
            } => InferenceCompletion {
                finish_reason: FinishReason::EndOfSequence,
                status: Some(status),
                widened,
                output: InferenceOutput {
                    values: vec![("token".into(), Value::Tokens(vec![token]))],
                },
                witness: None,
            },
            PredictOutcome::Abstain { widened } => InferenceCompletion {
                finish_reason: FinishReason::Abstained,
                status: None,
                widened,
                output: InferenceOutput::default(),
                witness: None,
            },
        })
    }

    /// Allocation-free single-step prediction into a caller-owned output.
    pub fn predict_next_into(
        &mut self,
        window: &[u32],
        prediction: &mut Prediction,
    ) -> AiResult<()> {
        self.engine.predict_next_into(window, prediction)
    }

    /// Allocation-free deterministic generation into a caller-owned buffer.
    pub fn generate_into(
        &mut self,
        seed: &[u32],
        output_tokens: &mut [u32],
        events: &mut dyn EventSink,
    ) -> AiResult<FinishInfo> {
        self.engine.generate_into(seed, output_tokens, events)
    }

    fn request_tokens(&self, request: &InferenceRequest, name: &str) -> AiResult<Vec<u32>> {
        if let Some(tokens) = request.tokens(name) {
            return Ok(tokens.to_vec());
        }
        if let Some(text) = request.text(name) {
            return self.engine.encode_text(text);
        }
        Err(AiError::invalid_argument(format!(
            "request is missing a '{name}' text or token input"
        )))
    }

    fn completion_from_finish(&self, tokens: &[u32], finish: &FinishInfo) -> InferenceCompletion {
        let text = self.engine.decode_tokens(tokens).unwrap_or_default();
        InferenceCompletion {
            finish_reason: finish.finish_reason,
            status: finish.status,
            widened: finish.widened,
            output: InferenceOutput {
                values: vec![
                    ("tokens".into(), Value::Tokens(tokens.to_vec())),
                    ("text".into(), Value::Text(text)),
                ],
            },
            witness: finish
                .witness
                .clone()
                .map(|(kind, data)| Witness { kind, data }),
        }
    }
}
