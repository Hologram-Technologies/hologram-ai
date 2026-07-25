//! Inference requests, completions, status, and streaming events.

use alloc::string::String;
use alloc::vec::Vec;

use crate::value::{MediaData, Payload, Value};

/// Resolution status of a produced token or answer, mapped from the
/// engine's status taxonomy. Abstention is **not** a status — it is a
/// [`FinishReason`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolutionStatus {
    /// Backed by an exact observed context.
    Exact,
    /// Backed by graph structure.
    Graph,
    /// Novel context (unobserved); only served if policy permits.
    Novel,
}

/// Why an inference operation finished.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    /// The engine emitted an end-of-sequence decision.
    EndOfSequence,
    /// The caller's output limit was reached.
    OutputLimit,
    /// The engine abstained under the status policy. No token is
    /// fabricated; `output` may be empty.
    Abstained,
    /// The operation was cancelled through the cancellation token.
    Cancelled,
}

/// An opaque witness/certification artifact produced alongside a result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Witness {
    /// Witness kind (engine-defined, e.g. `r4-score-step`).
    pub kind: String,
    /// Canonical witness bytes.
    pub data: Vec<u8>,
}

/// A named output value set.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct InferenceOutput {
    pub values: Vec<(String, Value)>,
}

impl InferenceOutput {
    /// First text output with the given name, if present.
    pub fn text(&self, name: &str) -> Option<&str> {
        self.values.iter().find_map(|(n, v)| match v {
            Value::Text(s) if n == name => Some(s.as_str()),
            _ => None,
        })
    }

    /// First token output with the given name, if present.
    pub fn tokens(&self, name: &str) -> Option<&[u32]> {
        self.values.iter().find_map(|(n, v)| match v {
            Value::Tokens(ts) if n == name => Some(ts.as_slice()),
            _ => None,
        })
    }
}

/// A typed successful completion. Abstention is represented here, never as
/// an error and never as a fabricated token.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InferenceCompletion {
    pub finish_reason: FinishReason,
    /// Status of the final produced output; `None` when abstained with no
    /// output produced.
    pub status: Option<ResolutionStatus>,
    /// Whether context widening occurred during resolution.
    pub widened: bool,
    pub output: InferenceOutput,
    pub witness: Option<Witness>,
}

/// A multimodal inference request. Built via [`InferenceRequest::builder`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct InferenceRequest {
    /// Named input values.
    pub values: Vec<(String, Value)>,
    /// Maximum number of output tokens to produce.
    pub max_output_tokens: Option<u32>,
}

impl InferenceRequest {
    pub fn builder() -> InferenceRequestBuilder {
        InferenceRequestBuilder::default()
    }

    /// First text input with the given name, if present.
    pub fn text(&self, name: &str) -> Option<&str> {
        self.values.iter().find_map(|(n, v)| match v {
            Value::Text(s) if n == name => Some(s.as_str()),
            _ => None,
        })
    }

    /// First token input with the given name, if present.
    pub fn tokens(&self, name: &str) -> Option<&[u32]> {
        self.values.iter().find_map(|(n, v)| match v {
            Value::Tokens(ts) if n == name => Some(ts.as_slice()),
            _ => None,
        })
    }
}

/// Builder for [`InferenceRequest`].
#[derive(Debug, Default)]
pub struct InferenceRequestBuilder {
    values: Vec<(String, Value)>,
    max_output_tokens: Option<u32>,
}

impl InferenceRequestBuilder {
    /// Add a text input (e.g. `"prompt"`).
    pub fn text(mut self, name: &str, text: &str) -> Self {
        self.values.push((name.into(), Value::Text(text.into())));
        self
    }

    /// Add a token-sequence input.
    pub fn tokens(mut self, name: &str, tokens: Vec<u32>) -> Self {
        self.values.push((name.into(), Value::Tokens(tokens)));
        self
    }

    /// Add an inline image input.
    pub fn image(mut self, name: &str, media_type: &str, data: Vec<u8>) -> Self {
        self.values.push((
            name.into(),
            Value::Image(MediaData {
                media_type: media_type.into(),
                payload: Payload::Inline(data),
            }),
        ));
        self
    }

    /// Add an inline audio input.
    pub fn audio(mut self, name: &str, media_type: &str, data: Vec<u8>) -> Self {
        self.values.push((
            name.into(),
            Value::Audio(MediaData {
                media_type: media_type.into(),
                payload: Payload::Inline(data),
            }),
        ));
        self
    }

    /// Add an image or audio input by content-addressed κ reference, so
    /// large media does not need to be copied between layers.
    pub fn content_ref(mut self, name: &str, media_type: &str, kappa: &str) -> Self {
        self.values.push((
            name.into(),
            Value::ContentRef {
                media_type: media_type.into(),
                kappa: kappa.into(),
            },
        ));
        self
    }

    /// Add an opaque inline-bytes input with a media type.
    pub fn bytes(mut self, name: &str, media_type: &str, data: Vec<u8>) -> Self {
        self.values.push((
            name.into(),
            Value::Bytes {
                media_type: media_type.into(),
                data,
            },
        ));
        self
    }

    /// Add a canonical-encoded structured input.
    pub fn structured(mut self, name: &str, data: Vec<u8>) -> Self {
        self.values.push((name.into(), Value::Structured(data)));
        self
    }

    pub fn max_output_tokens(mut self, n: u32) -> Self {
        self.max_output_tokens = Some(n);
        self
    }

    pub fn build(self) -> InferenceRequest {
        InferenceRequest {
            values: self.values,
            max_output_tokens: self.max_output_tokens,
        }
    }
}

/// A streaming event emitted during an operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamEvent {
    /// A produced output token.
    Token(u32),
    /// Decoded text for a produced token (when a tokenizer is present).
    Text(String),
    /// Resolution status of the most recent step.
    Status(ResolutionStatus),
    /// Context widening occurred.
    Widened,
    /// The operation finished.
    Finished(FinishReason),
}

/// Sink for streaming events. Implementations must not assume allocation
/// is free: engines call this from the steady-state path.
pub trait EventSink {
    fn on_event(&mut self, event: StreamEvent);
}

/// A no-op event sink.
pub struct NullEventSink;

impl EventSink for NullEventSink {
    fn on_event(&mut self, _event: StreamEvent) {}
}
