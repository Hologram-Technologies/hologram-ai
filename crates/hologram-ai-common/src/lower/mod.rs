pub mod builder;
pub mod dispatch;
pub mod strategy;

pub use builder::{lower, LoweringOptions, LoweringOutput, QuantStrategy};
pub use strategy::{ConcreteStrategy, DeferredStrategy, LoweringStrategy, SymbolicLowering};

/// Which phase of execution this graph is being lowered for.
///
/// For LLMs, the same `AiGraph` is lowered twice: once for prefill (prompt
/// processing) and once for single-token decode. The phase determines graph
/// I/O naming and the layer descriptor metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LowerPhase {
    /// Full-sequence forward pass (prompt processing).
    Prefill,
    /// Single-token autoregressive decode step.
    Decode,
    /// Non-LLM model: single forward pass with generic I/O names.
    Forward,
}

impl LowerPhase {
    /// Layer name for this phase in the archive.
    pub fn layer_name(&self) -> &'static str {
        match self {
            Self::Prefill => "lm.prefill",
            Self::Decode => "lm.decode",
            Self::Forward => "model.forward",
        }
    }
}
