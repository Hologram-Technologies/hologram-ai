pub mod builder;
pub mod dispatch;
pub mod dtype;

pub use builder::{lower, LoweringOptions, LoweringOutput, QuantStrategy};
pub use dispatch::{dispatch, AttrSpec, DesugarKind, OpPlan};

// The runtime shape-projection / strategy / op-resolver machinery is removed:
// hologram's compiler derives every op parameter from the concrete interned
// shapes hologram-ai supplies (architecture §5.1, §5.3). Quantization is a
// weight-encoding concern realized as `QuantAttrs` at the weight boundary.

/// Optimization profile for a compilation component.
///
/// Controls which optimization passes run. LLM-specific passes
/// (attention fusion, KV-cache injection) are skipped for non-transformer
/// components like autoencoders or generative heads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptProfile {
    /// Full MVP pipeline: attention fusion, KV injection, SwiGLU fusion, etc.
    Llm,
    /// Shape/data propagation + constant folding only. No attention-specific passes.
    Generic,
}

/// Which phase of execution this graph is being lowered for.
///
/// For LLMs, the same `AiGraph` is lowered twice: once for prefill (prompt
/// processing) and once for single-token decode. The phase determines graph
/// I/O naming and the layer descriptor metadata.
///
/// `Named` allows arbitrary component names for multi-component models
/// (e.g., "ae.encoder", "gen.head").
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LowerPhase {
    /// Full-sequence forward pass (prompt processing).
    Prefill,
    /// Single-token autoregressive decode step.
    Decode,
    /// Non-LLM model: single forward pass with generic I/O names.
    Forward,
    /// Arbitrary component name for multi-component pipelines.
    Named(String),
}

impl LowerPhase {
    /// Layer name for this phase in the archive.
    pub fn layer_name(&self) -> &str {
        match self {
            Self::Prefill => "lm.prefill",
            Self::Decode => "lm.decode",
            Self::Forward => "model.forward",
            Self::Named(name) => name.as_str(),
        }
    }
}
