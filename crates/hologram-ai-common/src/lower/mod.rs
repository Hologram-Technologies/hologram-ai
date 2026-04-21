pub mod builder;
pub mod dispatch;
pub mod quantize_graph;
pub mod shape_spec_bridge;
pub mod strategy;

pub use builder::{lower, LoweringOptions, LoweringOutput, QuantStrategy};
pub use quantize_graph::{quantize_graph, QuantLevel, QuantizeStats};
pub use shape_spec_bridge::{
    float_op_to_shape_spec_repr, resolve_spec, walk_shape_context, ShapeProjection,
};
pub use strategy::{ConcreteStrategy, DeferredStrategy, LoweringStrategy, SymbolicLowering};

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
