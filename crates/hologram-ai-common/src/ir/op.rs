use super::{dtype::DType, param::AiParam};
use hologram_ai_quant::QuantScheme;

/// Behavioral category for shape/dtype/value inference.
///
/// Most `AiOp` variants fall into a standard category with uniform inference
/// rules. Only `Custom` ops need per-variant logic in the propagation passes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpCategory {
    /// `output_shape = input[0].shape`, `output_dtype = input[0].dtype`.
    /// Value propagation: pass-through (values unchanged).
    UnaryElementwise,
    /// `output_shape = broadcast(input[0], input[1])`, `output_dtype = input[0].dtype`.
    /// Value propagation: elementwise arithmetic on i64 values.
    BinaryElementwise,
    /// `output_shape = broadcast(input[0], input[1])`, `output_dtype = BOOL`.
    /// Value propagation: none.
    BinaryComparison,
    /// `output_shape = input[0].shape` (extra inputs like weights are ignored).
    /// `output_dtype = input[0].dtype`. Value propagation: none.
    ShapePreserving,
    /// Op-specific shape/dtype/value rules. Each op gets a dedicated match arm
    /// in the propagation passes.
    Custom,
}

/// How scatter reduction is applied.
#[derive(Debug, Clone, PartialEq)]
pub enum ScatterReduce {
    None,
    Add,
    Mul,
    Min,
    Max,
}

/// Canonical AI IR operation.
///
/// This is the full operation set from `specs/docs/lowering.md`.
/// Variants produced by optimization passes carry a `Fused` prefix.
#[derive(Debug, Clone)]
pub enum AiOp {
    // ── Core linear algebra ────────────────────────────────────────────────
    MatMul,
    BatchMatMul,
    Gemm {
        alpha: f32,
        beta: f32,
        trans_a: bool,
        trans_b: bool,
    },
    Einsum {
        equation: String,
    },

    // ── Activations ────────────────────────────────────────────────────────
    Relu,
    Gelu,
    GeluApprox,
    Silu,
    Tanh,
    Sigmoid,
    Softmax {
        axis: i64,
    },
    LogSoftmax {
        axis: i64,
    },

    // ── Normalization ──────────────────────────────────────────────────────
    LayerNorm {
        axis: i64,
        epsilon: f32,
    },
    RmsNorm {
        epsilon: f32,
    },
    GroupNorm {
        num_groups: u32,
        epsilon: f32,
    },
    BatchNorm {
        epsilon: f32,
        momentum: f32,
        training: bool,
    },

    // ── High-level attention (semantic ops, pre-fusion) ────────────────────
    MultiHeadAttention {
        num_heads: u32,
        head_dim: u32,
        scale: Option<f32>,
        causal: bool,
    },
    GroupedQueryAttention {
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        scale: Option<f32>,
        causal: bool,
    },
    /// Hint from importer; lowering decides if flash attention is available.
    FlashAttentionHint,

    // ── Positional encoding ────────────────────────────────────────────────
    RotaryEmbedding {
        base: f32,
        dim: u32,
    },
    AlibiSlope,

    // ── Shape manipulation ─────────────────────────────────────────────────
    Reshape {
        allow_zero: bool,
    },
    Transpose {
        perm: Vec<u32>,
    },
    Concat {
        axis: i64,
    },
    Split {
        axis: i64,
        sizes: Vec<u64>,
    },
    Slice {
        axes: Vec<i64>,
        starts: Vec<i64>,
        ends: Vec<i64>,
        steps: Vec<i64>,
    },
    Gather {
        axis: i64,
    },
    GatherElements {
        axis: i64,
    },
    Scatter {
        axis: i64,
        reduce: ScatterReduce,
    },
    Unsqueeze {
        axes: Vec<i64>,
    },
    Squeeze {
        axes: Vec<i64>,
    },
    Expand,
    Tile {
        repeats: Vec<u64>,
    },
    GatherND {
        batch_dims: i64,
    },
    /// Extract shape of input tensor as a 1-D INT64 tensor.
    /// `start`/`end` (opset 15+) slice the output to a subrange of dims.
    Shape {
        start: Option<i64>,
        end: Option<i64>,
    },
    /// Conditional element selection: Where(cond, x, y).
    Where,
    /// Generate a range [start, limit) with step.
    Range,
    Flatten {
        axis: i64,
    },

    // ── Elementwise binary ─────────────────────────────────────────────────
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Mod,
    Min,
    Max,
    And,
    Or,
    Xor,
    Not,
    Equal,
    Less,
    LessOrEqual,
    Greater,
    GreaterOrEqual,

    // ── Elementwise unary ──────────────────────────────────────────────────
    Abs,
    Neg,
    Sqrt,
    Exp,
    Log,
    Sign,
    Floor,
    Ceil,
    Round,
    Clip,
    Erf,
    Reciprocal,
    Cos,
    Sin,
    IsNaN,

    // ── Reductions ─────────────────────────────────────────────────────────
    ReduceSum {
        axes: Vec<i64>,
        keepdims: bool,
    },
    ReduceMean {
        axes: Vec<i64>,
        keepdims: bool,
    },
    ReduceMax {
        axes: Vec<i64>,
        keepdims: bool,
    },
    ReduceMin {
        axes: Vec<i64>,
        keepdims: bool,
    },
    ArgMax {
        axis: i64,
        keepdims: bool,
    },
    ArgMin {
        axis: i64,
        keepdims: bool,
    },

    // ── Embeddings ─────────────────────────────────────────────────────────
    /// token_ids → embedding vectors via weight-table lookup.
    Embed,
    /// Generate causal attention mask.
    CausalMask,

    // ── Quantization (explicit in IR) ──────────────────────────────────────
    Quantize {
        scheme: QuantScheme,
    },
    Dequantize,
    QuantizedMatMul {
        lhs_scheme: QuantScheme,
        rhs_scheme: QuantScheme,
    },

    // ── KV-cache ─────────────────────────────────────────────────────────────
    /// Write K/V tensors into the KV-cache for a given layer.
    KvSlotWrite {
        layer: usize,
    },
    /// Read cached K/V tensors from the KV-cache for a given layer.
    KvSlotRead {
        layer: usize,
    },

    // ── Fused ops (produced by optimization passes) ────────────────────────
    /// gate × up → silu(gate) × up
    FusedSwiGLU,
    /// x + residual → layernorm
    FusedLayerNormResidual,

    // ── Type / control ─────────────────────────────────────────────────────
    Cast {
        to: DType,
    },
    Constant {
        value: AiParam,
    },
    Identity,

    /// Fallback for ops the importer could not map.
    Opaque {
        op_type: String,
        raw_attrs: Vec<u8>,
    },
}

impl AiOp {
    /// Behavioral category for shape/dtype/value inference.
    ///
    /// IMPORTANT: This match is exhaustive (no catch-all). When adding a new
    /// `AiOp` variant, the compiler forces you to assign a category, which
    /// automatically gives it correct shape/dtype/value propagation for the
    /// standard categories.
    pub fn category(&self) -> OpCategory {
        use OpCategory::*;
        match self {
            // ── Unary elementwise: output = input shape/dtype ─────────────
            AiOp::Relu
            | AiOp::Gelu
            | AiOp::GeluApprox
            | AiOp::Silu
            | AiOp::Tanh
            | AiOp::Sigmoid
            | AiOp::Abs
            | AiOp::Neg
            | AiOp::Sqrt
            | AiOp::Exp
            | AiOp::Log
            | AiOp::Sign
            | AiOp::Floor
            | AiOp::Ceil
            | AiOp::Round
            | AiOp::Clip
            | AiOp::Erf
            | AiOp::Reciprocal
            | AiOp::Cos
            | AiOp::Sin
            | AiOp::Not
            | AiOp::Identity
            | AiOp::Dequantize => UnaryElementwise,

            // ── Binary elementwise: broadcast shape, first-input dtype ────
            AiOp::Add
            | AiOp::Sub
            | AiOp::Mul
            | AiOp::Div
            | AiOp::Pow
            | AiOp::Mod
            | AiOp::Min
            | AiOp::Max
            | AiOp::And
            | AiOp::Or
            | AiOp::Xor => BinaryElementwise,

            // ── Binary comparison: broadcast shape, BOOL dtype ────────────
            AiOp::Equal
            | AiOp::Less
            | AiOp::LessOrEqual
            | AiOp::Greater
            | AiOp::GreaterOrEqual
            | AiOp::IsNaN => BinaryComparison,

            // ── Shape-preserving: output shape = first input shape ────────
            AiOp::Softmax { .. }
            | AiOp::LogSoftmax { .. }
            | AiOp::RmsNorm { .. }
            | AiOp::LayerNorm { .. }
            | AiOp::GroupNorm { .. }
            | AiOp::BatchNorm { .. }
            | AiOp::RotaryEmbedding { .. }
            | AiOp::FusedSwiGLU
            | AiOp::FusedLayerNormResidual
            | AiOp::KvSlotWrite { .. }
            | AiOp::KvSlotRead { .. }
            | AiOp::Quantize { .. } => ShapePreserving,

            // ── Custom: op-specific shape/dtype/value rules ───────────────
            AiOp::MatMul
            | AiOp::BatchMatMul
            | AiOp::Gemm { .. }
            | AiOp::Einsum { .. }
            | AiOp::MultiHeadAttention { .. }
            | AiOp::GroupedQueryAttention { .. }
            | AiOp::FlashAttentionHint
            | AiOp::AlibiSlope
            | AiOp::Reshape { .. }
            | AiOp::Transpose { .. }
            | AiOp::Concat { .. }
            | AiOp::Split { .. }
            | AiOp::Slice { .. }
            | AiOp::Gather { .. }
            | AiOp::GatherElements { .. }
            | AiOp::GatherND { .. }
            | AiOp::Scatter { .. }
            | AiOp::Unsqueeze { .. }
            | AiOp::Squeeze { .. }
            | AiOp::Expand
            | AiOp::Tile { .. }
            | AiOp::Flatten { .. }
            | AiOp::Shape { .. }
            | AiOp::Where
            | AiOp::Range
            | AiOp::ReduceSum { .. }
            | AiOp::ReduceMean { .. }
            | AiOp::ReduceMax { .. }
            | AiOp::ReduceMin { .. }
            | AiOp::ArgMax { .. }
            | AiOp::ArgMin { .. }
            | AiOp::Embed
            | AiOp::CausalMask
            | AiOp::QuantizedMatMul { .. }
            | AiOp::Cast { .. }
            | AiOp::Constant { .. }
            | AiOp::Opaque { .. } => Custom,
        }
    }
}
