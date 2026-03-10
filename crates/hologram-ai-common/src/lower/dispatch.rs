//! Op dispatch: `AiOp` → hologram `GraphOp` (all native float ops).
//!
//! Every supported `AiOp` maps to a native `GraphOp::Float(FloatOp::...)`.
//! No `CustomOpRegistry` is needed — archives are fully self-describing.

use crate::ir::AiOp;
use hologram::{f32_to_bits, FloatOp, GraphOp};
use hologram_ai_quant::QuantScheme;

/// Categorised dispatch target for a single `AiOp`.
///
/// Used only at model-compilation time (in `lower()`), not during inference.
#[derive(Debug)]
#[allow(clippy::large_enum_variant)]
pub enum DispatchTarget {
    /// Native hologram graph op (Lut, Prim, Float, etc.).
    GraphOp(GraphOp),
    /// Native float op that needs tensor shape resolution from `tensor_info`.
    /// Builder resolves shapes and constructs the `GraphOp::Float(FloatOp::...)`.
    FloatNeedsShape,
    /// Pass-through (identity: one input, same output).
    Identity,
    /// Lowering not yet supported.
    Unsupported { reason: &'static str },
}

/// Classify an `AiOp` into its dispatch target.
pub fn dispatch(op: &AiOp) -> DispatchTarget {
    use AiOp::*;
    use DispatchTarget as D;

    match op {
        // ── Native FloatOp: parameterless unary activations ──────────────
        Relu => D::GraphOp(GraphOp::Float(FloatOp::Relu)),
        Gelu => D::GraphOp(GraphOp::Float(FloatOp::Gelu)),
        GeluApprox => D::GraphOp(GraphOp::Float(FloatOp::Gelu)),
        Silu => D::GraphOp(GraphOp::Float(FloatOp::Silu)),
        Tanh => D::GraphOp(GraphOp::Float(FloatOp::Tanh)),
        Sigmoid => D::GraphOp(GraphOp::Float(FloatOp::Sigmoid)),

        // ── Native FloatOp: parameterless unary math ─────────────────────
        Exp => D::GraphOp(GraphOp::Float(FloatOp::Exp)),
        Log => D::GraphOp(GraphOp::Float(FloatOp::Log)),
        Sqrt => D::GraphOp(GraphOp::Float(FloatOp::Sqrt)),
        Abs => D::GraphOp(GraphOp::Float(FloatOp::Abs)),
        Neg => D::GraphOp(GraphOp::Float(FloatOp::Neg)),
        Reciprocal => D::GraphOp(GraphOp::Float(FloatOp::Reciprocal)),
        Cos => D::GraphOp(GraphOp::Float(FloatOp::Cos)),
        Sin => D::GraphOp(GraphOp::Float(FloatOp::Sin)),
        Sign => D::GraphOp(GraphOp::Float(FloatOp::Sign)),
        Floor => D::GraphOp(GraphOp::Float(FloatOp::Floor)),
        Ceil => D::GraphOp(GraphOp::Float(FloatOp::Ceil)),
        Round => D::GraphOp(GraphOp::Float(FloatOp::Round)),
        Erf => D::GraphOp(GraphOp::Float(FloatOp::Erf)),
        IsNaN => D::GraphOp(GraphOp::Float(FloatOp::IsNaN)),

        // Clip: no min/max in AiOp, use f32 full range as default
        Clip => D::GraphOp(GraphOp::Float(FloatOp::Clip {
            min: f32_to_bits(f32::NEG_INFINITY),
            max: f32_to_bits(f32::INFINITY),
        })),

        // ── Native FloatOp: parameterless binary arithmetic ──────────────
        Add => D::GraphOp(GraphOp::Float(FloatOp::Add)),
        Sub => D::GraphOp(GraphOp::Float(FloatOp::Sub)),
        Mul => D::GraphOp(GraphOp::Float(FloatOp::Mul)),
        Div => D::GraphOp(GraphOp::Float(FloatOp::Div)),
        Pow => D::GraphOp(GraphOp::Float(FloatOp::Pow)),
        Mod => D::GraphOp(GraphOp::Float(FloatOp::Mod)),
        Min => D::GraphOp(GraphOp::Float(FloatOp::Min)),
        Max => D::GraphOp(GraphOp::Float(FloatOp::Max)),

        // ── Boolean ops ──────────────────────────────────────────────────
        And => D::GraphOp(GraphOp::Float(FloatOp::And)),
        Or => D::GraphOp(GraphOp::Float(FloatOp::Or)),
        Xor => D::GraphOp(GraphOp::Float(FloatOp::Xor)),
        Not => D::GraphOp(GraphOp::Float(FloatOp::Not)),

        // ── Comparisons ──────────────────────────────────────────────────
        Equal => D::GraphOp(GraphOp::Float(FloatOp::Equal)),
        Less => D::GraphOp(GraphOp::Float(FloatOp::Less)),
        LessOrEqual => D::GraphOp(GraphOp::Float(FloatOp::LessOrEqual)),
        Greater => D::GraphOp(GraphOp::Float(FloatOp::Greater)),
        GreaterOrEqual => D::GraphOp(GraphOp::Float(FloatOp::GreaterOrEqual)),

        // ── Native FloatOp: parameterless fused ops ──────────────────────
        FusedSwiGLU => D::GraphOp(GraphOp::Float(FloatOp::FusedSwiGLU)),

        // ── Native FloatOp: params from AiOp (no shape resolution needed) ─
        RotaryEmbedding { base, dim } => D::GraphOp(GraphOp::Float(FloatOp::RotaryEmbedding {
            dim: *dim,
            base: f32_to_bits(*base),
        })),

        // ── Native FloatOp: pass-through / structural ────────────────────
        Reshape { .. } => D::GraphOp(GraphOp::Float(FloatOp::Reshape)),
        Transpose { perm } => {
            let mut arr = [0u8; 8];
            let ndim = perm.len().min(8) as u8;
            for (i, &p) in perm.iter().take(8).enumerate() {
                arr[i] = p as u8;
            }
            D::GraphOp(GraphOp::Float(FloatOp::Transpose { perm: arr, ndim }))
        }
        Cast { .. } => D::FloatNeedsShape,
        Shape { .. } => D::FloatNeedsShape,
        GatherND { .. } => D::GraphOp(GraphOp::Float(FloatOp::GatherND)),
        Where => D::GraphOp(GraphOp::Float(FloatOp::Where)),
        Range => D::GraphOp(GraphOp::Float(FloatOp::Range)),
        Dequantize => D::GraphOp(GraphOp::Float(FloatOp::Dequantize)),
        Flatten { .. } => D::GraphOp(GraphOp::Float(FloatOp::Reshape)),
        Embed => D::FloatNeedsShape,

        // ── KV-cache (structural, pass-through at graph level) ──────────
        KvSlotWrite { .. } => D::Identity,
        KvSlotRead { .. } => D::Identity,

        // ── Identity pass-through ────────────────────────────────────────
        Squeeze { .. } => D::Identity,
        Unsqueeze { .. } => D::Identity,
        Expand => D::GraphOp(GraphOp::Float(FloatOp::Reshape)),
        Slice { .. } => D::FloatNeedsShape,
        Split { .. } => D::Identity,
        Tile { .. } => D::Identity,
        Identity => D::Identity,

        // ── Native FloatOp: needs shape resolution from tensor_info ──────
        MatMul | BatchMatMul => D::FloatNeedsShape,
        Gemm { .. } => D::FloatNeedsShape,
        Softmax { .. } => D::FloatNeedsShape,
        LogSoftmax { .. } => D::FloatNeedsShape,
        RmsNorm { .. } => D::FloatNeedsShape,
        LayerNorm { .. } => D::FloatNeedsShape,
        ReduceSum { .. } => D::FloatNeedsShape,
        ReduceMean { .. } => D::FloatNeedsShape,
        ReduceMax { .. } => D::FloatNeedsShape,
        ReduceMin { .. } => D::FloatNeedsShape,
        Gather { .. } | GatherElements { .. } => D::FloatNeedsShape,
        Concat { .. } => D::FloatNeedsShape,

        // ── Attention (needs head_dim/scale/causal from AiOp) ───────────
        MultiHeadAttention { .. } => D::FloatNeedsShape,
        GroupedQueryAttention { .. } => D::FloatNeedsShape,
        FlashAttentionHint => D::FloatNeedsShape,

        // ── Quantized matmul (unsupported — use LUT path) ───────────────
        QuantizedMatMul {
            lhs_scheme: QuantScheme::Q4_0,
            ..
        } => D::Unsupported {
            reason: "QuantizedMatMul Q4_0: use builder.matmul_lut_4bit directly",
        },
        QuantizedMatMul {
            lhs_scheme: QuantScheme::Q8_0,
            ..
        } => D::Unsupported {
            reason: "QuantizedMatMul Q8_0: use builder.matmul_lut_8bit directly",
        },
        QuantizedMatMul { .. } => D::Unsupported {
            reason: "unsupported quant scheme for GEMM",
        },

        // ── Opaque ──────────────────────────────────────────────────────
        Opaque { .. } => D::Unsupported {
            reason: "opaque op cannot be lowered",
        },

        // ── Remaining ops: Phase 2/3 expansion ──────────────────────────
        _ => D::Unsupported {
            reason: "op not yet implemented in lowering",
        },
    }
}
