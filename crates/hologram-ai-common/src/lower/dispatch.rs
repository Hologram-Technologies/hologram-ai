//! Op dispatch: `AiOp` → hologram `GraphOp` or custom op ID.

use hologram::{CustomOpId, GraphOp, LutOp, PrimOp};
use hologram_ai_quant::QuantScheme;
use crate::ir::AiOp;

// ── Custom op numeric IDs ──────────────────────────────────────────────────────
// Stable IDs assigned to each custom handler type.

pub const ATTN_OP_ID:       CustomOpId = CustomOpId(1);
pub const GQA_OP_ID:        CustomOpId = CustomOpId(2);
pub const RMS_NORM_OP_ID:   CustomOpId = CustomOpId(3);
pub const LAYER_NORM_OP_ID: CustomOpId = CustomOpId(4);
pub const SOFTMAX_OP_ID:    CustomOpId = CustomOpId(5);
pub const ROPE_OP_ID:       CustomOpId = CustomOpId(6);
pub const EMBED_OP_ID:      CustomOpId = CustomOpId(7);
pub const SWIGLU_OP_ID:     CustomOpId = CustomOpId(8);
pub const DEQUANT_OP_ID:    CustomOpId = CustomOpId(9);
pub const RESHAPE_OP_ID:    CustomOpId = CustomOpId(10);
pub const CAST_OP_ID:       CustomOpId = CustomOpId(11);
pub const CONCAT_OP_ID:     CustomOpId = CustomOpId(12);

/// Categorised dispatch target for a single `AiOp`.
#[derive(Debug)]
pub enum DispatchTarget {
    /// Native hologram graph op (Lut, Prim, etc.).
    GraphOp(GraphOp),
    /// Custom op registered in `CustomOpRegistry`.
    Custom { id: CustomOpId, arity: u8 },
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
        // ── Activations → LUT ─────────────────────────────────────────────
        Relu       => D::GraphOp(GraphOp::Lut(LutOp::Relu)),
        Gelu       => D::GraphOp(GraphOp::Lut(LutOp::Gelu)),
        GeluApprox => D::GraphOp(GraphOp::Lut(LutOp::Gelu)),
        Silu       => D::GraphOp(GraphOp::Lut(LutOp::Silu)),
        Tanh       => D::GraphOp(GraphOp::Lut(LutOp::Tanh)),
        Sigmoid    => D::GraphOp(GraphOp::Lut(LutOp::Sigmoid)),

        // ── Binary elementwise → Prim ──────────────────────────────────────
        Add => D::GraphOp(GraphOp::Prim(PrimOp::Add)),
        Sub => D::GraphOp(GraphOp::Prim(PrimOp::Sub)),
        Mul => D::GraphOp(GraphOp::Prim(PrimOp::Mul)),
        Div => D::Unsupported { reason: "Div: no PrimOp::Div in hologram (Z/256Z arithmetic only)" },
        Neg => D::GraphOp(GraphOp::Prim(PrimOp::Neg)),

        // ── Quantized matmul (weight ConstantId injected by builder) ───────
        // Actual ConstantId assigned during param packing; placeholder 0 here.
        QuantizedMatMul { lhs_scheme: QuantScheme::Q4_0, .. } =>
            D::Unsupported { reason: "QuantizedMatMul Q4_0: use builder.matmul_lut_4bit directly" },
        QuantizedMatMul { lhs_scheme: QuantScheme::Q8_0, .. } =>
            D::Unsupported { reason: "QuantizedMatMul Q8_0: use builder.matmul_lut_8bit directly" },
        QuantizedMatMul { .. } =>
            D::Unsupported { reason: "unsupported quant scheme for GEMM" },

        // ── Attention ──────────────────────────────────────────────────────
        MultiHeadAttention { .. }    => D::Custom { id: ATTN_OP_ID,      arity: 3 },
        GroupedQueryAttention { .. } => D::Custom { id: GQA_OP_ID,       arity: 3 },
        FlashAttentionHint           => D::Custom { id: ATTN_OP_ID,      arity: 3 },

        // ── Norms ──────────────────────────────────────────────────────────
        RmsNorm { .. }   => D::Custom { id: RMS_NORM_OP_ID,   arity: 2 },
        LayerNorm { .. } => D::Custom { id: LAYER_NORM_OP_ID, arity: 3 },

        // ── Other AI ops ───────────────────────────────────────────────────
        Softmax { .. }         => D::Custom { id: SOFTMAX_OP_ID, arity: 1 },
        RotaryEmbedding { .. } => D::Custom { id: ROPE_OP_ID,    arity: 3 },
        Embed                  => D::Custom { id: EMBED_OP_ID,   arity: 2 },
        Dequantize             => D::Custom { id: DEQUANT_OP_ID, arity: 1 },
        FusedSwiGLU            => D::Custom { id: SWIGLU_OP_ID,  arity: 2 },

        // ── Shape + type ops ───────────────────────────────────────────────
        Reshape { .. }   => D::Custom { id: RESHAPE_OP_ID, arity: 1 },
        Transpose { .. } => D::Custom { id: RESHAPE_OP_ID, arity: 1 },
        Cast { .. }      => D::Custom { id: CAST_OP_ID,    arity: 1 },
        Concat { .. }    => D::Custom { id: CONCAT_OP_ID,  arity: 0 }, // variadic — arity set at call site

        // ── Control ────────────────────────────────────────────────────────
        Identity => D::Identity,

        // ── Opaque → always error ──────────────────────────────────────────
        Opaque { .. } => D::Unsupported { reason: "opaque op cannot be lowered" },

        // ── Remaining ops: Phase 2/3 expansion ────────────────────────────
        _ => D::Unsupported { reason: "op not yet implemented in lowering" },
    }
}
