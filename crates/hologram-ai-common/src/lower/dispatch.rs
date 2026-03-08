//! Op dispatch: `AiOp` → hologram `GraphOp` or custom op ID.

use hologram::{CustomOpId, GraphOp};
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
pub const GATHER_OP_ID:     CustomOpId = CustomOpId(13);
pub const MATMUL_OP_ID:     CustomOpId = CustomOpId(14);
pub const SHAPE_OP_ID:      CustomOpId = CustomOpId(15);
pub const WHERE_OP_ID:      CustomOpId = CustomOpId(16);
pub const RANGE_OP_ID:      CustomOpId = CustomOpId(17);
pub const GATHER_ND_OP_ID:  CustomOpId = CustomOpId(18);
pub const ISNAN_OP_ID:      CustomOpId = CustomOpId(19);
pub const FLATTEN_OP_ID:    CustomOpId = CustomOpId(20);
pub const POW_OP_ID:        CustomOpId = CustomOpId(21);
pub const RECIPROCAL_OP_ID: CustomOpId = CustomOpId(22);
pub const REDUCE_MEAN_OP_ID:CustomOpId = CustomOpId(23);
pub const AND_OP_ID:        CustomOpId = CustomOpId(24);
pub const MAX_OP_ID:        CustomOpId = CustomOpId(25);
pub const LESS_EQ_OP_ID:    CustomOpId = CustomOpId(26);
pub const DIV_OP_ID:        CustomOpId = CustomOpId(27);
pub const REDUCE_SUM_OP_ID: CustomOpId = CustomOpId(28);
pub const REDUCE_MAX_OP_ID: CustomOpId = CustomOpId(29);
pub const REDUCE_MIN_OP_ID: CustomOpId = CustomOpId(30);
pub const EQUAL_OP_ID:      CustomOpId = CustomOpId(31);
pub const LESS_OP_ID:       CustomOpId = CustomOpId(32);
pub const GREATER_OP_ID:    CustomOpId = CustomOpId(33);
pub const GREATER_EQ_OP_ID: CustomOpId = CustomOpId(34);
pub const OR_OP_ID:         CustomOpId = CustomOpId(35);
pub const XOR_OP_ID:        CustomOpId = CustomOpId(36);
pub const MOD_OP_ID:        CustomOpId = CustomOpId(37);
pub const MIN_OP_ID:        CustomOpId = CustomOpId(38);
pub const ERF_OP_ID:        CustomOpId = CustomOpId(39);
pub const CLIP_OP_ID:       CustomOpId = CustomOpId(40);
pub const NOT_OP_ID:        CustomOpId = CustomOpId(41);
pub const SIGN_OP_ID:       CustomOpId = CustomOpId(42);
pub const FLOOR_OP_ID:      CustomOpId = CustomOpId(43);
pub const CEIL_OP_ID:       CustomOpId = CustomOpId(44);
pub const ROUND_OP_ID:      CustomOpId = CustomOpId(45);
pub const LOG_SOFTMAX_OP_ID:CustomOpId = CustomOpId(46);
pub const ADD_OP_ID:        CustomOpId = CustomOpId(47);
pub const SUB_OP_ID:        CustomOpId = CustomOpId(48);
pub const MUL_OP_ID:        CustomOpId = CustomOpId(49);
pub const NEG_OP_ID:        CustomOpId = CustomOpId(50);
pub const RELU_OP_ID:       CustomOpId = CustomOpId(51);
pub const GELU_OP_ID:       CustomOpId = CustomOpId(52);
pub const SILU_OP_ID:       CustomOpId = CustomOpId(53);
pub const TANH_OP_ID:       CustomOpId = CustomOpId(54);
pub const SIGMOID_OP_ID:    CustomOpId = CustomOpId(55);
pub const EXP_OP_ID:        CustomOpId = CustomOpId(56);
pub const LOG_OP_ID:        CustomOpId = CustomOpId(57);
pub const SQRT_OP_ID:       CustomOpId = CustomOpId(58);
pub const ABS_OP_ID:        CustomOpId = CustomOpId(59);
pub const COS_OP_ID:        CustomOpId = CustomOpId(60);
pub const SIN_OP_ID:        CustomOpId = CustomOpId(61);

/// Categorised dispatch target for a single `AiOp`.
///
/// Used only at model-compilation time (in `lower()`), not during inference.
/// The size difference between variants is acceptable for a one-shot classification.
#[derive(Debug)]
#[allow(clippy::large_enum_variant)]
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
        // ── Activations (f32 custom handlers) ─────────────────────────────
        // Route through f32 handlers instead of byte-level LUT tables.
        Relu       => D::Custom { id: RELU_OP_ID,    arity: 1 },
        Gelu       => D::Custom { id: GELU_OP_ID,    arity: 1 },
        GeluApprox => D::Custom { id: GELU_OP_ID,    arity: 1 },
        Silu       => D::Custom { id: SILU_OP_ID,    arity: 1 },
        Tanh       => D::Custom { id: TANH_OP_ID,    arity: 1 },
        Sigmoid    => D::Custom { id: SIGMOID_OP_ID, arity: 1 },

        // ── Unary math (f32 custom handlers) ─────────────────────────────
        Exp  => D::Custom { id: EXP_OP_ID,  arity: 1 },
        Log  => D::Custom { id: LOG_OP_ID,  arity: 1 },
        Sqrt => D::Custom { id: SQRT_OP_ID, arity: 1 },
        Abs  => D::Custom { id: ABS_OP_ID,  arity: 1 },
        Cos  => D::Custom { id: COS_OP_ID,  arity: 1 },
        Sin  => D::Custom { id: SIN_OP_ID,  arity: 1 },

        // ── Binary elementwise ────────────────────────────────────────────
        // Route through custom f32 handlers instead of native PrimOp
        // (byte-level Z/256Z ring) so we get float arithmetic + broadcasting.
        Add => D::Custom { id: ADD_OP_ID,  arity: 2 },
        Sub => D::Custom { id: SUB_OP_ID,  arity: 2 },
        Mul => D::Custom { id: MUL_OP_ID,  arity: 2 },
        Div => D::Custom { id: DIV_OP_ID,  arity: 2 },
        Neg => D::Custom { id: NEG_OP_ID,  arity: 1 },
        Pow => D::Custom { id: POW_OP_ID,  arity: 2 },
        Mod => D::Custom { id: MOD_OP_ID,  arity: 2 },
        Min => D::Custom { id: MIN_OP_ID,  arity: 2 },
        Max => D::Custom { id: MAX_OP_ID,  arity: 2 },

        // ── Boolean ops ──────────────────────────────────────────────────────
        And => D::Custom { id: AND_OP_ID,  arity: 2 },
        Or  => D::Custom { id: OR_OP_ID,   arity: 2 },
        Xor => D::Custom { id: XOR_OP_ID,  arity: 2 },
        Not => D::Custom { id: NOT_OP_ID,  arity: 1 },

        // ── Comparisons ──────────────────────────────────────────────────────
        Equal          => D::Custom { id: EQUAL_OP_ID,      arity: 2 },
        Less           => D::Custom { id: LESS_OP_ID,       arity: 2 },
        LessOrEqual    => D::Custom { id: LESS_EQ_OP_ID,    arity: 2 },
        Greater        => D::Custom { id: GREATER_OP_ID,    arity: 2 },
        GreaterOrEqual => D::Custom { id: GREATER_EQ_OP_ID, arity: 2 },

        // ── Unary math (custom handlers) ─────────────────────────────────────
        Reciprocal => D::Custom { id: RECIPROCAL_OP_ID, arity: 1 },
        Sign       => D::Custom { id: SIGN_OP_ID,       arity: 1 },
        Floor      => D::Custom { id: FLOOR_OP_ID,      arity: 1 },
        Ceil       => D::Custom { id: CEIL_OP_ID,       arity: 1 },
        Round      => D::Custom { id: ROUND_OP_ID,      arity: 1 },
        Clip       => D::Custom { id: CLIP_OP_ID,       arity: 1 },
        Erf        => D::Custom { id: ERF_OP_ID,        arity: 1 },

        // ── Reductions ───────────────────────────────────────────────────────
        ReduceSum  { .. } => D::Custom { id: REDUCE_SUM_OP_ID,  arity: 1 },
        ReduceMean { .. } => D::Custom { id: REDUCE_MEAN_OP_ID, arity: 1 },
        ReduceMax  { .. } => D::Custom { id: REDUCE_MAX_OP_ID,  arity: 1 },
        ReduceMin  { .. } => D::Custom { id: REDUCE_MIN_OP_ID,  arity: 1 },

        // ── LogSoftmax ───────────────────────────────────────────────────────
        LogSoftmax { .. } => D::Custom { id: LOG_SOFTMAX_OP_ID, arity: 1 },

        // ── Quantized matmul (weight ConstantId injected by builder) ───────
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
        Squeeze { .. }   => D::Identity,
        Unsqueeze { .. } => D::Identity,
        Expand           => D::Identity,
        Slice { .. }     => D::Identity,
        Split { .. }     => D::Identity,
        Tile { .. }      => D::Identity,
        Cast { .. }      => D::Custom { id: CAST_OP_ID,    arity: 1 },
        Concat { .. }    => D::Custom { id: CONCAT_OP_ID,  arity: 0 }, // variadic

        // ── Gather / embedding lookup ──────────────────────────────────────
        Gather { .. } | GatherElements { .. } => D::Custom { id: GATHER_OP_ID, arity: 2 },
        GatherND { .. } => D::Custom { id: GATHER_ND_OP_ID, arity: 2 },

        // ── Shape / conditional / range ──────────────────────────────────
        Shape          => D::Custom { id: SHAPE_OP_ID,   arity: 1 },
        Where          => D::Custom { id: WHERE_OP_ID,   arity: 3 },
        Range          => D::Custom { id: RANGE_OP_ID,   arity: 3 },
        IsNaN          => D::Custom { id: ISNAN_OP_ID,   arity: 1 },
        Flatten { .. } => D::Custom { id: FLATTEN_OP_ID, arity: 1 },

        // ── Matrix multiply ────────────────────────────────────────────────
        MatMul | BatchMatMul => D::Custom { id: MATMUL_OP_ID, arity: 2 },
        Gemm { .. }          => D::Custom { id: MATMUL_OP_ID, arity: 3 },

        // ── Control ────────────────────────────────────────────────────────
        Identity => D::Identity,

        // ── Opaque → always error ──────────────────────────────────────────
        Opaque { .. } => D::Unsupported { reason: "opaque op cannot be lowered" },

        // ── Remaining ops: Phase 2/3 expansion ────────────────────────────
        _ => D::Unsupported { reason: "op not yet implemented in lowering" },
    }
}
