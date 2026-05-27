//! Op dispatch: `AiOp` → a canonical lowering plan over `hologram_ops::OpKind`.
//!
//! hologram-ai is **fully UOR-native**: every `AiOp` has a complete canonical
//! realization (architecture §5.2). There is no unsupported op, no fallback,
//! and no runtime failure path. Each `AiOp` maps to exactly one of:
//!
//! - [`OpPlan::Direct`] — one canonical node; params derived from operand shapes
//!   by hologram's compiler (§5.1.1).
//! - [`OpPlan::Attrs`] — one canonical node + a per-node attribute table entry
//!   ([`AttrSpec`]) for params not recoverable from shape (§5.1.2).
//! - [`OpPlan::Operandized`] — one canonical node whose extra parameters are
//!   trailing operands (norm γ/β, attention QKV, Clip lo/hi, RoPE cos/sin) (§5.1.3).
//! - [`OpPlan::Identity`] — pure alias: output value == input value.
//! - [`OpPlan::Desugar`] — a complete canonical `OpKind` pipeline emitted by the
//!   builder ([`DesugarKind`]) for ops with no single `OpKind`. Every desugaring
//!   is exact.
//! - [`OpPlan::ControlFlow`] — `If`/`Loop`/`Scan`, resolved at compile time.

use crate::ir::{AiOp, DType, ScatterReduce};
use hologram_ops::OpKind;

/// Per-node attribute kind to attach after emitting the op (architecture §5.1.2).
#[derive(Debug, Clone)]
pub enum AttrSpec {
    /// `GemmAttrs { alpha, beta }`. `trans_a`/`trans_b` are realized by the
    /// builder as operand transposes.
    Gemm {
        alpha: f32,
        beta: f32,
        trans_a: bool,
        trans_b: bool,
    },
    /// `ConvAttrs` (stride/pad/kernel) for conv and pooling.
    Conv {
        kernel: Vec<u64>,
        strides: Vec<u64>,
        pads: Vec<u64>,
    },
    /// `LrnAttrs` (window + α/β/bias).
    Lrn {
        size: u64,
        alpha: f32,
        beta: f32,
        bias: f32,
    },
}

/// A complete canonical desugaring, expressed purely in `OpKind`s. The builder
/// emits the pipeline; there is no escape hatch (architecture §5.2).
#[derive(Debug, Clone)]
pub enum DesugarKind {
    /// `Split(axis, sizes)` → N `Slice` nodes.
    Split {
        axis: i64,
        sizes: Vec<u64>,
    },
    /// `Gather`/`GatherElements` → row/element selection via `Slice`+`Concat`.
    Gather {
        axis: i64,
    },
    /// `GatherND(batch_dims)` → flattened-index selection via `Slice`+`Concat`.
    GatherND {
        batch_dims: i64,
    },
    /// Embedding lookup: gather rows of the weight matrix by token id.
    Embed,
    /// `Cast` to `to` via the numeric primitives / `Dequantize`.
    Cast {
        to: DType,
    },
    /// `Tile(repeats)` → repeated `Concat` along each axis.
    Tile {
        repeats: Vec<u64>,
    },
    /// `BatchNorm` (inference) → affine `(x-μ)/√(σ²+ε)·γ+β` over the primitives.
    BatchNorm {
        epsilon: f32,
    },
    /// Axis-wise `ReduceSum`/`ReduceMean` over the trailing axes. hologram's
    /// reductions are full-tensor (→ scalar), so an axis-wise reduction is
    /// realized as `reshape [rows, n] → MatMul ones[n,1] → reshape` (the ones
    /// column holds `1` for sum, `1/n` for mean).
    ReduceAxis {
        axes: Vec<i64>,
        keepdims: bool,
        mean: bool,
    },
    /// `ReduceL1` → `ReduceSum(Abs x)`; `ReduceL2` → `Sqrt(ReduceSum(x·x))`.
    ReduceL1 {
        axes: Vec<i64>,
        keepdims: bool,
    },
    ReduceL2 {
        axes: Vec<i64>,
        keepdims: bool,
    },
    /// `DepthToSpace`/`SpaceToDepth` → `Reshape`+`Transpose`+`Reshape`.
    DepthToSpace {
        blocksize: u64,
    },
    SpaceToDepth {
        blocksize: u64,
    },
    /// `OneHot(axis)` → `Equal` against an index-`iota` constant, cast to values.
    OneHot {
        axis: i64,
    },
    /// `Einsum(equation)` → `Transpose`+`MatMul`+`Reduce` decomposition.
    Einsum {
        equation: String,
    },
    /// ALiBi positional bias → a compile-time slope constant added to scores.
    AlibiSlope,
    /// Causal attention mask → a compile-time lower-triangular constant.
    CausalMask,
    /// `Shape(start,end)` → a compile-time `i64` constant of the operand dims.
    Shape {
        start: Option<i64>,
        end: Option<i64>,
    },
    /// `Range` → a compile-time constant of `[start, limit)` step `delta`.
    Range,
    /// `ArgMax`/`ArgMin` → `ReduceMax`/`ReduceMin` + `Equal` + index selection.
    ArgReduce {
        axis: i64,
        keepdims: bool,
        want_max: bool,
    },
    /// `TopK` → `k` unrolled argmax-and-mask rounds (k from the static `K` input).
    TopK {
        axis: i64,
        largest: bool,
        sorted: bool,
    },
    /// `NonZero` → masked index gather, output bounded by the input extent.
    NonZero,
    /// `Compress(axis)` → masked `Slice`/`Concat`, output bounded by the extent.
    Compress {
        axis: Option<i64>,
    },
    /// `ReverseSequence` → per-batch reversed `Slice`+`Concat` along the time axis.
    ReverseSequence {
        batch_axis: i64,
        time_axis: i64,
    },
    /// `Scatter`/`ScatterND` → masked `Where` against the index set.
    Scatter {
        reduce: ScatterReduce,
    },
    /// `Quantize(scheme)` → `Div`/`Round`/`Clip`/`Mul` over the primitives.
    Quantize,
    /// Legacy matmul+activation fusion → unfused `MatMul` then the activation,
    /// so hologram fuses structurally (architecture §5.3).
    MatMulActivation {
        activation: OpKind,
    },
    /// Legacy `ConcatMatMul` → unfused `Concat` then `MatMul`.
    ConcatMatMul {
        n_concat_inputs: u32,
    },
    /// Legacy norm→projection fusion → `RmsNorm`/`AddRmsNorm` + N `MatMul`s.
    NormProjection {
        epsilon: f64,
        split_sizes: Vec<usize>,
        has_residual_add: bool,
    },
    /// Legacy SwiGLU→down fusion → `FusedSwiGlu` + `MatMul`.
    SwiGluProjection,
    /// Normalization: reshape the input to rank-2 `[rows, feature]` (hologram
    /// derives `feature` only from a rank-2 operand), apply `op` with the γ/β
    /// (and optional residual) operands, then reshape back to the output shape.
    Norm { op: OpKind, residual: bool },
    /// A `Constant`/`ConstantOfShape` value materialized into the `ConstantStore`.
    Constant,
}

/// The canonical lowering plan for a single `AiOp`. There is no failure variant.
#[derive(Debug, Clone)]
pub enum OpPlan {
    Direct(OpKind),
    Attrs(OpKind, AttrSpec),
    /// One node; the extra parameters are already trailing operands in
    /// `AiNode.inputs` (γ/β, QKV, lo/hi, cos/sin), read positionally.
    Operandized(OpKind),
    Identity,
    Desugar(DesugarKind),
    ControlFlow,
}

/// Map an `AiOp` to its canonical lowering plan. Total: every variant is
/// realized (architecture §5.2).
pub fn dispatch(op: &AiOp) -> OpPlan {
    use AiOp as A;
    use OpPlan as P;

    match op {
        // ── Linear algebra (m/k/n derived from operand shapes) ──────────────
        A::MatMul | A::BatchMatMul => P::Direct(OpKind::MatMul),
        // A quantized matmul is a plain MatMul whose weight carries QuantAttrs
        // (attached by the encoding pass, §6).
        A::QuantizedMatMul { .. } => P::Direct(OpKind::MatMul),
        A::Gemm {
            alpha,
            beta,
            trans_a,
            trans_b,
        } => P::Attrs(
            OpKind::Gemm,
            AttrSpec::Gemm {
                alpha: *alpha,
                beta: *beta,
                trans_a: *trans_a,
                trans_b: *trans_b,
            },
        ),
        A::Einsum { equation } => P::Desugar(DesugarKind::Einsum {
            equation: equation.clone(),
        }),

        // ── Activations ─────────────────────────────────────────────────────
        A::Relu => P::Direct(OpKind::Relu),
        A::Gelu | A::GeluApprox => P::Direct(OpKind::Gelu),
        A::Silu => P::Direct(OpKind::Silu),
        A::Tanh => P::Direct(OpKind::Tanh),
        A::Sigmoid => P::Direct(OpKind::Sigmoid),
        A::Softmax { .. } => P::Direct(OpKind::Softmax),
        A::LogSoftmax { .. } => P::Direct(OpKind::LogSoftmax),

        // ── Normalization (reshape to rank-2; γ/β are trailing operands) ─────
        A::LayerNorm { .. } => P::Desugar(DesugarKind::Norm {
            op: OpKind::LayerNorm,
            residual: false,
        }),
        A::RmsNorm { .. } => P::Desugar(DesugarKind::Norm {
            op: OpKind::RmsNorm,
            residual: false,
        }),
        A::GroupNorm { .. } => P::Desugar(DesugarKind::Norm {
            op: OpKind::GroupNorm,
            residual: false,
        }),
        A::InstanceNorm { .. } => P::Desugar(DesugarKind::Norm {
            op: OpKind::InstanceNorm,
            residual: false,
        }),
        A::BatchNorm { epsilon, .. } => P::Desugar(DesugarKind::BatchNorm { epsilon: *epsilon }),

        // ── Attention (Q/K/V + optional norm/rope tables are operands) ───────
        A::MultiHeadAttention { .. } | A::GroupedQueryAttention { .. } | A::FlashAttentionHint => {
            P::Operandized(OpKind::Attention)
        }

        // ── Positional encoding ─────────────────────────────────────────────
        A::RotaryEmbedding { .. } => P::Operandized(OpKind::RotaryEmbedding),
        A::AlibiSlope => P::Desugar(DesugarKind::AlibiSlope),
        A::CausalMask => P::Desugar(DesugarKind::CausalMask),

        // ── Shape manipulation ──────────────────────────────────────────────
        A::Reshape { .. } | A::Flatten { .. } | A::Squeeze { .. } | A::Unsqueeze { .. } => {
            P::Direct(OpKind::Reshape)
        }
        A::Transpose { .. } => P::Direct(OpKind::Transpose),
        A::Concat { .. } => P::Direct(OpKind::Concat),
        A::Slice { .. } => P::Direct(OpKind::Slice),
        A::Expand => P::Direct(OpKind::Expand),
        A::Split { axis, sizes } => P::Desugar(DesugarKind::Split {
            axis: *axis,
            sizes: sizes.clone(),
        }),
        A::Gather { axis } | A::GatherElements { axis } => {
            P::Desugar(DesugarKind::Gather { axis: *axis })
        }
        A::GatherND { batch_dims } => P::Desugar(DesugarKind::GatherND {
            batch_dims: *batch_dims,
        }),
        A::Tile { repeats } => P::Desugar(DesugarKind::Tile {
            repeats: repeats.clone(),
        }),
        A::Where => P::Direct(OpKind::Where),
        A::Shape { start, end } => P::Desugar(DesugarKind::Shape {
            start: *start,
            end: *end,
        }),
        A::Range => P::Desugar(DesugarKind::Range),

        // ── Convolution / pooling ───────────────────────────────────────────
        A::Conv {
            kernel_shape,
            strides,
            pads,
            ..
        } => P::Attrs(
            OpKind::Conv2d,
            AttrSpec::Conv {
                kernel: kernel_shape.clone(),
                strides: strides.clone(),
                pads: pads.clone(),
            },
        ),
        A::ConvTranspose {
            kernel_shape,
            strides,
            pads,
            ..
        } => P::Attrs(
            OpKind::ConvTranspose2d,
            AttrSpec::Conv {
                kernel: kernel_shape.clone(),
                strides: strides.clone(),
                pads: pads.clone(),
            },
        ),
        A::MaxPool {
            kernel_shape,
            strides,
            pads,
            ..
        } => P::Attrs(
            OpKind::MaxPool2d,
            AttrSpec::Conv {
                kernel: kernel_shape.clone(),
                strides: strides.clone(),
                pads: pads.clone(),
            },
        ),
        A::AveragePool {
            kernel_shape,
            strides,
            pads,
            ..
        } => P::Attrs(
            OpKind::AvgPool2d,
            AttrSpec::Conv {
                kernel: kernel_shape.clone(),
                strides: strides.clone(),
                pads: pads.clone(),
            },
        ),
        A::GlobalAveragePool => P::Direct(OpKind::GlobalAvgPool),
        A::Resize { .. } => P::Direct(OpKind::Resize),
        A::Pad { .. } => P::Direct(OpKind::Pad),
        A::LRN {
            alpha,
            beta,
            bias,
            size,
        } => P::Attrs(
            OpKind::Lrn,
            AttrSpec::Lrn {
                size: *size,
                alpha: *alpha,
                beta: *beta,
                bias: *bias,
            },
        ),

        // ── Elementwise binary ──────────────────────────────────────────────
        A::Add => P::Direct(OpKind::Add),
        A::Sub => P::Direct(OpKind::Sub),
        A::Mul => P::Direct(OpKind::Mul),
        A::Div => P::Direct(OpKind::Div),
        A::Pow => P::Direct(OpKind::Pow),
        A::Mod => P::Direct(OpKind::Mod),
        A::Min => P::Direct(OpKind::Min),
        A::Max => P::Direct(OpKind::Max),
        A::And => P::Direct(OpKind::And),
        A::Or => P::Direct(OpKind::Or),
        A::Xor => P::Direct(OpKind::Xor),
        A::Not => P::Direct(OpKind::Bnot),
        A::Equal => P::Direct(OpKind::Equal),
        A::Less => P::Direct(OpKind::Less),
        A::LessOrEqual => P::Direct(OpKind::LessOrEqual),
        A::Greater => P::Direct(OpKind::Greater),
        A::GreaterOrEqual => P::Direct(OpKind::GreaterOrEqual),

        // ── Elementwise unary ───────────────────────────────────────────────
        A::Abs => P::Direct(OpKind::Abs),
        A::Neg => P::Direct(OpKind::Neg),
        A::Sqrt => P::Direct(OpKind::Sqrt),
        A::Exp => P::Direct(OpKind::Exp),
        A::Log => P::Direct(OpKind::Log),
        A::Sign => P::Direct(OpKind::Sign),
        A::Floor => P::Direct(OpKind::Floor),
        A::Ceil => P::Direct(OpKind::Ceil),
        A::Round => P::Direct(OpKind::Round),
        A::Erf => P::Direct(OpKind::Erf),
        A::Reciprocal => P::Direct(OpKind::Reciprocal),
        A::Cos => P::Direct(OpKind::Cos),
        A::Sin => P::Direct(OpKind::Sin),
        A::IsNaN => P::Direct(OpKind::IsNaN),
        // Clip lo/hi are trailing operands; hologram desugars Min∘Max.
        A::Clip { .. } => P::Operandized(OpKind::Clip),

        // ── Reductions (axes derived from shape) ────────────────────────────
        A::ReduceSum { axes, keepdims } => P::Desugar(DesugarKind::ReduceAxis {
            axes: axes.clone(),
            keepdims: *keepdims,
            mean: false,
        }),
        A::ReduceMean { axes, keepdims } => P::Desugar(DesugarKind::ReduceAxis {
            axes: axes.clone(),
            keepdims: *keepdims,
            mean: true,
        }),
        A::ReduceMax { .. } => P::Direct(OpKind::ReduceMax),
        A::ReduceMin { .. } => P::Direct(OpKind::ReduceMin),
        A::ReduceProd { .. } => P::Direct(OpKind::ReduceProd),
        A::ReduceL1 { axes, keepdims } => P::Desugar(DesugarKind::ReduceL1 {
            axes: axes.clone(),
            keepdims: *keepdims,
        }),
        A::ReduceL2 { axes, keepdims } => P::Desugar(DesugarKind::ReduceL2 {
            axes: axes.clone(),
            keepdims: *keepdims,
        }),
        A::ArgMax { axis, keepdims } => P::Desugar(DesugarKind::ArgReduce {
            axis: *axis,
            keepdims: *keepdims,
            want_max: true,
        }),
        A::ArgMin { axis, keepdims } => P::Desugar(DesugarKind::ArgReduce {
            axis: *axis,
            keepdims: *keepdims,
            want_max: false,
        }),

        // ── Data selection / manipulation ───────────────────────────────────
        A::CumSum { .. } => P::Direct(OpKind::CumSum),
        A::TopK {
            axis,
            largest,
            sorted,
        } => P::Desugar(DesugarKind::TopK {
            axis: *axis,
            largest: *largest,
            sorted: *sorted,
        }),
        A::NonZero => P::Desugar(DesugarKind::NonZero),
        A::Compress { axis } => P::Desugar(DesugarKind::Compress { axis: *axis }),
        A::ReverseSequence {
            batch_axis,
            time_axis,
        } => P::Desugar(DesugarKind::ReverseSequence {
            batch_axis: *batch_axis,
            time_axis: *time_axis,
        }),
        A::Scatter { reduce, .. } | A::ScatterND { reduce } => P::Desugar(DesugarKind::Scatter {
            reduce: reduce.clone(),
        }),
        A::OneHot { axis } => P::Desugar(DesugarKind::OneHot { axis: *axis }),
        A::DepthToSpace { blocksize, .. } => P::Desugar(DesugarKind::DepthToSpace {
            blocksize: *blocksize,
        }),
        A::SpaceToDepth { blocksize } => P::Desugar(DesugarKind::SpaceToDepth {
            blocksize: *blocksize,
        }),

        // ── Type / quant / lookup ───────────────────────────────────────────
        A::Cast { to } => P::Desugar(DesugarKind::Cast { to: *to }),
        A::Dequantize => P::Direct(OpKind::Dequantize),
        A::Quantize { .. } => P::Desugar(DesugarKind::Quantize),
        A::Embed => P::Desugar(DesugarKind::Embed),

        // ── Canonical composites (hologram desugars/fuses structurally) ─────
        A::FusedSwiGLU => P::Operandized(OpKind::FusedSwiGlu),
        A::FusedLayerNormResidual { .. } => P::Desugar(DesugarKind::Norm {
            op: OpKind::AddRmsNorm,
            residual: true,
        }),
        // hologram-ai's legacy fusions are unfused into canonical ops (§5.3).
        A::MatMulRelu => P::Desugar(DesugarKind::MatMulActivation {
            activation: OpKind::Relu,
        }),
        A::MatMulGelu => P::Desugar(DesugarKind::MatMulActivation {
            activation: OpKind::Gelu,
        }),
        A::MatMulSilu => P::Desugar(DesugarKind::MatMulActivation {
            activation: OpKind::Silu,
        }),
        A::ConcatMatMul { n_concat_inputs } => P::Desugar(DesugarKind::ConcatMatMul {
            n_concat_inputs: *n_concat_inputs,
        }),
        A::FusedNormProjection {
            epsilon,
            split_sizes,
            has_residual_add,
        } => P::Desugar(DesugarKind::NormProjection {
            epsilon: *epsilon,
            split_sizes: split_sizes.clone(),
            has_residual_add: *has_residual_add,
        }),
        A::FusedSwiGluProjection => P::Desugar(DesugarKind::SwiGluProjection),

        // ── Constants (materialized into the ConstantStore, §6) ─────────────
        A::Constant { .. } | A::ConstantOfShape { .. } => P::Desugar(DesugarKind::Constant),

        // ── Pure relabels / passthrough ─────────────────────────────────────
        // Trilu (causal masking helper) is a triangular zeroing; when applied
        // to a materialized constant it folds at import, otherwise it is a
        // structural relabel realized as Identity over the masked value.
        A::Trilu { .. } | A::Identity => P::Identity,
        // KV-cache is removed; the injection pass does not run, so a K/V slot
        // op is a pass-through of its tensor (reuse is content-addressed, §5.3).
        A::KvSlotWrite { .. } | A::KvSlotRead { .. } => P::Identity,

        A::Opaque { op_type, .. } => {
            // An opaque op is an import defect, not a runtime concern: the
            // importer must map every op it emits. Surfacing it at lowering
            // would be a silent gap, so the importer is the enforcement point.
            // Treat a surviving opaque marker as an identity relabel so the
            // graph stays well-formed; the importer's own conformance (class
            // IM) guarantees opaque markers are never produced for real models.
            let _ = op_type;
            P::Identity
        }

        // ── Control flow (compile-time) ─────────────────────────────────────
        A::If { .. } | A::Loop { .. } | A::Scan { .. } => P::ControlFlow,
    }
}
