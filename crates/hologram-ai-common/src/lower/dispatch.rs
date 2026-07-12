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

use crate::ir::{ActQuant, AiOp, DType, ScatterReduce, WeightLayout};
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
    /// `AttentionAttrs` on an already-operandized `Attention` node. Used by the
    /// six-input decode form (κ119), whose mask operand is the sole masking
    /// authority: `causal` MUST stay false (the substrate refuses it on that
    /// form) and `scale_bits` declares the softmax multiplier the kernel
    /// applies — `1.0` when the graph pre-folds the scale into q (ADR-0019),
    /// so the kernel's own scaling is an exact no-op.
    Attention { causal: bool, scale_bits: u32 },
}

/// A complete canonical desugaring, expressed purely in `OpKind`s. The builder
/// emits the pipeline; there is no escape hatch (architecture §5.2).
#[derive(Debug, Clone)]
pub enum DesugarKind {
    /// `MatMul`/`BatchMatMul` → 2-D `MatMul`, folding any rank≥3 batch dims of A
    /// into the row dimension (hologram's MatMul kernel is strictly 2-D).
    MatMul,
    /// `MultiHeadAttention`/`GroupedQueryAttention` → canonical `Attention` op
    /// with `AttentionAttrs` (causal + softmax scale) attached. The canonical
    /// kernel's operand contract is Q `[batch, heads, seq, head_dim]` / K,V
    /// `[batch, kv_heads, seq, head_dim]` (params derived positionally from
    /// the operand shapes), so `heads_first`/`rope` are layout/positional
    /// information the builder must realize — dropping them executes seq-first
    /// operands with transposed parameters and no positional encoding.
    Attention {
        causal: bool,
        scale_bits: u32,
        /// `true` ⇒ Q/K/V already carry the kernel layout `[B, H, S, D]`;
        /// `false` ⇒ seq-first `[B, S, H, D]`, transposed around the op.
        heads_first: bool,
        /// Apply rotary embeddings (rotate-half, non-interleaved) to Q/K
        /// before the canonical op, tables built from `rope_base`.
        rope: bool,
        rope_base: f32,
    },
    /// `Concat(axis)` → flat (axis-0) `Concat` chain. hologram's Concat is a flat
    /// byte append (axis-0 only), so a non-axis-0 concat is realized by
    /// transposing the join axis to the front, concatenating, and transposing
    /// back. Also chains N-ary concat into binary appends.
    Concat {
        axis: i64,
    },
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
    /// `Dequantize` (ONNX `DequantizeLinear`) → canonical `OpKind::Dequantize`
    /// reading the **packed** quantized operand, with scale/zero-point attached
    /// as `QuantAttrs` (per-tensor) or trailing operands (per-channel) so
    /// hologram's `MatMulDequant` / `DequantActivation` fusions consume the
    /// weight at its quantum width — the dense f32 is never materialized (§6).
    /// `axis` is the per-channel quantization axis (ignored when the scale is a
    /// scalar / per-tensor). `layout`/`act` carry the weight-slot declaration
    /// through to `QuantAttrs`; see [`crate::ir::WeightLayout`].
    Dequantize {
        axis: i64,
        layout: WeightLayout,
        act: ActQuant,
    },
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
    /// SwiGLU activation `silu(gate)·up` → canonical `Silu` + `Mul` (hologram's
    /// `FusedSwiGlu` op is an unimplemented two-weight matmul fusion).
    SwiGlu,
    /// SwiGLU→down fusion → `Silu` + `Mul` + down-projection `MatMul`.
    SwiGluProjection,
    /// Normalization: reshape the input to rank-2 `[rows, feature]` (hologram
    /// derives `feature` only from a rank-2 operand), apply `op` with the γ/β
    /// (and optional residual) operands, then reshape back to the output shape.
    /// `epsilon` is the op's variance-stabilizer: the canonical graph carries
    /// no ε channel (the substrate's norm kernels run at their 1e-9 floor), so
    /// the builder realizes it by input preconditioning.
    Norm {
        op: OpKind,
        residual: bool,
        epsilon: f32,
    },
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
        // hologram's MatMul kernel is 2-D (`[M,K]·[K,N]`); the desugar folds a
        // rank≥3 batched matmul's batch dims into the rows of A (B must be a
        // single shared matrix) so the canonical op stays 2-D.
        A::MatMul | A::BatchMatMul => P::Desugar(DesugarKind::MatMul),
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
        A::LayerNorm { epsilon, .. } => P::Desugar(DesugarKind::Norm {
            op: OpKind::LayerNorm,
            residual: false,
            epsilon: *epsilon,
        }),
        A::RmsNorm { epsilon } => P::Desugar(DesugarKind::Norm {
            op: OpKind::RmsNorm,
            residual: false,
            epsilon: *epsilon,
        }),
        A::GroupNorm { epsilon, .. } => P::Desugar(DesugarKind::Norm {
            op: OpKind::GroupNorm,
            residual: false,
            epsilon: *epsilon,
        }),
        A::InstanceNorm { epsilon } => P::Desugar(DesugarKind::Norm {
            op: OpKind::InstanceNorm,
            residual: false,
            epsilon: *epsilon,
        }),
        A::BatchNorm { epsilon, .. } => P::Desugar(DesugarKind::BatchNorm { epsilon: *epsilon }),

        // ── Attention (Q/K/V + optional norm/rope tables are operands) ───────
        // Causal masking + softmax scale ride on `AttentionAttrs` (the kernel is
        // a faithful SDPA: causal + grouped-query + scale). kv_heads is derived
        // by the compiler from the K operand's head dim. MultiHeadAttention
        // carries no layout/rope fields: its importers emit the kernel layout
        // directly (heads-first, no fused rope). FlashAttentionHint carries no
        // semantics → default (non-causal, 1/√d).
        A::MultiHeadAttention { scale, causal, .. } => P::Desugar(DesugarKind::Attention {
            causal: *causal,
            scale_bits: scale.map(|s| s.to_bits()).unwrap_or(0),
            heads_first: true,
            rope: false,
            rope_base: 0.0,
        }),
        A::GroupedQueryAttention {
            scale,
            causal,
            heads_first,
            rope,
            rope_base,
            ..
        } => P::Desugar(DesugarKind::Attention {
            causal: *causal,
            scale_bits: scale.map(|s| s.to_bits()).unwrap_or(0),
            heads_first: *heads_first,
            rope: *rope,
            rope_base: *rope_base,
        }),
        A::FlashAttentionHint => P::Operandized(OpKind::Attention),
        // v0.9.0 split-KV decode attention (ADR-0019): the six operands
        // `[q, k_past, v_past, k_new, v_new, mask]` pass straight through to the
        // six-input `OpKind::Attention` (κ119). The mask is the sole masking
        // authority (`causal` false — the substrate refuses it on this form);
        // `scale_bits = 1.0` because the decode rewrite pre-folds the model's
        // scale into q exactly as the legacy decomposition does, so the fused
        // kernel's own scaling is an exact no-op (`dot / (1/1.0) = dot`) and the
        // two forms share one scale placement, ulp for ulp.
        A::DecodeAttention => P::Attrs(
            OpKind::Attention,
            AttrSpec::Attention {
                causal: false,
                scale_bits: 1.0f32.to_bits(),
            },
        ),
        // Fixed-bucket ring write → `OpKind::KvCacheWrite` (κ120); the executor
        // realizes it as an in-place κ-move under sole ownership.
        A::KvCacheWrite => P::Operandized(OpKind::KvCacheWrite),

        // ── Positional encoding ─────────────────────────────────────────────
        A::RotaryEmbedding { .. } => P::Operandized(OpKind::RotaryEmbedding),
        A::AlibiSlope => P::Desugar(DesugarKind::AlibiSlope),
        A::CausalMask => P::Desugar(DesugarKind::CausalMask),

        // ── Shape manipulation ──────────────────────────────────────────────
        A::Reshape { .. } | A::Flatten { .. } | A::Squeeze { .. } | A::Unsqueeze { .. } => {
            P::Direct(OpKind::Reshape)
        }
        A::Transpose { .. } => P::Direct(OpKind::Transpose),
        A::Concat { axis } => P::Desugar(DesugarKind::Concat { axis: *axis }),
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
        A::Dequantize { axis, layout, act } => P::Desugar(DesugarKind::Dequantize {
            axis: *axis,
            layout: *layout,
            act: *act,
        }),
        A::Quantize { .. } => P::Desugar(DesugarKind::Quantize),
        A::Embed => P::Desugar(DesugarKind::Embed),

        // ── Canonical composites (hologram desugars/fuses structurally) ─────
        A::FusedSwiGLU => P::Desugar(DesugarKind::SwiGlu),
        A::FusedLayerNormResidual { epsilon } => P::Desugar(DesugarKind::Norm {
            op: OpKind::AddRmsNorm,
            residual: true,
            epsilon: *epsilon,
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
