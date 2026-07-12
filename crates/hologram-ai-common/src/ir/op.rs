use super::{dtype::DType, param::AiParam};
use hologram_ai_quant::QuantScheme;

/// Data layout for K/V tensors in the KV cache pipeline.
///
/// Determines how KvWrite transposes data before storing and how KvRead
/// transposes after reading. The layout must match the attention kernel's
/// `heads_first` flag to ensure correct attention computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvLayout {
    /// `[kv_heads, seq, head_dim]` — heads are the outermost dimension.
    /// Used by ONNX models after the head-reshaping Transpose.
    HeadsFirst,
    /// `[seq, kv_heads, head_dim]` — sequence positions are outermost.
    /// Used by GGUF models where weights are stored seq-first.
    SeqFirst,
}

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

/// **Layout of a quantized weight's bytes as they are bound** — a statement of
/// fact, not a request. Lowers to `hologram_types::weight_layout`.
///
/// A graph constant's bytes are already here, in `[k,n]`, so a constant may only
/// ever be [`WeightLayout::RowMajor`]; claiming otherwise is false and the
/// substrate's compiler rejects it. A **load-time-bound** weight — ours, whose
/// bytes we author at κ-materialization — may declare
/// [`WeightLayout::OutputMajor`], and that declaration is what lets the
/// *load-time* fusion emit the same fused output-major call the constant path
/// gets. It is the difference between a weightless compile reaching the fused
/// decode GEMV and being structurally excluded from it.
///
/// Declaring `OutputMajor` for bytes that are not `[n,k]` is not a performance
/// mistake, it is a wrong answer: no other kernel reads `[n,k]`, so a fallback
/// would transpose the weight by accident. The substrate refuses rather than
/// fall back, and `omajor_w8a8_servable` is the one predicate that decides both
/// the bytes and this declaration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightLayout {
    /// `[k, n]`: k-major, one row per input element. The archive's default.
    RowMajor,
    /// `[n, k]`: each output channel's k-vector contiguous.
    OutputMajor,
}

/// **How a fused `Dequantize → MatMul` treats the activation.** This changes the
/// computed value, so it is never implicit. Lowers to
/// `hologram_types::act_quant`.
///
/// [`ActQuant::W8A32`] is bit-identical to `dequantize → matmul`.
/// [`ActQuant::W8A8TokenSym`] quantizes each activation row to i8
/// (`scale = max|a|/127`) and accumulates the dot products in exact integer
/// arithmetic. It is a *different function*: it takes a distinct `op_signature`,
/// so a κ minted under one can never be served by the other, and switching is a
/// deliberate re-key visible in the archive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActQuant {
    /// f32 activation against the dequantized weight.
    W8A32,
    /// Per-token symmetric i8 activation quantization; exact i32 accumulation.
    W8A8TokenSym,
}

impl WeightLayout {
    /// The substrate's wire-stable tag.
    #[must_use]
    pub fn to_substrate(self) -> u8 {
        match self {
            Self::RowMajor => hologram_types::weight_layout::ROW_MAJOR,
            Self::OutputMajor => hologram_types::weight_layout::OUTPUT_MAJOR,
        }
    }
}

impl ActQuant {
    /// The substrate's wire-stable tag.
    #[must_use]
    pub fn to_substrate(self) -> u8 {
        match self {
            Self::W8A32 => hologram_types::act_quant::W8A32,
            Self::W8A8TokenSym => hologram_types::act_quant::W8A8_TOKEN_SYM,
        }
    }
}

/// Canonical AI IR operation.
///
/// This is the full operation set from the lowering docs (docs/architecture/ARCHITECTURE.md §4).
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
        /// If true, Q/K/V are `[n_heads, seq, head_dim]` (ONNX: pre-transposed).
        /// If false, Q/K/V are `[seq, n_heads, head_dim]` (GGUF: interleaved).
        heads_first: bool,
        /// If true, apply RmsNorm to Q and K before computing attention scores.
        /// Set by `PreAttentionFusion` pass; QK-norm weights are extra inputs.
        qk_norm: bool,
        /// If true, apply RoPE to Q and K before computing attention scores.
        /// Set by `PreAttentionFusion` pass.
        rope: bool,
        /// RoPE base frequency (default 10000.0). Only used when `rope=true`.
        rope_base: f32,
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

    // ── Convolution / pooling ────────────────────────────────────────────────
    /// N-D convolution (1D, 2D, 3D). Inputs: [X, W, bias?].
    Conv {
        kernel_shape: Vec<u64>,
        strides: Vec<u64>,
        pads: Vec<u64>,
        dilations: Vec<u64>,
        group: u64,
        auto_pad: String,
    },
    /// N-D transposed convolution. Inputs: [X, W, bias?].
    ConvTranspose {
        kernel_shape: Vec<u64>,
        strides: Vec<u64>,
        pads: Vec<u64>,
        output_padding: Vec<u64>,
        dilations: Vec<u64>,
        group: u64,
        auto_pad: String,
    },
    /// Max pooling. Inputs: `[X]`.
    MaxPool {
        kernel_shape: Vec<u64>,
        strides: Vec<u64>,
        pads: Vec<u64>,
        dilations: Vec<u64>,
        auto_pad: String,
        ceil_mode: bool,
    },
    /// Average pooling. Inputs: `[X]`.
    AveragePool {
        kernel_shape: Vec<u64>,
        strides: Vec<u64>,
        pads: Vec<u64>,
        count_include_pad: bool,
        auto_pad: String,
        ceil_mode: bool,
    },
    /// Global average pooling: spatial dims → 1.
    GlobalAveragePool,
    /// Resize/Upsample. Inputs: [X, roi?, scales?, sizes?].
    Resize {
        mode: String,
        coordinate_transform_mode: String,
        nearest_mode: String,
    },
    /// Padding. Inputs: [X, pads, constant_value?].
    Pad {
        mode: String,
    },
    /// Instance normalization. Inputs: [X, scale, bias].
    InstanceNorm {
        epsilon: f32,
    },
    /// Local response normalization.
    LRN {
        alpha: f32,
        beta: f32,
        bias: f32,
        size: u64,
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
    /// Clip to [min, max]. Defaults: min=-inf, max=+inf.
    /// ONNX opset 11+: min/max come as optional tensor inputs, resolved at import.
    Clip {
        min: f32,
        max: f32,
    },
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

    // ── Additional reductions ──────────────────────────────────────────────
    ReduceProd {
        axes: Vec<i64>,
        keepdims: bool,
    },
    ReduceL1 {
        axes: Vec<i64>,
        keepdims: bool,
    },
    ReduceL2 {
        axes: Vec<i64>,
        keepdims: bool,
    },

    // ── Data selection / manipulation ────────────────────────────────────────
    /// Return top-K largest or smallest elements. Inputs: [X, K].
    TopK {
        axis: i64,
        largest: bool,
        sorted: bool,
    },
    /// Scatter updates into a tensor using N-D indices. Inputs: [data, indices, updates].
    ScatterND {
        reduce: ScatterReduce,
    },
    /// Cumulative sum along an axis. Inputs: [X, axis_tensor].
    CumSum {
        exclusive: bool,
        reverse: bool,
    },
    /// Return indices of non-zero elements.
    NonZero,
    /// One-hot encoding. Inputs: [indices, depth, values].
    OneHot {
        axis: i64,
    },
    /// Rearrange depth to spatial blocks. Inputs: `[X]`.
    DepthToSpace {
        blocksize: u64,
        mode: String,
    },
    /// Rearrange spatial blocks to depth. Inputs: `[X]`.
    SpaceToDepth {
        blocksize: u64,
    },
    /// Select elements along an axis using a boolean mask. Inputs: [X, condition].
    Compress {
        axis: Option<i64>,
    },
    /// Reverse variable-length sequences. Inputs: [X, sequence_lens].
    ReverseSequence {
        batch_axis: i64,
        time_axis: i64,
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
    /// `DequantizeLinear`: dequantize a packed quantized operand. `axis` is the
    /// per-channel quantization axis (ONNX default 1); ignored when the scale
    /// is a scalar (per-tensor). Negative indexes from the end.
    ///
    /// `layout` and `act` are the weight-slot declaration this node carries down
    /// to `hologram_graph::QuantAttrs`. They are **assertions**, and they are
    /// deliberately not defaultable: only the pass that authors the weight's
    /// bytes knows how they are laid out, and only a pass that has decided to
    /// re-key knows whether the activation may be quantized. See
    /// [`WeightLayout`] and [`ActQuant`].
    Dequantize {
        axis: i64,
        layout: WeightLayout,
        act: ActQuant,
    },
    QuantizedMatMul {
        lhs_scheme: QuantScheme,
        rhs_scheme: QuantScheme,
    },

    // ── KV-cache ─────────────────────────────────────────────────────────────
    /// Write a K or V tensor into the KV-cache for a given layer.
    /// `is_key`: true for the K tensor, false for V.
    KvSlotWrite {
        layer: usize,
        is_key: bool,
        n_kv_heads: u32,
        head_dim: u32,
        /// Data layout of the K/V tensor entering KvWrite.
        layout: KvLayout,
    },
    /// Read cached K/V tensors from the KV-cache for a given layer.
    KvSlotRead {
        layer: usize,
        n_kv_heads: u32,
        head_dim: u32,
        /// Data layout expected by the downstream attention kernel.
        layout: KvLayout,
    },

    // ── v0.9.0 resident-KV decode (ADR-0019) ───────────────────────────────
    /// Fused decode / prefill attention over a **split** KV cache. Inputs are
    /// exactly six — `[q, k_past, v_past, k_new, v_new, mask]` — and the output
    /// is `q`'s shape. The realized context length rides the additive `mask`
    /// (`-inf` erases a key exactly); the bucket geometry is structure the
    /// compiler recovers from the operand shapes. Lowers to the six-input
    /// `OpKind::Attention` (κ discriminant 119). The substrate refuses a
    /// `causal` attr on this form — the mask is the one masking authority — so
    /// this op carries none, and the scale is the default `1/√head_dim`.
    DecodeAttention,
    /// Fixed-bucket **ring write** of `new` rows into a resident KV `cache` at a
    /// runtime `pos`. Inputs: `[cache, new, pos]`; output is `cache`'s shape.
    /// Lowers to `OpKind::KvCacheWrite` (κ discriminant 120), which the executor
    /// realizes as an in-place κ-**move** under sole ownership (no re-hash, no
    /// copy) and an honest copy while the cache is leased. The ring geometry
    /// (bucket / new / planes / row_bytes) is recovered by the compiler from the
    /// operand shapes, so this op carries no parameters.
    KvCacheWrite,

    // ── Fused ops (produced by optimization passes) ────────────────────────
    /// gate × up → silu(gate) × up
    FusedSwiGLU,
    /// x + residual → rmsnorm. Inputs: [x, residual, weight].
    FusedLayerNormResidual {
        epsilon: f32,
    },
    /// MatMul + Relu fused: out = relu(A × B). Inputs: [a, b].
    MatMulRelu,
    /// MatMul + Gelu fused: out = gelu(A × B). Inputs: [a, b].
    MatMulGelu,
    /// MatMul + Silu fused: out = silu(A × B). Inputs: [a, b].
    MatMulSilu,
    /// Concat + MatMul fused: out = concat(inputs) × W.
    /// Avoids materializing the concatenated buffer.
    /// Inputs: [h1, h2, ..., hN, W]. Last input is the weight matrix.
    ConcatMatMul {
        n_concat_inputs: u32,
    },

    // ── Deep decode fusions (Plan 054) ─────────────────────────────────────
    /// Fused: [Add +] RmsNorm → multi-output projection.
    /// When `has_residual_add` is false: inputs = [x, norm_weight, proj_weight].
    /// When `has_residual_add` is true:  inputs = [x, residual, norm_weight, proj_weight].
    /// Output: concatenated projection [M, sum(split_sizes)].
    /// Downstream Slice nodes split the output into individual projections.
    FusedNormProjection {
        epsilon: f64,
        split_sizes: Vec<usize>,
        has_residual_add: bool,
    },
    /// Fused: SwiGLU (silu(gate) * up) → down projection.
    /// Inputs: [gate, up, W_down].
    /// Output: [M, n] — activated values computed in-register, never materialized.
    FusedSwiGluProjection,

    // ── Control flow (subgraph ops) ────────────────────────────────────────
    /// Conditional: execute then_branch or else_branch subgraph based on
    /// condition input. Subgraph names reference `AiGraph.subgraphs` keys.
    If {
        then_branch: String,
        else_branch: Option<String>,
    },
    /// Loop: iterate body subgraph up to max_trip_count times (or until
    /// condition output is false). Carries state tensors between iterations.
    Loop {
        body: String,
        max_trip_count: Option<i64>,
    },
    /// Scan: iterate body subgraph over a sequence dimension, accumulating
    /// outputs. Similar to functional fold/scan.
    Scan {
        body: String,
        num_scan_inputs: u32,
    },

    // ── Type / control ─────────────────────────────────────────────────────
    Cast {
        to: DType,
    },
    Constant {
        value: AiParam,
    },
    Identity,
    /// Generates a tensor of a given shape filled with a constant value.
    /// Input: 1-D i64 shape tensor. Output: tensor of that shape filled
    /// with `fill_value` (stored as f32 bits). Standard ONNX opset 9+ op.
    ConstantOfShape {
        fill_value: u32, // f32::to_bits()
    },
    /// Upper or lower triangular part of a matrix (or batch of matrices).
    /// Input 0: tensor (at least 2-D). Input 1 (optional): scalar k (diagonal offset).
    /// `upper=true`: zero below the k-th diagonal (keep upper triangle).
    /// `upper=false`: zero above the k-th diagonal (keep lower triangle).
    /// Standard ONNX opset 14+ op; commonly used for causal attention masks.
    Trilu {
        upper: bool,
    },

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
            | AiOp::Clip { .. }
            | AiOp::Erf
            | AiOp::Reciprocal
            | AiOp::Cos
            | AiOp::Sin
            | AiOp::Not
            | AiOp::Identity
            | AiOp::Dequantize { .. } => UnaryElementwise,

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
            | AiOp::RotaryEmbedding { .. }
            | AiOp::FusedSwiGLU
            | AiOp::FusedLayerNormResidual { .. }
            | AiOp::KvSlotWrite { .. }
            | AiOp::KvSlotRead { .. }
            // Split-KV decode attention: output = q shape (first input).
            // KvCacheWrite: output = cache shape (first input). Both preserve
            // the first-input shape, so the generic ShapePreserving rule holds.
            | AiOp::DecodeAttention
            | AiOp::KvCacheWrite
            | AiOp::Quantize { .. }
            | AiOp::InstanceNorm { .. }
            | AiOp::LRN { .. }
            | AiOp::CumSum { .. }
            | AiOp::ReverseSequence { .. } => ShapePreserving,

            // ── Custom: op-specific shape/dtype/value rules ───────────────
            AiOp::MatMul
            | AiOp::BatchMatMul
            | AiOp::MatMulRelu
            | AiOp::MatMulGelu
            | AiOp::MatMulSilu
            | AiOp::ConcatMatMul { .. }
            | AiOp::FusedNormProjection { .. }
            | AiOp::FusedSwiGluProjection
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
            | AiOp::Opaque { .. }
            // Phase 1: vision ops
            | AiOp::Conv { .. }
            | AiOp::ConvTranspose { .. }
            | AiOp::MaxPool { .. }
            | AiOp::AveragePool { .. }
            | AiOp::GlobalAveragePool
            | AiOp::Resize { .. }
            | AiOp::Pad { .. }
            // Phase 2: utility ops
            | AiOp::ReduceProd { .. }
            | AiOp::ReduceL1 { .. }
            | AiOp::ReduceL2 { .. }
            | AiOp::TopK { .. }
            | AiOp::ScatterND { .. }
            | AiOp::NonZero
            | AiOp::OneHot { .. }
            | AiOp::DepthToSpace { .. }
            | AiOp::SpaceToDepth { .. }
            | AiOp::Compress { .. }
            // BatchNorm: training mode has 5 outputs
            | AiOp::BatchNorm { .. }
            // Phase 4: control flow
            | AiOp::If { .. }
            | AiOp::Loop { .. }
            | AiOp::Scan { .. }
            | AiOp::ConstantOfShape { .. }
            | AiOp::Trilu { .. } => Custom,
        }
    }
}
