//! Compile-time shape projection: `ShapeProjection` trait + runtime walker.
//!
//! Every op implements [`ShapeProjection`] to describe how its output shape
//! derives from its inputs. The builder queries this trait during lowering to
//! populate the [`ShapeContextGraph`], which is serialized into the `.holo`
//! archive and walked at runtime before dispatch.
//!
//! # Design
//!
//! - [`ShapeProjection`] — trait implemented for both `FloatOp` and `AiOp`.
//! - [`resolve_spec`] — runtime resolution of a spec against known input shapes.
//! - [`walk_shape_context`] — topological shape propagation before execution.

use std::collections::{BTreeSet, HashMap};

use hologram::FloatOp;

use crate::ir::AiOp;

pub use crate::exec_context::{ShapeContextGraph, ShapeDimRepr, ShapeProjectionEntry, ShapeSpecRepr};

// ── ShapeProjection trait ────────────────────────────────────────────────────

/// Trait for ops that describe how their output shape derives from input shapes.
///
/// Every op implements this trait to enable compile-time shape projection.
/// The builder queries this during lowering to populate the `ShapeContextGraph`.
///
/// - `impl ShapeProjection for FloatOp` — used for GraphOp and FloatNeedsShape
///   dispatch paths (carries concrete parameters like k, dim, etc.).
/// - `impl ShapeProjection for AiOp` — used for Identity dispatch path and as
///   documentation for all ops.
pub trait ShapeProjection {
    /// Returns the shape derivation spec and optional shape-value input index.
    /// Returns `None` only for ops that cannot describe their shape projection
    /// (constants, opaques, control flow).
    fn shape_spec(&self) -> Option<(ShapeSpecRepr, Option<u8>)>;
}

// ── FloatOp implementation ───────────────────────────────────────────────────

impl ShapeProjection for FloatOp {
    fn shape_spec(&self) -> Option<(ShapeSpecRepr, Option<u8>)> {
        Some(match self {
            // ── Unary elementwise — output same shape as input[0] ────────────
            FloatOp::Neg
            | FloatOp::Relu
            | FloatOp::Gelu
            | FloatOp::Silu
            | FloatOp::Tanh
            | FloatOp::Sigmoid
            | FloatOp::Exp
            | FloatOp::Log
            | FloatOp::Sqrt
            | FloatOp::Abs
            | FloatOp::Reciprocal
            | FloatOp::Cos
            | FloatOp::Sin
            | FloatOp::Sign
            | FloatOp::Floor
            | FloatOp::Ceil
            | FloatOp::Round
            | FloatOp::Erf
            | FloatOp::Clip { .. }
            | FloatOp::IsNaN
            | FloatOp::Not
            | FloatOp::Cast { .. }
            | FloatOp::Dequantize
            | FloatOp::CumSum { .. }
            | FloatOp::ScatterND
            | FloatOp::ReverseSequence { .. }
            | FloatOp::GatherND => (ShapeSpecRepr::SameAs(0), None),

            // ── Normalization / activation with size — same shape as input[0] ─
            FloatOp::Softmax { .. }
            | FloatOp::LogSoftmax { .. }
            | FloatOp::RmsNorm { .. }
            | FloatOp::AddRmsNorm { .. }
            | FloatOp::LayerNorm { .. }
            | FloatOp::InstanceNorm { .. }
            | FloatOp::GroupNorm { .. }
            | FloatOp::LRN { .. }
            | FloatOp::RotaryEmbedding { .. }
            | FloatOp::FusedSwiGLU => (ShapeSpecRepr::SameAs(0), None),

            // ── Binary elementwise — broadcast shape ─────────────────────────
            FloatOp::Add
            | FloatOp::Sub
            | FloatOp::Mul
            | FloatOp::Div
            | FloatOp::Pow
            | FloatOp::Mod
            | FloatOp::Min
            | FloatOp::Max
            | FloatOp::And
            | FloatOp::Or
            | FloatOp::Xor
            | FloatOp::Equal
            | FloatOp::Less
            | FloatOp::LessOrEqual
            | FloatOp::Greater
            | FloatOp::GreaterOrEqual => (ShapeSpecRepr::Broadcast(0, 1), None),

            // ── Reductions — removes last dim ─────────────────────────────────
            FloatOp::ReduceSum { .. }
            | FloatOp::ReduceMean { .. }
            | FloatOp::ReduceMax { .. }
            | FloatOp::ReduceMin { .. }
            | FloatOp::ReduceProd { .. } => (ShapeSpecRepr::DropLastDim(0), None),

            // ── Linear algebra ────────────────────────────────────────────────
            FloatOp::MatMul { k, .. } => (ShapeSpecRepr::MatMul { k_hint: *k }, None),
            FloatOp::Gemm { k, .. } => (ShapeSpecRepr::Gemm { k: *k }, None),

            // ── Gather / embed — indices shape ++ [dim] ──────────────────────
            FloatOp::Gather { dim, .. } | FloatOp::Embed { dim, .. } => {
                (ShapeSpecRepr::GatherEmbed { dim: *dim }, None)
            }

            // ── Reshape — shape from shape-value tensor (input[1]) ───────────
            FloatOp::Reshape => (ShapeSpecRepr::Reshape, Some(1)),

            // ── Transpose — permute dims ─────────────────────────────────────
            FloatOp::Transpose { perm, ndim } => {
                (ShapeSpecRepr::Transpose { perm: *perm, ndim: *ndim }, None)
            }

            // ── Concat — merge along last axis ────────────────────────────────
            FloatOp::Concat { .. } => (ShapeSpecRepr::Concat, None),

            // ── Slice — one-axis contiguous slice ─────────────────────────────
            FloatOp::Slice { axis_from_end, start, end, .. } => (
                ShapeSpecRepr::Slice {
                    axis_from_end: *axis_from_end,
                    start: *start,
                    end: *end,
                },
                None,
            ),

            // ── Attention — output is [num_q_heads, seq_q, head_dim] ──
            // Can't use SameAs(0) because Q's compiled shape may have
            // heads*head_dim merged. Encode explicitly; seq_q is Inferred
            // from total Q elements / (num_q_heads * head_dim).
            FloatOp::Attention {
                head_dim,
                num_q_heads,
                ..
            } => {
                let dims = vec![
                    ShapeDimRepr::Fixed(*num_q_heads),
                    ShapeDimRepr::Inferred, // seq_q = Q_elems / (num_q_heads * head_dim)
                    ShapeDimRepr::Fixed(*head_dim),
                ];
                (ShapeSpecRepr::Dims(dims), None)
            }

            // ── Shape op — output is [ndim_of_input] ──────────────────────────
            FloatOp::Shape { .. } => (ShapeSpecRepr::Shape, None),

            // ── Where — three-input broadcast ─────────────────────────────────
            FloatOp::Where => (ShapeSpecRepr::BroadcastAll, None),

            // ── Range — not resolvable from shapes alone ──────────────────────
            FloatOp::Range => (ShapeSpecRepr::Unknown, None),

            // ── Vision ops — spatial formulas (complex, flag as Unknown) ──────
            FloatOp::Conv2d { .. }
            | FloatOp::ConvTranspose { .. }
            | FloatOp::MaxPool2d { .. }
            | FloatOp::AvgPool2d { .. }
            | FloatOp::GlobalAvgPool { .. }
            | FloatOp::Resize { .. }
            | FloatOp::PadOp { .. } => (ShapeSpecRepr::Unknown, None),

            // ── Utility ops ──────────────────────────────────────────────────
            FloatOp::TopK { .. } | FloatOp::NonZero | FloatOp::Compress { .. } => {
                (ShapeSpecRepr::Unknown, None)
            }

            // ── KV cache ────────────────────────────────────────────────────
            FloatOp::KvWrite { .. } => (ShapeSpecRepr::SameAs(0), None),
            FloatOp::KvRead { .. } => (ShapeSpecRepr::Unknown, None),
        })
    }
}

// ── AiOp implementation ──────────────────────────────────────────────────────

impl ShapeProjection for AiOp {
    fn shape_spec(&self) -> Option<(ShapeSpecRepr, Option<u8>)> {
        Some(match self {
            // ── Unary elementwise → SameAs(0) ──────────────────────────────
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
            | AiOp::IsNaN
            | AiOp::Dequantize
            | AiOp::Identity => (ShapeSpecRepr::SameAs(0), None),

            // ── Binary elementwise → Broadcast(0, 1) ──────────────────────
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
            | AiOp::Xor => (ShapeSpecRepr::Broadcast(0, 1), None),

            // ── Binary comparison → Broadcast(0, 1) ──────────────────────
            AiOp::Equal
            | AiOp::Less
            | AiOp::LessOrEqual
            | AiOp::Greater
            | AiOp::GreaterOrEqual => (ShapeSpecRepr::Broadcast(0, 1), None),

            // ── Shape-preserving (output = input[0] shape) ────────────────
            AiOp::Softmax { .. }
            | AiOp::LogSoftmax { .. }
            | AiOp::RmsNorm { .. }
            | AiOp::LayerNorm { .. }
            | AiOp::GroupNorm { .. }
            | AiOp::BatchNorm { .. }
            | AiOp::RotaryEmbedding { .. }
            | AiOp::FusedSwiGLU
            | AiOp::FusedLayerNormResidual { .. }
            | AiOp::KvSlotWrite { .. }
            | AiOp::KvSlotRead { .. }
            | AiOp::Quantize { .. }
            | AiOp::InstanceNorm { .. }
            | AiOp::LRN { .. }
            | AiOp::CumSum { .. }
            | AiOp::ReverseSequence { .. }
            | AiOp::Cast { .. }
            | AiOp::GatherND { .. }
            | AiOp::ScatterND { .. }
            | AiOp::Scatter { .. } => (ShapeSpecRepr::SameAs(0), None),

            // ── Linear algebra (k_hint=0 at AiOp level; concrete in FloatOp) ─
            AiOp::MatMul
            | AiOp::BatchMatMul
            | AiOp::MatMulRelu
            | AiOp::MatMulGelu
            | AiOp::MatMulSilu
            | AiOp::ConcatMatMul { .. } => {
                (ShapeSpecRepr::MatMul { k_hint: 0 }, None)
            }
            AiOp::Gemm { .. } => (ShapeSpecRepr::Gemm { k: 0 }, None),

            // ── Attention (output = Q input shape) ─────────────────────────
            // At AiOp level, Q's shape should be correct ([batch, heads, seq, dim]),
            // so SameAs(0) is fine. The FloatOp level uses explicit Dims.
            AiOp::MultiHeadAttention { .. }
            | AiOp::FlashAttentionHint => (ShapeSpecRepr::SameAs(0), None),
            AiOp::GroupedQueryAttention { num_heads, head_dim, .. } => {
                let dims = vec![
                    ShapeDimRepr::Fixed(*num_heads),
                    ShapeDimRepr::Inferred,
                    ShapeDimRepr::Fixed(*head_dim),
                ];
                (ShapeSpecRepr::Dims(dims), None)
            }

            // ── Shape manipulation ─────────────────────────────────────────
            AiOp::Reshape { .. } | AiOp::Flatten { .. } => {
                (ShapeSpecRepr::Reshape, Some(1))
            }
            AiOp::Transpose { perm } => {
                let mut arr = [0u8; 8];
                let ndim = perm.len().min(8) as u8;
                for (i, &p) in perm.iter().take(8).enumerate() {
                    arr[i] = p as u8;
                }
                (ShapeSpecRepr::Transpose { perm: arr, ndim }, None)
            }
            AiOp::Concat { .. } => (ShapeSpecRepr::Concat, None),
            AiOp::Split { .. } => (ShapeSpecRepr::SameAs(0), None),
            AiOp::Slice { .. } => return None, // needs axis/start/end from strategy
            AiOp::Unsqueeze { axes } => (
                ShapeSpecRepr::Unsqueeze {
                    axes: axes.iter().map(|&a| a as i8).collect(),
                },
                None,
            ),
            AiOp::Squeeze { axes } => (
                ShapeSpecRepr::Squeeze {
                    axes: axes.iter().map(|&a| a as i8).collect(),
                },
                None,
            ),
            AiOp::Expand => (ShapeSpecRepr::Reshape, Some(1)),
            AiOp::Tile { repeats } => (
                ShapeSpecRepr::Tile {
                    repeats: repeats.iter().map(|&r| r as u32).collect(),
                },
                None,
            ),
            AiOp::Shape { .. } => (ShapeSpecRepr::Shape, None),
            AiOp::Where => (ShapeSpecRepr::BroadcastAll, None),
            AiOp::Range => (ShapeSpecRepr::Unknown, None),

            // ── Gather/Embed (dim=0 at AiOp level; concrete in FloatOp) ──
            AiOp::Gather { .. }
            | AiOp::GatherElements { .. }
            | AiOp::Embed => (ShapeSpecRepr::GatherEmbed { dim: 0 }, None),

            // ── Reductions ─────────────────────────────────────────────────
            AiOp::ReduceSum { .. }
            | AiOp::ReduceMean { .. }
            | AiOp::ReduceMax { .. }
            | AiOp::ReduceMin { .. }
            | AiOp::ReduceProd { .. }
            | AiOp::ReduceL1 { .. }
            | AiOp::ReduceL2 { .. }
            | AiOp::ArgMax { .. }
            | AiOp::ArgMin { .. } => (ShapeSpecRepr::DropLastDim(0), None),

            // ── Vision ops (spatial formulas — Unknown) ────────────────────
            AiOp::Conv { .. }
            | AiOp::ConvTranspose { .. }
            | AiOp::MaxPool { .. }
            | AiOp::AveragePool { .. }
            | AiOp::GlobalAveragePool
            | AiOp::Resize { .. }
            | AiOp::Pad { .. } => (ShapeSpecRepr::Unknown, None),

            // ── Utility ops ────────────────────────────────────────────────
            AiOp::TopK { .. }
            | AiOp::NonZero
            | AiOp::Compress { .. }
            | AiOp::OneHot { .. }
            | AiOp::DepthToSpace { .. }
            | AiOp::SpaceToDepth { .. } => (ShapeSpecRepr::Unknown, None),

            // ── Quantized matmul ───────────────────────────────────────────
            AiOp::QuantizedMatMul { .. } => {
                (ShapeSpecRepr::MatMul { k_hint: 0 }, None)
            }

            // ── Positional / mask ──────────────────────────────────────────
            AiOp::AlibiSlope | AiOp::CausalMask => {
                (ShapeSpecRepr::Unknown, None)
            }

            // ── Einsum (equation-dependent) ────────────────────────────────
            AiOp::Einsum { .. } => (ShapeSpecRepr::Unknown, None),

            // ── Control flow (subgraph — can't project) ────────────────────
            AiOp::If { .. } | AiOp::Loop { .. } | AiOp::Scan { .. } => {
                return None;
            }

            // ── Constant / Opaque (seeded directly, not projected) ────────
            AiOp::Constant { .. } | AiOp::Opaque { .. } => return None,
        })
    }
}

// ── Backward-compatible wrapper ──────────────────────────────────────────────

/// Convert a lowered `FloatOp` to a serializable `ShapeSpecRepr`.
///
/// Thin wrapper around [`ShapeProjection::shape_spec()`] for backward
/// compatibility. Panics if the `FloatOp` returns `None` (which no current
/// variant does).
pub fn float_op_to_shape_spec_repr(op: &FloatOp) -> (ShapeSpecRepr, Option<u8>) {
    op.shape_spec()
        .expect("all FloatOp variants have a shape spec")
}

// ── Runtime shape resolution ─────────────────────────────────────────────────

/// Resolve the output shape for a `ShapeSpecRepr` given input shapes.
///
/// `input_shapes`: shapes of each input node (indexed same as `input_node_ids`).
/// `shape_value_bytes`: raw bytes from the shape-value input (for Reshape), if available.
/// `input_elems`: element count of input[0] (from buffer size), used for Inferred dims.
///
/// Returns `None` if the shape cannot be determined from available information.
pub fn resolve_spec(
    spec: &ShapeSpecRepr,
    input_shapes: &[Vec<usize>],
    shape_value_bytes: Option<&[u8]>,
    input_elems: usize,
) -> Option<Vec<usize>> {
    match spec {
        ShapeSpecRepr::SameAs(i) => input_shapes.get(*i as usize).cloned(),

        ShapeSpecRepr::Broadcast(a, b) => {
            let sa = input_shapes.get(*a as usize)?;
            let sb = input_shapes.get(*b as usize)?;
            Some(broadcast_shapes(sa, sb))
        }

        ShapeSpecRepr::BroadcastAll => {
            let mut result = input_shapes.first()?.clone();
            for s in input_shapes.iter().skip(1) {
                result = broadcast_shapes(&result, s);
            }
            Some(result)
        }

        ShapeSpecRepr::DropLastDim(i) => {
            let s = input_shapes.get(*i as usize)?;
            if s.len() > 1 {
                Some(s[..s.len() - 1].to_vec())
            } else {
                Some(vec![1])
            }
        }

        ShapeSpecRepr::Dims(dims) => resolve_dims(dims, input_shapes, input_elems),

        ShapeSpecRepr::MatMul { k_hint } => resolve_matmul(input_shapes, *k_hint as usize),

        ShapeSpecRepr::Gemm { k } => resolve_gemm(input_shapes, *k as usize),

        ShapeSpecRepr::GatherEmbed { dim } => {
            let indices_shape = input_shapes.first()?;
            let mut out = indices_shape.clone();
            out.push(*dim as usize);
            Some(out)
        }

        ShapeSpecRepr::GatherAxis { axis } => {
            // Gather along axis: output = data.shape with data.shape[axis]
            // replaced by product(indices.shape).
            // In hologram, inputs are [indices, data].
            let indices_shape = input_shapes.first()?;
            let data_shape = input_shapes.get(1)?;
            let ax = *axis as usize;
            if ax >= data_shape.len() {
                return None;
            }
            let indices_count: usize = indices_shape.iter().product::<usize>().max(1);
            let mut out = data_shape.clone();
            out[ax] = indices_count;
            if indices_shape.is_empty() || (indices_shape.len() == 1 && indices_shape[0] == 1) {
                out.remove(ax);
            }
            Some(out)
        }

        ShapeSpecRepr::Reshape => resolve_reshape(input_shapes, shape_value_bytes, input_elems),

        ShapeSpecRepr::Transpose { perm, ndim } => {
            let in_shape = input_shapes.first()?;
            let nd = *ndim as usize;
            if nd == 0 || in_shape.len() < nd {
                return None;
            }
            let p = &perm[..nd];
            if p.iter().any(|&pi| (pi as usize) >= in_shape.len()) {
                return None;
            }
            Some(p.iter().map(|&pi| in_shape[pi as usize]).collect())
        }

        ShapeSpecRepr::Concat => {
            // Concat along the last axis across ALL inputs (not just 2).
            // output[-1] = sum of all input[-1] values.
            let first = input_shapes.first()?;
            if first.is_empty() {
                return None;
            }
            let mut out = first.clone();
            let last_idx = out.len() - 1;
            for other in input_shapes.iter().skip(1) {
                if let Some(&other_last) = other.last() {
                    out[last_idx] += other_last;
                }
            }
            Some(out)
        }

        ShapeSpecRepr::Slice { axis_from_end, start, end } => {
            let in_shape = input_shapes.first()?.clone();
            let ndim = in_shape.len();
            let afe = *axis_from_end as usize;
            let axis = ndim.saturating_sub(afe);
            let axis_size = in_shape.get(axis).copied().unwrap_or(1);
            let actual_end = (*end as usize).min(axis_size);
            let slice_len = actual_end.saturating_sub(*start as usize);
            let mut out = in_shape;
            if axis < out.len() {
                out[axis] = slice_len;
            }
            Some(out)
        }

        ShapeSpecRepr::Shape => {
            let ndim = input_shapes.first()?.len();
            Some(vec![ndim])
        }

        ShapeSpecRepr::Unsqueeze { axes } => {
            let in_shape = input_shapes.first()?;
            let out_rank = in_shape.len() + axes.len();
            let normalized: BTreeSet<usize> = axes
                .iter()
                .map(|&a| {
                    if a < 0 {
                        (out_rank as i8 + a) as usize
                    } else {
                        a as usize
                    }
                })
                .collect();
            let mut out = Vec::with_capacity(out_rank);
            let mut src = 0;
            for i in 0..out_rank {
                if normalized.contains(&i) {
                    out.push(1);
                } else {
                    out.push(*in_shape.get(src)?);
                    src += 1;
                }
            }
            Some(out)
        }

        ShapeSpecRepr::Squeeze { axes } => {
            let in_shape = input_shapes.first()?;
            let ndim = in_shape.len();
            let remove: BTreeSet<usize> = axes
                .iter()
                .map(|&a| {
                    if a < 0 {
                        (ndim as i8 + a) as usize
                    } else {
                        a as usize
                    }
                })
                .collect();
            Some(
                in_shape
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| !remove.contains(i))
                    .map(|(_, &d)| d)
                    .collect(),
            )
        }

        ShapeSpecRepr::Tile { repeats } => {
            let in_shape = input_shapes.first()?;
            Some(
                in_shape
                    .iter()
                    .zip(repeats.iter())
                    .map(|(&d, &r)| d * r as usize)
                    .collect(),
            )
        }

        ShapeSpecRepr::Unknown => None,
    }
}

// ── Internal helpers ──────────────────────────────────────────────────────────

fn broadcast_shapes(a: &[usize], b: &[usize]) -> Vec<usize> {
    let max_len = a.len().max(b.len());
    let mut result = Vec::with_capacity(max_len);
    for i in 0..max_len {
        let da = if i < max_len - a.len() { 1 } else { a[i - (max_len - a.len())] };
        let db = if i < max_len - b.len() { 1 } else { b[i - (max_len - b.len())] };
        result.push(da.max(db));
    }
    result
}

fn resolve_dims(
    dims: &[ShapeDimRepr],
    input_shapes: &[Vec<usize>],
    input_elems: usize,
) -> Option<Vec<usize>> {
    let mut shape = Vec::with_capacity(dims.len());
    let mut known_product = 1usize;
    let mut inferred_idx = None;

    for (i, dim) in dims.iter().enumerate() {
        match dim {
            ShapeDimRepr::Fixed(v) => {
                let v = *v as usize;
                shape.push(v);
                known_product = known_product.saturating_mul(v.max(1));
            }
            ShapeDimRepr::FromInput { input, axis } => {
                let v = input_shapes
                    .get(*input as usize)
                    .and_then(|s| {
                        let idx = if *axis < 0 {
                            s.len().wrapping_add(*axis as usize)
                        } else {
                            *axis as usize
                        };
                        s.get(idx).copied()
                    })
                    .unwrap_or(1);
                shape.push(v);
                known_product = known_product.saturating_mul(v.max(1));
            }
            ShapeDimRepr::Inferred => {
                shape.push(0);
                inferred_idx = Some(i);
            }
        }
    }

    if let Some(idx) = inferred_idx {
        if known_product > 0 && input_elems > 0 {
            shape[idx] = input_elems / known_product;
        } else {
            return None;
        }
    }

    if shape.contains(&0) {
        None
    } else {
        Some(shape)
    }
}

fn resolve_matmul(input_shapes: &[Vec<usize>], k_hint: usize) -> Option<Vec<usize>> {
    if input_shapes.len() < 2 {
        return None;
    }
    let a = &input_shapes[0];
    let b = &input_shapes[1];

    // Batched matmul (N-D).
    if a.len() >= 2 && b.len() >= 2 {
        let mut out = a[..a.len() - 1].to_vec();
        let n = *b.last()?;
        out.push(n);
        if !out.contains(&0) {
            return Some(out);
        }
    }

    // Fallback using k_hint — preserve A's leading dims when possible.
    if k_hint > 0 {
        let b_elems: usize = b.iter().product();
        let n = b_elems / k_hint;
        if n > 0 && a.len() >= 2 && *a.last().unwrap_or(&0) == k_hint {
            let mut out = a.clone();
            *out.last_mut().expect("non-empty shape") = n;
            return Some(out);
        }
        let a_elems: usize = a.iter().product();
        let m = a_elems / k_hint;
        if m > 0 && n > 0 {
            return Some(vec![m, n]);
        }
    }

    None
}

fn resolve_gemm(input_shapes: &[Vec<usize>], k: usize) -> Option<Vec<usize>> {
    if input_shapes.len() < 2 {
        return None;
    }
    let a = &input_shapes[0];
    let b = &input_shapes[1];
    if k == 0 {
        return None;
    }
    let b_elems: usize = b.iter().product();
    let n = b_elems / k;
    if n == 0 {
        return None;
    }
    // Preserve A's leading dims (batch, seq) — only replace last dim with n.
    // This prevents 3-D inputs like [1, seq, hidden] from collapsing to 2-D [m, n],
    // which would lose rank information and corrupt downstream shape resolution.
    if a.len() >= 2 && *a.last().unwrap_or(&0) == k {
        let mut out = a.clone();
        *out.last_mut().expect("non-empty shape") = n;
        return Some(out);
    }
    // Fallback: flat 2-D.
    let a_elems: usize = a.iter().product();
    let m = a_elems / k;
    if m > 0 {
        Some(vec![m, n])
    } else {
        None
    }
}

fn resolve_reshape(
    input_shapes: &[Vec<usize>],
    shape_bytes: Option<&[u8]>,
    input_elems: usize,
) -> Option<Vec<usize>> {
    let effective_elems = if input_elems > 0 {
        input_elems
    } else {
        input_shapes
            .first()
            .map(|s| s.iter().product())
            .unwrap_or(0)
    };
    if effective_elems == 0 {
        return None;
    }

    // Parse shape tensor bytes if available.
    if let Some(parsed) = shape_bytes.and_then(|b| parse_shape_i64(b, effective_elems)) {
        if !parsed.contains(&0) {
            let product: usize = parsed.iter().product();
            if product == effective_elems
                || (product > effective_elems
                    && effective_elems > 0
                    && product.is_multiple_of(effective_elems))
            {
                return Some(parsed);
            }
        }
        // Single-zero resolution.
        let zeros = parsed.iter().filter(|&&d| d == 0).count();
        if zeros == 1 {
            let known: usize = parsed
                .iter()
                .filter(|&&d| d > 0)
                .product::<usize>()
                .max(1);
            if effective_elems >= known && effective_elems.is_multiple_of(known) {
                return Some(
                    parsed
                        .iter()
                        .map(|&d| if d == 0 { effective_elems / known } else { d })
                        .collect(),
                );
            }
        }
    }

    None
}

/// Parse a shape tensor (raw bytes) to a `Vec<usize>`, resolving a single -1.
fn parse_shape_i64(bytes: &[u8], n_elems: usize) -> Option<Vec<usize>> {
    if bytes.is_empty() {
        return None;
    }
    let vals: Vec<i64> = if bytes.len().is_multiple_of(8) {
        bytes
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().expect("8-byte chunk")))
            .collect()
    } else if bytes.len().is_multiple_of(4) {
        bytes
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes(c.try_into().expect("4-byte chunk")) as i64)
            .collect()
    } else {
        return None;
    };

    let shape: Vec<usize> = vals
        .iter()
        .map(|&v| if v < 0 { 0 } else { v as usize })
        .collect();
    let zeros = shape.iter().filter(|&&d| d == 0).count();
    if zeros == 1 {
        let known: usize = shape
            .iter()
            .filter(|&&d| d > 0)
            .product::<usize>()
            .max(1);
        let unknown = if known > 0 { n_elems / known } else { n_elems };
        Some(
            shape
                .iter()
                .map(|&d| if d == 0 { unknown } else { d })
                .collect(),
        )
    } else {
        Some(shape)
    }
}

// ── Runtime shape context walker ─────────────────────────────────────────────

/// Walk the compile-time [`ShapeContextGraph`] to fully populate `shape_map`
/// before any op dispatch.
///
/// In addition to propagating output shapes, this walker maintains a secondary
/// map of concrete i64 values for shape-compute tensors (the outputs of `Shape`,
/// `Gather`, `Concat`, `Slice`, and `Cast` when applied to integer tensors).
/// This enables resolving `Reshape` ops whose target-shape input is a computed
/// intermediate (the `Shape→Gather→Concat→Reshape` pattern common in LLM
/// multi-head attention), even when the sequence length is not known at compile
/// time.
///
/// # Arguments
///
/// - `ctx_graph` — the serialized shape projection map from the `.holo` archive.
/// - `runtime_inputs` — user-supplied input shapes, keyed by graph node ID.
///   These override or supplement seeds for symbolic input nodes.
/// - `shape_value_bytes` — raw bytes for shape-value tensors (e.g., the second
///   input to a `Reshape` op). Keyed by the node ID of the shape-value node.
///   May be empty — the walker synthesizes bytes from propagated i64 values.
/// - `shape_map` — mutable map populated by this call. On return every node
///   for which a shape could be determined has an entry here.
///
/// Entries that cannot be resolved (e.g., `Unknown` specs) are skipped; the
/// caller is responsible for falling back to buffer-size inference for those.
pub fn walk_shape_context(
    ctx_graph: &ShapeContextGraph,
    runtime_inputs: &HashMap<u32, Vec<usize>>,
    shape_value_bytes: &HashMap<u32, Vec<u8>>,
    shape_map: &mut HashMap<u32, Vec<usize>>,
) {
    // Secondary map: concrete i64 values for shape-compute tensors.
    // Seeded from compile-time constants; extended during the topo walk.
    let mut i64_values: HashMap<u32, Vec<i64>> = HashMap::new();

    // Step 1 — seed from compile-time concrete shapes + known i64 values.
    for seed in &ctx_graph.seeds {
        let shape: Vec<usize> = seed.shape.iter().map(|&d| d as usize).collect();
        shape_map.entry(seed.node_id).or_insert(shape);
        if let Some(vals) = &seed.known_i64_values {
            // Flatten Option<i64> to i64 (use 0 as sentinel for unknown dims).
            let i64_vals: Vec<i64> = vals.iter().map(|v| v.unwrap_or(0)).collect();
            if !i64_vals.is_empty() {
                i64_values.insert(seed.node_id, i64_vals);
            }
        }
    }

    // Step 2 — inject / override with runtime-supplied input shapes.
    for (&node_id, shape) in runtime_inputs {
        shape_map.insert(node_id, shape.clone());
    }

    // Step 3 — topological projection (entries are already in topo order).
    for entry in &ctx_graph.projections {
        // Collect input shapes from the map (skip if any are missing).
        let input_shapes: Vec<Vec<usize>> = entry
            .input_node_ids
            .iter()
            .map(|id| shape_map.get(id).cloned().unwrap_or_default())
            .collect();

        // For Reshape: synthesize shape-value bytes from propagated i64 values
        // when the caller did not supply pre-computed bytes. This handles the
        // `Shape→Gather→Concat→Reshape` pattern for dynamic seq_len.
        let synthesized_sv: Option<Vec<u8>> = (|| {
            if !matches!(entry.spec, ShapeSpecRepr::Reshape) {
                return None;
            }
            let snid = entry
                .shape_value_input
                .and_then(|idx| entry.input_node_ids.get(idx as usize))
                .copied()?;
            let vals = i64_values.get(&snid)?;
            // Validate: the i64 count (= target rank) must match the Reshape
            // node's compiled output rank. When Shape(mask)=[batch,seq] feeds
            // directly into a Concat (without Gather to pick just seq), the
            // Concat produces more i64 values than the Reshape expects. This
            // causes the Reshape to change rank (e.g., 4-dim → 5-dim), breaking
            // downstream Gather index lookups.
            // Use the shape_value tensor's shape as the authoritative rank.
            if let Some(sv_shape) = shape_map.get(&snid) {
                let expected_elems: usize = sv_shape.iter().product();
                if expected_elems > 0 && vals.len() != expected_elems {
                    return None;
                }
            }
            // The i64 values are accepted — the Reshape may change rank at
            // runtime (e.g., from 4-dim to 5-dim) if the Concat assembled more
            // values from Shape(mask)=[batch,seq] than at compile time.
            // This is correct: the Reshape target is as the ONNX graph designed.
            Some(
                vals.iter()
                    .flat_map(|&v| v.to_le_bytes())
                    .collect::<Vec<u8>>(),
            )
        })();
        let sv_bytes: Option<&[u8]> = synthesized_sv.as_deref().or_else(|| {
            entry
                .shape_value_input
                .and_then(|idx| entry.input_node_ids.get(idx as usize))
                .and_then(|node_id| shape_value_bytes.get(node_id))
                .map(|v| v.as_slice())
        });

        // Element count of primary input (for Inferred dims / Reshape fallback).
        let input_elems: usize = input_shapes
            .first()
            .map(|s| s.iter().product())
            .unwrap_or(0);

        if let Some(out_shape) = resolve_spec(&entry.spec, &input_shapes, sv_bytes, input_elems) {
            // Propagate i64 values through shape-compute ops so downstream
            // Reshape nodes can determine their target shape.
            propagate_i64_values(entry, &input_shapes, &out_shape, &mut i64_values, shape_map);

            shape_map.insert(entry.node_id, out_shape);
        }
    }
}

/// Propagate concrete i64 values through shape-compute operations.
///
/// Called after each projection entry resolves its output shape. Updates
/// `i64_values` with the entry's output values when the op is a shape-compute
/// op (Shape, Gather, Concat, Slice, Cast/identity, Unsqueeze, Squeeze) and
/// the input values are available.
fn propagate_i64_values(
    entry: &ShapeProjectionEntry,
    input_shapes: &[Vec<usize>],
    _out_shape: &[usize],
    i64_values: &mut HashMap<u32, Vec<i64>>,
    shape_map: &HashMap<u32, Vec<usize>>,
) {
    match &entry.spec {
        // Shape op: output i64 values = the input tensor's shape dimensions.
        ShapeSpecRepr::Shape => {
            if let Some(in_shape) = input_shapes.first() {
                if !in_shape.is_empty() {
                    let vals: Vec<i64> = in_shape.iter().map(|&d| d as i64).collect();
                    i64_values.insert(entry.node_id, vals);
                }
            }
        }

        // Concat: concatenate i64 values from ALL inputs (only when all present).
        // Handles both 2-input and N-input concat (e.g., [seq, n_heads, head_dim]).
        ShapeSpecRepr::Concat => {
            let all: Option<Vec<i64>> = entry
                .input_node_ids
                .iter()
                .map(|id: &u32| i64_values.get(id).cloned())
                .collect::<Option<Vec<_>>>()
                .map(|vecs| vecs.into_iter().flatten().collect());
            if let Some(vals) = all {
                // Validate: the concatenated i64 count must match the output
                // tensor's element count. If a Shape upstream contributed too many
                // values (e.g., Shape(mask)=[batch,seq] instead of Gather'd [seq]),
                // the chain is broken — skip to prevent wrong Reshape targets.
                let expected = shape_map
                    .get(&entry.node_id)
                    .map(|s| s.iter().product::<usize>())
                    .unwrap_or(0);
                if expected == 0 || vals.len() == expected {
                    i64_values.insert(entry.node_id, vals);
                }
            }
        }

        // Slice: slice the i64 values along the specified axis.
        ShapeSpecRepr::Slice {
            axis_from_end,
            start,
            end,
        } => {
            if let Some(&src_id) = entry.input_node_ids.first() {
                if let Some(src_vals) = i64_values.get(&src_id).cloned() {
                    let n = src_vals.len();
                    let afe = *axis_from_end as usize;
                    let _axis = n.saturating_sub(afe);
                    let s = (*start as usize).min(n);
                    let e = (*end as usize).min(n);
                    if e > s {
                        i64_values.insert(entry.node_id, src_vals[s..e].to_vec());
                    }
                }
            }
        }

        // GatherAxis (shape-dim picking): pick i64 values at indices.
        // In hologram's representation, inputs are SWAPPED: (indices, data).
        ShapeSpecRepr::GatherAxis { .. } | ShapeSpecRepr::GatherEmbed { .. } => {
            let indices_id = entry.input_node_ids.first().copied();
            let data_id = entry.input_node_ids.get(1).copied();
            if let (Some(idx_id), Some(dat_id)) = (indices_id, data_id) {
                if let (Some(idx_vals), Some(data_vals)) =
                    (i64_values.get(&idx_id), i64_values.get(&dat_id))
                {
                    // Pick data_vals at each index in idx_vals.
                    let out: Vec<i64> = idx_vals
                        .iter()
                        .filter_map(|&idx| {
                            let i = if idx < 0 {
                                (data_vals.len() as i64 + idx) as usize
                            } else {
                                idx as usize
                            };
                            data_vals.get(i).copied()
                        })
                        .collect();
                    if !out.is_empty() {
                        i64_values.insert(entry.node_id, out);
                    }
                }
            }
        }

        // SameAs(0): pass i64 values through unary ops (Cast, identity).
        ShapeSpecRepr::SameAs(0) => {
            if let Some(&src_id) = entry.input_node_ids.first() {
                if let Some(vals) = i64_values.get(&src_id).cloned() {
                    i64_values.insert(entry.node_id, vals);
                }
            }
        }

        // Unsqueeze / Squeeze: i64 values pass through unchanged
        // (only shape metadata changes, not the actual data values).
        ShapeSpecRepr::Unsqueeze { .. } | ShapeSpecRepr::Squeeze { .. } => {
            if let Some(&src_id) = entry.input_node_ids.first() {
                if let Some(vals) = i64_values.get(&src_id).cloned() {
                    i64_values.insert(entry.node_id, vals);
                }
            }
        }

        _ => {}
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── FloatOp trait tests ──────────────────────────────────────────────

    #[test]
    fn trait_float_op_same_as() {
        let (spec, sv) = FloatOp::Relu.shape_spec().expect("should have spec");
        assert!(matches!(spec, ShapeSpecRepr::SameAs(0)));
        assert!(sv.is_none());
    }

    #[test]
    fn trait_float_op_broadcast() {
        let (spec, sv) = FloatOp::Add.shape_spec().expect("should have spec");
        assert!(matches!(spec, ShapeSpecRepr::Broadcast(0, 1)));
        assert!(sv.is_none());
    }

    #[test]
    fn trait_float_op_matmul() {
        let op = FloatOp::MatMul { m: 0, k: 2048, n: 4096 };
        let (spec, sv) = op.shape_spec().expect("should have spec");
        assert!(matches!(spec, ShapeSpecRepr::MatMul { k_hint: 2048 }));
        assert!(sv.is_none());
    }

    #[test]
    fn trait_matches_wrapper() {
        // Verify the backward-compat wrapper matches the trait.
        let ops: Vec<FloatOp> = vec![
            FloatOp::Relu,
            FloatOp::Add,
            FloatOp::MatMul { m: 1, k: 64, n: 128 },
            FloatOp::Reshape,
            FloatOp::Softmax { size: 32 },
        ];
        for op in &ops {
            let from_fn = float_op_to_shape_spec_repr(op);
            let from_trait = op.shape_spec().expect("should have spec");
            assert_eq!(from_fn, from_trait, "mismatch for {:?}", op);
        }
    }

    // ── AiOp trait tests ─────────────────────────────────────────────────

    #[test]
    fn trait_ai_op_unsqueeze() {
        let op = AiOp::Unsqueeze { axes: vec![1, 3] };
        let (spec, sv) = op.shape_spec().expect("should have spec");
        assert!(matches!(spec, ShapeSpecRepr::Unsqueeze { .. }));
        assert!(sv.is_none());
    }

    #[test]
    fn trait_ai_op_squeeze() {
        let op = AiOp::Squeeze { axes: vec![0, 2] };
        let (spec, sv) = op.shape_spec().expect("should have spec");
        assert!(matches!(spec, ShapeSpecRepr::Squeeze { .. }));
        assert!(sv.is_none());
    }

    #[test]
    fn trait_ai_op_tile() {
        let op = AiOp::Tile { repeats: vec![2, 3] };
        let (spec, sv) = op.shape_spec().expect("should have spec");
        assert!(matches!(spec, ShapeSpecRepr::Tile { .. }));
        assert!(sv.is_none());
    }

    #[test]
    fn trait_ai_op_returns_none_for_control_flow() {
        assert!(AiOp::If { then_branch: "t".into(), else_branch: None }.shape_spec().is_none());
        assert!(AiOp::Constant { value: crate::ir::AiParam::Inline {
            data: vec![],
            info: crate::ir::TensorInfo::new(crate::ir::DType::F32, smallvec::smallvec![]),
        }}.shape_spec().is_none());
    }

    // ── Backward compat tests ────────────────────────────────────────────

    #[test]
    fn same_as_unary() {
        let (spec, sv) = float_op_to_shape_spec_repr(&FloatOp::Relu);
        assert!(matches!(spec, ShapeSpecRepr::SameAs(0)));
        assert!(sv.is_none());
    }

    #[test]
    fn broadcast_binary() {
        let (spec, sv) = float_op_to_shape_spec_repr(&FloatOp::Add);
        assert!(matches!(spec, ShapeSpecRepr::Broadcast(0, 1)));
        assert!(sv.is_none());
    }

    #[test]
    fn drop_last_dim_reduce() {
        let (spec, sv) = float_op_to_shape_spec_repr(&FloatOp::ReduceSum { size: 512 });
        assert!(matches!(spec, ShapeSpecRepr::DropLastDim(0)));
        assert!(sv.is_none());
    }

    #[test]
    fn matmul_k_hint() {
        let (spec, sv) =
            float_op_to_shape_spec_repr(&FloatOp::MatMul { m: 0, k: 2048, n: 4096 });
        assert!(matches!(spec, ShapeSpecRepr::MatMul { k_hint: 2048 }));
        assert!(sv.is_none());
    }

    #[test]
    fn reshape_shape_value_input() {
        let (spec, sv) = float_op_to_shape_spec_repr(&FloatOp::Reshape);
        assert!(matches!(spec, ShapeSpecRepr::Reshape));
        assert_eq!(sv, Some(1));
    }

    #[test]
    fn gather_dim() {
        let (spec, sv) = float_op_to_shape_spec_repr(&FloatOp::Gather {
            dim: 2048,
            dtype: hologram::FloatDType::F32,
        });
        assert!(matches!(spec, ShapeSpecRepr::GatherEmbed { dim: 2048 }));
        assert!(sv.is_none());
    }

    // ── Resolve tests ────────────────────────────────────────────────────

    #[test]
    fn resolve_same_as() {
        let shapes = vec![vec![1, 128, 2048]];
        let result = resolve_spec(&ShapeSpecRepr::SameAs(0), &shapes, None, 0);
        assert_eq!(result, Some(vec![1, 128, 2048]));
    }

    #[test]
    fn resolve_broadcast() {
        let shapes = vec![vec![1, 128, 2048], vec![2048]];
        let result = resolve_spec(&ShapeSpecRepr::Broadcast(0, 1), &shapes, None, 0);
        assert_eq!(result, Some(vec![1, 128, 2048]));
    }

    #[test]
    fn resolve_drop_last_dim() {
        let shapes = vec![vec![1, 128, 2048]];
        let result = resolve_spec(&ShapeSpecRepr::DropLastDim(0), &shapes, None, 0);
        assert_eq!(result, Some(vec![1, 128]));
    }

    #[test]
    fn resolve_matmul_batched() {
        let shapes = vec![vec![1, 128, 2048], vec![2048, 4096]];
        let result = resolve_spec(&ShapeSpecRepr::MatMul { k_hint: 2048 }, &shapes, None, 0);
        assert_eq!(result, Some(vec![1, 128, 4096]));
    }

    #[test]
    fn resolve_matmul_symbolic_seq() {
        let shapes = vec![vec![1, 7, 2048], vec![2048, 4096]];
        let result = resolve_spec(&ShapeSpecRepr::MatMul { k_hint: 2048 }, &shapes, None, 0);
        assert_eq!(result, Some(vec![1, 7, 4096]));
    }

    #[test]
    fn resolve_gather_embed() {
        let shapes = vec![vec![1, 128]];
        let result = resolve_spec(&ShapeSpecRepr::GatherEmbed { dim: 2048 }, &shapes, None, 0);
        assert_eq!(result, Some(vec![1, 128, 2048]));
    }

    #[test]
    fn resolve_transpose() {
        let mut perm = [0u8; 8];
        perm[..3].copy_from_slice(&[1, 2, 0]);
        let shapes = vec![vec![2, 3, 4]];
        let result = resolve_spec(&ShapeSpecRepr::Transpose { perm, ndim: 3 }, &shapes, None, 0);
        assert_eq!(result, Some(vec![3, 4, 2]));
    }

    #[test]
    fn resolve_concat() {
        let shapes = vec![vec![1, 128, 1024], vec![1, 128, 1024]];
        let result = resolve_spec(&ShapeSpecRepr::Concat, &shapes, None, 0);
        assert_eq!(result, Some(vec![1, 128, 2048]));
    }

    #[test]
    fn resolve_reshape_from_bytes() {
        let shape_vals: Vec<i64> = vec![1, 7, 32, 64];
        let shape_bytes: Vec<u8> = shape_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let input_elems = 7 * 32 * 64;
        let shapes = vec![vec![input_elems]];
        let result =
            resolve_spec(&ShapeSpecRepr::Reshape, &shapes, Some(&shape_bytes), input_elems);
        assert_eq!(result, Some(vec![1, 7, 32, 64]));
    }

    // ── New variant resolve tests ────────────────────────────────────────

    #[test]
    fn resolve_unsqueeze() {
        let shapes = vec![vec![2, 3, 4]];
        let spec = ShapeSpecRepr::Unsqueeze { axes: vec![1] };
        let result = resolve_spec(&spec, &shapes, None, 0);
        assert_eq!(result, Some(vec![2, 1, 3, 4]));
    }

    #[test]
    fn resolve_unsqueeze_negative() {
        let shapes = vec![vec![2, 3]];
        let spec = ShapeSpecRepr::Unsqueeze { axes: vec![-1] };
        let result = resolve_spec(&spec, &shapes, None, 0);
        assert_eq!(result, Some(vec![2, 3, 1]));
    }

    #[test]
    fn resolve_unsqueeze_multiple() {
        let shapes = vec![vec![2, 3]];
        let spec = ShapeSpecRepr::Unsqueeze { axes: vec![0, 3] };
        let result = resolve_spec(&spec, &shapes, None, 0);
        assert_eq!(result, Some(vec![1, 2, 3, 1]));
    }

    #[test]
    fn resolve_squeeze() {
        let shapes = vec![vec![2, 1, 3, 1, 4]];
        let spec = ShapeSpecRepr::Squeeze { axes: vec![1, 3] };
        let result = resolve_spec(&spec, &shapes, None, 0);
        assert_eq!(result, Some(vec![2, 3, 4]));
    }

    #[test]
    fn resolve_squeeze_negative() {
        let shapes = vec![vec![2, 1, 3]];
        let spec = ShapeSpecRepr::Squeeze { axes: vec![-2] };
        let result = resolve_spec(&spec, &shapes, None, 0);
        assert_eq!(result, Some(vec![2, 3]));
    }

    #[test]
    fn resolve_tile() {
        let shapes = vec![vec![2, 3]];
        let spec = ShapeSpecRepr::Tile { repeats: vec![3, 2] };
        let result = resolve_spec(&spec, &shapes, None, 0);
        assert_eq!(result, Some(vec![6, 6]));
    }
}
