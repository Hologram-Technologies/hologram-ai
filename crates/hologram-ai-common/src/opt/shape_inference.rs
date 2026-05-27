//! Per-op shape inference dispatch.

use crate::ir::op::OpCategory;
use crate::ir::shape::DimExpr;
use crate::ir::{shape_from_concrete, AiOp, Shape};

use super::shape_helpers::{
    add_dims, broadcast_shape, normalize_axis, normalize_slice_bound, pool_output_dim,
    reduce_shape, resolve_reshape_shape,
};

/// Infer output shapes for a single op given input shapes.
///
/// `shape_known_values` is `Some` when the shape-input tensor (input[1] for
/// Reshape/Expand) has known constant values from `DataPropagation`.
pub(crate) fn infer_output_shapes(
    op: &AiOp,
    inputs: &[Shape],
    shape_known_values: Option<&[Option<i64>]>,
) -> Vec<Shape> {
    match op.category() {
        OpCategory::UnaryElementwise | OpCategory::ShapePreserving => {
            inputs.first().cloned().into_iter().collect()
        }
        OpCategory::BinaryElementwise | OpCategory::BinaryComparison => {
            if inputs.len() >= 2 {
                vec![broadcast_shape(&inputs[0], &inputs[1])]
            } else {
                inputs.first().cloned().into_iter().collect()
            }
        }
        OpCategory::Custom => infer_custom_output_shapes(op, inputs, shape_known_values),
    }
}

/// Shape inference for ops that need op-specific logic.
fn infer_custom_output_shapes(
    op: &AiOp,
    inputs: &[Shape],
    shape_known_values: Option<&[Option<i64>]>,
) -> Vec<Shape> {
    match op {
        // MatMul: [..., M, K] x [..., K, N] → [..., M, N]
        AiOp::MatMul | AiOp::BatchMatMul => {
            if inputs.len() >= 2 && inputs[0].len() >= 2 && inputs[1].len() >= 2 {
                let a = &inputs[0];
                let b = &inputs[1];
                let mut shape = a[..a.len() - 1].to_vec();
                shape.push(b[b.len() - 1].clone());
                vec![Shape::from(shape)]
            } else {
                vec![Shape::new()]
            }
        }

        // Concat along axis — sum that dimension.
        AiOp::Concat { axis } => {
            if inputs.is_empty() || inputs[0].is_empty() {
                return vec![Shape::new()];
            }
            let mut shape = inputs[0].clone();
            let ax = normalize_axis(*axis, shape.len());
            if ax < shape.len() {
                for inp in &inputs[1..] {
                    if ax < inp.len() {
                        shape[ax] = add_dims(&shape[ax], &inp[ax]);
                    }
                }
            }
            vec![shape]
        }

        // Embed: [batch, seq] → [batch, seq, embed_dim]
        AiOp::Embed => {
            if inputs.len() >= 2 && !inputs[1].is_empty() {
                let mut shape = inputs[0].clone();
                shape.push(inputs[1][inputs[1].len() - 1].clone());
                vec![Shape::from(shape)]
            } else {
                vec![Shape::new()]
            }
        }

        // Attention ops — output shape = [batch, seq, num_heads * head_dim]
        AiOp::MultiHeadAttention {
            num_heads: _,
            head_dim: _,
            ..
        }
        | AiOp::GroupedQueryAttention {
            num_heads: _,
            head_dim: _,
            ..
        } => {
            // Output shape = same as Q input shape.
            // Q is [batch, num_heads, seq, head_dim] (heads_first) or
            // [batch, seq, num_heads, head_dim] (seq_first).
            // Preserve Q's shape exactly — the kernel output matches Q's layout.
            if !inputs.is_empty() && !inputs[0].is_empty() {
                vec![Shape::from(inputs[0].clone())]
            } else {
                vec![Shape::new()]
            }
        }

        // Reductions.
        AiOp::ReduceSum { axes, keepdims }
        | AiOp::ReduceMean { axes, keepdims }
        | AiOp::ReduceMax { axes, keepdims }
        | AiOp::ReduceMin { axes, keepdims } => {
            if let Some(input) = inputs.first() {
                vec![reduce_shape(input, axes, *keepdims)]
            } else {
                vec![Shape::new()]
            }
        }

        // Cast preserves shape.
        AiOp::Cast { .. } => inputs.first().cloned().into_iter().collect(),

        // Shape op: output is a 1-D i64 tensor of length = rank(input).
        AiOp::Shape { start, end } => {
            if let Some(input) = inputs.first() {
                if !input.is_empty() {
                    let rank = input.len() as i64;
                    let s = start.unwrap_or(0);
                    let e = end.unwrap_or(rank);
                    let s = if s < 0 {
                        (rank + s).max(0) as usize
                    } else {
                        s as usize
                    };
                    let e = if e < 0 {
                        (rank + e).max(0) as usize
                    } else {
                        e.min(rank) as usize
                    };
                    let out_len = e.saturating_sub(s);
                    vec![shape_from_concrete(&[out_len as u64])]
                } else {
                    vec![Shape::new()]
                }
            } else {
                vec![Shape::new()]
            }
        }

        // Gather: replace axis dimension with indices shape.
        AiOp::Gather { axis } => {
            if inputs.len() >= 2 && !inputs[0].is_empty() {
                let data = &inputs[0];
                let indices = &inputs[1];
                let ax = normalize_axis(*axis, data.len());
                let mut shape = Vec::new();
                if ax < data.len() {
                    shape.extend_from_slice(&data[..ax]);
                }
                shape.extend_from_slice(indices);
                if ax + 1 < data.len() {
                    shape.extend_from_slice(&data[ax + 1..]);
                }
                vec![Shape::from(shape)]
            } else {
                vec![Shape::new()]
            }
        }

        // GatherElements preserves indices shape.
        AiOp::GatherElements { .. } => {
            if inputs.len() >= 2 && !inputs[1].is_empty() {
                vec![inputs[1].clone()]
            } else {
                vec![Shape::new()]
            }
        }

        // Unsqueeze: insert size-1 dims at specified axes.
        AiOp::Unsqueeze { axes } => {
            if let Some(input) = inputs.first() {
                let out_rank = input.len() + axes.len();
                let norm_axes: Vec<usize> =
                    axes.iter().map(|&a| normalize_axis(a, out_rank)).collect();
                let mut shape = Vec::with_capacity(out_rank);
                let mut in_idx = 0;
                for i in 0..out_rank {
                    if norm_axes.contains(&i) {
                        shape.push(DimExpr::Concrete(1));
                    } else if in_idx < input.len() {
                        shape.push(input[in_idx].clone());
                        in_idx += 1;
                    }
                }
                vec![Shape::from(shape)]
            } else {
                vec![Shape::new()]
            }
        }

        // Squeeze: remove dims at specified axes.
        AiOp::Squeeze { axes } => {
            if let Some(input) = inputs.first() {
                if input.is_empty() {
                    return vec![Shape::new()];
                }
                if axes.is_empty() {
                    let shape: Vec<DimExpr> = input
                        .iter()
                        .filter(|d| d.as_concrete() != Some(1))
                        .cloned()
                        .collect();
                    vec![Shape::from(shape)]
                } else {
                    let ndim = input.len();
                    let norm_axes: Vec<usize> =
                        axes.iter().map(|&a| normalize_axis(a, ndim)).collect();
                    let shape: Vec<DimExpr> = input
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| !norm_axes.contains(i))
                        .map(|(_, d)| d.clone())
                        .collect();
                    vec![Shape::from(shape)]
                }
            } else {
                vec![Shape::new()]
            }
        }

        // Transpose: permute dims.
        AiOp::Transpose { perm } => {
            if let Some(input) = inputs.first() {
                if input.is_empty() || perm.is_empty() {
                    return inputs.first().cloned().into_iter().collect();
                }
                let shape: Vec<DimExpr> = perm
                    .iter()
                    .map(|&p| input.get(p as usize).cloned().unwrap_or(DimExpr::Dynamic))
                    .collect();
                vec![Shape::from(shape)]
            } else {
                vec![Shape::new()]
            }
        }

        // Flatten: collapse to 2-D at axis.
        AiOp::Flatten { axis } => {
            if let Some(input) = inputs.first() {
                if input.is_empty() {
                    return vec![Shape::new()];
                }
                let ax = normalize_axis(*axis, input.len());
                let left: Option<u64> = input[..ax]
                    .iter()
                    .map(|d| d.as_concrete())
                    .collect::<Option<Vec<_>>>()
                    .map(|v| v.iter().product());
                let right: Option<u64> = input[ax..]
                    .iter()
                    .map(|d| d.as_concrete())
                    .collect::<Option<Vec<_>>>()
                    .map(|v| v.iter().product());
                match (left, right) {
                    (Some(l), Some(r)) => vec![shape_from_concrete(&[l, r])],
                    _ => vec![Shape::from(vec![DimExpr::Dynamic, DimExpr::Dynamic])],
                }
            } else {
                vec![Shape::new()]
            }
        }

        // Slice: compute output dims from static starts/ends/steps.
        AiOp::Slice {
            axes,
            starts,
            ends,
            steps,
        } => {
            if let Some(input) = inputs.first() {
                if input.is_empty() {
                    return vec![Shape::new()];
                }
                let mut shape = input.clone();
                for (i, &ax) in axes.iter().enumerate() {
                    let a = normalize_axis(ax, input.len());
                    if a < shape.len() {
                        if let Some(dim_val) = input[a].as_concrete() {
                            let s = normalize_slice_bound(starts[i], dim_val as i64);
                            let e = normalize_slice_bound(ends[i], dim_val as i64);
                            let step = steps.get(i).copied().unwrap_or(1).max(1);
                            let len = if e > s {
                                ((e - s + step - 1) / step) as u64
                            } else {
                                0
                            };
                            shape[a] = DimExpr::Concrete(len);
                        } else {
                            shape[a] = DimExpr::Dynamic;
                        }
                    }
                }
                vec![shape]
            } else {
                vec![Shape::new()]
            }
        }

        // Where: broadcast all three inputs.
        AiOp::Where => {
            if inputs.len() >= 3 {
                let bc = broadcast_shape(&inputs[0], &inputs[1]);
                vec![broadcast_shape(&bc, &inputs[2])]
            } else {
                vec![Shape::new()]
            }
        }

        // Reshape: use known_i64_values from the shape input if available.
        // ONNX Reshape preserves total element count. For entries that are:
        //   - Some(0): copy dim from data input at same position
        //   - Some(n>0): concrete dim
        //   - Some(-1): infer from element count (single -1 allowed)
        //   - None: unknown — inherit from data input, or infer via element count
        AiOp::Reshape { .. } => {
            if let Some(vals) = shape_known_values {
                let data_shape = inputs.first();
                let shape: Vec<DimExpr> = resolve_reshape_shape(vals, data_shape);
                if shape.is_empty() {
                    vec![Shape::new()]
                } else {
                    vec![Shape::from(shape)]
                }
            } else {
                vec![Shape::new()]
            }
        }

        // Expand (ONNX): the output is the *broadcast* of the input shape and
        // the given target shape — NOT the target shape verbatim. A target dim
        // of 1 broadcasts to the input's dim (e.g. `Expand([1,32,1], [1,1,1])`
        // → `[1,32,1]`, not `[1,1,1]`). This is the idiom HF dynamic exports use
        // for `.expand(batch, -1, 1)`: the `-1` is materialized as `1` via a
        // `Where`, and Expand's broadcast restores the kept dimension. Taking
        // the target verbatim silently dropped those dims (e.g. RoPE inv_freq's
        // head_dim/2), so it must broadcast against the input.
        AiOp::Expand => {
            if let Some(vals) = shape_known_values {
                let target: Shape = vals
                    .iter()
                    .map(|v| match v {
                        Some(n) if *n >= 0 => DimExpr::Concrete(*n as u64),
                        _ => DimExpr::Dynamic,
                    })
                    .collect();
                match (target.is_empty(), inputs.first()) {
                    (true, _) => inputs.first().cloned().into_iter().collect(),
                    (false, Some(input)) => vec![broadcast_shape(input, &target)],
                    (false, None) => vec![target],
                }
            } else {
                inputs.first().cloned().into_iter().collect()
            }
        }

        // ── Phase 1: Vision ops ──────────────────────────────────────────────

        // Conv: [N, C, *spatial] → [N, C_out, *conv_spatial]
        // out_dim = floor((in + pad_begin + pad_end - dilation*(kernel-1) - 1) / stride + 1)
        AiOp::Conv {
            kernel_shape,
            strides,
            pads,
            dilations,
            ..
        } => {
            if inputs.len() >= 2 && !inputs[0].is_empty() && inputs[0].len() >= 2 {
                let x = &inputs[0];
                let w = &inputs[1];
                let mut shape = Vec::new();
                shape.push(x[0].clone()); // N (batch)
                                          // C_out from weight[0]
                if !w.is_empty() {
                    shape.push(w[0].clone());
                } else {
                    shape.push(DimExpr::Dynamic);
                }
                // Spatial dims
                let spatial_rank = kernel_shape.len();
                for i in 0..spatial_rank {
                    if let Some(in_dim) = x.get(2 + i).and_then(|d| d.as_concrete()) {
                        let k = kernel_shape.get(i).copied().unwrap_or(1);
                        let s = strides.get(i).copied().unwrap_or(1);
                        let d = dilations.get(i).copied().unwrap_or(1);
                        let p_begin = pads.get(i).copied().unwrap_or(0);
                        let p_end = pads.get(spatial_rank + i).copied().unwrap_or(0);
                        let effective_k = d * (k - 1) + 1;
                        let out = (in_dim + p_begin + p_end).saturating_sub(effective_k) / s + 1;
                        shape.push(DimExpr::Concrete(out));
                    } else {
                        shape.push(DimExpr::Dynamic);
                    }
                }
                vec![Shape::from(shape)]
            } else {
                vec![Shape::new()]
            }
        }

        // ConvTranspose: out_dim = stride * (in - 1) + output_padding + dilation*(kernel-1) - pad_begin - pad_end + 1
        AiOp::ConvTranspose {
            kernel_shape,
            strides,
            pads,
            output_padding,
            dilations,
            ..
        } => {
            if inputs.len() >= 2 && !inputs[0].is_empty() && inputs[0].len() >= 2 {
                let x = &inputs[0];
                let w = &inputs[1];
                let mut shape = Vec::new();
                shape.push(x[0].clone()); // N
                if w.len() >= 2 {
                    shape.push(w[1].clone()); // C_out (weight dim 1 for conv transpose)
                } else {
                    shape.push(DimExpr::Dynamic);
                }
                let spatial_rank = kernel_shape.len();
                for i in 0..spatial_rank {
                    if let Some(in_dim) = x.get(2 + i).and_then(|d| d.as_concrete()) {
                        let k = kernel_shape.get(i).copied().unwrap_or(1);
                        let s = strides.get(i).copied().unwrap_or(1);
                        let d = dilations.get(i).copied().unwrap_or(1);
                        let p_begin = pads.get(i).copied().unwrap_or(0);
                        let p_end = pads.get(spatial_rank + i).copied().unwrap_or(0);
                        let out_pad = output_padding.get(i).copied().unwrap_or(0);
                        let out = s * (in_dim - 1) + out_pad + d * (k - 1) + 1 - p_begin - p_end;
                        shape.push(DimExpr::Concrete(out));
                    } else {
                        shape.push(DimExpr::Dynamic);
                    }
                }
                vec![Shape::from(shape)]
            } else {
                vec![Shape::new()]
            }
        }

        // MaxPool / AveragePool: same formula as Conv (spatial dims only, no weight).
        AiOp::MaxPool {
            kernel_shape,
            strides,
            pads,
            dilations,
            ceil_mode,
            ..
        } => {
            if let Some(x) = inputs.first() {
                if x.len() >= 2 {
                    let mut shape = vec![x[0].clone(), x[1].clone()]; // N, C
                    let spatial_rank = kernel_shape.len();
                    for i in 0..spatial_rank {
                        if let Some(in_dim) = x.get(2 + i).and_then(|d| d.as_concrete()) {
                            let k = kernel_shape.get(i).copied().unwrap_or(1);
                            let s = strides.get(i).copied().unwrap_or(1);
                            let d = dilations.get(i).copied().unwrap_or(1);
                            let p_begin = pads.get(i).copied().unwrap_or(0);
                            let p_end = pads.get(spatial_rank + i).copied().unwrap_or(0);
                            let effective_k = d * (k - 1) + 1;
                            let out =
                                pool_output_dim(in_dim, effective_k, s, p_begin, p_end, *ceil_mode);
                            shape.push(DimExpr::Concrete(out));
                        } else {
                            shape.push(DimExpr::Dynamic);
                        }
                    }
                    vec![Shape::from(shape)]
                } else {
                    vec![Shape::new()]
                }
            } else {
                vec![Shape::new()]
            }
        }

        AiOp::AveragePool {
            kernel_shape,
            strides,
            pads,
            ceil_mode,
            ..
        } => {
            if let Some(x) = inputs.first() {
                if x.len() >= 2 {
                    let mut shape = vec![x[0].clone(), x[1].clone()]; // N, C
                    let spatial_rank = kernel_shape.len();
                    for i in 0..spatial_rank {
                        if let Some(in_dim) = x.get(2 + i).and_then(|d| d.as_concrete()) {
                            let k = kernel_shape.get(i).copied().unwrap_or(1);
                            let s = strides.get(i).copied().unwrap_or(1);
                            let p_begin = pads.get(i).copied().unwrap_or(0);
                            let p_end = pads.get(spatial_rank + i).copied().unwrap_or(0);
                            let out = pool_output_dim(in_dim, k, s, p_begin, p_end, *ceil_mode);
                            shape.push(DimExpr::Concrete(out));
                        } else {
                            shape.push(DimExpr::Dynamic);
                        }
                    }
                    vec![Shape::from(shape)]
                } else {
                    vec![Shape::new()]
                }
            } else {
                vec![Shape::new()]
            }
        }

        // GlobalAveragePool: [N, C, *spatial] → [N, C, 1, 1, ...]
        AiOp::GlobalAveragePool => {
            if let Some(x) = inputs.first() {
                if x.len() >= 2 {
                    let mut shape = vec![x[0].clone(), x[1].clone()];
                    for _ in 2..x.len() {
                        shape.push(DimExpr::Concrete(1));
                    }
                    vec![Shape::from(shape)]
                } else {
                    vec![Shape::new()]
                }
            } else {
                vec![Shape::new()]
            }
        }

        // Resize: output shape from sizes or scales (passed via shape_known_values).
        AiOp::Resize { .. } => {
            if let Some(vals) = shape_known_values {
                let shape: Vec<DimExpr> = vals
                    .iter()
                    .map(|v| match v {
                        Some(n) if *n > 0 => DimExpr::Concrete(*n as u64),
                        _ => DimExpr::Dynamic,
                    })
                    .collect();
                if shape.is_empty() {
                    inputs.first().cloned().into_iter().collect()
                } else {
                    vec![Shape::from(shape)]
                }
            } else {
                // No sizes or scales — preserve input shape.
                inputs.first().cloned().into_iter().collect()
            }
        }

        // Pad: add pad amounts per dim. Pads from known_i64_values (input[1]).
        AiOp::Pad { .. } => {
            if let Some(x) = inputs.first() {
                if let Some(pad_vals) = shape_known_values {
                    // ONNX pads format: [x1_begin, x2_begin, ..., x1_end, x2_end, ...]
                    let ndim = x.len();
                    if pad_vals.len() == 2 * ndim {
                        let shape: Vec<DimExpr> = (0..ndim)
                            .map(|i| {
                                let p_begin = pad_vals[i].unwrap_or(0);
                                let p_end = pad_vals[ndim + i].unwrap_or(0);
                                if let Some(d) = x[i].as_concrete() {
                                    DimExpr::Concrete((d as i64 + p_begin + p_end) as u64)
                                } else {
                                    DimExpr::Dynamic
                                }
                            })
                            .collect();
                        vec![Shape::from(shape)]
                    } else {
                        inputs.first().cloned().into_iter().collect()
                    }
                } else {
                    inputs.first().cloned().into_iter().collect()
                }
            } else {
                vec![Shape::new()]
            }
        }

        // ── Phase 2: Utility ops ────────────────────────────────────────────

        // Additional reductions: same pattern as ReduceSum.
        AiOp::ReduceProd { axes, keepdims }
        | AiOp::ReduceL1 { axes, keepdims }
        | AiOp::ReduceL2 { axes, keepdims } => {
            if let Some(input) = inputs.first() {
                vec![reduce_shape(input, axes, *keepdims)]
            } else {
                vec![Shape::new()]
            }
        }

        // TopK: axis dim → K (from input[1] known value). Two outputs: values, indices.
        AiOp::TopK { axis, .. } => {
            if let Some(x) = inputs.first() {
                if x.is_empty() {
                    return vec![Shape::new(), Shape::new()];
                }
                let ax = normalize_axis(*axis, x.len());
                let mut shape = x.clone();
                // K comes from input[1] known_i64_values; if unavailable, dim is dynamic.
                if ax < shape.len() {
                    shape[ax] = DimExpr::Dynamic; // K is dynamic unless we have it
                }
                vec![shape.clone(), shape] // values, indices
            } else {
                vec![Shape::new(), Shape::new()]
            }
        }

        // ScatterND: output = data shape (input[0]).
        AiOp::ScatterND { .. } => inputs.first().cloned().into_iter().collect(),

        // NonZero: output = [rank, num_nonzero] (dynamic second dim).
        AiOp::NonZero => {
            if let Some(x) = inputs.first() {
                let rank = x.len() as u64;
                vec![Shape::from(vec![DimExpr::Concrete(rank), DimExpr::Dynamic])]
            } else {
                vec![Shape::new()]
            }
        }

        // OneHot: indices_shape + [depth] inserted at axis.
        AiOp::OneHot { axis } => {
            if let Some(indices) = inputs.first() {
                let out_rank = indices.len() + 1;
                let ax = normalize_axis(*axis, out_rank);
                let mut shape = indices.to_vec();
                shape.insert(ax, DimExpr::Dynamic); // depth is from input[1]
                vec![Shape::from(shape)]
            } else {
                vec![Shape::new()]
            }
        }

        // DepthToSpace: [N, C, H, W] → [N, C/bs², H*bs, W*bs]
        AiOp::DepthToSpace { blocksize, .. } => {
            if let Some(x) = inputs.first() {
                if x.len() == 4 {
                    let bs = *blocksize;
                    let shape = vec![
                        x[0].clone(), // N
                        match x[1].as_concrete() {
                            Some(c) => DimExpr::Concrete(c / (bs * bs)),
                            None => DimExpr::Dynamic,
                        },
                        match x[2].as_concrete() {
                            Some(h) => DimExpr::Concrete(h * bs),
                            None => DimExpr::Dynamic,
                        },
                        match x[3].as_concrete() {
                            Some(w) => DimExpr::Concrete(w * bs),
                            None => DimExpr::Dynamic,
                        },
                    ];
                    vec![Shape::from(shape)]
                } else {
                    inputs.first().cloned().into_iter().collect()
                }
            } else {
                vec![Shape::new()]
            }
        }

        // SpaceToDepth: [N, C, H, W] → [N, C*bs², H/bs, W/bs]
        AiOp::SpaceToDepth { blocksize } => {
            if let Some(x) = inputs.first() {
                if x.len() == 4 {
                    let bs = *blocksize;
                    let shape = vec![
                        x[0].clone(),
                        match x[1].as_concrete() {
                            Some(c) => DimExpr::Concrete(c * bs * bs),
                            None => DimExpr::Dynamic,
                        },
                        match x[2].as_concrete() {
                            Some(h) => DimExpr::Concrete(h / bs),
                            None => DimExpr::Dynamic,
                        },
                        match x[3].as_concrete() {
                            Some(w) => DimExpr::Concrete(w / bs),
                            None => DimExpr::Dynamic,
                        },
                    ];
                    vec![Shape::from(shape)]
                } else {
                    inputs.first().cloned().into_iter().collect()
                }
            } else {
                vec![Shape::new()]
            }
        }

        // Compress: dynamic on compressed axis (or flattened if axis=None).
        AiOp::Compress { axis } => {
            if let Some(x) = inputs.first() {
                if let Some(ax) = axis {
                    let a = normalize_axis(*ax, x.len());
                    let mut shape = x.clone();
                    if a < shape.len() {
                        shape[a] = DimExpr::Dynamic;
                    }
                    vec![shape]
                } else {
                    // No axis: flatten and return 1-D dynamic.
                    vec![Shape::from(vec![DimExpr::Dynamic])]
                }
            } else {
                vec![Shape::new()]
            }
        }

        // ── Phase 2: Scatter (already has ScatterND above) ──────────────────
        AiOp::Scatter { .. } => {
            // Output shape = data shape (input[0]).
            inputs.first().cloned().into_iter().collect()
        }

        // ArgMax/ArgMin: reduce axis to 1 (or remove if !keepdims).
        AiOp::ArgMax { axis, keepdims } | AiOp::ArgMin { axis, keepdims } => {
            if let Some(x) = inputs.first() {
                if x.is_empty() {
                    return vec![Shape::new()];
                }
                let ax = normalize_axis(*axis, x.len());
                let mut shape = Vec::new();
                for (i, d) in x.iter().enumerate() {
                    if i == ax {
                        if *keepdims {
                            shape.push(DimExpr::Concrete(1));
                        }
                    } else {
                        shape.push(d.clone());
                    }
                }
                vec![Shape::from(shape)]
            } else {
                vec![Shape::new()]
            }
        }

        // Split: divide axis dim into parts. Multiple outputs.
        AiOp::Split { axis, sizes } => {
            if let Some(x) = inputs.first() {
                if x.is_empty() {
                    return sizes.iter().map(|_| Shape::new()).collect();
                }
                let ax = normalize_axis(*axis, x.len());
                sizes
                    .iter()
                    .map(|&s| {
                        let mut shape = x.clone();
                        if ax < shape.len() {
                            shape[ax] = DimExpr::Concrete(s);
                        }
                        shape
                    })
                    .collect()
            } else {
                sizes.iter().map(|_| Shape::new()).collect()
            }
        }

        // Tile: multiply each dim by repeats.
        AiOp::Tile { repeats } => {
            if let Some(x) = inputs.first() {
                let shape: Vec<DimExpr> = x
                    .iter()
                    .enumerate()
                    .map(|(i, d)| {
                        let r = repeats.get(i).copied().unwrap_or(1);
                        match d.as_concrete() {
                            Some(v) => DimExpr::Concrete(v * r),
                            None => DimExpr::Dynamic,
                        }
                    })
                    .collect();
                vec![Shape::from(shape)]
            } else {
                vec![Shape::new()]
            }
        }

        // Gemm: [M, K] x [K, N] → [M, N] (with optional transposes).
        AiOp::Gemm {
            trans_a, trans_b, ..
        } => {
            if inputs.len() >= 2 && inputs[0].len() == 2 && inputs[1].len() == 2 {
                let a = &inputs[0];
                let b = &inputs[1];
                let m = if *trans_a { a[1].clone() } else { a[0].clone() };
                let n = if *trans_b { b[0].clone() } else { b[1].clone() };
                vec![Shape::from(vec![m, n])]
            } else {
                vec![Shape::new()]
            }
        }

        // GatherND: complex shape rule.
        AiOp::GatherND { batch_dims } => {
            if inputs.len() >= 2 && !inputs[0].is_empty() && !inputs[1].is_empty() {
                let indices = &inputs[1];
                let bd = *batch_dims as usize;
                // Output = batch_dims from data + indices shape except last dim.
                let mut shape: Vec<DimExpr> = inputs[0][..bd.min(inputs[0].len())].to_vec();
                if indices.len() > 1 {
                    shape.extend_from_slice(&indices[bd..indices.len() - 1]);
                }
                // If last dim of indices < data rank - batch_dims, append remaining data dims.
                if let Some(last) = indices.last().and_then(|d| d.as_concrete()) {
                    let data_remaining = inputs[0].len().saturating_sub(bd + last as usize);
                    for i in 0..data_remaining {
                        shape.push(inputs[0][bd + last as usize + i].clone());
                    }
                }
                vec![Shape::from(shape)]
            } else {
                vec![Shape::new()]
            }
        }

        // BatchNorm: inference mode → 1 output (input shape).
        // Training mode → 5 outputs: Y, mean, var, saved_mean, saved_var.
        // mean/var/saved_mean/saved_var have shape [C] (from input dim 1).
        AiOp::BatchNorm { training, .. } => {
            if let Some(x) = inputs.first() {
                let y_shape = x.clone();
                if *training && x.len() >= 2 {
                    let c_shape = Shape::from(vec![x[1].clone()]);
                    vec![
                        y_shape,
                        c_shape.clone(),
                        c_shape.clone(),
                        c_shape.clone(),
                        c_shape,
                    ]
                } else {
                    vec![y_shape]
                }
            } else {
                vec![Shape::new()]
            }
        }

        // ── Phase 4: Control flow ops ───────────────────────────────────────
        // If/Loop/Scan: we can't infer shapes without recursing into subgraphs.
        // Return empty for now — Phase 4 will add subgraph shape prop.
        AiOp::If { .. } | AiOp::Loop { .. } | AiOp::Scan { .. } => {
            vec![Shape::new()]
        }

        // Remaining custom ops: return empty (unknown shape).
        _ => vec![Shape::new()],
    }
}
