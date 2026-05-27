//! Individual compile-time constant evaluation functions.

use crate::ir::{AiOp, DType, TensorInfo};

// ── N-D broadcast infrastructure ─────────────────────────────────────────────

/// Compute the broadcast output shape for two input shapes (numpy rules).
pub(crate) fn broadcast_shape(a: &[usize], b: &[usize]) -> Option<Vec<usize>> {
    let ndims = a.len().max(b.len());
    let mut result = vec![0usize; ndims];

    for i in 0..ndims {
        let da = if i < ndims - a.len() {
            1
        } else {
            a[i - (ndims - a.len())]
        };
        let db = if i < ndims - b.len() {
            1
        } else {
            b[i - (ndims - b.len())]
        };
        if da == db {
            result[i] = da;
        } else if da == 1 {
            result[i] = db;
        } else if db == 1 {
            result[i] = da;
        } else {
            return None; // incompatible
        }
    }
    Some(result)
}

/// Broadcast three shapes (for Where: cond, x, y).
pub(crate) fn broadcast_shape_3(a: &[usize], b: &[usize], c: &[usize]) -> Option<Vec<usize>> {
    let ab = broadcast_shape(a, b)?;
    broadcast_shape(&ab, c)
}

/// Map a flat output index to a flat input index, given the padded input shape
/// and output strides. Handles broadcasting by clamping dim=1 coords to 0.
#[inline]
pub(crate) fn broadcast_input_index(
    out_flat: usize,
    out_strides: &[usize],
    in_shape_padded: &[usize],
    in_strides: &[usize],
    ndims: usize,
) -> usize {
    let mut remaining = out_flat;
    let mut in_flat = 0usize;
    for d in 0..ndims {
        let coord = remaining / out_strides[d];
        remaining %= out_strides[d];
        let in_coord = if in_shape_padded[d] == 1 { 0 } else { coord };
        in_flat += in_coord * in_strides[d];
    }
    in_flat
}

/// Precompute strides for a shape (row-major).
pub(crate) fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// Pad a shape with leading 1s to match the target rank.
pub(crate) fn pad_shape(shape: &[usize], target_ndims: usize) -> Vec<usize> {
    let mut padded = vec![1usize; target_ndims.saturating_sub(shape.len())];
    padded.extend_from_slice(shape);
    padded
}

// ── Value extraction ─────────────────────────────────────────────────────────

/// Read f32 values from bytes, handling F32 and INT64 dtypes.
pub(crate) fn read_as_f64(data: &[u8], dtype: DType) -> Option<Vec<f64>> {
    match dtype {
        DType::F32 => {
            if !data.len().is_multiple_of(4) {
                return None;
            }
            Some(
                data.chunks_exact(4)
                    .map(|c| f32::from_le_bytes(c.try_into().unwrap()) as f64)
                    .collect(),
            )
        }
        DType::INT64 => {
            if !data.len().is_multiple_of(8) {
                return None;
            }
            Some(
                data.chunks_exact(8)
                    .map(|c| i64::from_le_bytes(c.try_into().unwrap()) as f64)
                    .collect(),
            )
        }
        DType::INT32 => {
            if !data.len().is_multiple_of(4) {
                return None;
            }
            Some(
                data.chunks_exact(4)
                    .map(|c| i32::from_le_bytes(c.try_into().unwrap()) as f64)
                    .collect(),
            )
        }
        DType::BOOL | DType::U8 => Some(data.iter().map(|&b| b as f64).collect()),
        _ => None,
    }
}

/// Read as i64 values.
pub(crate) fn read_as_i64(data: &[u8], dtype: DType) -> Option<Vec<i64>> {
    match dtype {
        DType::INT64 => {
            if !data.len().is_multiple_of(8) {
                return None;
            }
            Some(
                data.chunks_exact(8)
                    .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
                    .collect(),
            )
        }
        DType::INT32 => {
            if !data.len().is_multiple_of(4) {
                return None;
            }
            Some(
                data.chunks_exact(4)
                    .map(|c| i32::from_le_bytes(c.try_into().unwrap()) as i64)
                    .collect(),
            )
        }
        DType::F32 => {
            if !data.len().is_multiple_of(4) {
                return None;
            }
            Some(
                data.chunks_exact(4)
                    .map(|c| f32::from_le_bytes(c.try_into().unwrap()) as i64)
                    .collect(),
            )
        }
        DType::BOOL | DType::U8 => Some(data.iter().map(|&b| b as i64).collect()),
        _ => None,
    }
}

// ── Op evaluators ────────────────────────────────────────────────────────────

/// Safety limit: don't materialize tensors larger than this.
const MAX_OUTPUT_ELEMS: usize = 100_000_000;

pub(crate) fn eval_expand(
    inputs: &[(&[u8], &TensorInfo)],
    input_shapes: &[Vec<usize>],
) -> Option<(Vec<u8>, DType, Vec<usize>)> {
    if inputs.len() < 2 {
        return None;
    }
    let (data_bytes, data_info) = inputs[0];
    let (shape_bytes, shape_info) = inputs[1];
    let elem_size = data_info.logical_dtype.byte_size()?;
    if elem_size == 0 || data_bytes.is_empty() {
        return None;
    }

    // Read target shape from the shape tensor.
    let target_shape: Vec<usize> =
        if shape_info.logical_dtype == DType::INT64 && shape_bytes.len() % 8 == 0 {
            shape_bytes
                .chunks_exact(8)
                .map(|c| i64::from_le_bytes(c.try_into().unwrap()) as usize)
                .collect()
        } else {
            return None;
        };

    let target_elems: usize = target_shape.iter().product();
    if target_elems == 0 || target_elems > MAX_OUTPUT_ELEMS {
        return None;
    }

    let data_shape = &input_shapes[0];
    let data_elems: usize = data_shape.iter().product();

    // If element counts match, it's just a reshape — skip (handled by ConstantFolding).
    if target_elems == data_elems {
        return None;
    }

    let expanded = broadcast_expand_bytes(data_bytes, data_shape, &target_shape, elem_size)?;
    Some((expanded, data_info.logical_dtype, target_shape))
}

pub(crate) fn eval_binary_f32(
    inputs: &[(&[u8], &TensorInfo)],
    input_shapes: &[Vec<usize>],
    f: impl Fn(f32, f32) -> f32,
) -> Option<(Vec<u8>, DType, Vec<usize>)> {
    if inputs.len() < 2 {
        return None;
    }
    let a_vals = read_as_f64(inputs[0].0, inputs[0].1.logical_dtype)?;
    let b_vals = read_as_f64(inputs[1].0, inputs[1].1.logical_dtype)?;

    let out_shape = broadcast_shape(&input_shapes[0], &input_shapes[1])?;
    let out_elems: usize = out_shape.iter().product();
    if out_elems > MAX_OUTPUT_ELEMS {
        return None;
    }

    let ndims = out_shape.len();
    let out_strides = compute_strides(&out_shape);
    let a_padded = pad_shape(&input_shapes[0], ndims);
    let b_padded = pad_shape(&input_shapes[1], ndims);
    let a_strides = compute_strides(&a_padded);
    let b_strides = compute_strides(&b_padded);

    let mut result = Vec::with_capacity(out_elems * 4);
    for i in 0..out_elems {
        let ai = broadcast_input_index(i, &out_strides, &a_padded, &a_strides, ndims);
        let bi = broadcast_input_index(i, &out_strides, &b_padded, &b_strides, ndims);
        let val = f(a_vals[ai] as f32, b_vals[bi] as f32);
        result.extend_from_slice(&val.to_le_bytes());
    }
    Some((result, DType::F32, out_shape))
}

pub(crate) fn eval_comparison(
    inputs: &[(&[u8], &TensorInfo)],
    input_shapes: &[Vec<usize>],
    f: impl Fn(f64, f64) -> bool,
) -> Option<(Vec<u8>, DType, Vec<usize>)> {
    if inputs.len() < 2 {
        return None;
    }
    let a_vals = read_as_f64(inputs[0].0, inputs[0].1.logical_dtype)?;
    let b_vals = read_as_f64(inputs[1].0, inputs[1].1.logical_dtype)?;

    let out_shape = broadcast_shape(&input_shapes[0], &input_shapes[1])?;
    let out_elems: usize = out_shape.iter().product();
    if out_elems > MAX_OUTPUT_ELEMS {
        return None;
    }

    let ndims = out_shape.len();
    let out_strides = compute_strides(&out_shape);
    let a_padded = pad_shape(&input_shapes[0], ndims);
    let b_padded = pad_shape(&input_shapes[1], ndims);
    let a_strides = compute_strides(&a_padded);
    let b_strides = compute_strides(&b_padded);

    // Output as F32 (0.0 or 1.0). The hologram runtime works exclusively
    // with f32 buffers, so logical results must be f32.
    let mut result = Vec::with_capacity(out_elems * 4);
    for i in 0..out_elems {
        let ai = broadcast_input_index(i, &out_strides, &a_padded, &a_strides, ndims);
        let bi = broadcast_input_index(i, &out_strides, &b_padded, &b_strides, ndims);
        let val: f32 = if ai < a_vals.len() && bi < b_vals.len() && f(a_vals[ai], b_vals[bi]) {
            1.0
        } else {
            0.0
        };
        result.extend_from_slice(&val.to_le_bytes());
    }
    Some((result, DType::F32, out_shape))
}

pub(crate) fn eval_logical(
    inputs: &[(&[u8], &TensorInfo)],
    input_shapes: &[Vec<usize>],
    f: impl Fn(i64, i64) -> bool,
) -> Option<(Vec<u8>, DType, Vec<usize>)> {
    if inputs.len() < 2 {
        return None;
    }
    let a_vals = read_as_i64(inputs[0].0, inputs[0].1.logical_dtype)?;
    let b_vals = read_as_i64(inputs[1].0, inputs[1].1.logical_dtype)?;

    let out_shape = broadcast_shape(&input_shapes[0], &input_shapes[1])?;
    let out_elems: usize = out_shape.iter().product();
    if out_elems > MAX_OUTPUT_ELEMS {
        return None;
    }

    let ndims = out_shape.len();
    let out_strides = compute_strides(&out_shape);
    let a_padded = pad_shape(&input_shapes[0], ndims);
    let b_padded = pad_shape(&input_shapes[1], ndims);
    let a_strides = compute_strides(&a_padded);
    let b_strides = compute_strides(&b_padded);

    // Output as F32 (0.0 or 1.0). The hologram runtime works exclusively
    // with f32 buffers, so logical results must be f32.
    let mut result = Vec::with_capacity(out_elems * 4);
    for i in 0..out_elems {
        let ai = broadcast_input_index(i, &out_strides, &a_padded, &a_strides, ndims);
        let bi = broadcast_input_index(i, &out_strides, &b_padded, &b_strides, ndims);
        let val: f32 = if ai < a_vals.len() && bi < b_vals.len() && f(a_vals[ai], b_vals[bi]) {
            1.0
        } else {
            0.0
        };
        result.extend_from_slice(&val.to_le_bytes());
    }
    Some((result, DType::F32, out_shape))
}

pub(crate) fn eval_where(
    inputs: &[(&[u8], &TensorInfo)],
    input_shapes: &[Vec<usize>],
) -> Option<(Vec<u8>, DType, Vec<usize>)> {
    if inputs.len() < 3 {
        return None;
    }
    let cond_vals = read_as_i64(inputs[0].0, inputs[0].1.logical_dtype)?;
    let x_bytes = inputs[1].0;
    let y_bytes = inputs[2].0;
    let x_dtype = inputs[1].1.logical_dtype;
    let y_dtype = inputs[2].1.logical_dtype;

    // x and y should have the same dtype.
    if x_dtype != y_dtype {
        return None;
    }
    let elem_size = x_dtype.byte_size()?;
    if elem_size == 0 {
        return None;
    }

    let out_shape = broadcast_shape_3(&input_shapes[0], &input_shapes[1], &input_shapes[2])?;
    let out_elems: usize = out_shape.iter().product();
    if out_elems > MAX_OUTPUT_ELEMS {
        return None;
    }

    let ndims = out_shape.len();
    let out_strides = compute_strides(&out_shape);
    let c_padded = pad_shape(&input_shapes[0], ndims);
    let x_padded = pad_shape(&input_shapes[1], ndims);
    let y_padded = pad_shape(&input_shapes[2], ndims);
    let c_strides = compute_strides(&c_padded);
    let x_strides = compute_strides(&x_padded);
    let y_strides = compute_strides(&y_padded);

    let mut result = Vec::with_capacity(out_elems * elem_size);
    for i in 0..out_elems {
        let ci = broadcast_input_index(i, &out_strides, &c_padded, &c_strides, ndims);
        let xi = broadcast_input_index(i, &out_strides, &x_padded, &x_strides, ndims);
        let yi = broadcast_input_index(i, &out_strides, &y_padded, &y_strides, ndims);

        let src = if cond_vals[ci] != 0 {
            &x_bytes[xi * elem_size..(xi + 1) * elem_size]
        } else {
            &y_bytes[yi * elem_size..(yi + 1) * elem_size]
        };
        result.extend_from_slice(src);
    }
    Some((result, x_dtype, out_shape))
}

pub(crate) fn eval_cast(
    inputs: &[(&[u8], &TensorInfo)],
    to: DType,
) -> Option<(Vec<u8>, DType, Vec<usize>)> {
    if inputs.is_empty() {
        return None;
    }
    let (data, info) = inputs[0];
    let from = info.logical_dtype;
    let from_size = from.byte_size()?;
    if from_size == 0 || data.is_empty() {
        return None;
    }
    let n = data.len() / from_size;
    if n > MAX_OUTPUT_ELEMS {
        return None;
    }

    // Read source values as f64 (universal intermediate).
    let vals = read_as_f64(data, from)?;

    let to_size = to.byte_size()?;
    let mut result = Vec::with_capacity(n * to_size);
    for &v in &vals {
        match to {
            DType::F32 => result.extend_from_slice(&(v as f32).to_le_bytes()),
            DType::INT64 => result.extend_from_slice(&(v as i64).to_le_bytes()),
            DType::INT32 => result.extend_from_slice(&(v as i32).to_le_bytes()),
            DType::BOOL => result.push(if v != 0.0 { 1u8 } else { 0u8 }),
            DType::U8 => result.push(v as u8),
            _ => return None,
        }
    }

    // Shape is preserved.
    let shape: Vec<usize> = info
        .shape
        .iter()
        .filter_map(|d| d.as_concrete().map(|n| n as usize))
        .collect();

    Some((result, to, shape))
}

pub(crate) fn eval_not(inputs: &[(&[u8], &TensorInfo)]) -> Option<(Vec<u8>, DType, Vec<usize>)> {
    if inputs.is_empty() {
        return None;
    }
    let vals = read_as_i64(inputs[0].0, inputs[0].1.logical_dtype)?;
    let mut result = Vec::with_capacity(vals.len() * 4);
    for &v in &vals {
        let r: f32 = if v == 0 { 1.0 } else { 0.0 };
        result.extend_from_slice(&r.to_le_bytes());
    }

    let shape: Vec<usize> = inputs[0]
        .1
        .shape
        .iter()
        .filter_map(|d| d.as_concrete().map(|n| n as usize))
        .collect();

    Some((result, DType::F32, shape))
}

pub(crate) fn eval_unary_f32(
    inputs: &[(&[u8], &TensorInfo)],
    f: impl Fn(f32) -> f32,
) -> Option<(Vec<u8>, DType, Vec<usize>)> {
    if inputs.is_empty() {
        return None;
    }
    let vals = read_as_f64(inputs[0].0, inputs[0].1.logical_dtype)?;
    let mut result = Vec::with_capacity(vals.len() * 4);
    for &v in &vals {
        result.extend_from_slice(&f(v as f32).to_le_bytes());
    }

    let shape: Vec<usize> = inputs[0]
        .1
        .shape
        .iter()
        .filter_map(|d| d.as_concrete().map(|n| n as usize))
        .collect();

    Some((result, DType::F32, shape))
}

/// Evaluate ONNX Gather(data, indices, axis).
///
/// Output shape = data_shape[:axis] + indices_shape + data_shape[axis+1:]
/// For each index position, gathers the slice from the data tensor along the axis.
pub(crate) fn eval_gather(
    inputs: &[(&[u8], &TensorInfo)],
    input_shapes: &[Vec<usize>],
    axis: i64,
) -> Option<(Vec<u8>, DType, Vec<usize>)> {
    if inputs.len() < 2 {
        return None;
    }
    let (data_bytes, data_info) = inputs[0];
    let elem_size = data_info.logical_dtype.byte_size()?;
    if elem_size == 0 || data_bytes.is_empty() {
        return None;
    }

    let data_shape = &input_shapes[0];
    let indices_shape = &input_shapes[1];

    // Read indices as i64.
    let indices = read_as_i64(inputs[1].0, inputs[1].1.logical_dtype)?;

    let ndim = data_shape.len();
    if ndim == 0 {
        return None;
    }
    let ax = if axis < 0 {
        (ndim as i64 + axis).max(0) as usize
    } else {
        (axis as usize).min(ndim - 1)
    };

    let axis_size = data_shape[ax];

    // Output shape = data_shape[:ax] + indices_shape + data_shape[ax+1:]
    let mut out_shape = Vec::new();
    out_shape.extend_from_slice(&data_shape[..ax]);
    out_shape.extend_from_slice(indices_shape);
    out_shape.extend_from_slice(&data_shape[ax + 1..]);

    let out_elems: usize = out_shape.iter().product();
    if out_elems > MAX_OUTPUT_ELEMS {
        return None;
    }

    // Compute strides for the data tensor.
    let data_strides = compute_strides(data_shape);

    // For each output element, compute which data element to read.
    let pre_axis_dims: usize = data_shape[..ax].iter().product::<usize>().max(1);
    let post_axis_dims: usize = data_shape[ax + 1..].iter().product::<usize>().max(1);
    let indices_total: usize = indices_shape.iter().product::<usize>().max(1);

    let mut result = vec![0u8; out_elems * elem_size];

    for pre in 0..pre_axis_dims {
        for (idx_flat, &index_val) in indices.iter().enumerate() {
            // Handle negative indices.
            let index = if index_val < 0 {
                (axis_size as i64 + index_val).max(0) as usize
            } else {
                index_val as usize
            };
            if index >= axis_size {
                continue; // out of bounds, leave as zero
            }

            for post in 0..post_axis_dims {
                let data_flat = pre * data_strides[ax.min(data_strides.len() - 1)]
                    + index * post_axis_dims
                    + post;
                let out_flat =
                    pre * (indices_total * post_axis_dims) + idx_flat * post_axis_dims + post;

                let src = data_flat * elem_size;
                let dst = out_flat * elem_size;
                if src + elem_size <= data_bytes.len() && dst + elem_size <= result.len() {
                    result[dst..dst + elem_size].copy_from_slice(&data_bytes[src..src + elem_size]);
                }
            }
        }
    }

    Some((result, data_info.logical_dtype, out_shape))
}

// ── Structural op evaluators ──────────────────────────────────────────────────

/// Identity: pass through data unchanged.
pub(crate) fn eval_identity(
    inputs: &[(&[u8], &TensorInfo)],
    input_shapes: &[Vec<usize>],
) -> Option<(Vec<u8>, DType, Vec<usize>)> {
    if inputs.is_empty() {
        return None;
    }
    Some((
        inputs[0].0.to_vec(),
        inputs[0].1.logical_dtype,
        input_shapes[0].clone(),
    ))
}

/// Unsqueeze/Squeeze/Flatten: data bytes unchanged, shape changes.
pub(crate) fn eval_structural_reshape(
    inputs: &[(&[u8], &TensorInfo)],
    input_shapes: &[Vec<usize>],
    op: &AiOp,
) -> Option<(Vec<u8>, DType, Vec<usize>)> {
    if inputs.is_empty() {
        return None;
    }
    let (data, info) = inputs[0];
    let in_shape = &input_shapes[0];
    let in_elems: usize = in_shape.iter().product();

    let out_shape = match op {
        AiOp::Unsqueeze { axes } => {
            let mut shape = in_shape.to_vec();
            let mut sorted_axes: Vec<i64> = axes.clone();
            sorted_axes.sort();
            for &ax in &sorted_axes {
                let ndim = shape.len() as i64;
                let pos = if ax < 0 {
                    (ndim + 1 + ax).max(0) as usize
                } else {
                    ax as usize
                };
                shape.insert(pos.min(shape.len()), 1);
            }
            shape
        }
        AiOp::Squeeze { axes } => {
            if axes.is_empty() {
                // Squeeze all dims of size 1.
                in_shape.iter().copied().filter(|&d| d != 1).collect()
            } else {
                let ndim = in_shape.len() as i64;
                let remove: Vec<usize> = axes
                    .iter()
                    .map(|&a| {
                        if a < 0 {
                            (ndim + a).max(0) as usize
                        } else {
                            a as usize
                        }
                    })
                    .collect();
                in_shape
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| !remove.contains(i))
                    .map(|(_, &d)| d)
                    .collect()
            }
        }
        AiOp::Flatten { axis } => {
            let ndim = in_shape.len() as i64;
            let ax = if *axis < 0 {
                (ndim + axis).max(0) as usize
            } else {
                *axis as usize
            };
            let ax = ax.min(in_shape.len());
            let d0: usize = in_shape[..ax].iter().product::<usize>().max(1);
            let d1: usize = in_shape[ax..].iter().product::<usize>().max(1);
            vec![d0, d1]
        }
        _ => return None,
    };

    // Verify element count unchanged.
    let out_elems: usize = out_shape.iter().product();
    if out_elems != in_elems {
        return None;
    }

    Some((data.to_vec(), info.logical_dtype, out_shape))
}

/// Reshape: data bytes unchanged, shape from second input or tensor_info.
pub(crate) fn eval_reshape(
    inputs: &[(&[u8], &TensorInfo)],
    input_shapes: &[Vec<usize>],
) -> Option<(Vec<u8>, DType, Vec<usize>)> {
    if inputs.len() < 2 {
        return None;
    }
    let (data, info) = inputs[0];
    let in_elems: usize = input_shapes[0].iter().product();

    // Read target shape from the shape tensor (INT64).
    let shape_vals = read_as_i64(inputs[1].0, inputs[1].1.logical_dtype)?;

    // Resolve -1 (infer) and 0 (keep) dimensions.
    let mut out_shape: Vec<usize> = Vec::with_capacity(shape_vals.len());
    let mut infer_idx: Option<usize> = None;
    let mut known_product: usize = 1;

    for (i, &v) in shape_vals.iter().enumerate() {
        if v == -1 {
            if infer_idx.is_some() {
                return None; // Only one -1 allowed.
            }
            infer_idx = Some(i);
            out_shape.push(0); // Placeholder.
        } else if v == 0 {
            // Keep original dim.
            let orig = input_shapes[0].get(i).copied().unwrap_or(1);
            out_shape.push(orig);
            known_product *= orig;
        } else {
            out_shape.push(v as usize);
            known_product *= v as usize;
        }
    }

    if let Some(idx) = infer_idx {
        if known_product == 0 {
            return None;
        }
        out_shape[idx] = in_elems / known_product;
    }

    let out_elems: usize = out_shape.iter().product();
    if out_elems != in_elems {
        return None;
    }

    Some((data.to_vec(), info.logical_dtype, out_shape))
}

/// Transpose: physically permute elements.
pub(crate) fn eval_transpose(
    inputs: &[(&[u8], &TensorInfo)],
    input_shapes: &[Vec<usize>],
    perm: &[u32],
) -> Option<(Vec<u8>, DType, Vec<usize>)> {
    if inputs.is_empty() {
        return None;
    }
    let (data, info) = inputs[0];
    let in_shape = &input_shapes[0];
    let elem_size = info.logical_dtype.byte_size()?;
    if elem_size == 0 || data.is_empty() {
        return None;
    }

    let ndim = in_shape.len();
    if perm.len() != ndim {
        return None;
    }

    let in_elems: usize = in_shape.iter().product();
    if in_elems > MAX_OUTPUT_ELEMS {
        return None;
    }

    let out_shape: Vec<usize> = perm.iter().map(|&p| in_shape[p as usize]).collect();
    let in_strides = compute_strides(in_shape);
    let out_strides = compute_strides(&out_shape);

    let mut result = vec![0u8; in_elems * elem_size];
    for out_flat in 0..in_elems {
        // Decompose output flat index into output coords.
        let mut remaining = out_flat;
        let mut in_flat = 0usize;
        for d in 0..ndim {
            let out_coord = remaining / out_strides[d];
            remaining %= out_strides[d];
            // out_coord on dimension d of output = coord on dimension perm[d] of input.
            in_flat += out_coord * in_strides[perm[d] as usize];
        }

        let src = in_flat * elem_size;
        let dst = out_flat * elem_size;
        if src + elem_size <= data.len() && dst + elem_size <= result.len() {
            result[dst..dst + elem_size].copy_from_slice(&data[src..src + elem_size]);
        }
    }

    Some((result, info.logical_dtype, out_shape))
}

/// Slice: extract a sub-tensor along specified axes.
pub(crate) fn eval_slice(
    inputs: &[(&[u8], &TensorInfo)],
    input_shapes: &[Vec<usize>],
    axes: &[i64],
    starts: &[i64],
    ends: &[i64],
    steps: &[i64],
) -> Option<(Vec<u8>, DType, Vec<usize>)> {
    if inputs.is_empty() {
        return None;
    }
    let (data, info) = inputs[0];
    let in_shape = &input_shapes[0];
    let elem_size = info.logical_dtype.byte_size()?;
    if elem_size == 0 || data.is_empty() {
        return None;
    }

    let ndim = in_shape.len();

    // Build per-dimension slice specs: (start, end, step) for each dim.
    let mut dim_specs: Vec<(usize, usize, usize)> = in_shape.iter().map(|&d| (0, d, 1)).collect();

    for i in 0..axes.len() {
        let ax = if axes[i] < 0 {
            (ndim as i64 + axes[i]).max(0) as usize
        } else {
            axes[i] as usize
        };
        if ax >= ndim {
            continue;
        }
        let dim_size = in_shape[ax] as i64;
        let step = if i < steps.len() { steps[i] } else { 1 };
        if step == 0 {
            return None;
        }
        // Only handle positive steps for now.
        if step < 0 {
            return None;
        }
        let s = normalize_slice_bound(starts[i], dim_size);
        let e = normalize_slice_bound(ends[i], dim_size);
        if s >= e {
            return None;
        }
        dim_specs[ax] = (s as usize, e as usize, step as usize);
    }

    // Compute output shape.
    let out_shape: Vec<usize> = dim_specs
        .iter()
        .map(|&(s, e, step)| (e - s).div_ceil(step))
        .collect();
    let out_elems: usize = out_shape.iter().product();
    if out_elems > MAX_OUTPUT_ELEMS || out_elems == 0 {
        return None;
    }

    let in_strides = compute_strides(in_shape);
    let out_strides = compute_strides(&out_shape);

    let mut result = vec![0u8; out_elems * elem_size];
    for out_flat in 0..out_elems {
        // Decompose output index, map to input index.
        let mut remaining = out_flat;
        let mut in_flat = 0usize;
        for d in 0..ndim {
            let out_coord = remaining / out_strides[d];
            remaining %= out_strides[d];
            let (start, _, step) = dim_specs[d];
            let in_coord = start + out_coord * step;
            in_flat += in_coord * in_strides[d];
        }

        let src = in_flat * elem_size;
        let dst = out_flat * elem_size;
        if src + elem_size <= data.len() && dst + elem_size <= result.len() {
            result[dst..dst + elem_size].copy_from_slice(&data[src..src + elem_size]);
        }
    }

    Some((result, info.logical_dtype, out_shape))
}

pub(crate) fn normalize_slice_bound(val: i64, dim_size: i64) -> i64 {
    let v = if val < 0 { dim_size + val } else { val };
    v.clamp(0, dim_size)
}

/// Concat: concatenate tensors along an axis.
pub(crate) fn eval_concat(
    inputs: &[(&[u8], &TensorInfo)],
    input_shapes: &[Vec<usize>],
    axis: i64,
) -> Option<(Vec<u8>, DType, Vec<usize>)> {
    if inputs.len() < 2 {
        return None;
    }
    let dtype = inputs[0].1.logical_dtype;
    let elem_size = dtype.byte_size()?;
    if elem_size == 0 {
        return None;
    }

    let ndim = input_shapes[0].len();
    let ax = if axis < 0 {
        (ndim as i64 + axis).max(0) as usize
    } else {
        axis as usize
    };
    if ax >= ndim {
        return None;
    }

    // Verify all inputs have same shape except at concat axis, and same dtype.
    for i in 1..inputs.len() {
        if inputs[i].1.logical_dtype != dtype {
            return None;
        }
        if input_shapes[i].len() != ndim {
            return None;
        }
        for (d, (&dim_i, &dim_0)) in input_shapes[i]
            .iter()
            .zip(input_shapes[0].iter())
            .enumerate()
        {
            if d != ax && dim_i != dim_0 {
                return None;
            }
        }
    }

    // Output shape: same as input except concat axis is sum of all.
    let mut out_shape = input_shapes[0].clone();
    out_shape[ax] = input_shapes.iter().map(|s| s[ax]).sum();
    let out_elems: usize = out_shape.iter().product();
    if out_elems > MAX_OUTPUT_ELEMS {
        return None;
    }

    // Pre-axis size and post-axis size (these are the same for all inputs).
    let pre: usize = input_shapes[0][..ax].iter().product::<usize>().max(1);
    let post: usize = input_shapes[0][ax + 1..].iter().product::<usize>().max(1);

    let mut result = Vec::with_capacity(out_elems * elem_size);

    for p in 0..pre {
        for (input_idx, (data, _info)) in inputs.iter().enumerate() {
            let axis_size = input_shapes[input_idx][ax];
            let chunk_size = axis_size * post * elem_size;
            let offset = p * axis_size * post * elem_size;
            if offset + chunk_size <= data.len() {
                result.extend_from_slice(&data[offset..offset + chunk_size]);
            } else {
                // Pad with zeros if data is too short.
                result.extend(std::iter::repeat_n(0u8, chunk_size));
            }
        }
    }

    Some((result, dtype, out_shape))
}

/// Broadcast-expand raw bytes from data_shape to target_shape.
pub(crate) fn broadcast_expand_bytes(
    data: &[u8],
    data_shape: &[usize],
    target_shape: &[usize],
    elem_size: usize,
) -> Option<Vec<u8>> {
    let ndims = target_shape.len();
    let padded = pad_shape(data_shape, ndims);

    // Validate broadcast compatibility.
    for (d, t) in padded.iter().zip(target_shape.iter()) {
        if *d != *t && *d != 1 {
            return None;
        }
    }

    let target_elems: usize = target_shape.iter().product();
    let out_strides = compute_strides(target_shape);
    let in_strides = compute_strides(&padded);

    let mut output = vec![0u8; target_elems * elem_size];
    for i in 0..target_elems {
        let src_idx = broadcast_input_index(i, &out_strides, &padded, &in_strides, ndims);
        let src_start = src_idx * elem_size;
        let dst_start = i * elem_size;
        if src_start + elem_size <= data.len() {
            output[dst_start..dst_start + elem_size]
                .copy_from_slice(&data[src_start..src_start + elem_size]);
        }
    }
    Some(output)
}
