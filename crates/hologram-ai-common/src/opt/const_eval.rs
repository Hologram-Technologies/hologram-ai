//! Compile-time constant evaluation pass.
//!
//! Evaluates any node whose inputs are ALL materialized constants (AiParam::Inline),
//! with proper N-D broadcasting support. This eliminates entire constant subgraphs
//! (causal masks, position embeddings, comparison matrices) that the runtime cannot
//! handle due to lack of N-D broadcast support.
//!
//! Runs after DataPropagation (which materializes shape-computation results) and
//! before ConstantFolding (which removes nodes whose outputs are params).

use super::pipeline::Pass;
use crate::ir::{shape_from_concrete, AiGraph, AiOp, AiParam, DType, TensorInfo};
use std::collections::HashMap;

/// Evaluate constant subgraphs at compile time.
pub struct ConstantEvaluation;

impl Pass for ConstantEvaluation {
    fn name(&self) -> &str {
        "ConstantEvaluation"
    }

    fn run(&self, mut graph: AiGraph) -> anyhow::Result<AiGraph> {
        let order = graph.topo_order();
        let node_map: HashMap<u32, usize> = graph
            .nodes
            .iter()
            .enumerate()
            .map(|(i, n)| (n.id, i))
            .collect();

        let mut materialized = 0u32;

        for &nid in &order {
            let idx = match node_map.get(&nid) {
                Some(&i) => i,
                None => continue,
            };

            let node = &graph.nodes[idx];
            if node.outputs.is_empty() {
                continue;
            }
            let out_tid = node.outputs[0];

            // Skip if already materialized.
            if graph.params.contains_key(&out_tid) {
                continue;
            }

            // Check if ALL inputs are inline constants.
            let inputs: Vec<(&[u8], &TensorInfo)> = node
                .inputs
                .iter()
                .filter_map(|tid| {
                    graph.params.get(tid).and_then(|p| match p {
                        AiParam::Inline { data, info } => Some((data.as_slice(), info)),
                        _ => None,
                    })
                })
                .collect();

            if inputs.len() != node.inputs.len() {
                continue; // not all inputs are constants
            }

            // Get input shapes from tensor_info (prefer) or param info.
            let input_shapes: Vec<Vec<usize>> = node
                .inputs
                .iter()
                .zip(inputs.iter())
                .map(|(tid, (_data, param_info))| {
                    graph
                        .tensor_info
                        .get(tid)
                        .and_then(|ti| concrete_shape(&ti.shape))
                        .or_else(|| concrete_shape(&param_info.shape))
                        .unwrap_or_else(|| {
                            let elem_sz = param_info.logical_dtype.byte_size().unwrap_or(1);
                            if elem_sz > 0 {
                                vec![_data.len() / elem_sz]
                            } else {
                                vec![_data.len()]
                            }
                        })
                })
                .collect();

            // Try to evaluate.
            if let Some((result_bytes, result_dtype, result_shape)) =
                eval_node(&node.op, &inputs, &input_shapes)
            {
                // Skip empty results: a 0-element tensor means a dynamic dim
                // was substituted with 0 (e.g. seq_len sentinel). Materializing
                // it as an empty constant would fail validation.
                if result_shape.contains(&0) || result_bytes.is_empty() {
                    continue;
                }

                let byte_len = result_bytes.len();

                let shape = shape_from_concrete(
                    &result_shape.iter().map(|&d| d as u64).collect::<Vec<_>>(),
                );
                let info = TensorInfo::new(result_dtype, shape);
                graph
                    .params
                    .insert(out_tid, AiParam::inline(result_bytes, info.clone()));
                graph.tensor_info.insert(out_tid, info);

                tracing::trace!(nid, ?node.op, out_tid, byte_len, ?result_shape, "const-eval: materialized node");
                materialized += 1;
            }
        }

        if materialized > 0 {
            tracing::debug!(materialized, "const-eval: materialized nodes");
        }

        Ok(graph)
    }
}


/// Extract a fully-concrete shape from DimExpr slice.
fn concrete_shape(shape: &[crate::ir::Dim]) -> Option<Vec<usize>> {
    shape
        .iter()
        .map(|d| d.as_concrete().map(|n| n as usize))
        .collect()
}

/// Evaluate a node at compile time. Returns (bytes, dtype, shape) or None.
fn eval_node(
    op: &AiOp,
    inputs: &[(&[u8], &TensorInfo)],
    input_shapes: &[Vec<usize>],
) -> Option<(Vec<u8>, DType, Vec<usize>)> {
    match op {
        // Expand: broadcast data to target shape.
        AiOp::Expand => eval_expand(inputs, input_shapes),

        // Element-wise binary arithmetic with N-D broadcast.
        AiOp::Add => eval_binary_f32(inputs, input_shapes, |a, b| a + b),
        AiOp::Sub => eval_binary_f32(inputs, input_shapes, |a, b| a - b),
        AiOp::Mul => eval_binary_f32(inputs, input_shapes, |a, b| a * b),
        AiOp::Div => eval_binary_f32(inputs, input_shapes, |a, b| {
            if b != 0.0 { a / b } else { 0.0 }
        }),
        AiOp::Pow => eval_binary_f32(inputs, input_shapes, |a, b| a.powf(b)),

        // Comparisons with N-D broadcast (output: INT64, 0 or 1).
        AiOp::LessOrEqual => eval_comparison(inputs, input_shapes, |a, b| a <= b),
        AiOp::Less => eval_comparison(inputs, input_shapes, |a, b| a < b),
        AiOp::Greater => eval_comparison(inputs, input_shapes, |a, b| a > b),
        AiOp::GreaterOrEqual => eval_comparison(inputs, input_shapes, |a, b| a >= b),
        AiOp::Equal => eval_comparison(inputs, input_shapes, |a, b| (a - b).abs() < f64::EPSILON),

        // Logical ops with N-D broadcast (input/output: INT64 or BOOL, 0 or 1).
        AiOp::And => eval_logical(inputs, input_shapes, |a, b| a != 0 && b != 0),
        AiOp::Or => eval_logical(inputs, input_shapes, |a, b| a != 0 || b != 0),

        // Where(cond, x, y) with N-D broadcast.
        AiOp::Where => eval_where(inputs, input_shapes),

        // Cast to different dtype.
        AiOp::Cast { to } => eval_cast(inputs, *to),

        // Not (unary logical).
        AiOp::Not => eval_not(inputs),

        // Neg (unary arithmetic).
        AiOp::Neg => eval_unary_f32(inputs, |x| -x),

        // Abs, Sqrt, etc.
        // Gather along axis (ONNX semantics).
        AiOp::Gather { axis } => eval_gather(inputs, input_shapes, *axis),
        AiOp::GatherElements { axis } => eval_gather(inputs, input_shapes, *axis),

        AiOp::Abs => eval_unary_f32(inputs, |x| x.abs()),
        AiOp::Sqrt => eval_unary_f32(inputs, |x| x.sqrt()),
        AiOp::Ceil => eval_unary_f32(inputs, |x| x.ceil()),
        AiOp::Floor => eval_unary_f32(inputs, |x| x.floor()),
        AiOp::Exp => eval_unary_f32(inputs, |x| x.exp()),
        AiOp::Log => eval_unary_f32(inputs, |x| x.ln()),
        AiOp::Cos => eval_unary_f32(inputs, |x| x.cos()),
        AiOp::Sin => eval_unary_f32(inputs, |x| x.sin()),
        AiOp::Reciprocal => eval_unary_f32(inputs, |x| 1.0 / x),

        // ── Structural ops: copy bytes, change shape ──────────────────────
        // These let constants propagate through shape manipulation chains
        // (e.g., cos_cached → Unsqueeze → Expand → Slice all become constants).
        AiOp::Unsqueeze { .. } | AiOp::Squeeze { .. } | AiOp::Flatten { .. } => {
            eval_structural_reshape(inputs, input_shapes, op)
        }
        AiOp::Reshape { .. } => eval_reshape(inputs, input_shapes),
        AiOp::Transpose { perm } => eval_transpose(inputs, input_shapes, perm),
        AiOp::Slice { axes, starts, ends, steps } => {
            eval_slice(inputs, input_shapes, axes, starts, ends, steps)
        }
        AiOp::Concat { axis } => eval_concat(inputs, input_shapes, *axis),
        AiOp::Identity => eval_identity(inputs, input_shapes),

        _ => None,
    }
}

// ── N-D broadcast infrastructure ─────────────────────────────────────────────

/// Compute the broadcast output shape for two input shapes (numpy rules).
fn broadcast_shape(a: &[usize], b: &[usize]) -> Option<Vec<usize>> {
    let ndims = a.len().max(b.len());
    let mut result = vec![0usize; ndims];

    for i in 0..ndims {
        let da = if i < ndims - a.len() { 1 } else { a[i - (ndims - a.len())] };
        let db = if i < ndims - b.len() { 1 } else { b[i - (ndims - b.len())] };
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
fn broadcast_shape_3(a: &[usize], b: &[usize], c: &[usize]) -> Option<Vec<usize>> {
    let ab = broadcast_shape(a, b)?;
    broadcast_shape(&ab, c)
}

/// Map a flat output index to a flat input index, given the padded input shape
/// and output strides. Handles broadcasting by clamping dim=1 coords to 0.
#[inline]
fn broadcast_input_index(
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
fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// Pad a shape with leading 1s to match the target rank.
fn pad_shape(shape: &[usize], target_ndims: usize) -> Vec<usize> {
    let mut padded = vec![1usize; target_ndims.saturating_sub(shape.len())];
    padded.extend_from_slice(shape);
    padded
}

// ── Value extraction ─────────────────────────────────────────────────────────

/// Read f32 values from bytes, handling F32 and INT64 dtypes.
fn read_as_f64(data: &[u8], dtype: DType) -> Option<Vec<f64>> {
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
fn read_as_i64(data: &[u8], dtype: DType) -> Option<Vec<i64>> {
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

fn eval_expand(
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
    let target_shape: Vec<usize> = if shape_info.logical_dtype == DType::INT64 && shape_bytes.len() % 8 == 0 {
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

fn eval_binary_f32(
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

fn eval_comparison(
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
        let val: f32 = if f(a_vals[ai], b_vals[bi]) { 1.0 } else { 0.0 };
        result.extend_from_slice(&val.to_le_bytes());
    }
    Some((result, DType::F32, out_shape))
}

fn eval_logical(
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
        let val: f32 = if f(a_vals[ai], b_vals[bi]) { 1.0 } else { 0.0 };
        result.extend_from_slice(&val.to_le_bytes());
    }
    Some((result, DType::F32, out_shape))
}

fn eval_where(
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

fn eval_cast(
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

fn eval_not(inputs: &[(&[u8], &TensorInfo)]) -> Option<(Vec<u8>, DType, Vec<usize>)> {
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

fn eval_unary_f32(
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
fn eval_gather(
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
                let out_flat = pre * (indices_total * post_axis_dims)
                    + idx_flat * post_axis_dims
                    + post;

                let src = data_flat * elem_size;
                let dst = out_flat * elem_size;
                if src + elem_size <= data_bytes.len() && dst + elem_size <= result.len() {
                    result[dst..dst + elem_size]
                        .copy_from_slice(&data_bytes[src..src + elem_size]);
                }
            }
        }
    }

    Some((result, data_info.logical_dtype, out_shape))
}

// ── Structural op evaluators ──────────────────────────────────────────────────

/// Identity: pass through data unchanged.
fn eval_identity(
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
fn eval_structural_reshape(
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
                let pos = if ax < 0 { (ndim + 1 + ax).max(0) as usize } else { ax as usize };
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
                    .map(|&a| if a < 0 { (ndim + a).max(0) as usize } else { a as usize })
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
            let ax = if *axis < 0 { (ndim + axis).max(0) as usize } else { *axis as usize };
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
fn eval_reshape(
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
fn eval_transpose(
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
fn eval_slice(
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
    let mut dim_specs: Vec<(usize, usize, usize)> = in_shape
        .iter()
        .map(|&d| (0, d, 1))
        .collect();

    for i in 0..axes.len() {
        let ax = if axes[i] < 0 { (ndim as i64 + axes[i]).max(0) as usize } else { axes[i] as usize };
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

fn normalize_slice_bound(val: i64, dim_size: i64) -> i64 {
    let v = if val < 0 { dim_size + val } else { val };
    v.clamp(0, dim_size)
}

/// Concat: concatenate tensors along an axis.
fn eval_concat(
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
    let ax = if axis < 0 { (ndim as i64 + axis).max(0) as usize } else { axis as usize };
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
        for (d, (&dim_i, &dim_0)) in input_shapes[i].iter().zip(input_shapes[0].iter()).enumerate() {
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
fn broadcast_expand_bytes(
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{shape_from_concrete, AiNode, TensorId};

    fn make_ti(dtype: DType, shape: &[u64]) -> TensorInfo {
        TensorInfo::new(dtype, shape_from_concrete(shape))
    }

    fn make_graph_with_params(
        nodes: Vec<AiNode>,
        params: HashMap<TensorId, AiParam>,
        tensor_info: HashMap<TensorId, TensorInfo>,
        outputs: Vec<TensorId>,
    ) -> AiGraph {
        AiGraph {
            name: "test".into(),
            nodes,
            inputs: vec![],
            outputs,
            input_names: vec![],
            output_names: vec![],
            params,
            tensor_info,
            metadata: HashMap::new(),
            warnings: vec![],
            dim_vars: Default::default(),
            shape_constraints: Default::default(),
            subgraphs: HashMap::new(),
            tensor_names: HashMap::new(),
            topo_cache: Default::default(),
        }
    }

    #[test]
    fn test_broadcast_shape() {
        assert_eq!(broadcast_shape(&[3, 1], &[1, 4]), Some(vec![3, 4]));
        assert_eq!(broadcast_shape(&[2048], &[2048]), Some(vec![2048]));
        assert_eq!(
            broadcast_shape(&[1, 1, 2048, 1], &[1, 1, 1, 2048]),
            Some(vec![1, 1, 2048, 2048])
        );
        assert_eq!(broadcast_shape(&[3], &[4]), None); // incompatible
    }

    #[test]
    fn test_eval_less_or_equal_broadcast() {
        // a = [0, 1, 2] shape [3, 1]
        // b = [0, 1, 2] shape [1, 3]
        // result[i][j] = (a[i] <= b[j]) → lower triangular
        let a_bytes: Vec<u8> = [0i64, 1, 2]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let b_bytes: Vec<u8> = [0i64, 1, 2]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();

        let a_info = make_ti(DType::INT64, &[3, 1]);
        let b_info = make_ti(DType::INT64, &[1, 3]);

        let inputs = vec![
            (a_bytes.as_slice(), &a_info),
            (b_bytes.as_slice(), &b_info),
        ];
        let shapes = vec![vec![3, 1], vec![1, 3]];

        let (result, dtype, shape) = eval_comparison(&inputs, &shapes, |a, b| a <= b).unwrap();
        assert_eq!(dtype, DType::F32);
        assert_eq!(shape, vec![3, 3]);

        // Read result as f32.
        let vals: Vec<f32> = result
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        // [0<=0, 0<=1, 0<=2, 1<=0, 1<=1, 1<=2, 2<=0, 2<=1, 2<=2]
        assert_eq!(vals, vec![1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_eval_and_broadcast() {
        // a = [1, 0] shape [2]
        // b = [1, 1] shape [2]
        let a_bytes: Vec<u8> = [1i64, 0].iter().flat_map(|v| v.to_le_bytes()).collect();
        let b_bytes: Vec<u8> = [1i64, 1].iter().flat_map(|v| v.to_le_bytes()).collect();

        let a_info = make_ti(DType::INT64, &[2]);
        let b_info = make_ti(DType::INT64, &[2]);

        let inputs = vec![
            (a_bytes.as_slice(), &a_info),
            (b_bytes.as_slice(), &b_info),
        ];
        let shapes = vec![vec![2], vec![2]];

        let (result, dtype, shape) = eval_logical(&inputs, &shapes, |a, b| a != 0 && b != 0).unwrap();
        assert_eq!(dtype, DType::F32);
        assert_eq!(shape, vec![2]);
        let vals: Vec<f32> = result
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(vals, vec![1.0, 0.0]);
    }

    #[test]
    fn test_const_eval_pass_materializes_expand() {
        // Expand([0], shape=[1,1,1,4]) → [0,0,0,0]
        let data_bytes: Vec<u8> = 0i64.to_le_bytes().to_vec();
        let shape_bytes: Vec<u8> = [1i64, 1, 1, 4]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();

        let mut params = HashMap::new();
        params.insert(
            10u32,
            AiParam::inline(data_bytes, make_ti(DType::INT64, &[1])),
        );
        params.insert(
            11u32,
            AiParam::inline(shape_bytes, make_ti(DType::INT64, &[4])),
        );

        let mut ti = HashMap::new();
        ti.insert(10u32, make_ti(DType::INT64, &[1]));
        ti.insert(11u32, make_ti(DType::INT64, &[4]));
        ti.insert(12u32, make_ti(DType::INT64, &[1, 1, 1, 4]));

        let g = make_graph_with_params(
            vec![AiNode::new(0, AiOp::Expand, vec![10, 11], vec![12])],
            params,
            ti,
            vec![12],
        );

        let pass = ConstantEvaluation;
        let g2 = pass.run(g).unwrap();

        // Output should be materialized as a param.
        assert!(g2.params.contains_key(&12));
        let param = &g2.params[&12];
        let bytes = match param {
            AiParam::Inline { data, .. } => data,
            _ => panic!("expected inline"),
        };
        // 4 i64 zeros = 32 bytes.
        assert_eq!(bytes.len(), 32);
        let vals: Vec<i64> = bytes
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(vals, vec![0, 0, 0, 0]);
    }

    #[test]
    fn test_const_eval_causal_mask_pattern() {
        // Simulates causal mask: LessOrEqual(range_col[1,1,3,1], range_row[1,1,1,3])
        // → 3x3 lower-triangular mask
        let col_bytes: Vec<u8> = [0i64, 1, 2]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let row_bytes: Vec<u8> = [0i64, 1, 2]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();

        let mut params = HashMap::new();
        params.insert(
            10u32,
            AiParam::inline(col_bytes, make_ti(DType::INT64, &[1, 1, 3, 1])),
        );
        params.insert(
            11u32,
            AiParam::inline(row_bytes, make_ti(DType::INT64, &[1, 1, 1, 3])),
        );

        let mut ti = HashMap::new();
        ti.insert(10u32, make_ti(DType::INT64, &[1, 1, 3, 1]));
        ti.insert(11u32, make_ti(DType::INT64, &[1, 1, 1, 3]));
        ti.insert(12u32, make_ti(DType::INT64, &[1, 1, 3, 3]));

        let g = make_graph_with_params(
            vec![AiNode::new(
                0,
                AiOp::LessOrEqual,
                vec![10, 11],
                vec![12],
            )],
            params,
            ti,
            vec![12],
        );

        let pass = ConstantEvaluation;
        let g2 = pass.run(g).unwrap();

        assert!(g2.params.contains_key(&12));
        let param = &g2.params[&12];
        let bytes = match param {
            AiParam::Inline { data, .. } => data,
            _ => panic!("expected inline"),
        };
        // 9 f32 values.
        let vals: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        // col <= row: [0<=0, 0<=1, 0<=2, 1<=0, 1<=1, 1<=2, 2<=0, 2<=1, 2<=2]
        assert_eq!(vals, vec![1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]);
    }
}
