//! Forward shape propagation pass.
//!
//! Infers output shapes from input shapes for each node in topological order.
//! Collects `ShapeConstraint` entries into `graph.shape_constraints`.

use super::pipeline::Pass;
use crate::ir::dtype::DType;
use crate::ir::op::OpCategory;
use crate::ir::shape::DimExpr;
use crate::ir::{shape_from_concrete, AiGraph, AiOp, Shape};

/// Propagate shapes forward through the graph.
///
/// For each node, computes output shapes from input shapes. Unknown shapes
/// are left as-is (no error). Shape constraints are recorded for later
/// validation.
///
/// When `known_i64_values` are available on shape-input tensors (populated by
/// `DataPropagation`), this pass can resolve Reshape and Expand output shapes.
pub struct ShapePropagation;

/// Aggressive shape propagation that always overwrites existing shapes
/// with inferred shapes (when the inferred shape is non-empty).
///
/// Used in post-concretization repair where all dims are concrete and
/// the pipeline needs to re-derive shapes from scratch.
pub struct AggressiveShapePropagation;

impl Pass for ShapePropagation {
    fn name(&self) -> &str {
        "ShapePropagation"
    }

    fn run(&self, graph: AiGraph) -> anyhow::Result<AiGraph> {
        propagate_shapes(graph, false)
    }
}

impl Pass for AggressiveShapePropagation {
    fn name(&self) -> &str {
        "AggressiveShapePropagation"
    }

    fn run(&self, graph: AiGraph) -> anyhow::Result<AiGraph> {
        propagate_shapes(graph, true)
    }
}

fn propagate_shapes(mut graph: AiGraph, overwrite: bool) -> anyhow::Result<AiGraph> {
        let order = graph.topo_order();

        // Build node lookup.
        let node_idx: std::collections::HashMap<u32, usize> = graph
            .nodes
            .iter()
            .enumerate()
            .map(|(i, n)| (n.id, i))
            .collect();

        // Shape propagation: single pass in topological order.
        for &nid in &order {
            let idx = match node_idx.get(&nid) {
                Some(&i) => i,
                None => continue,
            };

            let input_shapes: Vec<Shape> = graph.nodes[idx]
                .inputs
                .iter()
                .map(|tid| {
                    graph
                        .tensor_info
                        .get(tid)
                        .map(|ti| ti.shape.clone())
                        .unwrap_or_default()
                })
                .collect();

            // For Reshape/Expand: try to get known_i64_values from the shape input.
            let shape_known_values: Option<Vec<Option<i64>>> = match &graph.nodes[idx].op {
                AiOp::Reshape { .. } | AiOp::Expand => {
                    graph.nodes[idx].inputs.get(1).and_then(|tid| {
                        graph
                            .tensor_info
                            .get(tid)
                            .and_then(|ti| ti.known_i64_values.clone())
                    })
                }
                _ => None,
            };

            let output_tids = graph.nodes[idx].outputs.clone();
            let op = graph.nodes[idx].op.clone();

            let inferred = infer_output_shapes(&op, &input_shapes, shape_known_values.as_deref());

            for (i, tid) in output_tids.iter().enumerate() {
                if let Some(shape) = inferred.get(i) {
                    if let Some(info) = graph.tensor_info.get_mut(tid) {
                        let should_write = overwrite
                            || info.shape.is_empty()
                            || info.shape.iter().all(|d| matches!(d, DimExpr::Dynamic));
                        if should_write && !shape.is_empty() {
                            info.shape = shape.clone();
                        }
                    }
                }
            }
        }

    // Dtype propagation: fixpoint loop until no changes.
    // Single pass is insufficient because intermediate tensors default to F32
    // and cascading updates (Shape→I64 → Gather→I64 → Concat→I64) may not
    // propagate through a single topological pass if some inputs haven't been
    // updated yet when their consumers are processed.
    loop {
        let mut changed = false;
        for &nid in &order {
            let idx = match node_idx.get(&nid) {
                Some(&i) => i,
                None => continue,
            };

            let input_dtypes: Vec<DType> = graph.nodes[idx]
                .inputs
                .iter()
                .map(|tid| {
                    graph
                        .tensor_info
                        .get(tid)
                        .map(|ti| ti.logical_dtype)
                        .unwrap_or(DType::F32)
                })
                .collect();

            let output_tids = graph.nodes[idx].outputs.clone();
            let op = graph.nodes[idx].op.clone();
            let inferred_dtype = infer_output_dtype(&op, &input_dtypes);

            for tid in &output_tids {
                if let Some(info) = graph.tensor_info.get_mut(tid) {
                    // Update if the current dtype differs from inferred AND
                    // the inferred dtype is more specific than F32 default.
                    if info.logical_dtype != inferred_dtype && inferred_dtype != DType::F32 {
                        info.logical_dtype = inferred_dtype;
                        info.storage_dtype = inferred_dtype;
                        changed = true;
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }

    Ok(graph)
}

/// Infer output shapes for a single op given input shapes.
///
/// `shape_known_values` is `Some` when the shape-input tensor (input[1] for
/// Reshape/Expand) has known constant values from `DataPropagation`.
fn infer_output_shapes(
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
            num_heads,
            head_dim,
            ..
        }
        | AiOp::GroupedQueryAttention {
            num_heads,
            head_dim,
            ..
        } => {
            if !inputs.is_empty() && inputs[0].len() >= 2 {
                let mut shape = inputs[0][..inputs[0].len() - 1].to_vec();
                shape.push(DimExpr::Concrete((*num_heads as u64) * (*head_dim as u64)));
                vec![Shape::from(shape)]
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

        // Expand: use known_i64_values from the shape input if available.
        AiOp::Expand => {
            if let Some(vals) = shape_known_values {
                let shape: Vec<DimExpr> = vals
                    .iter()
                    .map(|v| match v {
                        Some(n) if *n >= 0 => DimExpr::Concrete(*n as u64),
                        _ => DimExpr::Dynamic,
                    })
                    .collect();
                if shape.is_empty() {
                    inputs.first().cloned().into_iter().collect()
                } else {
                    vec![Shape::from(shape)]
                }
            } else {
                inputs.first().cloned().into_iter().collect()
            }
        }

        // Remaining custom ops: return empty (unknown shape).
        _ => vec![Shape::new()],
    }
}

fn normalize_axis(axis: i64, ndim: usize) -> usize {
    if axis < 0 {
        (ndim as i64 + axis).max(0) as usize
    } else {
        axis as usize
    }
}

fn add_dims(a: &DimExpr, b: &DimExpr) -> DimExpr {
    match (a.as_concrete(), b.as_concrete()) {
        (Some(av), Some(bv)) => DimExpr::Concrete(av + bv),
        _ => DimExpr::Dynamic,
    }
}

/// Normalize a slice start/end bound, clamping to [0, dim_size].
/// Resolve a Reshape target shape from known_i64_values and the data input shape.
///
/// Handles ONNX Reshape semantics:
///   - `Some(0)`: copy dim from data input at same position
///   - `Some(n>0)`: concrete dim value
///   - `Some(-1)`: infer from element count conservation (at most one allowed)
///   - `None`: unknown — try to inherit from data input, else mark for inference
///
/// Uses element count conservation to resolve -1 and unknown entries when
/// the data input shape provides enough information.
fn resolve_reshape_shape(vals: &[Option<i64>], data_shape: Option<&Shape>) -> Vec<DimExpr> {
    // First pass: resolve all deterministic entries.
    // Track which indices need inference (None or -1).
    let mut shape: Vec<DimExpr> = Vec::with_capacity(vals.len());
    let mut unknown_indices: Vec<usize> = Vec::new();

    for (i, v) in vals.iter().enumerate() {
        match v {
            Some(0) => {
                // ONNX Reshape: 0 means "copy from data input at same position".
                shape.push(
                    data_shape
                        .and_then(|ds| ds.get(i).cloned())
                        .unwrap_or(DimExpr::Dynamic),
                );
            }
            Some(n) if *n > 0 => {
                shape.push(DimExpr::Concrete(*n as u64));
            }
            Some(-1) | None => {
                // Placeholder — resolved via element count conservation below.
                shape.push(DimExpr::Concrete(0));
                unknown_indices.push(i);
            }
            Some(_) => {
                shape.push(DimExpr::Dynamic);
            }
        }
    }

    if unknown_indices.is_empty() {
        return shape;
    }

    // Element count conservation: data elements == output elements.
    // Separate concrete and symbolic dims in the data shape.
    let ds = match data_shape {
        Some(ds) if !ds.is_empty() => ds,
        _ => {
            // No data shape — fall back to position-based inheritance.
            for &idx in &unknown_indices {
                shape[idx] = data_shape
                    .and_then(|ds| ds.get(idx).cloned())
                    .unwrap_or(DimExpr::Dynamic);
            }
            return shape;
        }
    };

    let data_concrete: u64 = ds
        .iter()
        .filter_map(|d| d.as_concrete())
        .product::<u64>()
        .max(1);
    let data_symbolic: Vec<&DimExpr> = ds
        .iter()
        .filter(|d| d.as_concrete().is_none())
        .collect();

    // Product of already-resolved output dims (excluding unknowns).
    let out_concrete: u64 = shape
        .iter()
        .enumerate()
        .filter(|(i, _)| !unknown_indices.contains(i))
        .filter_map(|(_, d)| d.as_concrete())
        .product::<u64>()
        .max(1);
    let out_symbolic: Vec<(usize, &DimExpr)> = shape
        .iter()
        .enumerate()
        .filter(|(i, _)| !unknown_indices.contains(i))
        .filter(|(_, d)| d.as_concrete().is_none())
        .collect();

    // If there's exactly 1 unknown and all symbolic dims cancel
    // (same Var dims on both sides), we can solve for the unknown.
    if unknown_indices.len() == 1 {
        let idx = unknown_indices[0];

        // Check if symbolic dims cancel between input and output.
        // E.g., data=[batch, 32, seq, 64], known_out=[32, 64]
        // → unknowns have symbolic_data=[batch, seq], symbolic_out=[]
        // → unknown = data_concrete / out_concrete * (batch * seq) / 1
        // But since we can't compute the symbolic part, if all symbolics
        // appear on the data side only and out has none, the unknown carries them.
        if data_symbolic.is_empty() && out_symbolic.is_empty() {
            // Fully concrete: simple division.
            let resolved = data_concrete / out_concrete;
            shape[idx] = DimExpr::Concrete(resolved.max(1));
        } else if out_symbolic.is_empty() && data_concrete > 0 && out_concrete > 0 {
            // Output is all-concrete except the unknown. Data has symbolic dims.
            // The unknown absorbs both the concrete ratio AND the symbolic dims.
            let concrete_ratio = data_concrete / out_concrete;
            if data_symbolic.len() == 1 {
                // Single symbolic dim: unknown = sym * concrete_ratio.
                let sym = data_symbolic[0];
                if concrete_ratio == 1 {
                    shape[idx] = sym.clone();
                } else {
                    shape[idx] = DimExpr::Mul(
                        Box::new(sym.clone()),
                        Box::new(DimExpr::Concrete(concrete_ratio)),
                    );
                }
            } else if data_symbolic.is_empty() {
                // No symbolic dims (shouldn't reach here, but safety).
                shape[idx] = DimExpr::Concrete(concrete_ratio.max(1));
            } else {
                // Multiple symbolic dims in data — can't resolve cleanly.
                // Use concrete ratio as best guess.
                shape[idx] = DimExpr::Concrete(concrete_ratio.max(1));
            }
        } else {
            // Both sides have symbolic dims or can't resolve.
            // Fall back to position-based inheritance.
            shape[idx] = data_shape
                .and_then(|d| d.get(idx).cloned())
                .unwrap_or(DimExpr::Dynamic);
        }
    } else {
        // Multiple unknowns — split into -1 entries and None entries.
        // None entries inherit symbolic dims from the data input (in order).
        // Then the -1 entry (if any) is resolved via element count conservation.
        let neg1_positions: Vec<usize> = unknown_indices
            .iter()
            .copied()
            .filter(|&i| vals[i] == Some(-1))
            .collect();
        let none_positions: Vec<usize> = unknown_indices
            .iter()
            .copied()
            .filter(|&i| vals[i].is_none())
            .collect();

        // Collect symbolic dims from data input not accounted for by known output dims.
        let mut available_symbolic: Vec<DimExpr> = ds
            .iter()
            .filter(|d| d.as_concrete().is_none())
            .cloned()
            .collect();

        // Assign symbolic dims to None positions in order.
        for &idx in &none_positions {
            if let Some(sym) = available_symbolic.first().cloned() {
                available_symbolic.remove(0);
                shape[idx] = sym;
            } else {
                // No more symbolic dims — try position-based inheritance.
                shape[idx] = ds.get(idx).cloned().unwrap_or(DimExpr::Dynamic);
            }
        }

        // Now resolve -1 entries via element count conservation.
        if neg1_positions.len() == 1 {
            let idx = neg1_positions[0];
            let out_known: u64 = shape
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != idx)
                .filter_map(|(_, d)| d.as_concrete())
                .product::<u64>()
                .max(1);
            if data_concrete > 0 && out_known > 0 {
                let ratio = data_concrete / out_known;
                if available_symbolic.is_empty() {
                    shape[idx] = DimExpr::Concrete(ratio.max(1));
                } else if available_symbolic.len() == 1 {
                    let sym = &available_symbolic[0];
                    if ratio == 1 {
                        shape[idx] = sym.clone();
                    } else {
                        shape[idx] = DimExpr::Mul(
                            Box::new(sym.clone()),
                            Box::new(DimExpr::Concrete(ratio)),
                        );
                    }
                } else {
                    shape[idx] = DimExpr::Concrete(ratio.max(1));
                }
            } else {
                shape[idx] = DimExpr::Dynamic;
            }
        } else {
            // Multiple -1 entries (invalid ONNX, but handle gracefully).
            for &idx in &neg1_positions {
                shape[idx] = DimExpr::Dynamic;
            }
        }
    }

    shape
}

fn normalize_slice_bound(val: i64, dim_size: i64) -> i64 {
    let v = if val < 0 { dim_size + val } else { val };
    v.clamp(0, dim_size)
}

fn broadcast_shape(a: &Shape, b: &Shape) -> Shape {
    let len = a.len().max(b.len());
    let mut result = Shape::new();
    for i in 0..len {
        let ad = if i < a.len() {
            &a[a.len() - 1 - i]
        } else {
            &DimExpr::Concrete(1)
        };
        let bd = if i < b.len() {
            &b[b.len() - 1 - i]
        } else {
            &DimExpr::Concrete(1)
        };
        let dim = match (ad.as_concrete(), bd.as_concrete()) {
            (Some(1), _) => bd.clone(),
            (_, Some(1)) => ad.clone(),
            (Some(av), Some(bv)) if av == bv => ad.clone(),
            _ => DimExpr::Dynamic,
        };
        result.push(dim);
    }
    result.reverse();
    result
}

fn reduce_shape(input: &Shape, axes: &[i64], keepdims: bool) -> Shape {
    if axes.is_empty() {
        // Reduce all axes.
        if keepdims {
            Shape::from(vec![DimExpr::Concrete(1); input.len()])
        } else {
            shape_from_concrete(&[1])
        }
    } else {
        let ndim = input.len();
        let mut shape = Vec::new();
        for (i, dim) in input.iter().enumerate() {
            let is_reduced = axes.iter().any(|&ax| normalize_axis(ax, ndim) == i);
            if is_reduced {
                if keepdims {
                    shape.push(DimExpr::Concrete(1));
                }
            } else {
                shape.push(dim.clone());
            }
        }
        Shape::from(shape)
    }
}

/// Infer the output dtype for a single op given input dtypes.
fn infer_output_dtype(op: &AiOp, inputs: &[DType]) -> DType {
    match op.category() {
        OpCategory::UnaryElementwise
        | OpCategory::BinaryElementwise
        | OpCategory::ShapePreserving => inputs.first().copied().unwrap_or(DType::F32),
        OpCategory::BinaryComparison => DType::BOOL,
        OpCategory::Custom => match op {
            AiOp::Shape { .. } | AiOp::Range => DType::INT64,
            AiOp::Cast { to, .. } => *to,
            _ => inputs.first().copied().unwrap_or(DType::F32),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{shape_from_concrete, AiGraph, AiNode, AiOp, DType, TensorInfo};
    use std::collections::HashMap;

    #[test]
    fn propagate_matmul_shape() {
        let mut ti = HashMap::new();
        ti.insert(
            0u32,
            TensorInfo::new(DType::F32, shape_from_concrete(&[1, 4, 8])),
        );
        ti.insert(
            1u32,
            TensorInfo::new(DType::F32, shape_from_concrete(&[8, 16])),
        );
        // Output starts with unknown shape.
        ti.insert(2u32, TensorInfo::new(DType::F32, Shape::new()));

        let g = AiGraph {
            name: "test".into(),
            nodes: vec![AiNode::new(0, AiOp::MatMul, vec![0, 1], vec![2])],
            inputs: vec![0, 1],
            outputs: vec![2],
            input_names: vec![],
            output_names: vec![],
            params: HashMap::new(),
            tensor_info: ti,
            metadata: HashMap::new(),
            warnings: vec![],
            dim_vars: Default::default(),
            shape_constraints: Default::default(),
        };

        let pass = ShapePropagation;
        let g2 = pass.run(g).unwrap();
        let out_shape = &g2.tensor_info[&2].shape;
        // [1, 4, 8] x [8, 16] → [1, 4, 16]
        assert_eq!(out_shape.len(), 3);
        assert_eq!(out_shape[0].as_concrete(), Some(1));
        assert_eq!(out_shape[1].as_concrete(), Some(4));
        assert_eq!(out_shape[2].as_concrete(), Some(16));
    }

    #[test]
    fn propagate_elementwise_broadcast() {
        let mut ti = HashMap::new();
        ti.insert(
            0u32,
            TensorInfo::new(DType::F32, shape_from_concrete(&[4, 1])),
        );
        ti.insert(
            1u32,
            TensorInfo::new(DType::F32, shape_from_concrete(&[1, 8])),
        );
        ti.insert(2u32, TensorInfo::new(DType::F32, Shape::new()));

        let g = AiGraph {
            name: "test".into(),
            nodes: vec![AiNode::new(0, AiOp::Add, vec![0, 1], vec![2])],
            inputs: vec![0, 1],
            outputs: vec![2],
            input_names: vec![],
            output_names: vec![],
            params: HashMap::new(),
            tensor_info: ti,
            metadata: HashMap::new(),
            warnings: vec![],
            dim_vars: Default::default(),
            shape_constraints: Default::default(),
        };

        let pass = ShapePropagation;
        let g2 = pass.run(g).unwrap();
        let out_shape = &g2.tensor_info[&2].shape;
        // [4, 1] + [1, 8] → [4, 8]
        assert_eq!(out_shape.len(), 2);
        assert_eq!(out_shape[0].as_concrete(), Some(4));
        assert_eq!(out_shape[1].as_concrete(), Some(8));
    }

    /// Simulates the ONNX shape subgraph: Shape → Gather → Unsqueeze → Concat.
    /// This is the exact pattern that causes i64 shape corruption at runtime.
    #[test]
    fn propagate_shape_subgraph_chain() {
        let mut ti = HashMap::new();
        // input_ids: [batch=1, seq=2] I64
        ti.insert(
            0u32,
            TensorInfo::new(DType::INT64, shape_from_concrete(&[1, 2])),
        );
        // Shape op output: should be [2] I64 (rank of input)
        ti.insert(1u32, TensorInfo::new(DType::F32, Shape::new()));
        // Gather indices: scalar constant (value=0)
        ti.insert(
            2u32,
            TensorInfo::new(DType::INT64, Shape::new()), // scalar
        );
        // Gather output: scalar I64
        ti.insert(3u32, TensorInfo::new(DType::F32, Shape::new()));
        // Unsqueeze output: [1] I64
        ti.insert(4u32, TensorInfo::new(DType::F32, Shape::new()));

        let g = AiGraph {
            name: "test".into(),
            nodes: vec![
                AiNode::new(
                    0,
                    AiOp::Shape {
                        start: None,
                        end: None,
                    },
                    vec![0],
                    vec![1],
                ),
                AiNode::new(1, AiOp::Gather { axis: 0 }, vec![1, 2], vec![3]),
                AiNode::new(2, AiOp::Unsqueeze { axes: vec![0] }, vec![3], vec![4]),
            ],
            inputs: vec![0],
            outputs: vec![4],
            input_names: vec![],
            output_names: vec![],
            params: HashMap::new(),
            tensor_info: ti,
            metadata: HashMap::new(),
            warnings: vec![],
            dim_vars: Default::default(),
            shape_constraints: Default::default(),
        };

        let pass = ShapePropagation;
        let g2 = pass.run(g).unwrap();

        // Shape output: [2] (rank of [1, 2] input)
        let shape_out = &g2.tensor_info[&1].shape;
        assert_eq!(shape_out.len(), 1);
        assert_eq!(shape_out[0].as_concrete(), Some(2));
        assert_eq!(g2.tensor_info[&1].logical_dtype, DType::INT64);

        // Gather output: scalar (empty shape) — axis=0 dim removed, scalar indices add nothing
        let gather_out = &g2.tensor_info[&3].shape;
        assert_eq!(gather_out.len(), 0);
        assert_eq!(g2.tensor_info[&3].logical_dtype, DType::INT64);

        // Unsqueeze output: [1] — scalar + unsqueeze(axis=0)
        let unsqueeze_out = &g2.tensor_info[&4].shape;
        assert_eq!(unsqueeze_out.len(), 1);
        assert_eq!(unsqueeze_out[0].as_concrete(), Some(1));
        assert_eq!(g2.tensor_info[&4].logical_dtype, DType::INT64);
    }

    #[test]
    fn propagate_squeeze_shape() {
        let mut ti = HashMap::new();
        ti.insert(
            0u32,
            TensorInfo::new(DType::F32, shape_from_concrete(&[1, 4, 1, 8])),
        );
        ti.insert(1u32, TensorInfo::new(DType::F32, Shape::new()));

        let g = AiGraph {
            name: "test".into(),
            nodes: vec![AiNode::new(
                0,
                AiOp::Squeeze { axes: vec![0, 2] },
                vec![0],
                vec![1],
            )],
            inputs: vec![0],
            outputs: vec![1],
            input_names: vec![],
            output_names: vec![],
            params: HashMap::new(),
            tensor_info: ti,
            metadata: HashMap::new(),
            warnings: vec![],
            dim_vars: Default::default(),
            shape_constraints: Default::default(),
        };

        let pass = ShapePropagation;
        let g2 = pass.run(g).unwrap();
        let out_shape = &g2.tensor_info[&1].shape;
        // [1, 4, 1, 8] squeeze axes [0, 2] → [4, 8]
        assert_eq!(out_shape.len(), 2);
        assert_eq!(out_shape[0].as_concrete(), Some(4));
        assert_eq!(out_shape[1].as_concrete(), Some(8));
    }

    #[test]
    fn propagate_transpose_shape() {
        let mut ti = HashMap::new();
        ti.insert(
            0u32,
            TensorInfo::new(DType::F32, shape_from_concrete(&[2, 3, 4])),
        );
        ti.insert(1u32, TensorInfo::new(DType::F32, Shape::new()));

        let g = AiGraph {
            name: "test".into(),
            nodes: vec![AiNode::new(
                0,
                AiOp::Transpose {
                    perm: vec![2, 0, 1],
                },
                vec![0],
                vec![1],
            )],
            inputs: vec![0],
            outputs: vec![1],
            input_names: vec![],
            output_names: vec![],
            params: HashMap::new(),
            tensor_info: ti,
            metadata: HashMap::new(),
            warnings: vec![],
            dim_vars: Default::default(),
            shape_constraints: Default::default(),
        };

        let pass = ShapePropagation;
        let g2 = pass.run(g).unwrap();
        let out_shape = &g2.tensor_info[&1].shape;
        // [2, 3, 4] perm [2, 0, 1] → [4, 2, 3]
        assert_eq!(out_shape.len(), 3);
        assert_eq!(out_shape[0].as_concrete(), Some(4));
        assert_eq!(out_shape[1].as_concrete(), Some(2));
        assert_eq!(out_shape[2].as_concrete(), Some(3));
    }
}
