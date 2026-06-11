//! Pure helper functions for shape inference.
//!
//! These are stateless utilities used by `shape_inference` and `shape_prop`.

use crate::ir::dtype::DType;
use crate::ir::op::OpCategory;
use crate::ir::shape::DimExpr;
use crate::ir::{shape_from_concrete, AiOp, Shape, SymbolicShapeExt};

/// Compute pooling output dimension, with optional ceil_mode.
pub(crate) fn pool_output_dim(
    in_dim: u64,
    effective_kernel: u64,
    stride: u64,
    p_begin: u64,
    p_end: u64,
    ceil_mode: bool,
) -> u64 {
    let padded = in_dim + p_begin + p_end;
    if padded < effective_kernel {
        return 0;
    }
    let numerator = padded - effective_kernel;
    if ceil_mode {
        numerator.div_ceil(stride) + 1
    } else {
        numerator / stride + 1
    }
}

pub(crate) fn normalize_axis(axis: i64, ndim: usize) -> usize {
    if axis < 0 {
        (ndim as i64 + axis).max(0) as usize
    } else {
        axis as usize
    }
}

pub(crate) fn add_dims(a: &DimExpr, b: &DimExpr) -> DimExpr {
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
pub(crate) fn resolve_reshape_shape(
    vals: &[Option<i64>],
    data_shape: Option<&Shape>,
) -> Vec<DimExpr> {
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
    let data_symbolic: Vec<&DimExpr> = ds.iter().filter(|d| d.as_concrete().is_none()).collect();

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
                        shape[idx] =
                            DimExpr::Mul(Box::new(sym.clone()), Box::new(DimExpr::Concrete(ratio)));
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

pub(crate) fn normalize_slice_bound(val: i64, dim_size: i64) -> i64 {
    let v = if val < 0 { dim_size + val } else { val };
    v.clamp(0, dim_size)
}

pub(crate) fn broadcast_shape(a: &Shape, b: &Shape) -> Shape {
    a.as_slice().broadcast_shape(b.as_slice())
}

pub(crate) fn reduce_shape(input: &Shape, axes: &[i64], keepdims: bool) -> Shape {
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

/// Infer per-output dtypes for an op given its input dtypes.
///
/// UOR-native: returns `None` when an output dtype cannot be determined
/// from the inputs (no fabricated F32 fallback). Callers must accept
/// the input dtypes are real (i.e. drawn from `tensor_info`, not made
/// up). Output is `Some(Vec)` only when every output has a derivable
/// dtype.
pub(crate) fn infer_output_dtypes(
    op: &AiOp,
    inputs: &[DType],
    num_outputs: usize,
) -> Option<Vec<DType>> {
    match op.category() {
        OpCategory::UnaryElementwise
        | OpCategory::BinaryElementwise
        | OpCategory::ShapePreserving => {
            // Output dtype is the first input's dtype. Refuse to infer
            // when no inputs were supplied.
            let dt = *inputs.first()?;
            Some(vec![dt; num_outputs])
        }
        OpCategory::BinaryComparison => Some(vec![DType::BOOL; num_outputs]),
        OpCategory::Custom => match op {
            AiOp::Shape { .. } | AiOp::Range | AiOp::NonZero => {
                Some(vec![DType::INT64; num_outputs])
            }
            AiOp::Cast { to, .. } => Some(vec![*to; num_outputs]),
            // TopK: output[0]=values (input dtype), output[1]=indices (INT64).
            AiOp::TopK { .. } => {
                let dt = *inputs.first()?;
                Some(vec![dt, DType::INT64])
            }
            _ => {
                // Generic Custom op — output dtype = first input dtype.
                let dt = *inputs.first()?;
                Some(vec![dt; num_outputs])
            }
        },
    }
}
