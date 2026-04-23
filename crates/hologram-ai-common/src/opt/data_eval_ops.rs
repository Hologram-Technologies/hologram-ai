//! Individual data-propagation evaluators for shape-computation ops.

use crate::ir::dtype::DType;
use crate::ir::param::AiParam;
use crate::ir::shape::DimExpr;
use crate::ir::AiOp;

/// A partially-known value: `Some(n)` = concrete, `None` = dynamic.
pub(crate) type KnownValues = Vec<Option<i64>>;

/// Evaluate custom ops that need per-variant logic for value propagation.
pub(crate) fn eval_custom_op(
    op: &AiOp,
    inputs: &[Option<&KnownValues>],
    input_shapes: &[Option<&[DimExpr]>],
) -> Option<KnownValues> {
    match op {
        // Shape: output = dimension values of the input tensor's shape.
        AiOp::Shape { start, end } => {
            let shape = input_shapes.first().copied().flatten()?;
            if shape.is_empty() {
                return None;
            }
            let rank = shape.len() as i64;
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
            if s > e || s >= shape.len() {
                return None;
            }
            Some(
                shape[s..e]
                    .iter()
                    .map(|dim| match dim {
                        // Concrete(0) is a 0-sentinel for dynamic dims (seq_len etc.) —
                        // treat as unknown so the shape subgraph stays live at runtime.
                        DimExpr::Concrete(0) => None,
                        DimExpr::Concrete(n) => Some(*n as i64),
                        _ => None,
                    })
                    .collect(),
            )
        }

        // Gather(axis=0): index into the data values.
        AiOp::Gather { axis } if *axis == 0 => {
            let data = inputs.first().copied().flatten()?;
            let indices = inputs.get(1).copied().flatten()?;
            if indices.len() == 1 {
                let idx = (*indices.first()?)? as usize;
                if idx < data.len() {
                    Some(vec![data[idx]])
                } else {
                    None
                }
            } else if indices.is_empty() {
                None
            } else {
                let result: Vec<Option<i64>> = indices
                    .iter()
                    .map(|idx_opt| {
                        let idx = (*idx_opt)? as usize;
                        if idx < data.len() {
                            data[idx]
                        } else {
                            None
                        }
                    })
                    .collect();
                Some(result)
            }
        }

        // Unsqueeze/Squeeze: values pass through — only shape metadata changes.
        AiOp::Unsqueeze { .. } | AiOp::Squeeze { .. } => inputs.first().copied().flatten().cloned(),

        // Concat: concatenate value arrays (1-D shape tensors).
        AiOp::Concat { .. } => {
            let mut result = Vec::new();
            for inp in inputs {
                let vals = (*inp)?;
                result.extend_from_slice(vals);
            }
            Some(result)
        }

        // Cast: pass through for integer-to-integer casts.
        AiOp::Cast {
            to: DType::INT64 | DType::INT32,
        } => inputs.first().copied().flatten().cloned(),

        // Constant: extract values from the inline AiParam.
        AiOp::Constant { value } => {
            extract_i64_param(value).map(|vals| vals.into_iter().map(Some).collect())
        }

        // Slice: extract subrange of values (common in attention shape subgraphs).
        //
        // Two forms:
        // 1. Attribute-based: axes/starts/ends/steps in the AiOp struct fields
        // 2. ONNX opset 10+: struct fields are empty, values come from input
        //    tensors: (data, starts, ends, [axes], [steps])
        AiOp::Slice {
            axes,
            starts,
            ends,
            steps,
        } => {
            let data = inputs.first().copied().flatten()?;

            // Resolve the actual slice parameters.
            let (r_starts, r_ends, r_axes, r_steps) = if !starts.is_empty() {
                // Attribute-based form.
                (starts.clone(), ends.clone(), axes.clone(), steps.clone())
            } else {
                // ONNX opset 10+: read from input tensors.
                let inp_starts: Vec<i64> = inputs
                    .get(1)
                    .copied()
                    .flatten()?
                    .iter()
                    .map(|v| (*v).unwrap_or(0))
                    .collect();
                let inp_ends: Vec<i64> = inputs
                    .get(2)
                    .copied()
                    .flatten()?
                    .iter()
                    .map(|v| (*v).unwrap_or(i64::MAX))
                    .collect();
                let inp_axes: Vec<i64> = inputs
                    .get(3)
                    .and_then(|v| *v)
                    .map(|v| v.iter().map(|x| (*x).unwrap_or(0)).collect())
                    .unwrap_or_else(|| (0..inp_starts.len() as i64).collect());
                let inp_steps: Vec<i64> = inputs
                    .get(4)
                    .and_then(|v| *v)
                    .map(|v| v.iter().map(|x| (*x).unwrap_or(1)).collect())
                    .unwrap_or_else(|| vec![1; inp_starts.len()]);
                (inp_starts, inp_ends, inp_axes, inp_steps)
            };

            // For 1-D shape tensors with a single axis=0 slice, apply directly.
            if r_axes.len() == 1 && r_axes[0] == 0 {
                let len = data.len() as i64;
                let s = normalize_slice_bound(r_starts[0], len);
                let e = normalize_slice_bound(r_ends[0], len);
                let step = r_steps.first().copied().unwrap_or(1).max(1) as usize;
                if s < e && (s as usize) <= data.len() {
                    let end_idx = (e as usize).min(data.len());
                    let result: Vec<Option<i64>> = data[s as usize..end_idx]
                        .iter()
                        .step_by(step)
                        .cloned()
                        .collect();
                    Some(result)
                } else {
                    None
                }
            } else {
                None
            }
        }

        // Range: generate [start, limit) with step.
        AiOp::Range => {
            let start = inputs
                .first()
                .copied()
                .flatten()?
                .first()
                .copied()
                .flatten()?;
            let limit = inputs
                .get(1)
                .copied()
                .flatten()?
                .first()
                .copied()
                .flatten()?;
            let delta = inputs
                .get(2)
                .copied()
                .flatten()?
                .first()
                .copied()
                .flatten()?;
            if delta == 0 {
                return None;
            }
            let mut vals = Vec::new();
            let mut v = start;
            while (delta > 0 && v < limit) || (delta < 0 && v > limit) {
                vals.push(Some(v));
                v += delta;
                // Safety limit to avoid runaway loops.
                if vals.len() > 10_000 {
                    return None;
                }
            }
            Some(vals)
        }

        // ConstantOfShape: when the shape input has known values, we can
        // materialize the output as a param. Return empty known_values to
        // signal the output is determined (ConstantFolding removes the node).
        AiOp::ConstantOfShape { .. } => {
            let _shape_known = inputs.first().copied().flatten()?;
            Some(vec![])
        }

        // Trilu: if the input is fully known, the output is fully known.
        // Return empty to signal determined (materialized in the pass body).
        AiOp::Trilu { .. } => {
            // We only signal "determined" here; actual materialization happens
            // in the pass body alongside ConstantOfShape.
            Some(vec![])
        }

        _ => None,
    }
}

/// Normalize a slice start/end bound, clamping to [0, dim_size].
pub(crate) fn normalize_slice_bound(val: i64, dim_size: i64) -> i64 {
    let v = if val < 0 { dim_size + val } else { val };
    v.clamp(0, dim_size)
}

/// Evaluate a binary elementwise op on known i64 values with broadcasting.
pub(crate) fn eval_binary(
    inputs: &[Option<&KnownValues>],
    f: impl Fn(i64, i64) -> Option<i64>,
) -> Option<KnownValues> {
    let a = inputs.first().copied().flatten()?;
    let b = inputs.get(1).copied().flatten()?;

    // Scalar broadcasting.
    if a.len() == 1 && b.len() > 1 {
        return Some(
            b.iter()
                .map(|bv| {
                    let av = (*a.first()?)?;
                    let bv = (*bv)?;
                    f(av, bv)
                })
                .collect(),
        );
    }
    if b.len() == 1 && a.len() > 1 {
        return Some(
            a.iter()
                .map(|av| {
                    let av = (*av)?;
                    let bv = (*b.first()?)?;
                    f(av, bv)
                })
                .collect(),
        );
    }

    // Elementwise (same length).
    if a.len() == b.len() {
        return Some(
            a.iter()
                .zip(b.iter())
                .map(|(av, bv)| {
                    let av = (*av)?;
                    let bv = (*bv)?;
                    f(av, bv)
                })
                .collect(),
        );
    }

    None
}

/// Extract i64 values from an inline parameter (supports INT64, INT32, INT8, F32).
///
/// F32 support is needed for ONNX models that use float-typed scalars as
/// Range inputs (e.g., `Range(0.0, 2048.0, 1.0)` for causal mask generation).
pub(crate) fn extract_i64_param(param: &AiParam) -> Option<Vec<i64>> {
    let (data, info) = match param {
        AiParam::Inline { data, info } => (data.as_slice(), info),
        _ => return None,
    };
    match info.logical_dtype {
        DType::INT64 => {
            if data.len() % 8 != 0 {
                return None;
            }
            Some(
                data.chunks_exact(8)
                    .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
                    .collect(),
            )
        }
        DType::INT32 => {
            if data.len() % 4 != 0 {
                return None;
            }
            Some(
                data.chunks_exact(4)
                    .map(|c| i32::from_le_bytes(c.try_into().unwrap()) as i64)
                    .collect(),
            )
        }
        DType::INT8 => Some(data.iter().map(|&b| b as i8 as i64).collect()),
        DType::F32 => {
            if data.len() % 4 != 0 {
                return None;
            }
            Some(
                data.chunks_exact(4)
                    .map(|c| f32::from_le_bytes(c.try_into().unwrap()) as i64)
                    .collect(),
            )
        }
        _ => None,
    }
}
