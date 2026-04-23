//! Compile-time constant evaluation pass.
//!
//! Evaluates any node whose inputs are ALL materialized constants (AiParam::Inline),
//! with proper N-D broadcasting support. This eliminates entire constant subgraphs
//! (causal masks, position embeddings, comparison matrices) that the runtime cannot
//! handle due to lack of N-D broadcast support.
//!
//! Runs after DataPropagation (which materializes shape-computation results) and
//! before ConstantFolding (which removes nodes whose outputs are params).

use super::const_eval_ops::*;
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

        for &nid in order.iter() {
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

            // Shape ops produce the shape of their input as an INT64 tensor.
            // The input data is not needed — only the shape from tensor_info.
            // Evaluate eagerly when the input shape is fully concrete.
            if let Some(bytes) = try_eval_shape(&node.op, &node.inputs, &graph.tensor_info) {
                let n_dims = bytes.len() / 8;
                let out_shape = shape_from_concrete(&[n_dims as u64]);
                let info = TensorInfo::new(DType::INT64, out_shape);
                graph
                    .params
                    .insert(out_tid, AiParam::inline(bytes, info.clone()));
                graph.tensor_info.insert(out_tid, info);
                materialized += 1;
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

/// Evaluate an `AiOp::Shape` node when the input has a fully-concrete shape.
/// Returns the shape values serialized as little-endian INT64 bytes, or None
/// if the op is not Shape or the input shape is not fully concrete.
fn try_eval_shape(
    op: &AiOp,
    inputs: &[crate::ir::TensorId],
    tensor_info: &HashMap<crate::ir::TensorId, TensorInfo>,
) -> Option<Vec<u8>> {
    let (start, end) = match op {
        AiOp::Shape { start, end } => (*start, *end),
        _ => return None,
    };
    let in_tid = *inputs.first()?;
    let ti = tensor_info.get(&in_tid)?;
    let shape = concrete_shape(&ti.shape)?;
    let rank = shape.len() as i64;
    let s = normalize_axis(start.unwrap_or(0), rank);
    let e = normalize_axis(end.unwrap_or(rank), rank).min(shape.len());
    if s > e {
        return None;
    }
    let bytes: Vec<u8> = shape[s..e]
        .iter()
        .flat_map(|&d| (d as i64).to_le_bytes())
        .collect();
    if bytes.is_empty() {
        return None;
    }
    Some(bytes)
}

/// Normalize a potentially-negative axis index to a non-negative usize.
fn normalize_axis(axis: i64, rank: i64) -> usize {
    if axis < 0 {
        (rank + axis).max(0) as usize
    } else {
        axis as usize
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
        AiOp::Div => eval_binary_f32(
            inputs,
            input_shapes,
            |a, b| {
                if b != 0.0 {
                    a / b
                } else {
                    0.0
                }
            },
        ),
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
        AiOp::Slice {
            axes,
            starts,
            ends,
            steps,
        } => eval_slice(inputs, input_shapes, axes, starts, ends, steps),
        AiOp::Concat { axis } => eval_concat(inputs, input_shapes, *axis),
        AiOp::Identity => eval_identity(inputs, input_shapes),

        _ => None,
    }
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
        let a_bytes: Vec<u8> = [0i64, 1, 2].iter().flat_map(|v| v.to_le_bytes()).collect();
        let b_bytes: Vec<u8> = [0i64, 1, 2].iter().flat_map(|v| v.to_le_bytes()).collect();

        let a_info = make_ti(DType::INT64, &[3, 1]);
        let b_info = make_ti(DType::INT64, &[1, 3]);

        let inputs = vec![(a_bytes.as_slice(), &a_info), (b_bytes.as_slice(), &b_info)];
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

        let inputs = vec![(a_bytes.as_slice(), &a_info), (b_bytes.as_slice(), &b_info)];
        let shapes = vec![vec![2], vec![2]];

        let (result, dtype, shape) =
            eval_logical(&inputs, &shapes, |a, b| a != 0 && b != 0).unwrap();
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
        let col_bytes: Vec<u8> = [0i64, 1, 2].iter().flat_map(|v| v.to_le_bytes()).collect();
        let row_bytes: Vec<u8> = [0i64, 1, 2].iter().flat_map(|v| v.to_le_bytes()).collect();

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
            vec![AiNode::new(0, AiOp::LessOrEqual, vec![10, 11], vec![12])],
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
