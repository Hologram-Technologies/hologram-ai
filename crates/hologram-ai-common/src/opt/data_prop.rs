//! Data propagation pass — evaluates shape-computation subgraphs at compile time.
//!
//! ONNX models compute Reshape target shapes at runtime via subgraphs like
//! `Shape → Gather → Unsqueeze → Concat`. This pass evaluates those ops when
//! their inputs have known constant values, populating `TensorInfo::known_i64_values`
//! so that the lowering pass can bake correct shapes into the compiled graph.

use super::pipeline::Pass;
use crate::ir::dtype::DType;
use crate::ir::graph::TensorInfo;
use crate::ir::node::TensorId;
use crate::ir::op::OpCategory;
use crate::ir::param::AiParam;
use crate::ir::shape::DimExpr;
use crate::ir::{AiGraph, AiOp};
use std::collections::HashMap;

/// Propagate known constant values through shape-computation subgraphs.
pub struct DataPropagation;

/// A partially-known value: `Some(n)` = concrete, `None` = dynamic.
type KnownValues = Vec<Option<i64>>;

impl Pass for DataPropagation {
    fn name(&self) -> &str {
        "DataPropagation"
    }

    fn run(&self, mut graph: AiGraph) -> anyhow::Result<AiGraph> {
        let order = graph.topo_order();
        let node_idx: HashMap<u32, usize> = graph
            .nodes
            .iter()
            .enumerate()
            .map(|(i, n)| (n.id, i))
            .collect();

        // Seed known values from constant params (AiParam::Inline with integer dtype).
        let mut known: HashMap<TensorId, KnownValues> = HashMap::new();
        for (&tid, param) in &graph.params {
            if let Some(vals) = extract_i64_param(param) {
                tracing::trace!(tid, ?vals, "DataProp: seeded param");
                known.insert(tid, vals.into_iter().map(Some).collect());
            }
        }
        // Also seed from any pre-existing known_i64_values in tensor_info.
        for (&tid, info) in &graph.tensor_info {
            if let Some(vals) = &info.known_i64_values {
                known.entry(tid).or_insert_with(|| vals.clone());
            }
        }

        // Forward pass in topological order.
        for &nid in &order {
            let idx = match node_idx.get(&nid) {
                Some(&i) => i,
                None => continue,
            };

            let op = &graph.nodes[idx].op;
            let input_tids = &graph.nodes[idx].inputs;
            let output_tids = &graph.nodes[idx].outputs;

            // Gather known values for each input.
            let input_known: Vec<Option<&KnownValues>> =
                input_tids.iter().map(|tid| known.get(tid)).collect();

            // Gather tensor_info shapes for Shape op.
            let input_shapes: Vec<Option<&[DimExpr]>> = input_tids
                .iter()
                .map(|tid| graph.tensor_info.get(tid).map(|ti| ti.shape.as_slice()))
                .collect();

            if let Some(result) = eval_op(op, &input_known, &input_shapes) {
                if let Some(&out_tid) = output_tids.first() {
                    tracing::trace!(nid, ?op, out_tid, ?result, "DataProp: evaluated op");
                    known.insert(out_tid, result);
                }
            } else if matches!(
                op,
                AiOp::Shape { .. }
                    | AiOp::Gather { .. }
                    | AiOp::Unsqueeze { .. }
                    | AiOp::Squeeze { .. }
                    | AiOp::Concat { .. }
                    | AiOp::Cast { .. }
                    | AiOp::Slice { .. }
                    | AiOp::Range
            ) {
                tracing::debug!(
                    nid,
                    ?op,
                    input_tids = ?input_tids,
                    inputs_known = ?input_known.iter().map(|v| v.is_some()).collect::<Vec<_>>(),
                    "DataProp: could not evaluate shape-relevant op"
                );
            }
        }

        // Post-process: resolve -1 values in Reshape shape tensors.
        // If a Reshape's shape tensor has [Some(batch), Some(seq), Some(-1), Some(64)]
        // and the data input has concrete dims (e.g. shape [batch, seq, 2048]),
        // resolve -1 = product_of_known_data_dims / product_of_known_shape_values.
        //
        // IMPORTANT: Multiple Reshape nodes may share the same shape tensor (e.g.,
        // Q/K/V in GQA attention). The -1 resolves to DIFFERENT values depending on
        // the data tensor (Q=32, K/V=4 for head count). We must resolve per-consumer
        // and store the result in each Reshape's OUTPUT tensor, not mutate the shared
        // shape tensor.
        for &nid in &order {
            let idx = match node_idx.get(&nid) {
                Some(&i) => i,
                None => continue,
            };
            if !matches!(graph.nodes[idx].op, AiOp::Reshape { .. }) {
                continue;
            }
            let inputs = &graph.nodes[idx].inputs;
            let outputs = &graph.nodes[idx].outputs;
            if inputs.len() < 2 || outputs.is_empty() {
                continue;
            }
            let shape_tid = inputs[1];
            let data_tid = inputs[0];
            let out_tid = outputs[0];

            if let Some(vals) = known.get(&shape_tid) {
                let neg_one_count = vals.iter().filter(|v| **v == Some(-1)).count();
                if neg_one_count == 1 {
                    // The -1 in ONNX Reshape means "infer from total elements".
                    // Dynamic dims (batch, seq) appear in BOTH data shape and shape
                    // tensor, so they cancel: -1 = data_concrete / shape_concrete.
                    // E.g., data=[batch,seq,2048], shape=[batch,seq,-1,64] → -1=2048/64=32
                    let data_concrete: u64 = graph
                        .tensor_info
                        .get(&data_tid)
                        .map(|ti| {
                            ti.shape
                                .iter()
                                .filter_map(|d| d.as_concrete())
                                .product::<u64>()
                                .max(1)
                        })
                        .unwrap_or(1);

                    let shape_concrete: i64 = vals
                        .iter()
                        .filter_map(|v| *v)
                        .filter(|&v| v > 0)
                        .product::<i64>()
                        .max(1);

                    tracing::trace!(
                        nid,
                        data_tid,
                        shape_tid,
                        data_concrete,
                        shape_concrete,
                        ?vals,
                        "DataProp: resolving -1 in Reshape"
                    );
                    if data_concrete > 1 && shape_concrete > 0 {
                        let resolved = data_concrete as i64 / shape_concrete;
                        if resolved > 0 {
                            tracing::trace!(nid, resolved, out_tid, "DataProp: resolved -1");
                            // Create a per-consumer resolved copy; do NOT mutate shared shape tensor.
                            let mut resolved_vals = vals.clone();
                            for v in resolved_vals.iter_mut() {
                                if *v == Some(-1) {
                                    *v = Some(resolved);
                                    break;
                                }
                            }
                            // Store the resolved shape as the Reshape output's known values.
                            // This feeds into shape_prop for the compiled output shape.
                            known.insert(out_tid, resolved_vals);
                        }
                    }
                }
            }
        }

        // Write known values back to tensor_info.
        for (tid, vals) in &known {
            if let Some(info) = graph.tensor_info.get_mut(tid) {
                info.known_i64_values = Some(vals.clone());
            }
        }

        // Materialize fully-concrete known values as AiParam::Inline
        // so that ConstantFolding/DeadNodeElimination can prune the subgraph.
        //
        // IMPORTANT: Only materialize tensors whose dtype is an integer type
        // (INT64, INT32, INT8). Known i64 values are only meaningful for
        // shape/index tensors. Materializing F32 tensors as INT64 corrupts
        // downstream data paths (e.g., attention Q/K/V become INT64 garbage).
        for (tid, vals) in &known {
            // Skip tensors that are already params.
            if graph.params.contains_key(tid) {
                continue;
            }
            // Only materialize integer-typed tensors (shape/index subgraphs).
            let is_integer_tensor = graph
                .tensor_info
                .get(tid)
                .map(|ti| matches!(ti.logical_dtype, DType::INT64 | DType::INT32 | DType::INT8))
                .unwrap_or(false);
            if !is_integer_tensor {
                continue;
            }
            // Only materialize if all values are concrete.
            if vals.iter().all(|v| v.is_some()) {
                let concrete: Vec<i64> = vals.iter().map(|v| v.unwrap()).collect();
                let bytes: Vec<u8> = concrete.iter().flat_map(|v| v.to_le_bytes()).collect();
                let shape = crate::ir::shape_from_concrete(&[concrete.len() as u64]);
                let info = TensorInfo::new(DType::INT64, shape);
                graph.params.insert(*tid, AiParam::inline(bytes, info));
            }
        }

        Ok(graph)
    }
}

/// Extract i64 values from an inline parameter (supports INT64, INT32, INT8, F32).
///
/// F32 support is needed for ONNX models that use float-typed scalars as
/// Range inputs (e.g., `Range(0.0, 2048.0, 1.0)` for causal mask generation).
fn extract_i64_param(param: &AiParam) -> Option<Vec<i64>> {
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

/// Evaluate a single op to produce known output values, if possible.
fn eval_op(
    op: &AiOp,
    inputs: &[Option<&KnownValues>],
    input_shapes: &[Option<&[DimExpr]>],
) -> Option<KnownValues> {
    match op.category() {
        // Unary elementwise: only Identity passes through known i64 values.
        // Other unary ops (Cos, Sin, Exp, etc.) transform data — their i64
        // representation is meaningless, and propagating would leak shape
        // values into data paths.
        OpCategory::UnaryElementwise => {
            if matches!(op, AiOp::Identity) {
                inputs.first().copied().flatten().cloned()
            } else {
                None
            }
        }
        // Binary elementwise: arithmetic on known i64 values.
        OpCategory::BinaryElementwise => eval_binary_by_op(op, inputs),
        // Comparisons and shape-preserving ops don't propagate i64 values.
        OpCategory::BinaryComparison | OpCategory::ShapePreserving => None,
        // Custom ops need per-variant logic.
        OpCategory::Custom => eval_custom_op(op, inputs, input_shapes),
    }
}

/// Dispatch binary elementwise ops to the appropriate arithmetic operation.
fn eval_binary_by_op(op: &AiOp, inputs: &[Option<&KnownValues>]) -> Option<KnownValues> {
    match op {
        AiOp::Add => eval_binary(inputs, |a, b| a.checked_add(b)),
        AiOp::Sub => eval_binary(inputs, |a, b| a.checked_sub(b)),
        AiOp::Mul => eval_binary(inputs, |a, b| a.checked_mul(b)),
        AiOp::Div => eval_binary(inputs, |a, b| if b != 0 { Some(a / b) } else { None }),
        // Other binary ops (Pow, Mod, Min, Max, And, Or, Xor) don't appear
        // in shape-computation subgraphs; skip for now.
        _ => None,
    }
}

/// Evaluate custom ops that need per-variant logic for value propagation.
fn eval_custom_op(
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

        _ => None,
    }
}

/// Normalize a slice start/end bound, clamping to [0, dim_size].
fn normalize_slice_bound(val: i64, dim_size: i64) -> i64 {
    let v = if val < 0 { dim_size + val } else { val };
    v.clamp(0, dim_size)
}

/// Evaluate a binary elementwise op on known i64 values with broadcasting.
fn eval_binary(
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::shape::{ConstraintStore, DimVarTable, Shape};
    use crate::ir::{node::AiNode, shape::shape_from_concrete, AiGraph, AiOp, DType, TensorInfo};

    fn make_graph(
        nodes: Vec<AiNode>,
        inputs: Vec<TensorId>,
        outputs: Vec<TensorId>,
        params: HashMap<TensorId, AiParam>,
        tensor_info: HashMap<TensorId, TensorInfo>,
    ) -> AiGraph {
        AiGraph {
            name: "test".into(),
            nodes,
            inputs,
            outputs,
            input_names: vec![],
            output_names: vec![],
            params,
            tensor_info,
            metadata: HashMap::new(),
            warnings: vec![],
            dim_vars: DimVarTable::default(),
            shape_constraints: ConstraintStore::default(),
        }
    }

    /// Core test: Shape → Gather → Unsqueeze → Concat chain.
    /// input_ids: [1, 2] (batch=1, seq=2)
    /// Shape → [1, 2]  (rank 2)
    /// Gather(0, idx=1) → 2  (seq dim)
    /// Unsqueeze(0) → [2]
    /// Concat([2], [-1, 16, 64]) → [2, -1, 16, 64]
    #[test]
    fn propagate_shape_gather_unsqueeze_concat() {
        let mut ti = HashMap::new();
        // input_ids: [batch=1, seq=2]
        ti.insert(
            0,
            TensorInfo::new(DType::INT64, shape_from_concrete(&[1, 2])),
        );
        // Shape output
        ti.insert(1, TensorInfo::new(DType::INT64, shape_from_concrete(&[2])));
        // Gather index: scalar constant = 1
        ti.insert(2, TensorInfo::new(DType::INT64, shape_from_concrete(&[1])));
        // Gather output: scalar
        ti.insert(3, TensorInfo::new(DType::INT64, Shape::new()));
        // Unsqueeze output: [1]
        ti.insert(4, TensorInfo::new(DType::INT64, shape_from_concrete(&[1])));
        // Constant shape part: [-1, 16, 64]
        ti.insert(5, TensorInfo::new(DType::INT64, shape_from_concrete(&[3])));
        // Concat output: [4]
        ti.insert(6, TensorInfo::new(DType::INT64, shape_from_concrete(&[4])));

        let mut params = HashMap::new();
        // Gather index = 1 (pick seq dim)
        params.insert(
            2,
            AiParam::inline(
                1i64.to_le_bytes().to_vec(),
                TensorInfo::new(DType::INT64, shape_from_concrete(&[1])),
            ),
        );
        // Constant [-1, 16, 64]
        let const_bytes: Vec<u8> = [-1i64, 16, 64]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        params.insert(
            5,
            AiParam::inline(
                const_bytes,
                TensorInfo::new(DType::INT64, shape_from_concrete(&[3])),
            ),
        );

        let nodes = vec![
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
            AiNode::new(3, AiOp::Concat { axis: 0 }, vec![4, 5], vec![6]),
        ];

        let g = make_graph(nodes, vec![0], vec![6], params, ti);
        let pass = DataPropagation;
        let g2 = pass.run(g).unwrap();

        // Shape output: [Some(1), Some(2)]
        let shape_vals = g2.tensor_info[&1].known_i64_values.as_ref().unwrap();
        assert_eq!(shape_vals, &[Some(1), Some(2)]);

        // Gather output: [Some(2)] (picked index 1 → seq dim = 2)
        let gather_vals = g2.tensor_info[&3].known_i64_values.as_ref().unwrap();
        assert_eq!(gather_vals, &[Some(2)]);

        // Unsqueeze output: [Some(2)]
        let unsqueeze_vals = g2.tensor_info[&4].known_i64_values.as_ref().unwrap();
        assert_eq!(unsqueeze_vals, &[Some(2)]);

        // Concat output: [Some(2), Some(-1), Some(16), Some(64)]
        let concat_vals = g2.tensor_info[&6].known_i64_values.as_ref().unwrap();
        assert_eq!(concat_vals, &[Some(2), Some(-1), Some(16), Some(64)]);

        // Concat output should also be materialized as AiParam::Inline
        // since all values are concrete.
        assert!(g2.params.contains_key(&6));
    }

    /// Test with dynamic dims: Shape on a tensor with symbolic batch dim.
    #[test]
    fn propagate_with_dynamic_dims() {
        let mut ti = HashMap::new();
        // input: [Dynamic, 768]
        let shape = Shape::from(vec![DimExpr::Dynamic, DimExpr::Concrete(768)]);
        ti.insert(0, TensorInfo::new(DType::F32, shape));
        // Shape output
        ti.insert(1, TensorInfo::new(DType::INT64, shape_from_concrete(&[2])));
        // Gather index (pick dim 1 = 768)
        ti.insert(2, TensorInfo::new(DType::INT64, shape_from_concrete(&[1])));
        // Gather output
        ti.insert(3, TensorInfo::new(DType::INT64, Shape::new()));

        let mut params = HashMap::new();
        params.insert(
            2,
            AiParam::inline(
                1i64.to_le_bytes().to_vec(),
                TensorInfo::new(DType::INT64, shape_from_concrete(&[1])),
            ),
        );

        let nodes = vec![
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
        ];

        let g = make_graph(nodes, vec![0], vec![3], params, ti);
        let pass = DataPropagation;
        let g2 = pass.run(g).unwrap();

        // Shape output: [None, Some(768)] (batch is dynamic)
        let shape_vals = g2.tensor_info[&1].known_i64_values.as_ref().unwrap();
        assert_eq!(shape_vals, &[None, Some(768)]);

        // Gather(axis=0, idx=1): picks dim 1 → Some(768)
        let gather_vals = g2.tensor_info[&3].known_i64_values.as_ref().unwrap();
        assert_eq!(gather_vals, &[Some(768)]);
    }

    /// Test that INT32 Gather index constants are properly seeded.
    /// This is the real-world ONNX pattern where Gather indices are INT32.
    #[test]
    fn propagate_int32_gather_index() {
        let mut ti = HashMap::new();
        // input: [batch=1, seq=2, embed=2048]
        ti.insert(
            0,
            TensorInfo::new(DType::F32, shape_from_concrete(&[1, 2, 2048])),
        );
        // Shape output: rank 3
        ti.insert(1, TensorInfo::new(DType::INT64, shape_from_concrete(&[3])));
        // Gather index: INT32 constant = 2 (picks embed dim)
        ti.insert(2, TensorInfo::new(DType::INT32, shape_from_concrete(&[1])));
        // Gather output
        ti.insert(3, TensorInfo::new(DType::INT64, Shape::new()));

        let mut params = HashMap::new();
        // INT32 index = 2
        params.insert(
            2,
            AiParam::inline(
                2i32.to_le_bytes().to_vec(),
                TensorInfo::new(DType::INT32, shape_from_concrete(&[1])),
            ),
        );

        let nodes = vec![
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
        ];

        let g = make_graph(nodes, vec![0], vec![3], params, ti);
        let pass = DataPropagation;
        let g2 = pass.run(g).unwrap();

        // Gather should pick dim 2 = 2048
        let gather_vals = g2.tensor_info[&3].known_i64_values.as_ref().unwrap();
        assert_eq!(gather_vals, &[Some(2048)]);
    }

    /// Test -1 resolution in Reshape shape tensor with Concat producing the shape.
    /// data=[batch, seq, 2048], shape_concat=[batch, seq, -1, 64] → -1 = 2048/64 = 32
    #[test]
    fn propagate_reshape_neg1_resolution() {
        let mut ti = HashMap::new();
        // T0: data tensor (input to Reshape)
        let data_shape = Shape::from(vec![
            DimExpr::Dynamic,
            DimExpr::Dynamic,
            DimExpr::Concrete(2048),
        ]);
        ti.insert(0, TensorInfo::new(DType::F32, data_shape));
        // T1: Shape op output
        ti.insert(1, TensorInfo::new(DType::INT64, shape_from_concrete(&[3])));
        // T2: Gather index 0 (batch) — constant
        ti.insert(2, TensorInfo::new(DType::INT64, shape_from_concrete(&[1])));
        // T3: Gather output (batch dim value)
        ti.insert(3, TensorInfo::new(DType::INT64, Shape::new()));
        // T4: Unsqueeze output [batch]
        ti.insert(4, TensorInfo::new(DType::INT64, shape_from_concrete(&[1])));
        // T5: Gather index 1 (seq) — constant
        ti.insert(5, TensorInfo::new(DType::INT64, shape_from_concrete(&[1])));
        // T6: Gather output (seq dim value)
        ti.insert(6, TensorInfo::new(DType::INT64, Shape::new()));
        // T7: Unsqueeze output [seq]
        ti.insert(7, TensorInfo::new(DType::INT64, shape_from_concrete(&[1])));
        // T8: Constant [-1, 64]
        ti.insert(8, TensorInfo::new(DType::INT64, shape_from_concrete(&[2])));
        // T9: Concat output [batch, seq, -1, 64]
        ti.insert(9, TensorInfo::new(DType::INT64, shape_from_concrete(&[4])));
        // T10: Reshape output
        ti.insert(10, TensorInfo::new(DType::F32, Shape::new()));

        let params = HashMap::from([
            (
                2,
                AiParam::inline(
                    0i64.to_le_bytes().to_vec(),
                    TensorInfo::new(DType::INT64, shape_from_concrete(&[1])),
                ),
            ),
            (
                5,
                AiParam::inline(
                    1i64.to_le_bytes().to_vec(),
                    TensorInfo::new(DType::INT64, shape_from_concrete(&[1])),
                ),
            ),
            (8, {
                let const_bytes: Vec<u8> =
                    [-1i64, 64].iter().flat_map(|v| v.to_le_bytes()).collect();
                AiParam::inline(
                    const_bytes,
                    TensorInfo::new(DType::INT64, shape_from_concrete(&[2])),
                )
            }),
        ]);

        let nodes = vec![
            AiNode::new(
                0,
                AiOp::Shape {
                    start: None,
                    end: None,
                },
                vec![0],
                vec![1],
            ),
            AiNode::new(1, AiOp::Gather { axis: 0 }, vec![1, 2], vec![3]), // batch
            AiNode::new(2, AiOp::Unsqueeze { axes: vec![0] }, vec![3], vec![4]),
            AiNode::new(3, AiOp::Gather { axis: 0 }, vec![1, 5], vec![6]), // seq
            AiNode::new(4, AiOp::Unsqueeze { axes: vec![0] }, vec![6], vec![7]),
            AiNode::new(5, AiOp::Concat { axis: 0 }, vec![4, 7, 8], vec![9]),
            AiNode::new(6, AiOp::Reshape { allow_zero: false }, vec![0, 9], vec![10]),
        ];

        let g = make_graph(nodes, vec![0], vec![10], params, ti);
        let pass = DataPropagation;
        let g2 = pass.run(g).unwrap();

        // Concat output (shape tensor): [None, None, Some(-1), Some(64)]
        let concat_vals = g2.tensor_info[&9].known_i64_values.as_ref().unwrap();
        assert_eq!(concat_vals, &[None, None, Some(-1), Some(64)]);

        // Reshape output (per-consumer resolved): -1 → 2048/64 = 32
        let reshape_vals = g2.tensor_info[&10].known_i64_values.as_ref().unwrap();
        assert_eq!(reshape_vals, &[None, None, Some(32), Some(64)]);
    }

    /// Test arithmetic: Mul on known i64 values.
    #[test]
    fn propagate_arithmetic() {
        let mut ti = HashMap::new();
        ti.insert(0, TensorInfo::new(DType::INT64, shape_from_concrete(&[2])));
        ti.insert(1, TensorInfo::new(DType::INT64, shape_from_concrete(&[1])));
        ti.insert(2, TensorInfo::new(DType::INT64, shape_from_concrete(&[2])));

        let mut params = HashMap::new();
        // [4, 8]
        let a_bytes: Vec<u8> = [4i64, 8].iter().flat_map(|v| v.to_le_bytes()).collect();
        params.insert(
            0,
            AiParam::inline(
                a_bytes,
                TensorInfo::new(DType::INT64, shape_from_concrete(&[2])),
            ),
        );
        // [2] (scalar broadcast)
        params.insert(
            1,
            AiParam::inline(
                2i64.to_le_bytes().to_vec(),
                TensorInfo::new(DType::INT64, shape_from_concrete(&[1])),
            ),
        );

        let nodes = vec![AiNode::new(0, AiOp::Mul, vec![0, 1], vec![2])];

        let g = make_graph(nodes, vec![], vec![2], params, ti);
        let pass = DataPropagation;
        let g2 = pass.run(g).unwrap();

        let mul_vals = g2.tensor_info[&2].known_i64_values.as_ref().unwrap();
        assert_eq!(mul_vals, &[Some(8), Some(16)]);
    }

    /// Test Slice on known values — extracts subrange from shape tensor.
    /// This is the pattern in attention reshapes: Slice([1,2,32,64], start=2) → [32, 64]
    #[test]
    fn propagate_slice() {
        let mut ti = HashMap::new();
        // data: [1, 2, 32, 64] (4 elements)
        ti.insert(0, TensorInfo::new(DType::INT64, shape_from_concrete(&[4])));
        // slice output: [32, 64] (2 elements)
        ti.insert(1, TensorInfo::new(DType::INT64, shape_from_concrete(&[2])));

        let data_bytes: Vec<u8> = [1i64, 2, 32, 64]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let params = HashMap::from([(
            0,
            AiParam::inline(
                data_bytes,
                TensorInfo::new(DType::INT64, shape_from_concrete(&[4])),
            ),
        )]);

        let nodes = vec![AiNode::new(
            0,
            AiOp::Slice {
                axes: vec![0],
                starts: vec![2],
                ends: vec![4],
                steps: vec![1],
            },
            vec![0],
            vec![1],
        )];

        let g = make_graph(nodes, vec![], vec![1], params, ti);
        let g2 = DataPropagation.run(g).unwrap();

        let vals = g2.tensor_info[&1].known_i64_values.as_ref().unwrap();
        assert_eq!(vals, &[Some(32), Some(64)]);
    }

    /// Test Slice with negative indices (Slice(data, start=-2, end=MAX)).
    #[test]
    fn propagate_slice_negative() {
        let mut ti = HashMap::new();
        ti.insert(0, TensorInfo::new(DType::INT64, shape_from_concrete(&[4])));
        ti.insert(1, TensorInfo::new(DType::INT64, shape_from_concrete(&[2])));

        let data_bytes: Vec<u8> = [1i64, 2, 32, 64]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let params = HashMap::from([(
            0,
            AiParam::inline(
                data_bytes,
                TensorInfo::new(DType::INT64, shape_from_concrete(&[4])),
            ),
        )]);

        let nodes = vec![AiNode::new(
            0,
            AiOp::Slice {
                axes: vec![0],
                starts: vec![-2],
                ends: vec![i64::MAX],
                steps: vec![1],
            },
            vec![0],
            vec![1],
        )];

        let g = make_graph(nodes, vec![], vec![1], params, ti);
        let g2 = DataPropagation.run(g).unwrap();

        let vals = g2.tensor_info[&1].known_i64_values.as_ref().unwrap();
        assert_eq!(vals, &[Some(32), Some(64)]);
    }

    /// Test Range: Range(0, 5, 1) → [0, 1, 2, 3, 4]
    #[test]
    fn propagate_range() {
        let mut ti = HashMap::new();
        ti.insert(0, TensorInfo::new(DType::INT64, shape_from_concrete(&[1])));
        ti.insert(1, TensorInfo::new(DType::INT64, shape_from_concrete(&[1])));
        ti.insert(2, TensorInfo::new(DType::INT64, shape_from_concrete(&[1])));
        ti.insert(3, TensorInfo::new(DType::INT64, Shape::new()));

        let params = HashMap::from([
            (
                0,
                AiParam::inline(
                    0i64.to_le_bytes().to_vec(),
                    TensorInfo::new(DType::INT64, shape_from_concrete(&[1])),
                ),
            ),
            (
                1,
                AiParam::inline(
                    5i64.to_le_bytes().to_vec(),
                    TensorInfo::new(DType::INT64, shape_from_concrete(&[1])),
                ),
            ),
            (
                2,
                AiParam::inline(
                    1i64.to_le_bytes().to_vec(),
                    TensorInfo::new(DType::INT64, shape_from_concrete(&[1])),
                ),
            ),
        ]);

        let nodes = vec![AiNode::new(0, AiOp::Range, vec![0, 1, 2], vec![3])];

        let g = make_graph(nodes, vec![], vec![3], params, ti);
        let g2 = DataPropagation.run(g).unwrap();

        let vals = g2.tensor_info[&3].known_i64_values.as_ref().unwrap();
        assert_eq!(vals, &[Some(0), Some(1), Some(2), Some(3), Some(4)]);
    }

    /// Test Slice→Concat chain (the attention reshape pattern).
    /// Shape=[1,2,32,64] → Slice(start=0,end=2)→[1,2] + Slice(start=2,end=4)→[32,64]
    /// → Concat → [1,2,32,64]
    #[test]
    fn propagate_slice_concat_chain() {
        let mut ti = HashMap::new();
        ti.insert(0, TensorInfo::new(DType::INT64, shape_from_concrete(&[4])));
        ti.insert(1, TensorInfo::new(DType::INT64, shape_from_concrete(&[2])));
        ti.insert(2, TensorInfo::new(DType::INT64, shape_from_concrete(&[2])));
        ti.insert(3, TensorInfo::new(DType::INT64, shape_from_concrete(&[4])));

        let data_bytes: Vec<u8> = [1i64, 2, 32, 64]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let params = HashMap::from([(
            0,
            AiParam::inline(
                data_bytes,
                TensorInfo::new(DType::INT64, shape_from_concrete(&[4])),
            ),
        )]);

        let nodes = vec![
            AiNode::new(
                0,
                AiOp::Slice {
                    axes: vec![0],
                    starts: vec![0],
                    ends: vec![2],
                    steps: vec![1],
                },
                vec![0],
                vec![1],
            ),
            AiNode::new(
                1,
                AiOp::Slice {
                    axes: vec![0],
                    starts: vec![2],
                    ends: vec![4],
                    steps: vec![1],
                },
                vec![0],
                vec![2],
            ),
            AiNode::new(2, AiOp::Concat { axis: 0 }, vec![1, 2], vec![3]),
        ];

        let g = make_graph(nodes, vec![], vec![3], params, ti);
        let g2 = DataPropagation.run(g).unwrap();

        let vals = g2.tensor_info[&3].known_i64_values.as_ref().unwrap();
        assert_eq!(vals, &[Some(1), Some(2), Some(32), Some(64)]);
    }
}
