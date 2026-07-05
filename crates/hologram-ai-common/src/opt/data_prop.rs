//! Data propagation pass — evaluates shape-computation subgraphs at compile time.
//!
//! ONNX models compute Reshape target shapes at runtime via subgraphs like
//! `Shape → Gather → Unsqueeze → Concat`. This pass evaluates those ops when
//! their inputs have known constant values, populating `TensorInfo::known_i64_values`
//! so that the lowering pass can bake correct shapes into the compiled graph.

use super::data_eval_ops::{eval_binary, eval_custom_op, extract_i64_param, KnownValues};
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
        //
        // Only shape/index data feeds data propagation: small integer vectors
        // (reshape targets, axes, slice bounds, gather indices). A model's
        // weights are large constants that are *never* shape operands, so skip
        // any param above a generous shape-data ceiling — materializing a
        // multi-MB weight as `Vec<Option<i64>>` (16 B/elem) would blow up memory
        // by an order of magnitude for values that propagation never reads (a
        // 135M-param f32 model would otherwise seed ~2 GB here).
        const MAX_SHAPE_DATA_BYTES: usize = 256 * 1024;
        let mut known: HashMap<TensorId, KnownValues> = HashMap::new();
        for (&tid, param) in &graph.params {
            if let AiParam::Inline { data, .. } = param {
                if data.len() > MAX_SHAPE_DATA_BYTES {
                    continue;
                }
            }
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

        // Track which TIDs were freshly computed by this forward pass.
        // Used in materialization to allow re-writing a DataProp-created param
        // when the shape computation is re-evaluated with updated input shapes.
        let mut computed_tids: std::collections::HashSet<TensorId> =
            std::collections::HashSet::new();

        // Forward pass in topological order.
        for &nid in order.iter() {
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
                    computed_tids.insert(out_tid);
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
        // the data tensor (Q=32, K/V=4 for head count). Resolution is therefore
        // per-consumer and applied to the consuming Reshape's OUTPUT SHAPE in
        // tensor_info — never to `known` and never to the shared shape tensor.
        // `known` carries the runtime VALUES flowing through a tensor; a Reshape's
        // output carries the reshaped DATA, not its own target shape. Writing the
        // target shape into the output's value channel corrupted integer data
        // tensors (e.g. position ids): the materializer inlined a rank-N tensor as
        // a tiny "shape vector" param and downstream constant folding computed with
        // garbage (Qwen2.5 regression — caught by the architecture matrix).
        for &nid in order.iter() {
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
                            // Strengthen the Reshape output's SHAPE (the correct
                            // channel for a target shape): concretize dims the
                            // resolution pinned, leave the rest untouched. An
                            // empty (unknown-rank) shape is replaced wholesale,
                            // with Dynamic holding unresolved positions.
                            if let Some(info) = graph.tensor_info.get_mut(&out_tid) {
                                if info.shape.is_empty() {
                                    info.shape = resolved_vals
                                        .iter()
                                        .map(|v| match v {
                                            Some(n) if *n > 0 => {
                                                crate::ir::Dim::Concrete(*n as u64)
                                            }
                                            _ => crate::ir::Dim::Dynamic,
                                        })
                                        .collect();
                                } else if info.shape.len() == resolved_vals.len() {
                                    for (dim, v) in info.shape.iter_mut().zip(&resolved_vals) {
                                        if let Some(v) = v {
                                            if *v > 0 && dim.as_concrete().is_none() {
                                                *dim = crate::ir::Dim::Concrete(*v as u64);
                                            }
                                        }
                                    }
                                }
                            }
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
        //
        // Re-materialization policy: if this pass computed a value for a TID
        // that already exists as a param (from a previous DataProp run), we
        // overwrite it with the freshly-computed value. This handles multi-level
        // shape dependencies where a second DataProp pass (after AggressiveProp
        // has propagated DataProp's first-pass results) produces better values.
        // We NEVER overwrite original model params (not in computed_tids).
        for (tid, vals) in &known {
            // Skip original model params that DataProp did not recompute.
            if graph.params.contains_key(tid) && !computed_tids.contains(tid) {
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
            if vals.is_empty() {
                continue;
            }
            if vals.iter().all(|v| v.is_some()) {
                // All values concrete — materialize directly.
                let concrete: Vec<i64> = vals.iter().map(|v| v.unwrap()).collect();
                // Preserve the tensor's existing dtype and (rank≥2) shape when
                // both are known. The default fallback below treats the result
                // as a 1-D INT64 shape vector — correct for the original use
                // case (RoPE/index subgraphs) but wrong for rank≥2 integer
                // weights that DataProp can now fold through MatMulInteger's
                // Cast→Sub→Cast decomposition. A 2-D INT8 weight like [896,128]
                // would otherwise be materialized as [114688] INT64, which
                // breaks downstream MatMul shape inference.
                let existing = graph.tensor_info.get(tid);
                let preserve_shape: Option<crate::Shape> = existing.and_then(|ti| {
                    let concrete_dims: Vec<u64> =
                        ti.shape.iter().filter_map(|d| d.as_concrete()).collect();
                    if concrete_dims.len() == ti.shape.len() && !ti.shape.is_empty() {
                        let prod: u64 = concrete_dims.iter().product();
                        if prod as usize == concrete.len() {
                            return Some(crate::ir::shape_from_concrete(&concrete_dims));
                        }
                    }
                    None
                });
                let preserve_dtype = existing.map(|ti| ti.logical_dtype).filter(|dt| {
                    matches!(dt, DType::INT64 | DType::INT32 | DType::INT8 | DType::U8)
                });
                // A fully-concrete tensor_info shape whose element count
                // disagrees with the value count means these are NOT the
                // tensor's runtime values (e.g. a stale or smuggled shape
                // vector on a data tensor). Materializing would inline
                // garbage data under a real tensor's id — skip loudly.
                let contradicts = existing.is_some_and(|ti| {
                    let dims: Vec<u64> = ti.shape.iter().filter_map(|d| d.as_concrete()).collect();
                    dims.len() == ti.shape.len()
                        && !ti.shape.is_empty()
                        && dims.iter().product::<u64>() as usize != concrete.len()
                });
                if contradicts {
                    tracing::warn!(
                        tid,
                        vals = concrete.len(),
                        shape = ?existing.map(|ti| ti.shape.clone()),
                        "DataProp: known values contradict the tensor's concrete shape; not materializing"
                    );
                    continue;
                }
                if let (Some(shape), Some(dtype)) = (preserve_shape, preserve_dtype) {
                    // Re-encode in the tensor's native dtype so byte counts
                    // and downstream interpretation stay consistent.
                    let bytes: Vec<u8> = match dtype {
                        DType::INT64 => concrete.iter().flat_map(|v| v.to_le_bytes()).collect(),
                        DType::INT32 => concrete
                            .iter()
                            .flat_map(|v| (*v as i32).to_le_bytes())
                            .collect(),
                        DType::INT8 => concrete.iter().map(|v| *v as i8 as u8).collect(),
                        DType::U8 => concrete.iter().map(|v| *v as u8).collect(),
                        _ => unreachable!("filter above"),
                    };
                    let info = TensorInfo::new(dtype, shape);
                    graph.params.insert(*tid, AiParam::inline(bytes, info));
                } else {
                    // Fallback: shape vector (1-D INT64). Original behavior.
                    let bytes: Vec<u8> = concrete.iter().flat_map(|v| v.to_le_bytes()).collect();
                    let shape = crate::ir::shape_from_concrete(&[concrete.len() as u64]);
                    let info = TensorInfo::new(DType::INT64, shape);
                    graph.params.insert(*tid, AiParam::inline(bytes, info));
                }
            }
        }

        // Materialize ConstantOfShape outputs as F32 params.
        // ConstantOfShape produces F32 data (not shape metadata), so the
        // integer-only materialization above skips it. We handle it separately:
        // read the shape from the input's known_i64_values, fill with the
        // op's fill_value, and register as a param.
        for node in &graph.nodes {
            let AiOp::ConstantOfShape { fill_value } = &node.op else {
                continue;
            };
            let Some(&out_tid) = node.outputs.first() else {
                continue;
            };
            // Already materialized (e.g. from import-time folding).
            if graph.params.contains_key(&out_tid) {
                continue;
            }
            // Read the shape from the input tensor's known_i64_values.
            let shape_vals: Vec<i64> = node
                .inputs
                .first()
                .and_then(|tid| graph.tensor_info.get(tid))
                .and_then(|info| info.known_i64_values.as_ref())
                .map(|vals| vals.iter().filter_map(|v| *v).collect::<Vec<_>>())
                .unwrap_or_default();
            if shape_vals.is_empty() {
                continue;
            }
            let fill = f32::from_bits(*fill_value);
            let n_elements: usize = shape_vals.iter().map(|&d| d.max(0) as usize).product();
            let data_f32 = vec![fill; n_elements];
            let data_bytes: Vec<u8> = data_f32.iter().flat_map(|f| f.to_le_bytes()).collect();
            let dim_shape: crate::Shape = shape_vals
                .iter()
                .map(|&d| crate::Dim::Concrete(d as u64))
                .collect();
            let info = TensorInfo::new(DType::F32, dim_shape);
            graph
                .tensor_info
                .entry(out_tid)
                .and_modify(|existing| *existing = info.clone())
                .or_insert_with(|| info.clone());
            graph
                .params
                .insert(out_tid, AiParam::inline(data_bytes, info));
        }

        // Materialize Trilu outputs when the input is a known constant param.
        // Trilu(upper=true): zero below the k-th diagonal.
        // Trilu(upper=false): zero above the k-th diagonal.
        for node in &graph.nodes {
            let AiOp::Trilu { upper } = &node.op else {
                continue;
            };
            let Some(&out_tid) = node.outputs.first() else {
                continue;
            };
            if graph.params.contains_key(&out_tid) {
                continue;
            }
            // Input 0: the matrix. Must be a known f32 param.
            let input_tid = match node.inputs.first() {
                Some(tid) => *tid,
                None => continue,
            };
            let input_param = match graph.params.get(&input_tid) {
                Some(p) => p,
                None => continue,
            };
            let (input_data, input_info) = match input_param {
                AiParam::Inline { data, info } => (data.as_slice(), info),
                _ => continue,
            };
            if input_info.logical_dtype != DType::F32 {
                continue;
            }
            let shape: Vec<usize> = input_info
                .shape
                .iter()
                .filter_map(|d| match d {
                    crate::Dim::Concrete(v) => Some(*v as usize),
                    _ => None,
                })
                .collect();
            if shape.len() < 2 {
                continue;
            }
            // Read the diagonal offset k from input 1 (default 0).
            let k: i64 = node
                .inputs
                .get(1)
                .and_then(|tid| graph.params.get(tid))
                .and_then(extract_i64_param)
                .and_then(|vals| vals.first().copied())
                .unwrap_or(0);

            let input_f32: &[f32] = match bytemuck::try_cast_slice(input_data) {
                Ok(s) => s,
                Err(_) => continue,
            };
            let rows = shape[shape.len() - 2];
            let cols = shape[shape.len() - 1];
            let batch_size: usize = shape[..shape.len() - 2].iter().product::<usize>().max(1);
            let mut output = input_f32.to_vec();
            for b in 0..batch_size {
                let offset = b * rows * cols;
                for r in 0..rows {
                    for c in 0..cols {
                        let diag = c as i64 - r as i64;
                        let zero_it = if *upper { diag < k } else { diag > k };
                        if zero_it {
                            output[offset + r * cols + c] = 0.0;
                        }
                    }
                }
            }
            let out_bytes: Vec<u8> = output.iter().flat_map(|f| f.to_le_bytes()).collect();
            let out_info = input_info.clone();
            graph
                .tensor_info
                .entry(out_tid)
                .and_modify(|existing| *existing = out_info.clone())
                .or_insert_with(|| out_info.clone());
            graph
                .params
                .insert(out_tid, AiParam::inline(out_bytes, out_info));
        }

        Ok(graph)
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
            subgraphs: HashMap::new(),
            tensor_names: HashMap::new(),
            topo_cache: Default::default(),
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

        // Reshape output (per-consumer resolved): -1 → 2048/64 = 32. The
        // resolution lands in the output's SHAPE — the value channel stays
        // untouched (the output carries reshaped DATA, not its target shape),
        // and no param may be materialized for it.
        let out_info = &g2.tensor_info[&10];
        assert_eq!(
            out_info
                .shape
                .iter()
                .map(|d| d.as_concrete())
                .collect::<Vec<_>>(),
            vec![None, None, Some(32), Some(64)],
            "resolved -1 concretizes the output shape per-consumer"
        );
        assert!(
            out_info.known_i64_values.is_none(),
            "a Reshape output's value channel must not carry its target shape"
        );
        assert!(
            !g2.params.contains_key(&10),
            "no inline param may be materialized for a data-carrying output"
        );
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
