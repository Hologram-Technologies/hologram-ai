use super::{graph_utils::next_node_id, pipeline::Pass};
use crate::ir::{
    shape_from_concrete, AiGraph, AiNode, AiOp, AiOpSemantics, AiParam, DType, Shape,
    SymbolicShapeExt, TensorId, TensorInfo,
};

pub struct ExplicitBroadcastBinary;

impl Pass for ExplicitBroadcastBinary {
    fn name(&self) -> &str {
        "ExplicitBroadcastBinary"
    }

    fn should_run(&self, graph: &AiGraph) -> bool {
        graph.nodes.iter().any(|node| {
            node.op.is_broadcast_binary()
                && node
                    .outputs
                    .first()
                    .and_then(|tid| graph.tensor_info.get(tid))
                    .is_some_and(|info| {
                        let out_shape = &info.shape;
                        node.inputs.iter().any(|input_tid| {
                            graph.tensor_info.get(input_tid).is_some_and(|input_info| {
                                input_info.shape != *out_shape
                                    && input_info.shape.as_slice().is_concrete()
                                    && out_shape.as_slice().is_concrete()
                            })
                        })
                    })
        })
    }

    fn run(&self, mut graph: AiGraph) -> anyhow::Result<AiGraph> {
        let mut next_tid = next_tensor_id(&graph);
        let mut next_nid = next_node_id(&graph);
        let original = std::mem::take(&mut graph.nodes);
        let mut rewritten = Vec::with_capacity(original.len());

        for mut node in original {
            if !node.op.is_broadcast_binary() || node.inputs.len() < 2 {
                rewritten.push(node);
                continue;
            }

            let Some(output_tid) = node.outputs.first().copied() else {
                rewritten.push(node);
                continue;
            };
            let Some(output_shape) = graph
                .tensor_info
                .get(&output_tid)
                .map(|info| info.shape.clone())
            else {
                rewritten.push(node);
                continue;
            };
            if !output_shape.as_slice().is_concrete() {
                rewritten.push(node);
                continue;
            }

            for input_tid in &mut node.inputs {
                let Some(input_shape) = graph
                    .tensor_info
                    .get(input_tid)
                    .map(|info| info.shape.clone())
                else {
                    continue;
                };
                if input_shape == output_shape || !input_shape.as_slice().is_concrete() {
                    continue;
                }
                if input_shape
                    .as_slice()
                    .broadcast_to(output_shape.as_slice())
                    .is_none()
                {
                    continue;
                }

                let output_dims = output_shape
                    .as_slice()
                    .concrete_dims()
                    .expect("explicit broadcast requires concrete output shape");
                let input_dims = input_shape
                    .as_slice()
                    .concrete_dims()
                    .expect("explicit broadcast requires concrete input shape");
                let aligned_input_dims = align_input_dims(&input_dims, output_dims.len());
                let mut current_tid = *input_tid;

                if aligned_input_dims != input_dims {
                    let reshape_shape_tid = next_tid;
                    next_tid += 1;
                    let reshaped_tid = next_tid;
                    next_tid += 1;
                    insert_shape_param(&mut graph, reshape_shape_tid, &aligned_input_dims);
                    clone_tensor_info(
                        &mut graph,
                        current_tid,
                        reshaped_tid,
                        shape_from_concrete(&aligned_input_dims),
                    );
                    graph
                        .tensor_names
                        .insert(reshaped_tid, format!("tensor_{current_tid}.reshape"));
                    rewritten.push(AiNode::new(
                        next_nid,
                        AiOp::Reshape { allow_zero: false },
                        vec![current_tid, reshape_shape_tid],
                        vec![reshaped_tid],
                    ));
                    next_nid += 1;
                    current_tid = reshaped_tid;
                }

                if aligned_input_dims == output_dims {
                    *input_tid = current_tid;
                    continue;
                }

                let shape_tid = next_tid;
                next_tid += 1;
                let expanded_tid = next_tid;
                next_tid += 1;
                insert_shape_param(&mut graph, shape_tid, &output_dims);
                clone_tensor_info(&mut graph, current_tid, expanded_tid, output_shape.clone());
                graph
                    .tensor_names
                    .insert(expanded_tid, format!("tensor_{current_tid}.broadcast"));

                rewritten.push(AiNode::new(
                    next_nid,
                    AiOp::Expand,
                    vec![current_tid, shape_tid],
                    vec![expanded_tid],
                ));
                next_nid += 1;
                *input_tid = expanded_tid;
            }

            rewritten.push(node);
        }

        graph.nodes = rewritten;
        graph.invalidate_topo_cache();
        Ok(graph)
    }
}

fn next_tensor_id(graph: &AiGraph) -> TensorId {
    let mut next_tid = graph
        .nodes
        .iter()
        .flat_map(|node| node.inputs.iter().chain(node.outputs.iter()))
        .copied()
        .max()
        .unwrap_or(0)
        + 1;
    if let Some(&max_param) = graph.params.keys().max() {
        next_tid = next_tid.max(max_param + 1);
    }
    if let Some(&max_input) = graph.inputs.iter().max() {
        next_tid = next_tid.max(max_input + 1);
    }
    if let Some(&max_output) = graph.outputs.iter().max() {
        next_tid = next_tid.max(max_output + 1);
    }
    next_tid
}

fn align_input_dims(input_dims: &[u64], target_rank: usize) -> Vec<u64> {
    let mut aligned = vec![1; target_rank.saturating_sub(input_dims.len())];
    aligned.extend_from_slice(input_dims);
    aligned
}

fn insert_shape_param(graph: &mut AiGraph, tid: TensorId, dims: &[u64]) {
    let mut shape_info = TensorInfo::new(DType::INT64, shape_from_concrete(&[dims.len() as u64]));
    let shape_values: Vec<i64> = dims.iter().map(|&dim| dim as i64).collect();
    shape_info.known_i64_values = Some(shape_values.iter().copied().map(Some).collect());
    let shape_bytes: Vec<u8> = shape_values.iter().flat_map(|v| v.to_le_bytes()).collect();
    graph
        .params
        .insert(tid, AiParam::inline(shape_bytes, shape_info.clone()));
    graph.tensor_info.insert(tid, shape_info);
    graph
        .tensor_names
        .insert(tid, format!("broadcast_shape_{tid}"));
}

fn clone_tensor_info(graph: &mut AiGraph, src_tid: TensorId, dst_tid: TensorId, shape: Shape) {
    if let Some(mut info) = graph.tensor_info.get(&src_tid).cloned() {
        info.shape = shape;
        info.known_i64_values = None;
        graph.tensor_info.insert(dst_tid, info);
    }
}

#[cfg(test)]
mod tests {
    use super::ExplicitBroadcastBinary;
    use crate::ir::{
        shape_from_concrete, AiGraph, AiNode, AiOp, ConstraintStore, DType, DimVarTable, TensorInfo,
    };
    use crate::Pass;
    use std::collections::HashMap;

    #[test]
    fn inserts_expand_for_bias_add() {
        let mut graph = AiGraph {
            name: "broadcast_add".into(),
            nodes: vec![AiNode::new(0, AiOp::Add, vec![0, 1], vec![2])],
            inputs: vec![0],
            outputs: vec![2],
            input_names: vec![],
            output_names: vec![],
            params: HashMap::new(),
            tensor_info: HashMap::new(),
            metadata: HashMap::new(),
            warnings: vec![],
            dim_vars: DimVarTable::default(),
            shape_constraints: ConstraintStore::default(),
            subgraphs: HashMap::new(),
            tensor_names: HashMap::new(),
            topo_cache: Default::default(),
        };
        graph
            .tensor_info
            .insert(0, TensorInfo::new(DType::F32, shape_from_concrete(&[768])));
        graph.tensor_info.insert(
            1,
            TensorInfo::new(DType::F32, shape_from_concrete(&[1, 8, 768])),
        );
        graph.tensor_info.insert(
            2,
            TensorInfo::new(DType::F32, shape_from_concrete(&[1, 8, 768])),
        );

        let out = ExplicitBroadcastBinary.run(graph).expect("pass succeeds");
        assert_eq!(out.nodes.len(), 3);
        assert!(matches!(
            out.nodes[0].op,
            AiOp::Reshape { allow_zero: false }
        ));
        assert!(matches!(out.nodes[1].op, AiOp::Expand));
        assert!(matches!(out.nodes[2].op, AiOp::Add));
        assert_ne!(out.nodes[2].inputs[0], 0);
    }

    #[test]
    fn inserts_only_expand_for_same_rank_broadcast() {
        let mut graph = AiGraph {
            name: "broadcast_mul".into(),
            nodes: vec![AiNode::new(0, AiOp::Mul, vec![0, 1], vec![2])],
            inputs: vec![0],
            outputs: vec![2],
            input_names: vec![],
            output_names: vec![],
            params: HashMap::new(),
            tensor_info: HashMap::new(),
            metadata: HashMap::new(),
            warnings: vec![],
            dim_vars: DimVarTable::default(),
            shape_constraints: ConstraintStore::default(),
            subgraphs: HashMap::new(),
            tensor_names: HashMap::new(),
            topo_cache: Default::default(),
        };
        graph.tensor_info.insert(
            0,
            TensorInfo::new(DType::F32, shape_from_concrete(&[1, 1, 768])),
        );
        graph.tensor_info.insert(
            1,
            TensorInfo::new(DType::F32, shape_from_concrete(&[1, 8, 768])),
        );
        graph.tensor_info.insert(
            2,
            TensorInfo::new(DType::F32, shape_from_concrete(&[1, 8, 768])),
        );

        let out = ExplicitBroadcastBinary.run(graph).expect("pass succeeds");
        assert_eq!(out.nodes.len(), 2);
        assert!(matches!(out.nodes[0].op, AiOp::Expand));
        assert!(matches!(out.nodes[1].op, AiOp::Mul));
    }
}
