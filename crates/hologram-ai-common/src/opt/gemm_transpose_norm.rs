use super::pipeline::Pass;
use crate::ir::{AiGraph, AiNode, AiOp, TensorId, TensorInfo};

/// Normalize `Gemm { trans_a/trans_b }` into explicit `Transpose` nodes.
///
/// hologram's lowering/runtime boundary does not carry transpose flags on the
/// final kernel call, so any surviving ONNX `Gemm` transpose attribute must be
/// realized structurally before lowering.
pub struct GemmTransposeNormalization;

impl Pass for GemmTransposeNormalization {
    fn name(&self) -> &str {
        "GemmTransposeNormalization"
    }

    fn should_run(&self, graph: &AiGraph) -> bool {
        graph.nodes.iter().any(|node| {
            matches!(
                node.op,
                AiOp::Gemm { trans_a: true, .. } | AiOp::Gemm { trans_b: true, .. }
            )
        })
    }

    fn run(&self, mut graph: AiGraph) -> anyhow::Result<AiGraph> {
        let mut next_tid = graph
            .tensor_info
            .keys()
            .chain(graph.params.keys())
            .max()
            .copied()
            .unwrap_or(0);
        let mut next_nid = graph.nodes.iter().map(|n| n.id).max().unwrap_or(0);
        let original_nodes = graph.nodes.clone();
        let mut rewritten = Vec::with_capacity(original_nodes.len());

        for node in &original_nodes {
            let AiOp::Gemm {
                alpha,
                beta,
                trans_a,
                trans_b,
            } = &node.op
            else {
                rewritten.push(node.clone());
                continue;
            };

            if !*trans_a && !*trans_b {
                rewritten.push(node.clone());
                continue;
            }

            let mut inputs = node.inputs.clone();
            if *trans_a {
                inputs[0] = insert_transpose(
                    &mut graph,
                    &mut rewritten,
                    &mut next_tid,
                    &mut next_nid,
                    node.inputs[0],
                )?;
            }
            if *trans_b {
                inputs[1] = insert_transpose(
                    &mut graph,
                    &mut rewritten,
                    &mut next_tid,
                    &mut next_nid,
                    node.inputs[1],
                )?;
            }

            let mut normalized = node.clone();
            normalized.inputs = inputs;
            normalized.op = AiOp::Gemm {
                alpha: *alpha,
                beta: *beta,
                trans_a: false,
                trans_b: false,
            };
            rewritten.push(normalized);
        }

        graph.nodes = rewritten;
        graph.invalidate_topo_cache();
        Ok(graph)
    }
}

fn insert_transpose(
    graph: &mut AiGraph,
    rewritten: &mut Vec<AiNode>,
    next_tid: &mut TensorId,
    next_nid: &mut u32,
    input_tid: TensorId,
) -> anyhow::Result<TensorId> {
    let input_info = graph.tensor_info.get(&input_tid).cloned().ok_or_else(|| {
        anyhow::anyhow!("transpose normalization: missing TensorInfo for {input_tid}")
    })?;
    let transposed_shape = swap_last_two_dims(&input_info)?;

    *next_tid += 1;
    *next_nid += 1;
    let out_tid = *next_tid;
    let out_nid = *next_nid;

    let mut out_info = input_info.clone();
    out_info.shape = transposed_shape;
    graph.tensor_info.insert(out_tid, out_info);

    if let Some(name) = graph.tensor_names.get(&input_tid).cloned() {
        graph.tensor_names.insert(out_tid, format!("{name}.T"));
    }

    rewritten.push(AiNode::new(
        out_nid,
        AiOp::Transpose { perm: vec![1, 0] },
        vec![input_tid],
        vec![out_tid],
    ));

    Ok(out_tid)
}

fn swap_last_two_dims(info: &TensorInfo) -> anyhow::Result<crate::Shape> {
    let mut shape = info.shape.clone();
    if shape.len() < 2 {
        anyhow::bail!(
            "transpose normalization requires rank >= 2, got {}",
            shape.len()
        );
    }
    let last = shape.len() - 1;
    shape.swap(last - 1, last);
    Ok(shape)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{shape_from_concrete, DType};
    use std::collections::HashMap;

    #[test]
    fn normalizes_transposed_gemm() {
        let mut tensor_info = HashMap::new();
        tensor_info.insert(0, TensorInfo::new(DType::F32, shape_from_concrete(&[2, 3])));
        tensor_info.insert(1, TensorInfo::new(DType::F32, shape_from_concrete(&[4, 3])));
        tensor_info.insert(2, TensorInfo::new(DType::F32, shape_from_concrete(&[4])));
        tensor_info.insert(3, TensorInfo::new(DType::F32, shape_from_concrete(&[2, 4])));

        let graph = AiGraph {
            name: "gemm_norm".into(),
            nodes: vec![AiNode::new(
                0,
                AiOp::Gemm {
                    alpha: 1.0,
                    beta: 1.0,
                    trans_a: false,
                    trans_b: true,
                },
                vec![0, 1, 2],
                vec![3],
            )],
            inputs: vec![0, 1, 2],
            outputs: vec![3],
            input_names: vec![],
            output_names: vec![],
            params: HashMap::new(),
            tensor_info,
            metadata: HashMap::new(),
            warnings: vec![],
            dim_vars: Default::default(),
            shape_constraints: Default::default(),
            subgraphs: HashMap::new(),
            tensor_names: HashMap::new(),
            topo_cache: std::sync::Mutex::new(None),
        };

        let out = GemmTransposeNormalization
            .run(graph)
            .expect("normalization should succeed");

        assert_eq!(out.nodes.len(), 2);
        assert!(matches!(out.nodes[0].op, AiOp::Transpose { .. }));
        match &out.nodes[1].op {
            AiOp::Gemm {
                trans_a, trans_b, ..
            } => {
                assert!(!*trans_a);
                assert!(!*trans_b);
            }
            other => panic!("expected normalized Gemm, got {other:?}"),
        }
    }
}
