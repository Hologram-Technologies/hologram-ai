//! Op decomposition pass: rewrite compound AiOps into primitive subgraphs.
//!
//! Runs before lowering to ensure all ops map to existing `FloatOp` variants.
//! Decompositions:
//! - `ReduceL1` → `Abs` + `ReduceSum`
//! - `ReduceL2` → `Mul(x,x)` + `ReduceSum` + `Sqrt`
//! - `DepthToSpace` → `Reshape` + `Transpose` + `Reshape`
//! - `SpaceToDepth` → `Reshape` + `Transpose` + `Reshape`

use super::pipeline::Pass;
use crate::ir::{AiGraph, AiNode, AiOp, DType, TensorId, TensorInfo};

pub struct OpDecomposition;

impl Pass for OpDecomposition {
    fn name(&self) -> &str {
        "OpDecomposition"
    }

    fn run(&self, mut graph: AiGraph) -> anyhow::Result<AiGraph> {
        // Find next free tensor ID and node ID.
        let mut next_tid: TensorId = graph
            .nodes
            .iter()
            .flat_map(|n| n.outputs.iter().copied())
            .max()
            .unwrap_or(0)
            .max(
                graph
                    .nodes
                    .iter()
                    .flat_map(|n| n.inputs.iter().copied())
                    .max()
                    .unwrap_or(0),
            )
            + 1;
        // Also account for params and graph inputs/outputs.
        if let Some(&max_param) = graph.params.keys().max() {
            next_tid = next_tid.max(max_param + 1);
        }
        if let Some(&max_inp) = graph.inputs.iter().max() {
            next_tid = next_tid.max(max_inp + 1);
        }
        if let Some(&max_out) = graph.outputs.iter().max() {
            next_tid = next_tid.max(max_out + 1);
        }

        let mut next_nid: u32 = graph.nodes.iter().map(|n| n.id).max().unwrap_or(0) + 1;

        let mut new_nodes: Vec<AiNode> = Vec::new();

        for node in graph.nodes.drain(..) {
            match &node.op {
                AiOp::ReduceL1 { axes, keepdims } => {
                    // ReduceL1(x) = ReduceSum(Abs(x))
                    let x = node.inputs[0];
                    let out = node.outputs[0];

                    // Abs node: x → abs_out
                    let abs_out = next_tid;
                    next_tid += 1;
                    let abs_nid = next_nid;
                    next_nid += 1;
                    new_nodes.push(AiNode::new(abs_nid, AiOp::Abs, vec![x], vec![abs_out]));
                    // Copy tensor info from input for intermediate.
                    if let Some(info) = graph.tensor_info.get(&x) {
                        graph.tensor_info.insert(abs_out, info.clone());
                    }

                    // ReduceSum node: abs_out → out
                    let reduce_nid = next_nid;
                    next_nid += 1;
                    new_nodes.push(AiNode::new(
                        reduce_nid,
                        AiOp::ReduceSum {
                            axes: axes.clone(),
                            keepdims: *keepdims,
                        },
                        vec![abs_out],
                        vec![out],
                    ));
                }
                AiOp::ReduceL2 { axes, keepdims } => {
                    // ReduceL2(x) = Sqrt(ReduceSum(Mul(x, x)))
                    let x = node.inputs[0];
                    let out = node.outputs[0];

                    // Mul node: x * x → sq_out
                    let sq_out = next_tid;
                    next_tid += 1;
                    let sq_nid = next_nid;
                    next_nid += 1;
                    new_nodes.push(AiNode::new(sq_nid, AiOp::Mul, vec![x, x], vec![sq_out]));
                    if let Some(info) = graph.tensor_info.get(&x) {
                        graph.tensor_info.insert(sq_out, info.clone());
                    }

                    // ReduceSum node: sq_out → sum_out
                    let sum_out = next_tid;
                    next_tid += 1;
                    let sum_nid = next_nid;
                    next_nid += 1;
                    new_nodes.push(AiNode::new(
                        sum_nid,
                        AiOp::ReduceSum {
                            axes: axes.clone(),
                            keepdims: *keepdims,
                        },
                        vec![sq_out],
                        vec![sum_out],
                    ));
                    // Infer reduced shape from output tensor info if available.
                    if let Some(info) = graph.tensor_info.get(&out) {
                        graph.tensor_info.insert(sum_out, info.clone());
                    }

                    // Sqrt node: sum_out → out
                    let sqrt_nid = next_nid;
                    next_nid += 1;
                    new_nodes.push(AiNode::new(
                        sqrt_nid,
                        AiOp::Sqrt,
                        vec![sum_out],
                        vec![out],
                    ));
                }
                AiOp::DepthToSpace { blocksize, .. } => {
                    // DepthToSpace: [N, C, H, W] → [N, C/bs², H*bs, W*bs]
                    // Decomposition: Reshape → Transpose → Reshape
                    let x = node.inputs[0];
                    let out = node.outputs[0];
                    let bs = *blocksize;

                    let in_info = graph.tensor_info.get(&x);
                    if let Some(info) = in_info {
                        if info.shape.len() == 4 {
                            if let (Some(n), Some(c), Some(h), Some(w)) = (
                                info.shape[0].as_concrete(),
                                info.shape[1].as_concrete(),
                                info.shape[2].as_concrete(),
                                info.shape[3].as_concrete(),
                            ) {
                                let c_out = c / (bs * bs);
                                // Step 1: Reshape [N, C, H, W] → [N, C/bs², bs, bs, H, W]
                                let r1_out = next_tid;
                                next_tid += 1;
                                let r1_nid = next_nid;
                                next_nid += 1;
                                let r1_shape = [n, c_out, bs, bs, h, w];
                                graph.tensor_info.insert(
                                    r1_out,
                                    TensorInfo::new(
                                        DType::F32,
                                        r1_shape.iter().map(|&d| crate::ir::DimExpr::Concrete(d)).collect(),
                                    ),
                                );
                                new_nodes.push(AiNode::new(
                                    r1_nid,
                                    AiOp::Reshape { allow_zero: false },
                                    vec![x],
                                    vec![r1_out],
                                ));

                                // Step 2: Transpose [N, C', bs, bs, H, W] → [N, C', H, bs, W, bs]
                                let t_out = next_tid;
                                next_tid += 1;
                                let t_nid = next_nid;
                                next_nid += 1;
                                let t_shape = [n, c_out, h, bs, w, bs];
                                graph.tensor_info.insert(
                                    t_out,
                                    TensorInfo::new(
                                        DType::F32,
                                        t_shape.iter().map(|&d| crate::ir::DimExpr::Concrete(d)).collect(),
                                    ),
                                );
                                new_nodes.push(AiNode::new(
                                    t_nid,
                                    AiOp::Transpose {
                                        perm: vec![0, 1, 4, 2, 5, 3],
                                    },
                                    vec![r1_out],
                                    vec![t_out],
                                ));

                                // Step 3: Reshape [N, C', H, bs, W, bs] → [N, C', H*bs, W*bs]
                                let r2_nid = next_nid;
                                next_nid += 1;
                                new_nodes.push(AiNode::new(
                                    r2_nid,
                                    AiOp::Reshape { allow_zero: false },
                                    vec![t_out],
                                    vec![out],
                                ));
                                continue;
                            }
                        }
                    }
                    // Fallback: keep as-is (will fail at lowering with Unsupported)
                    new_nodes.push(node);
                    continue;
                }
                AiOp::SpaceToDepth { blocksize } => {
                    // SpaceToDepth: [N, C, H, W] → [N, C*bs², H/bs, W/bs]
                    // Decomposition: Reshape → Transpose → Reshape
                    let x = node.inputs[0];
                    let out = node.outputs[0];
                    let bs = *blocksize;

                    let in_info = graph.tensor_info.get(&x);
                    if let Some(info) = in_info {
                        if info.shape.len() == 4 {
                            if let (Some(n), Some(c), Some(h), Some(w)) = (
                                info.shape[0].as_concrete(),
                                info.shape[1].as_concrete(),
                                info.shape[2].as_concrete(),
                                info.shape[3].as_concrete(),
                            ) {
                                let h_out = h / bs;
                                let w_out = w / bs;
                                // Step 1: Reshape [N, C, H, W] → [N, C, H/bs, bs, W/bs, bs]
                                let r1_out = next_tid;
                                next_tid += 1;
                                let r1_nid = next_nid;
                                next_nid += 1;
                                let r1_shape = [n, c, h_out, bs, w_out, bs];
                                graph.tensor_info.insert(
                                    r1_out,
                                    TensorInfo::new(
                                        DType::F32,
                                        r1_shape.iter().map(|&d| crate::ir::DimExpr::Concrete(d)).collect(),
                                    ),
                                );
                                new_nodes.push(AiNode::new(
                                    r1_nid,
                                    AiOp::Reshape { allow_zero: false },
                                    vec![x],
                                    vec![r1_out],
                                ));

                                // Step 2: Transpose [N, C, H/bs, bs, W/bs, bs] → [N, C, bs, bs, H/bs, W/bs]
                                let t_out = next_tid;
                                next_tid += 1;
                                let t_nid = next_nid;
                                next_nid += 1;
                                let t_shape = [n, c, bs, bs, h_out, w_out];
                                graph.tensor_info.insert(
                                    t_out,
                                    TensorInfo::new(
                                        DType::F32,
                                        t_shape.iter().map(|&d| crate::ir::DimExpr::Concrete(d)).collect(),
                                    ),
                                );
                                new_nodes.push(AiNode::new(
                                    t_nid,
                                    AiOp::Transpose {
                                        perm: vec![0, 1, 3, 5, 2, 4],
                                    },
                                    vec![r1_out],
                                    vec![t_out],
                                ));

                                // Step 3: Reshape [N, C, bs, bs, H/bs, W/bs] → [N, C*bs², H/bs, W/bs]
                                let r2_nid = next_nid;
                                next_nid += 1;
                                new_nodes.push(AiNode::new(
                                    r2_nid,
                                    AiOp::Reshape { allow_zero: false },
                                    vec![t_out],
                                    vec![out],
                                ));
                                continue;
                            }
                        }
                    }
                    // Fallback: keep as-is
                    new_nodes.push(node);
                    continue;
                }
                _ => {
                    new_nodes.push(node);
                }
            }
        }

        graph.nodes = new_nodes;
        Ok(graph)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{shape_from_concrete, AiGraph, AiNode, AiOp, DType, TensorInfo};
    use std::collections::HashMap;

    #[test]
    fn reduce_l1_decomposition() {
        let mut ti = HashMap::new();
        ti.insert(0u32, TensorInfo::new(DType::F32, shape_from_concrete(&[2, 4])));
        ti.insert(1u32, TensorInfo::new(DType::F32, shape_from_concrete(&[2, 1])));

        let g = AiGraph {
            name: "test".into(),
            nodes: vec![AiNode::new(
                0,
                AiOp::ReduceL1 {
                    axes: vec![-1],
                    keepdims: true,
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
            subgraphs: HashMap::new(),
            tensor_names: HashMap::new(),
        };

        let pass = OpDecomposition;
        let g2 = pass.run(g).unwrap();
        assert_eq!(g2.nodes.len(), 2); // Abs + ReduceSum
        assert!(matches!(g2.nodes[0].op, AiOp::Abs));
        assert!(matches!(g2.nodes[1].op, AiOp::ReduceSum { .. }));
    }

    #[test]
    fn reduce_l2_decomposition() {
        let mut ti = HashMap::new();
        ti.insert(0u32, TensorInfo::new(DType::F32, shape_from_concrete(&[2, 4])));
        ti.insert(1u32, TensorInfo::new(DType::F32, shape_from_concrete(&[2, 1])));

        let g = AiGraph {
            name: "test".into(),
            nodes: vec![AiNode::new(
                0,
                AiOp::ReduceL2 {
                    axes: vec![-1],
                    keepdims: true,
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
            subgraphs: HashMap::new(),
            tensor_names: HashMap::new(),
        };

        let pass = OpDecomposition;
        let g2 = pass.run(g).unwrap();
        assert_eq!(g2.nodes.len(), 3); // Mul + ReduceSum + Sqrt
        assert!(matches!(g2.nodes[0].op, AiOp::Mul));
        assert!(matches!(g2.nodes[1].op, AiOp::ReduceSum { .. }));
        assert!(matches!(g2.nodes[2].op, AiOp::Sqrt));
    }
}
