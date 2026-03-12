//! Compile-time shape consistency validation pass.
//!
//! Runs after concretization to catch shape errors before they reach the
//! executor. Validates weight/shape consistency, MatMul inner dimensions,
//! and dynamic dim resolution.
//!
//! This is a **read-only** pass — it does not modify the graph. Errors are
//! collected and returned as warnings on the graph. Zero runtime cost.

use super::pipeline::Pass;
use crate::ir::graph::ImportWarning;
use crate::ir::node::TensorId;
use crate::ir::op::AiOp;
use crate::ir::shape::DimExpr;
use crate::ir::AiGraph;

/// Validate shape consistency of a fully-concretized AiGraph.
///
/// Checks:
/// 1. Weight tensors match their declared shapes
/// 2. MatMul/Gemm inner dimensions match
/// 3. No remaining Dynamic dims after concretization
/// 4. No zero-product shapes (except graph inputs which may be dynamic)
pub struct ShapeConsistencyCheck;

impl Pass for ShapeConsistencyCheck {
    fn name(&self) -> &str {
        "ShapeConsistencyCheck"
    }

    fn run(&self, mut graph: AiGraph) -> anyhow::Result<AiGraph> {
        let errors = validate_shape_consistency(&graph);
        for err in errors {
            graph.warnings.push(ImportWarning {
                message: format!("[shape-consistency] {}", err.message),
                node_name: err.node_name,
            });
        }
        Ok(graph)
    }
}

/// A shape consistency error found during validation.
#[derive(Debug, Clone)]
pub struct ShapeError {
    pub message: String,
    pub node_name: Option<String>,
}

/// Run all shape consistency checks on the graph.
pub fn validate_shape_consistency(graph: &AiGraph) -> Vec<ShapeError> {
    let mut errors = Vec::new();

    check_param_shapes(graph, &mut errors);
    check_matmul_dims(graph, &mut errors);
    check_dynamic_dims(graph, &mut errors);
    check_zero_product_shapes(graph, &mut errors);

    errors
}

/// Check 1: Weight tensor byte sizes match their declared shapes.
fn check_param_shapes(graph: &AiGraph, errors: &mut Vec<ShapeError>) {
    for (&tid, param) in &graph.params {
        let info = param.info();

        // Try to evaluate shape to concrete dimensions.
        let concrete_dims: Option<Vec<u64>> = info
            .shape
            .iter()
            .map(|d| d.evaluate())
            .collect();

        let concrete_dims = match concrete_dims {
            Some(dims) => dims,
            None => continue, // Shape has symbolic dims — can't validate statically.
        };

        let shape_product: u64 = concrete_dims.iter().product();

        // Get byte size of dtype.
        let elem_bytes = match info.storage_dtype.byte_size() {
            Some(b) => b as u64,
            None => continue, // Sub-byte type (INT4) — skip for now.
        };

        let expected_bytes = shape_product * elem_bytes;

        let actual_bytes = match param {
            crate::ir::param::AiParam::Inline { data, .. } => data.len() as u64,
            crate::ir::param::AiParam::Mmap { len, .. } => *len,
        };

        if expected_bytes > 0 && actual_bytes > 0 && expected_bytes != actual_bytes {
            let shape_str: Vec<String> = concrete_dims.iter().map(|d| d.to_string()).collect();
            errors.push(ShapeError {
                message: format!(
                    "param tid={tid}: shape [{}] ({:?}) expects {expected_bytes} bytes, \
                     got {actual_bytes} bytes (ratio: {:.2}x)",
                    shape_str.join(", "),
                    info.storage_dtype,
                    actual_bytes as f64 / expected_bytes as f64,
                ),
                node_name: None,
            });
        }
    }
}

/// Check 2: MatMul/Gemm inner dimensions match between inputs.
fn check_matmul_dims(graph: &AiGraph, errors: &mut Vec<ShapeError>) {
    for node in &graph.nodes {
        match &node.op {
            AiOp::MatMul | AiOp::BatchMatMul => {
                if node.inputs.len() < 2 {
                    continue;
                }
                check_matmul_inner_dim(
                    graph,
                    node.id,
                    node.inputs[0],
                    node.inputs[1],
                    false,
                    false,
                    errors,
                );
            }
            AiOp::Gemm {
                trans_a, trans_b, ..
            } => {
                if node.inputs.len() < 2 {
                    continue;
                }
                check_matmul_inner_dim(
                    graph,
                    node.id,
                    node.inputs[0],
                    node.inputs[1],
                    *trans_a,
                    *trans_b,
                    errors,
                );
            }
            _ => {}
        }

        // Also check output shape consistency for MatMul.
        if matches!(node.op, AiOp::MatMul | AiOp::BatchMatMul) && !node.outputs.is_empty() {
            check_matmul_output_shape(graph, node.id, &node.inputs, node.outputs[0], errors);
        }
    }
}

/// Verify that the inner dimension (k) matches between two MatMul inputs.
fn check_matmul_inner_dim(
    graph: &AiGraph,
    node_id: u32,
    lhs_tid: TensorId,
    rhs_tid: TensorId,
    trans_a: bool,
    trans_b: bool,
    errors: &mut Vec<ShapeError>,
) {
    let lhs_shape = match graph.tensor_info.get(&lhs_tid) {
        Some(info) => &info.shape,
        None => return,
    };
    let rhs_shape = match graph.tensor_info.get(&rhs_tid) {
        Some(info) => &info.shape,
        None => return,
    };

    if lhs_shape.is_empty() || rhs_shape.is_empty() {
        return;
    }

    // For A: k is last dim (or second-to-last if transposed)
    let lhs_k_idx = if trans_a {
        lhs_shape.len().saturating_sub(2)
    } else {
        lhs_shape.len() - 1
    };

    // For B: k is second-to-last dim (or last if transposed)
    let rhs_k_idx = if trans_b {
        rhs_shape.len() - 1
    } else {
        rhs_shape.len().saturating_sub(2)
    };

    let lhs_k = lhs_shape.get(lhs_k_idx).and_then(|d| d.evaluate());
    let rhs_k = rhs_shape.get(rhs_k_idx).and_then(|d| d.evaluate());

    if let (Some(lk), Some(rk)) = (lhs_k, rhs_k) {
        if lk != rk {
            let lhs_dims: Vec<String> = lhs_shape.iter().map(|d| format!("{d:?}")).collect();
            let rhs_dims: Vec<String> = rhs_shape.iter().map(|d| format!("{d:?}")).collect();
            errors.push(ShapeError {
                message: format!(
                    "node {node_id} MatMul inner dim mismatch: \
                     A[{}] k={lk} (idx {lhs_k_idx}) vs B[{}] k={rk} (idx {rhs_k_idx})\
                     {}",
                    lhs_dims.join(", "),
                    rhs_dims.join(", "),
                    if trans_a || trans_b {
                        format!(" (trans_a={trans_a}, trans_b={trans_b})")
                    } else {
                        String::new()
                    },
                ),
                node_name: Some(format!("node_{node_id}")),
            });
        }
    }
}

/// Verify MatMul output shape is consistent with input shapes.
fn check_matmul_output_shape(
    graph: &AiGraph,
    node_id: u32,
    input_tids: &[TensorId],
    output_tid: TensorId,
    errors: &mut Vec<ShapeError>,
) {
    if input_tids.len() < 2 {
        return;
    }

    let lhs_info = match graph.tensor_info.get(&input_tids[0]) {
        Some(info) => info,
        None => return,
    };
    let rhs_info = match graph.tensor_info.get(&input_tids[1]) {
        Some(info) => info,
        None => return,
    };
    let out_info = match graph.tensor_info.get(&output_tid) {
        Some(info) => info,
        None => return,
    };

    // For 2D MatMul: [m,k] x [k,n] -> [m,n]
    // For batched: [...,m,k] x [...,k,n] -> [...,m,n]
    let lhs = &lhs_info.shape;
    let rhs = &rhs_info.shape;
    let out = &out_info.shape;

    if lhs.len() < 2 || rhs.len() < 2 || out.is_empty() {
        return;
    }

    // Check output's last dim matches rhs's last dim (n)
    let rhs_n = rhs.last().and_then(|d| d.evaluate());
    let out_n = out.last().and_then(|d| d.evaluate());
    if let (Some(rn), Some(on)) = (rhs_n, out_n) {
        if rn != on {
            errors.push(ShapeError {
                message: format!(
                    "node {node_id} MatMul output last dim {on} != B last dim {rn}",
                ),
                node_name: Some(format!("node_{node_id}")),
            });
        }
    }

    // Check output's second-to-last dim matches lhs's second-to-last dim (m)
    if out.len() >= 2 && lhs.len() >= 2 {
        let lhs_m = lhs.get(lhs.len() - 2).and_then(|d| d.evaluate());
        let out_m = out.get(out.len() - 2).and_then(|d| d.evaluate());
        if let (Some(lm), Some(om)) = (lhs_m, out_m) {
            if lm != om {
                errors.push(ShapeError {
                    message: format!(
                        "node {node_id} MatMul output M dim {om} != A M dim {lm}",
                    ),
                    node_name: Some(format!("node_{node_id}")),
                });
            }
        }
    }
}

/// Check 3: No remaining Dynamic dims after concretization.
fn check_dynamic_dims(graph: &AiGraph, errors: &mut Vec<ShapeError>) {
    // Skip graph inputs — they may legitimately have dynamic dims.
    let input_tids: std::collections::HashSet<TensorId> =
        graph.inputs.iter().copied().collect();

    for (&tid, info) in &graph.tensor_info {
        if input_tids.contains(&tid) {
            continue;
        }

        for (dim_idx, dim) in info.shape.iter().enumerate() {
            if matches!(dim, DimExpr::Dynamic) {
                errors.push(ShapeError {
                    message: format!(
                        "tid={tid} has Dynamic dim at index {dim_idx} after concretization \
                         (shape: {:?})",
                        info.shape,
                    ),
                    node_name: None,
                });
            }
        }
    }
}

/// Check 4: No zero-product shapes (except graph inputs and shape-computation tensors).
fn check_zero_product_shapes(graph: &AiGraph, errors: &mut Vec<ShapeError>) {
    let input_tids: std::collections::HashSet<TensorId> =
        graph.inputs.iter().copied().collect();

    for (&tid, info) in &graph.tensor_info {
        if input_tids.contains(&tid) {
            continue;
        }

        // Skip scalar tensors (empty shape = scalar, product = 1).
        if info.shape.is_empty() {
            continue;
        }

        let concrete_dims: Option<Vec<u64>> = info
            .shape
            .iter()
            .map(|d| d.evaluate())
            .collect();

        if let Some(dims) = concrete_dims {
            let product: u64 = dims.iter().product();
            if product == 0 && !dims.is_empty() {
                let shape_str: Vec<String> = dims.iter().map(|d| d.to_string()).collect();
                errors.push(ShapeError {
                    message: format!(
                        "tid={tid} has zero-product shape [{}] ({:?})",
                        shape_str.join(", "),
                        info.logical_dtype,
                    ),
                    node_name: None,
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::graph::TensorInfo;
    use crate::ir::node::AiNode;
    use crate::ir::shape::{shape_from_concrete, ConstraintStore, DimVarTable};
    use crate::ir::{dtype::DType, param::AiParam};
    use std::collections::HashMap;

    fn make_graph() -> AiGraph {
        AiGraph {
            name: "test".into(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
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
        }
    }

    #[test]
    fn param_shape_mismatch_detected() {
        let mut g = make_graph();
        let info = TensorInfo::new(DType::F32, shape_from_concrete(&[4, 8]));
        // 4*8*4 = 128 bytes expected, but provide 64 bytes.
        g.params.insert(10, AiParam::inline(vec![0u8; 64], info.clone()));
        g.tensor_info.insert(10, info);

        let errs = validate_shape_consistency(&g);
        assert!(!errs.is_empty(), "should detect param shape mismatch");
        assert!(errs[0].message.contains("expects 128 bytes"));
        assert!(errs[0].message.contains("got 64 bytes"));
    }

    #[test]
    fn param_shape_match_no_error() {
        let mut g = make_graph();
        let info = TensorInfo::new(DType::F32, shape_from_concrete(&[4, 8]));
        g.params.insert(10, AiParam::inline(vec![0u8; 128], info.clone()));
        g.tensor_info.insert(10, info);

        let errs = validate_shape_consistency(&g);
        assert!(errs.is_empty(), "correct param should not error: {errs:?}");
    }

    #[test]
    fn matmul_k_mismatch_detected() {
        let mut g = make_graph();
        // A: [2, 64], B: [128, 32] — k=64 vs k=128, mismatch!
        g.tensor_info
            .insert(0, TensorInfo::new(DType::F32, shape_from_concrete(&[2, 64])));
        g.tensor_info
            .insert(1, TensorInfo::new(DType::F32, shape_from_concrete(&[128, 32])));
        g.tensor_info
            .insert(2, TensorInfo::new(DType::F32, shape_from_concrete(&[2, 32])));
        g.nodes.push(AiNode::new(0, AiOp::MatMul, vec![0, 1], vec![2]));

        let errs = validate_shape_consistency(&g);
        assert!(
            errs.iter().any(|e| e.message.contains("inner dim mismatch")),
            "should detect k mismatch: {errs:?}"
        );
    }

    #[test]
    fn matmul_k_match_no_error() {
        let mut g = make_graph();
        // A: [2, 64], B: [64, 32] — k=64, matches
        g.tensor_info
            .insert(0, TensorInfo::new(DType::F32, shape_from_concrete(&[2, 64])));
        g.tensor_info
            .insert(1, TensorInfo::new(DType::F32, shape_from_concrete(&[64, 32])));
        g.tensor_info
            .insert(2, TensorInfo::new(DType::F32, shape_from_concrete(&[2, 32])));
        g.nodes.push(AiNode::new(0, AiOp::MatMul, vec![0, 1], vec![2]));

        let errs = validate_shape_consistency(&g);
        let matmul_errs: Vec<_> = errs
            .iter()
            .filter(|e| e.message.contains("MatMul"))
            .collect();
        assert!(
            matmul_errs.is_empty(),
            "matching k should not error: {matmul_errs:?}"
        );
    }

    #[test]
    fn dynamic_dim_detected() {
        let mut g = make_graph();
        let mut info = TensorInfo::new(DType::F32, shape_from_concrete(&[1, 64]));
        info.shape[0] = DimExpr::Dynamic;
        g.tensor_info.insert(5, info);

        let errs = validate_shape_consistency(&g);
        assert!(
            errs.iter().any(|e| e.message.contains("Dynamic dim")),
            "should detect Dynamic dim: {errs:?}"
        );
    }

    #[test]
    fn zero_product_shape_detected() {
        let mut g = make_graph();
        g.tensor_info
            .insert(5, TensorInfo::new(DType::F32, shape_from_concrete(&[1, 0, 64])));

        let errs = validate_shape_consistency(&g);
        assert!(
            errs.iter().any(|e| e.message.contains("zero-product")),
            "should detect zero-product shape: {errs:?}"
        );
    }

    #[test]
    fn graph_input_dynamic_dims_skipped() {
        let mut g = make_graph();
        let mut info = TensorInfo::new(DType::F32, shape_from_concrete(&[1, 64]));
        info.shape[0] = DimExpr::Dynamic;
        g.tensor_info.insert(0, info);
        g.inputs.push(0); // Mark as graph input — should be skipped.

        let errs = validate_shape_consistency(&g);
        let dynamic_errs: Vec<_> = errs.iter().filter(|e| e.message.contains("Dynamic")).collect();
        assert!(
            dynamic_errs.is_empty(),
            "graph input dynamic dims should be skipped: {dynamic_errs:?}"
        );
    }

    #[test]
    fn gemm_transb_k_check() {
        let mut g = make_graph();
        // Gemm with trans_b=true: A=[2,64], B=[32,64] (stored as [N,K])
        // k should be A's last dim = 64, B's last dim (trans_b) = 64 — match
        g.tensor_info
            .insert(0, TensorInfo::new(DType::F32, shape_from_concrete(&[2, 64])));
        g.tensor_info
            .insert(1, TensorInfo::new(DType::F32, shape_from_concrete(&[32, 64])));
        g.tensor_info
            .insert(2, TensorInfo::new(DType::F32, shape_from_concrete(&[2, 32])));
        g.tensor_info
            .insert(3, TensorInfo::new(DType::F32, shape_from_concrete(&[2, 32])));
        g.nodes.push(AiNode::new(
            0,
            AiOp::Gemm {
                alpha: 1.0,
                beta: 1.0,
                trans_a: false,
                trans_b: true,
            },
            vec![0, 1, 2],
            vec![3],
        ));

        let errs = validate_shape_consistency(&g);
        let matmul_errs: Vec<_> = errs
            .iter()
            .filter(|e| e.message.contains("inner dim"))
            .collect();
        assert!(
            matmul_errs.is_empty(),
            "Gemm trans_b=true with matching k should not error: {matmul_errs:?}"
        );
    }
}
