//! End-to-end pipeline integration tests.
//!
//! These tests verify the full import → optimize → lower → compile → execute path.
//! Sprint 001 focus: ONNX import through to AiGraph validation.

use std::collections::HashMap;
use hologram_ai_common::{
    AiGraph, AiNode, AiOp, DType, TensorInfo, OptPipeline, MemoryPlanner,
    shape_from_concrete, QuantDescriptor,
};
use hologram_ai_onnx::{import_onnx, OnnxImportOptions};

// ── ONNX import tests ─────────────────────────────────────────────────────────

#[test]
fn onnx_identity_imports_cleanly() {
    let bytes = std::fs::read("tests/fixtures/onnx/identity.onnx")
        .expect("identity.onnx fixture missing — run scripts/gen-fixtures.py");
    let graph = import_onnx(&bytes, OnnxImportOptions::default()).expect("import failed");
    assert!(graph.validate().is_empty(), "unexpected validation errors: {:?}", graph.validate());
}

#[test]
fn onnx_tiny_mlp_imports_cleanly() {
    let bytes = std::fs::read("tests/fixtures/onnx/tiny-mlp.onnx")
        .expect("tiny-mlp.onnx fixture missing — run scripts/gen-fixtures.py");
    let graph = import_onnx(&bytes, OnnxImportOptions::default()).expect("import failed");
    let errs = graph.validate();
    assert!(errs.is_empty(), "validation errors: {:?}", errs);
    // Should have at least 2 nodes: Gather + MatMul.
    assert!(graph.nodes.len() >= 2, "expected >= 2 nodes, got {}", graph.nodes.len());
}

// ── Optimization pass tests ───────────────────────────────────────────────────

#[test]
fn opt_pipeline_mvp_runs_on_tiny_mlp() {
    let bytes = std::fs::read("tests/fixtures/onnx/tiny-mlp.onnx")
        .expect("tiny-mlp.onnx fixture missing");
    let graph = import_onnx(&bytes, OnnxImportOptions::default()).expect("import failed");
    let node_count_before = graph.nodes.len();

    let optimised = OptPipeline::mvp().run(graph).expect("opt failed");
    // Dead-node elimination should not remove live nodes.
    assert!(optimised.nodes.len() <= node_count_before);
    assert!(optimised.validate().is_empty());
}

// ── Memory planner tests ──────────────────────────────────────────────────────

#[test]
fn memory_planner_produces_non_zero_weight_budget() {
    let bytes = std::fs::read("tests/fixtures/onnx/tiny-mlp.onnx")
        .expect("tiny-mlp.onnx fixture missing");
    let graph = import_onnx(&bytes, OnnxImportOptions::default()).expect("import failed");
    let plan = MemoryPlanner.plan(&graph).expect("planning failed");
    assert!(plan.total_weight_bytes > 0, "expected non-zero weight bytes");
}

// ── Programmatic graph tests ──────────────────────────────────────────────────

/// Build an `AiGraph` entirely from Rust code (no file loading required).
fn make_identity_graph() -> AiGraph {
    let mut tensor_info = HashMap::new();
    let shape = shape_from_concrete(&[1, 64]);
    tensor_info.insert(0u32, TensorInfo::new(DType::F32, shape.clone()));
    tensor_info.insert(1u32, TensorInfo::new(DType::F32, shape));

    AiGraph {
        name: "test_identity".into(),
        nodes: vec![AiNode::new(0, AiOp::Identity, vec![0], vec![1])],
        inputs: vec![0],
        outputs: vec![1],
        input_names: vec![],
        output_names: vec![],
        params: HashMap::new(),
        tensor_info,
        metadata: HashMap::new(),
        warnings: vec![],
        dim_vars: Default::default(),
        shape_constraints: Default::default(),
    }
}

#[test]
fn programmatic_graph_validates() {
    let g = make_identity_graph();
    assert!(g.validate().is_empty());
}

#[test]
fn dead_node_elimination_removes_dead_nodes() {
    let mut g = make_identity_graph();
    // Add a dead node that produces tensor 2 (not connected to any output).
    let shape = shape_from_concrete(&[1, 64]);
    g.tensor_info.insert(2u32, TensorInfo::new(DType::F32, shape));
    g.nodes.push(AiNode::new(1, AiOp::Identity, vec![0], vec![2]));

    assert_eq!(g.nodes.len(), 2);
    let g2 = OptPipeline::mvp().run(g).expect("opt failed");
    assert_eq!(g2.nodes.len(), 1, "dead node should have been removed");
}

#[test]
fn topo_order_matches_node_count() {
    let bytes = std::fs::read("tests/fixtures/onnx/tiny-mlp.onnx")
        .expect("tiny-mlp.onnx fixture missing");
    let graph = import_onnx(&bytes, OnnxImportOptions::default()).expect("import failed");
    let order = graph.topo_order();
    assert_eq!(order.len(), graph.nodes.len(), "topo order must cover all nodes");
}

// ── Quant descriptor tests ────────────────────────────────────────────────────

#[test]
fn quant_descriptor_none_has_zero_block_size() {
    let q = QuantDescriptor::none();
    assert_eq!(q.block_size, 0);
}
