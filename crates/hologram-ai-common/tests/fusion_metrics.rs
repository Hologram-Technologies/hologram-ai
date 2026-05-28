//! Fusion metrics tests (Plan 054).
//!
//! These tests construct realistic transformer block subgraphs and verify
//! that the fusion passes fire correctly, reducing node counts and producing
//! the expected fused op types.
//!
//! These are compile-time metrics — they validate the compiler's fusion
//! decisions, not runtime performance. Runtime benchmarks should wait until
//! fused kernels handle quantized weights (Q4/Q8).

use hologram_ai_common::{
    shape_from_concrete, AiGraph, AiNode, AiOp, AiParam, DType, SemanticHint, TensorInfo,
};
use std::collections::HashMap;

fn empty_graph() -> AiGraph {
    AiGraph {
        name: "fusion_bench".to_string(),
        nodes: Vec::new(),
        inputs: Vec::new(),
        outputs: Vec::new(),
        input_names: Vec::new(),
        output_names: Vec::new(),
        params: Default::default(),
        tensor_info: Default::default(),
        metadata: Default::default(),
        warnings: Vec::new(),
        dim_vars: Default::default(),
        shape_constraints: Default::default(),
        subgraphs: Default::default(),
        tensor_names: Default::default(),
        topo_cache: Default::default(),
    }
}

fn f32_param(k: usize, n: usize) -> (AiParam, TensorInfo) {
    let data = vec![0u8; k * n * 4];
    let info = TensorInfo {
        shape: shape_from_concrete(&[k as u64, n as u64]),
        logical_dtype: DType::F32,
        storage_dtype: DType::F32,
        quant: hologram_ai_quant::QuantDescriptor::none(),
        known_i64_values: None,
        semantic: SemanticHint::Unknown,
    };
    (
        AiParam::Inline {
            data: data.into(),
            info: info.clone(),
        },
        info,
    )
}

fn input_info(dims: &[u64]) -> TensorInfo {
    TensorInfo {
        shape: shape_from_concrete(dims),
        logical_dtype: DType::F32,
        storage_dtype: DType::F32,
        quant: hologram_ai_quant::QuantDescriptor::none(),
        known_i64_values: None,
        semantic: SemanticHint::Unknown,
    }
}

/// Count occurrences of each op type in a graph.
fn op_counts(graph: &AiGraph) -> HashMap<String, usize> {
    let mut counts = HashMap::new();
    for node in &graph.nodes {
        let name = match &node.op {
            AiOp::RmsNorm { .. } => "RmsNorm",
            AiOp::FusedLayerNormResidual { .. } => "FusedLayerNormResidual",
            AiOp::MatMul => "MatMul",
            AiOp::Gemm { .. } => "Gemm",
            AiOp::FusedSwiGLU => "FusedSwiGLU",
            AiOp::FusedNormProjection { .. } => "FusedNormProjection",
            AiOp::FusedSwiGluProjection => "FusedSwiGluProjection",
            AiOp::Silu => "Silu",
            AiOp::Mul => "Mul",
            AiOp::Add => "Add",
            AiOp::Slice { .. } => "Slice",
            _ => "Other",
        };
        *counts.entry(name.to_string()).or_default() += 1;
    }
    counts
}

/// Tensor IDs and dimensions for a single FFN block.
///
/// Uses the Rust builder pattern so the test helper doesn't take ten
/// positional arguments: [`FfnBlockSpec::new`] fixes the five weight
/// tensor IDs + the two dimensions, and the intermediate tensor/node ID
/// bases default to 100 / 0 and can be overridden via
/// [`Self::with_base_tid`] / [`Self::with_base_node_id`].
struct FfnBlockSpec {
    hidden_tid: u32,
    norm_weight_tid: u32,
    w_gate_tid: u32,
    w_up_tid: u32,
    w_down_tid: u32,
    /// First tensor ID to allocate for intermediates — the helper claims
    /// `base_tid..base_tid + 6`.
    base_tid: u32,
    /// First node ID to assign to the 6 emitted nodes.
    base_node_id: u32,
    hidden_dim: usize,
    ffn_dim: usize,
}

impl FfnBlockSpec {
    /// New spec with the five weight tensor IDs and the FFN shape. The
    /// `base_tid` (first intermediate tensor ID) defaults to `100` and
    /// `base_node_id` defaults to `0`.
    fn new(
        hidden_tid: u32,
        norm_weight_tid: u32,
        w_gate_tid: u32,
        w_up_tid: u32,
        w_down_tid: u32,
        hidden_dim: usize,
        ffn_dim: usize,
    ) -> Self {
        Self {
            hidden_tid,
            norm_weight_tid,
            w_gate_tid,
            w_up_tid,
            w_down_tid,
            base_tid: 100,
            base_node_id: 0,
            hidden_dim,
            ffn_dim,
        }
    }

    fn with_base_tid(mut self, base_tid: u32) -> Self {
        self.base_tid = base_tid;
        self
    }

    fn with_base_node_id(mut self, base_node_id: u32) -> Self {
        self.base_node_id = base_node_id;
        self
    }
}

/// Build a single transformer FFN block:
///   hidden → RmsNorm → MatMul(W_gate) → SiLU → Mul(up) → MatMul(W_down) → out
///                     → MatMul(W_up)   ──────↗
///
/// With f32 weights so NormProjectionFusion and SwiGluProjectionFusion can fire.
fn build_ffn_block(graph: &mut AiGraph, spec: FfnBlockSpec) -> (u32, u32) {
    let FfnBlockSpec {
        hidden_tid,
        norm_weight_tid,
        w_gate_tid,
        w_up_tid,
        w_down_tid,
        base_tid,
        base_node_id,
        hidden_dim,
        ffn_dim,
    } = spec;
    // tid allocation
    let normed = base_tid;
    let gate_out = base_tid + 1;
    let up_out = base_tid + 2;
    let silu_out = base_tid + 3;
    let swiglu_out = base_tid + 4;
    let ffn_out = base_tid + 5;

    // node id allocation
    let mut nid = base_node_id;

    // RmsNorm(hidden, norm_weight) → normed
    graph.nodes.push(AiNode::new(
        nid,
        AiOp::RmsNorm { epsilon: 1e-5 },
        vec![hidden_tid, norm_weight_tid],
        vec![normed],
    ));
    graph
        .tensor_info
        .insert(normed, input_info(&[1, hidden_dim as u64]));
    nid += 1;

    // MatMul(normed, W_gate) → gate_out
    graph.nodes.push(AiNode::new(
        nid,
        AiOp::MatMul,
        vec![normed, w_gate_tid],
        vec![gate_out],
    ));
    graph
        .tensor_info
        .insert(gate_out, input_info(&[1, ffn_dim as u64]));
    nid += 1;

    // MatMul(normed, W_up) → up_out
    graph.nodes.push(AiNode::new(
        nid,
        AiOp::MatMul,
        vec![normed, w_up_tid],
        vec![up_out],
    ));
    graph
        .tensor_info
        .insert(up_out, input_info(&[1, ffn_dim as u64]));
    nid += 1;

    // SiLU(gate_out) → silu_out
    graph
        .nodes
        .push(AiNode::new(nid, AiOp::Silu, vec![gate_out], vec![silu_out]));
    graph
        .tensor_info
        .insert(silu_out, input_info(&[1, ffn_dim as u64]));
    nid += 1;

    // Mul(silu_out, up_out) → swiglu_out
    graph.nodes.push(AiNode::new(
        nid,
        AiOp::Mul,
        vec![silu_out, up_out],
        vec![swiglu_out],
    ));
    graph
        .tensor_info
        .insert(swiglu_out, input_info(&[1, ffn_dim as u64]));
    nid += 1;

    // MatMul(swiglu_out, W_down) → ffn_out
    graph.nodes.push(AiNode::new(
        nid,
        AiOp::MatMul,
        vec![swiglu_out, w_down_tid],
        vec![ffn_out],
    ));
    graph
        .tensor_info
        .insert(ffn_out, input_info(&[1, hidden_dim as u64]));
    nid += 1;

    (ffn_out, nid)
}

/// Test that SwiGluFusion + SwiGluProjectionFusion fires on a single FFN block.
#[test]
fn ffn_block_swiglu_projection_fusion() {
    let hidden_dim = 512;
    let ffn_dim = 1024;

    let mut g = empty_graph();

    // Inputs and params
    let hidden_tid = 1u32;
    let norm_w_tid = 2u32;
    let w_gate_tid = 3u32;
    let w_up_tid = 4u32;
    let w_down_tid = 5u32;

    g.inputs = vec![hidden_tid];
    g.tensor_info
        .insert(hidden_tid, input_info(&[1, hidden_dim as u64]));

    // Norm weight (1D)
    let norm_data = vec![0u8; hidden_dim * 4];
    let norm_info = TensorInfo {
        shape: shape_from_concrete(&[hidden_dim as u64]),
        logical_dtype: DType::F32,
        storage_dtype: DType::F32,
        quant: hologram_ai_quant::QuantDescriptor::none(),
        known_i64_values: None,
        semantic: SemanticHint::Unknown,
    };
    g.params.insert(
        norm_w_tid,
        AiParam::Inline {
            data: norm_data.into(),
            info: norm_info.clone(),
        },
    );
    g.tensor_info.insert(norm_w_tid, norm_info);

    // Projection weights
    for &(tid, k, n) in &[
        (w_gate_tid, hidden_dim, ffn_dim),
        (w_up_tid, hidden_dim, ffn_dim),
        (w_down_tid, ffn_dim, hidden_dim),
    ] {
        let (param, info) = f32_param(k, n);
        g.params.insert(tid, param);
        g.tensor_info.insert(tid, info);
    }

    let (ffn_out, _) = build_ffn_block(
        &mut g,
        FfnBlockSpec::new(
            hidden_tid, norm_w_tid, w_gate_tid, w_up_tid, w_down_tid, hidden_dim, ffn_dim,
        ),
    );
    g.outputs = vec![ffn_out];

    let before_nodes = g.nodes.len();
    let before_counts = op_counts(&g);

    // Run just the fusion passes (not the full pipeline which needs shape prop).
    // SwiGluFusion is the ADR-0018 declarative rule set (RulePass over
    // `pattern_rules::swiglu_rules()`); the others are still imperative
    // passes pending their own rule-set port.
    use hologram_ai_common::opt::{
        norm_projection_fusion::NormProjectionFusion, pipeline::Pass, rmsnorm_fusion::RmsNormFusion,
    };
    use hologram_ai_common::rules::{
        pattern_rules::{swiglu_projection_rules, swiglu_rules},
        RulePass,
    };

    let g = RmsNormFusion.run(g).expect("RmsNormFusion");
    let g = RulePass::new("SwiGluFusion", swiglu_rules())
        .run(g)
        .expect("SwiGluFusion");
    let g = NormProjectionFusion.run(g).expect("NormProjectionFusion");
    let g = RulePass::new("SwiGluProjectionFusion", swiglu_projection_rules())
        .run(g)
        .expect("SwiGluProjectionFusion");

    let after_nodes = g.nodes.len();
    let after_counts = op_counts(&g);

    // ── Metrics ──
    eprintln!("=== FFN Block Fusion Metrics ===");
    eprintln!("Before: {before_nodes} nodes — {before_counts:?}");
    eprintln!("After:  {after_nodes} nodes — {after_counts:?}");
    eprintln!(
        "Reduction: {} nodes eliminated",
        before_nodes as i32 - after_nodes as i32
    );

    // SwiGluFusion should fire: SiLU + Mul → FusedSwiGLU
    assert_eq!(
        after_counts.get("Silu"),
        None,
        "SiLU should be consumed by SwiGluFusion"
    );
    assert_eq!(
        after_counts.get("Mul"),
        None,
        "Mul should be consumed by SwiGluFusion"
    );

    // SwiGluProjectionFusion should fire: FusedSwiGLU + MatMul(W_down) → FusedSwiGluProjection
    assert_eq!(
        after_counts.get("FusedSwiGLU"),
        None,
        "FusedSwiGLU should be consumed by SwiGluProjectionFusion"
    );
    assert_eq!(
        after_counts
            .get("FusedSwiGluProjection")
            .copied()
            .unwrap_or(0),
        1,
        "should have exactly 1 FusedSwiGluProjection"
    );

    // NormProjectionFusion should fire: RmsNorm + 2 MatMuls → 1 FusedNormProjection (multi-output)
    assert_eq!(
        after_counts
            .get("FusedNormProjection")
            .copied()
            .unwrap_or(0),
        1,
        "should have exactly 1 FusedNormProjection"
    );

    // No standalone MatMul should remain (all absorbed into fusions).
    assert_eq!(
        after_counts.get("MatMul").copied().unwrap_or(0),
        0,
        "all MatMuls should be absorbed into fused ops"
    );

    // Net node reduction: 6 original → 2 (FusedNormProj + FusedSwiGluProj)
    assert!(
        after_nodes <= before_nodes,
        "fusion should not increase node count: {after_nodes} > {before_nodes}"
    );
}

/// Test a 2-layer transformer FFN stack — verifies fusions scale across layers.
#[test]
fn multi_layer_ffn_fusion_scaling() {
    let hidden_dim = 512;
    let ffn_dim = 1024;
    let n_layers = 4;

    let mut g = empty_graph();
    let hidden_tid = 1u32;
    g.inputs = vec![hidden_tid];
    g.tensor_info
        .insert(hidden_tid, input_info(&[1, hidden_dim as u64]));

    let mut next_param_tid = 10u32;
    let mut next_tid = 200u32;
    let mut next_nid = 0u32;
    let mut current_hidden = hidden_tid;

    for _layer in 0..n_layers {
        // Allocate per-layer params
        let norm_w = next_param_tid;
        next_param_tid += 1;
        let w_gate = next_param_tid;
        next_param_tid += 1;
        let w_up = next_param_tid;
        next_param_tid += 1;
        let w_down = next_param_tid;
        next_param_tid += 1;

        // Register norm weight
        let norm_data = vec![0u8; hidden_dim * 4];
        let norm_info = TensorInfo {
            shape: shape_from_concrete(&[hidden_dim as u64]),
            logical_dtype: DType::F32,
            storage_dtype: DType::F32,
            quant: hologram_ai_quant::QuantDescriptor::none(),
            known_i64_values: None,
            semantic: SemanticHint::Unknown,
        };
        g.params.insert(
            norm_w,
            AiParam::Inline {
                data: norm_data.into(),
                info: norm_info.clone(),
            },
        );
        g.tensor_info.insert(norm_w, norm_info);

        for &(tid, k, n) in &[
            (w_gate, hidden_dim, ffn_dim),
            (w_up, hidden_dim, ffn_dim),
            (w_down, ffn_dim, hidden_dim),
        ] {
            let (param, info) = f32_param(k, n);
            g.params.insert(tid, param);
            g.tensor_info.insert(tid, info);
        }

        let (out, nid) = build_ffn_block(
            &mut g,
            FfnBlockSpec::new(
                current_hidden,
                norm_w,
                w_gate,
                w_up,
                w_down,
                hidden_dim,
                ffn_dim,
            )
            .with_base_tid(next_tid)
            .with_base_node_id(next_nid),
        );
        next_tid += 10;
        next_nid = nid;
        current_hidden = out;
    }

    g.outputs = vec![current_hidden];
    let before_nodes = g.nodes.len();

    use hologram_ai_common::opt::{
        norm_projection_fusion::NormProjectionFusion, pipeline::Pass, rmsnorm_fusion::RmsNormFusion,
    };
    use hologram_ai_common::rules::{
        pattern_rules::{swiglu_projection_rules, swiglu_rules},
        RulePass,
    };

    let g = RmsNormFusion.run(g).expect("RmsNormFusion");
    let g = RulePass::new("SwiGluFusion", swiglu_rules())
        .run(g)
        .expect("SwiGluFusion");
    let g = NormProjectionFusion.run(g).expect("NormProjectionFusion");
    let g = RulePass::new("SwiGluProjectionFusion", swiglu_projection_rules())
        .run(g)
        .expect("SwiGluProjectionFusion");

    let after_nodes = g.nodes.len();
    let after_counts = op_counts(&g);

    eprintln!("=== {n_layers}-Layer FFN Fusion Metrics ===");
    eprintln!(
        "Before: {before_nodes} nodes ({} per layer)",
        before_nodes / n_layers
    );
    eprintln!(
        "After:  {after_nodes} nodes ({} per layer)",
        after_nodes / n_layers
    );
    eprintln!(
        "Reduction: {} nodes ({:.0}%)",
        before_nodes - after_nodes,
        (1.0 - after_nodes as f64 / before_nodes as f64) * 100.0
    );
    eprintln!("Op counts: {after_counts:?}");

    // Per layer: 1 FusedNormProjection (multi-output) + 1 FusedSwiGluProjection = 2 nodes
    // vs original: 6 nodes per layer
    assert_eq!(
        after_counts
            .get("FusedNormProjection")
            .copied()
            .unwrap_or(0),
        n_layers,
        "should have {n_layers} FusedNormProjection ops"
    );
    assert_eq!(
        after_counts
            .get("FusedSwiGluProjection")
            .copied()
            .unwrap_or(0),
        n_layers,
        "should have {n_layers} FusedSwiGluProjection ops"
    );
    assert_eq!(
        after_counts.get("MatMul").copied().unwrap_or(0),
        0,
        "no standalone MatMul should remain"
    );

    // Verify linear scaling: each layer should contribute the same reduction.
    let nodes_per_layer = after_nodes / n_layers;
    assert_eq!(
        nodes_per_layer, 2,
        "expected 2 nodes per layer after fusion (NormProj + SwiGluProj)"
    );
}
