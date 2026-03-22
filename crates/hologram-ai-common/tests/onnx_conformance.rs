//! ONNX conformance tests: verify that each supported op's shape propagation
//! resolves output shapes correctly through the full optimization pipeline.
//!
//! Each test builds a minimal AiGraph with a single op, runs the optimization
//! pipeline, and checks that output tensor_info has the expected shape.

use hologram_ai_common::{
    shape_from_concrete, AiGraph, AiNode, AiOp, AiParam, DType, OptPipeline, Shape, TensorInfo,
};
use std::collections::HashMap;

/// Build a single-op graph and run the opt pipeline.
fn run_single_op(
    op: AiOp,
    inputs: Vec<(u32, DType, &[u64])>,
    outputs: Vec<(u32, DType)>,
    params: Vec<(u32, DType, Vec<u8>)>,
) -> AiGraph {
    let mut tensor_info = HashMap::new();
    let mut input_ids = Vec::new();
    let mut param_map = HashMap::new();
    let mut all_input_tids = Vec::new();
    let mut all_output_tids = Vec::new();

    for (tid, dtype, shape) in &inputs {
        tensor_info.insert(*tid, TensorInfo::new(*dtype, shape_from_concrete(shape)));
        input_ids.push(*tid);
        all_input_tids.push(*tid);
    }

    for (tid, dtype, data) in &params {
        let elem_size = match dtype {
            DType::F32 => 4,
            DType::INT64 => 8,
            DType::INT32 => 4,
            _ => 1,
        };
        let num_elems = data.len() / elem_size;
        tensor_info.insert(
            *tid,
            TensorInfo::new(*dtype, shape_from_concrete(&[num_elems as u64])),
        );
        param_map.insert(
            *tid,
            AiParam::Inline {
                data: data.clone(),
                info: TensorInfo::new(*dtype, shape_from_concrete(&[num_elems as u64])),
            },
        );
        all_input_tids.push(*tid);
    }

    for (tid, dtype) in &outputs {
        tensor_info.insert(*tid, TensorInfo::new(*dtype, Shape::new()));
        all_output_tids.push(*tid);
    }

    let graph = AiGraph {
        name: "conformance_test".into(),
        nodes: vec![AiNode::new(0, op, all_input_tids, all_output_tids.clone())],
        inputs: input_ids,
        outputs: all_output_tids,
        input_names: vec![],
        output_names: vec![],
        params: param_map,
        tensor_info,
        metadata: HashMap::new(),
        warnings: vec![],
        dim_vars: Default::default(),
        shape_constraints: Default::default(),
        subgraphs: HashMap::new(),
        tensor_names: HashMap::new(),
            topo_cache: Default::default(),
    };

    OptPipeline::mvp().run(graph).expect("opt pipeline failed")
}

fn assert_shape(graph: &AiGraph, tid: u32, expected: &[u64]) {
    let info = graph
        .tensor_info
        .get(&tid)
        .unwrap_or_else(|| panic!("tensor {} not found in tensor_info", tid));
    let actual: Vec<u64> = info
        .shape
        .iter()
        .filter_map(|d| d.as_concrete())
        .collect();
    assert_eq!(
        actual, expected,
        "tensor {} shape mismatch: got {:?}, expected {:?}",
        tid, info.shape, expected
    );
}

// ── Unary elementwise ops ────────────────────────────────────────────────────

#[test]
fn conformance_relu() {
    let g = run_single_op(
        AiOp::Relu,
        vec![(0, DType::F32, &[2, 3])],
        vec![(1, DType::F32)],
        vec![],
    );
    assert_shape(&g, 1, &[2, 3]);
}

#[test]
fn conformance_sigmoid() {
    let g = run_single_op(
        AiOp::Sigmoid,
        vec![(0, DType::F32, &[4, 5])],
        vec![(1, DType::F32)],
        vec![],
    );
    assert_shape(&g, 1, &[4, 5]);
}

#[test]
fn conformance_abs() {
    let g = run_single_op(
        AiOp::Abs,
        vec![(0, DType::F32, &[3])],
        vec![(1, DType::F32)],
        vec![],
    );
    assert_shape(&g, 1, &[3]);
}

// ── Binary elementwise ops ───────────────────────────────────────────────────

#[test]
fn conformance_add() {
    let g = run_single_op(
        AiOp::Add,
        vec![(0, DType::F32, &[2, 3]), (1, DType::F32, &[2, 3])],
        vec![(2, DType::F32)],
        vec![],
    );
    assert_shape(&g, 2, &[2, 3]);
}

#[test]
fn conformance_add_broadcast() {
    let g = run_single_op(
        AiOp::Add,
        vec![(0, DType::F32, &[2, 3]), (1, DType::F32, &[3])],
        vec![(2, DType::F32)],
        vec![],
    );
    assert_shape(&g, 2, &[2, 3]);
}

#[test]
fn conformance_mul() {
    let g = run_single_op(
        AiOp::Mul,
        vec![(0, DType::F32, &[4, 1]), (1, DType::F32, &[1, 5])],
        vec![(2, DType::F32)],
        vec![],
    );
    assert_shape(&g, 2, &[4, 5]);
}

// ── Reduction ops ────────────────────────────────────────────────────────────

#[test]
fn conformance_reduce_sum_keepdims() {
    let g = run_single_op(
        AiOp::ReduceSum {
            axes: vec![1],
            keepdims: true,
        },
        vec![(0, DType::F32, &[3, 4])],
        vec![(1, DType::F32)],
        vec![],
    );
    assert_shape(&g, 1, &[3, 1]);
}

#[test]
fn conformance_reduce_mean_no_keepdims() {
    let g = run_single_op(
        AiOp::ReduceMean {
            axes: vec![0],
            keepdims: false,
        },
        vec![(0, DType::F32, &[3, 4])],
        vec![(1, DType::F32)],
        vec![],
    );
    assert_shape(&g, 1, &[4]);
}

#[test]
fn conformance_reduce_prod() {
    let g = run_single_op(
        AiOp::ReduceProd {
            axes: vec![-1],
            keepdims: true,
        },
        vec![(0, DType::F32, &[2, 5])],
        vec![(1, DType::F32)],
        vec![],
    );
    assert_shape(&g, 1, &[2, 1]);
}

// ── MatMul ───────────────────────────────────────────────────────────────────

#[test]
fn conformance_matmul() {
    let g = run_single_op(
        AiOp::MatMul,
        vec![(0, DType::F32, &[2, 3]), (1, DType::F32, &[3, 4])],
        vec![(2, DType::F32)],
        vec![],
    );
    assert_shape(&g, 2, &[2, 4]);
}

#[test]
fn conformance_matmul_batched() {
    let g = run_single_op(
        AiOp::MatMul,
        vec![(0, DType::F32, &[2, 3, 4]), (1, DType::F32, &[4, 5])],
        vec![(2, DType::F32)],
        vec![],
    );
    assert_shape(&g, 2, &[2, 3, 5]);
}

// ── Shape-changing ops ───────────────────────────────────────────────────────

#[test]
fn conformance_transpose() {
    let g = run_single_op(
        AiOp::Transpose {
            perm: vec![1, 0, 2],
        },
        vec![(0, DType::F32, &[2, 3, 4])],
        vec![(1, DType::F32)],
        vec![],
    );
    assert_shape(&g, 1, &[3, 2, 4]);
}

#[test]
fn conformance_concat() {
    let g = run_single_op(
        AiOp::Concat { axis: 1 },
        vec![(0, DType::F32, &[2, 3]), (1, DType::F32, &[2, 5])],
        vec![(2, DType::F32)],
        vec![],
    );
    assert_shape(&g, 2, &[2, 8]);
}

#[test]
fn conformance_split() {
    let g = run_single_op(
        AiOp::Split {
            axis: 1,
            sizes: vec![2, 3],
        },
        vec![(0, DType::F32, &[4, 5])],
        vec![(1, DType::F32), (2, DType::F32)],
        vec![],
    );
    assert_shape(&g, 1, &[4, 2]);
    assert_shape(&g, 2, &[4, 3]);
}

#[test]
fn conformance_flatten() {
    let g = run_single_op(
        AiOp::Flatten { axis: 1 },
        vec![(0, DType::F32, &[2, 3, 4])],
        vec![(1, DType::F32)],
        vec![],
    );
    assert_shape(&g, 1, &[2, 12]);
}

// ── Vision ops ───────────────────────────────────────────────────────────────

#[test]
fn conformance_conv2d() {
    let g = run_single_op(
        AiOp::Conv {
            kernel_shape: vec![3, 3],
            strides: vec![1, 1],
            pads: vec![1, 1, 1, 1],
            dilations: vec![1, 1],
            group: 1,
            auto_pad: String::new(),
        },
        vec![
            (0, DType::F32, &[1, 3, 32, 32]),
            (1, DType::F32, &[16, 3, 3, 3]),
        ],
        vec![(2, DType::F32)],
        vec![],
    );
    assert_shape(&g, 2, &[1, 16, 32, 32]);
}

#[test]
fn conformance_conv2d_stride2() {
    let g = run_single_op(
        AiOp::Conv {
            kernel_shape: vec![3, 3],
            strides: vec![2, 2],
            pads: vec![1, 1, 1, 1],
            dilations: vec![1, 1],
            group: 1,
            auto_pad: String::new(),
        },
        vec![
            (0, DType::F32, &[1, 3, 32, 32]),
            (1, DType::F32, &[16, 3, 3, 3]),
        ],
        vec![(2, DType::F32)],
        vec![],
    );
    assert_shape(&g, 2, &[1, 16, 16, 16]);
}

#[test]
fn conformance_max_pool() {
    let g = run_single_op(
        AiOp::MaxPool {
            kernel_shape: vec![2, 2],
            strides: vec![2, 2],
            pads: vec![0, 0, 0, 0],
            dilations: vec![1, 1],
            auto_pad: String::new(),
            ceil_mode: false,
        },
        vec![(0, DType::F32, &[1, 3, 32, 32])],
        vec![(1, DType::F32)],
        vec![],
    );
    assert_shape(&g, 1, &[1, 3, 16, 16]);
}

#[test]
fn conformance_global_avg_pool() {
    let g = run_single_op(
        AiOp::GlobalAveragePool,
        vec![(0, DType::F32, &[1, 64, 7, 7])],
        vec![(1, DType::F32)],
        vec![],
    );
    assert_shape(&g, 1, &[1, 64, 1, 1]);
}

// ── Normalization ops ────────────────────────────────────────────────────────

#[test]
fn conformance_batch_norm_inference() {
    let g = run_single_op(
        AiOp::BatchNorm {
            epsilon: 1e-5,
            momentum: 0.1,
            training: false,
        },
        vec![
            (0, DType::F32, &[2, 3, 4, 4]),
            (1, DType::F32, &[3]),
            (2, DType::F32, &[3]),
            (3, DType::F32, &[3]),
            (4, DType::F32, &[3]),
        ],
        vec![(5, DType::F32)],
        vec![],
    );
    assert_shape(&g, 5, &[2, 3, 4, 4]);
}

#[test]
fn conformance_batch_norm_training() {
    let g = run_single_op(
        AiOp::BatchNorm {
            epsilon: 1e-5,
            momentum: 0.1,
            training: true,
        },
        vec![
            (0, DType::F32, &[2, 3, 4, 4]),
            (1, DType::F32, &[3]),
            (2, DType::F32, &[3]),
            (3, DType::F32, &[3]),
            (4, DType::F32, &[3]),
        ],
        vec![
            (5, DType::F32),
            (6, DType::F32),
            (7, DType::F32),
            (8, DType::F32),
            (9, DType::F32),
        ],
        vec![],
    );
    assert_shape(&g, 5, &[2, 3, 4, 4]);
    assert_shape(&g, 6, &[3]);
    assert_shape(&g, 7, &[3]);
    assert_shape(&g, 8, &[3]);
    assert_shape(&g, 9, &[3]);
}

// ── TopK (multi-output) ─────────────────────────────────────────────────────

#[test]
fn conformance_topk() {
    let g = run_single_op(
        AiOp::TopK {
            axis: -1,
            largest: true,
            sorted: true,
        },
        vec![(0, DType::F32, &[3, 10]), (1, DType::INT64, &[1])],
        vec![(2, DType::F32), (3, DType::INT64)],
        vec![],
    );
    let info = g.tensor_info.get(&2).expect("values tensor missing");
    assert_eq!(info.shape.len(), 2);
    assert_eq!(info.shape[0].as_concrete(), Some(3));
    let idx_info = g.tensor_info.get(&3).expect("indices tensor missing");
    assert_eq!(idx_info.logical_dtype, DType::INT64);
}

// ── Decomposition ops ────────────────────────────────────────────────────────

#[test]
fn conformance_reduce_l1_decomposed() {
    let g = run_single_op(
        AiOp::ReduceL1 {
            axes: vec![1],
            keepdims: true,
        },
        vec![(0, DType::F32, &[2, 4])],
        vec![(1, DType::F32)],
        vec![],
    );
    assert_shape(&g, 1, &[2, 1]);
    assert!(
        !g.nodes.iter().any(|n| matches!(n.op, AiOp::ReduceL1 { .. })),
        "ReduceL1 should have been decomposed"
    );
}

#[test]
fn conformance_reduce_l2_decomposed() {
    let g = run_single_op(
        AiOp::ReduceL2 {
            axes: vec![-1],
            keepdims: false,
        },
        vec![(0, DType::F32, &[3, 5])],
        vec![(1, DType::F32)],
        vec![],
    );
    assert_shape(&g, 1, &[3]);
    assert!(
        !g.nodes.iter().any(|n| matches!(n.op, AiOp::ReduceL2 { .. })),
        "ReduceL2 should have been decomposed"
    );
}

// ── Clip with min/max ────────────────────────────────────────────────────────

#[test]
fn conformance_clip() {
    let g = run_single_op(
        AiOp::Clip {
            min: 0.0,
            max: 6.0,
        },
        vec![(0, DType::F32, &[2, 3])],
        vec![(1, DType::F32)],
        vec![],
    );
    assert_shape(&g, 1, &[2, 3]);
}

// ── Gemm ─────────────────────────────────────────────────────────────────────

#[test]
fn conformance_gemm() {
    let g = run_single_op(
        AiOp::Gemm {
            alpha: 1.0,
            beta: 1.0,
            trans_a: false,
            trans_b: false,
        },
        vec![
            (0, DType::F32, &[3, 4]),
            (1, DType::F32, &[4, 5]),
            (2, DType::F32, &[5]),
        ],
        vec![(3, DType::F32)],
        vec![],
    );
    assert_shape(&g, 3, &[3, 5]);
}

// ── Comparison ops ───────────────────────────────────────────────────────────

#[test]
fn conformance_equal() {
    let g = run_single_op(
        AiOp::Equal,
        vec![(0, DType::F32, &[2, 3]), (1, DType::F32, &[2, 3])],
        vec![(2, DType::BOOL)],
        vec![],
    );
    assert_shape(&g, 2, &[2, 3]);
    let info = g.tensor_info.get(&2).unwrap();
    assert_eq!(info.logical_dtype, DType::BOOL);
}

// ── Softmax ──────────────────────────────────────────────────────────────────

#[test]
fn conformance_softmax() {
    let g = run_single_op(
        AiOp::Softmax { axis: -1 },
        vec![(0, DType::F32, &[2, 10])],
        vec![(1, DType::F32)],
        vec![],
    );
    assert_shape(&g, 1, &[2, 10]);
}

// ── Cast ─────────────────────────────────────────────────────────────────────

#[test]
fn conformance_cast() {
    let g = run_single_op(
        AiOp::Cast {
            to: DType::INT32,
        },
        vec![(0, DType::F32, &[4, 5])],
        vec![(1, DType::INT32)],
        vec![],
    );
    assert_shape(&g, 1, &[4, 5]);
    let info = g.tensor_info.get(&1).unwrap();
    assert_eq!(info.logical_dtype, DType::INT32);
}

// ── Subgraph lowering tests ─────────────────────────────────────────────────

use hologram_ai_common::{lower, KvCacheLayout, LowerPhase, LoweringOptions};

/// Build an AiGraph with subgraphs for control flow testing.
fn build_if_graph() -> AiGraph {
    let mut tensor_info = HashMap::new();

    // Main graph: condition (bool), x (f32 [2,3]) → If → output (f32 [2,3])
    tensor_info.insert(0u32, TensorInfo::new(DType::BOOL, shape_from_concrete(&[1])));
    tensor_info.insert(1, TensorInfo::new(DType::F32, shape_from_concrete(&[2, 3])));
    tensor_info.insert(10, TensorInfo::new(DType::F32, shape_from_concrete(&[2, 3])));

    // Then branch: input(0) → Relu → output
    let mut then_ti = HashMap::new();
    then_ti.insert(100u32, TensorInfo::new(DType::F32, shape_from_concrete(&[2, 3])));
    then_ti.insert(101, TensorInfo::new(DType::F32, shape_from_concrete(&[2, 3])));

    let then_graph = AiGraph {
        name: "then_branch".into(),
        nodes: vec![AiNode::new(0, AiOp::Relu, vec![100], vec![101])],
        inputs: vec![100],
        outputs: vec![101],
        input_names: vec!["x".into()],
        output_names: vec!["y".into()],
        params: HashMap::new(),
        tensor_info: then_ti,
        metadata: HashMap::new(),
        warnings: vec![],
        dim_vars: Default::default(),
        shape_constraints: Default::default(),
        subgraphs: HashMap::new(),
        tensor_names: HashMap::new(),
            topo_cache: Default::default(),
    };

    // Else branch: input(0) → Neg → output
    let mut else_ti = HashMap::new();
    else_ti.insert(200u32, TensorInfo::new(DType::F32, shape_from_concrete(&[2, 3])));
    else_ti.insert(201, TensorInfo::new(DType::F32, shape_from_concrete(&[2, 3])));

    let else_graph = AiGraph {
        name: "else_branch".into(),
        nodes: vec![AiNode::new(0, AiOp::Neg, vec![200], vec![201])],
        inputs: vec![200],
        outputs: vec![201],
        input_names: vec!["x".into()],
        output_names: vec!["y".into()],
        params: HashMap::new(),
        tensor_info: else_ti,
        metadata: HashMap::new(),
        warnings: vec![],
        dim_vars: Default::default(),
        shape_constraints: Default::default(),
        subgraphs: HashMap::new(),
        tensor_names: HashMap::new(),
            topo_cache: Default::default(),
    };

    let mut subgraphs = HashMap::new();
    subgraphs.insert("if_then_0".to_string(), then_graph);
    subgraphs.insert("if_else_0".to_string(), else_graph);

    AiGraph {
        name: "if_test".into(),
        nodes: vec![AiNode::new(
            0,
            AiOp::If {
                then_branch: "if_then_0".into(),
                else_branch: Some("if_else_0".into()),
            },
            vec![0, 1], // condition, feed input
            vec![10],   // output
        )],
        inputs: vec![0, 1],
        outputs: vec![10],
        input_names: vec!["condition".into(), "x".into()],
        output_names: vec!["result".into()],
        params: HashMap::new(),
        tensor_info,
        metadata: HashMap::new(),
        warnings: vec![],
        dim_vars: Default::default(),
        shape_constraints: Default::default(),
        subgraphs,
        tensor_names: HashMap::new(),
            topo_cache: Default::default(),
    }
}

#[test]
fn subgraph_if_lowering() {
    let graph = build_if_graph();
    let kv = KvCacheLayout::none();
    let opts = LoweringOptions::default();
    let result = lower(&graph, &kv, &opts, &LowerPhase::Forward);

    // Lowering should succeed — both branches flatten + Where selects.
    assert!(
        result.is_ok(),
        "If lowering failed: {:?}",
        result.err()
    );

    let output = result.unwrap();
    // The graph should have nodes from both branches + Where + I/O.
    assert!(output.graph.node_count() > 0, "lowered graph is empty");
}

#[test]
fn subgraph_if_then_only() {
    // If with no else branch: outputs are just then branch outputs.
    let mut graph = build_if_graph();
    // Remove else branch.
    graph.subgraphs.remove("if_else_0");
    graph.nodes[0].op = AiOp::If {
        then_branch: "if_then_0".into(),
        else_branch: None,
    };

    let kv = KvCacheLayout::none();
    let opts = LoweringOptions::default();
    let result = lower(&graph, &kv, &opts, &LowerPhase::Forward);
    assert!(
        result.is_ok(),
        "If (then-only) lowering failed: {:?}",
        result.err()
    );
}

#[test]
fn subgraph_loop_known_trip_count() {
    let mut tensor_info = HashMap::new();
    // Loop inputs: [max_trip_count, condition, ...carry_state]
    tensor_info.insert(0u32, TensorInfo::new(DType::INT64, shape_from_concrete(&[1])));
    tensor_info.insert(1, TensorInfo::new(DType::BOOL, shape_from_concrete(&[1])));
    tensor_info.insert(2, TensorInfo::new(DType::F32, shape_from_concrete(&[4])));
    tensor_info.insert(10, TensorInfo::new(DType::F32, shape_from_concrete(&[4])));

    // Body: input[2] (carry) → Relu → output
    // Body inputs: [iter_num, condition, carry_state]
    // Body outputs: [condition, updated_carry]
    let mut body_ti = HashMap::new();
    body_ti.insert(300u32, TensorInfo::new(DType::INT64, shape_from_concrete(&[1])));
    body_ti.insert(301, TensorInfo::new(DType::BOOL, shape_from_concrete(&[1])));
    body_ti.insert(302, TensorInfo::new(DType::F32, shape_from_concrete(&[4])));
    body_ti.insert(310, TensorInfo::new(DType::BOOL, shape_from_concrete(&[1])));
    body_ti.insert(311, TensorInfo::new(DType::F32, shape_from_concrete(&[4])));

    let body_graph = AiGraph {
        name: "loop_body".into(),
        nodes: vec![
            // Pass through condition.
            AiNode::new(0, AiOp::Identity, vec![301], vec![310]),
            // Apply Relu to carry state.
            AiNode::new(1, AiOp::Relu, vec![302], vec![311]),
        ],
        inputs: vec![300, 301, 302],
        outputs: vec![310, 311],
        input_names: vec!["iter".into(), "cond".into(), "carry".into()],
        output_names: vec!["cond_out".into(), "carry_out".into()],
        params: HashMap::new(),
        tensor_info: body_ti,
        metadata: HashMap::new(),
        warnings: vec![],
        dim_vars: Default::default(),
        shape_constraints: Default::default(),
        subgraphs: HashMap::new(),
        tensor_names: HashMap::new(),
            topo_cache: Default::default(),
    };

    let mut subgraphs = HashMap::new();
    subgraphs.insert("loop_body_0".to_string(), body_graph);

    let graph = AiGraph {
        name: "loop_test".into(),
        nodes: vec![AiNode::new(
            0,
            AiOp::Loop {
                body: "loop_body_0".into(),
                max_trip_count: Some(3),
            },
            vec![0, 1, 2],
            vec![10],
        )],
        inputs: vec![0, 1, 2],
        outputs: vec![10],
        input_names: vec!["trip".into(), "cond".into(), "init_carry".into()],
        output_names: vec!["result".into()],
        params: HashMap::new(),
        tensor_info,
        metadata: HashMap::new(),
        warnings: vec![],
        dim_vars: Default::default(),
        shape_constraints: Default::default(),
        subgraphs,
        tensor_names: HashMap::new(),
            topo_cache: Default::default(),
    };

    let kv = KvCacheLayout::none();
    let opts = LoweringOptions::default();
    let result = lower(&graph, &kv, &opts, &LowerPhase::Forward);
    assert!(
        result.is_ok(),
        "Loop lowering failed: {:?}",
        result.err()
    );

    let output = result.unwrap();
    // 3 iterations of unrolling: should have Relu nodes from each iteration.
    assert!(
        output.graph.node_count() > 3,
        "expected unrolled loop to have multiple nodes, got {}",
        output.graph.node_count()
    );
}

#[test]
fn subgraph_loop_zero_trip() {
    let mut tensor_info = HashMap::new();
    tensor_info.insert(0u32, TensorInfo::new(DType::INT64, shape_from_concrete(&[1])));
    tensor_info.insert(1, TensorInfo::new(DType::BOOL, shape_from_concrete(&[1])));
    tensor_info.insert(2, TensorInfo::new(DType::F32, shape_from_concrete(&[4])));
    tensor_info.insert(10, TensorInfo::new(DType::F32, shape_from_concrete(&[4])));

    // Body is irrelevant for 0-trip loop.
    let mut body_ti = HashMap::new();
    body_ti.insert(300u32, TensorInfo::new(DType::INT64, shape_from_concrete(&[1])));
    body_ti.insert(301, TensorInfo::new(DType::BOOL, shape_from_concrete(&[1])));
    body_ti.insert(302, TensorInfo::new(DType::F32, shape_from_concrete(&[4])));
    body_ti.insert(310, TensorInfo::new(DType::BOOL, shape_from_concrete(&[1])));
    body_ti.insert(311, TensorInfo::new(DType::F32, shape_from_concrete(&[4])));

    let body_graph = AiGraph {
        name: "loop_body".into(),
        nodes: vec![
            AiNode::new(0, AiOp::Identity, vec![301], vec![310]),
            AiNode::new(1, AiOp::Relu, vec![302], vec![311]),
        ],
        inputs: vec![300, 301, 302],
        outputs: vec![310, 311],
        input_names: vec![],
        output_names: vec![],
        params: HashMap::new(),
        tensor_info: body_ti,
        metadata: HashMap::new(),
        warnings: vec![],
        dim_vars: Default::default(),
        shape_constraints: Default::default(),
        subgraphs: HashMap::new(),
        tensor_names: HashMap::new(),
            topo_cache: Default::default(),
    };

    let mut subgraphs = HashMap::new();
    subgraphs.insert("loop_body_0".to_string(), body_graph);

    let graph = AiGraph {
        name: "loop_zero".into(),
        nodes: vec![AiNode::new(
            0,
            AiOp::Loop {
                body: "loop_body_0".into(),
                max_trip_count: Some(0),
            },
            vec![0, 1, 2],
            vec![10],
        )],
        inputs: vec![0, 1, 2],
        outputs: vec![10],
        input_names: vec![],
        output_names: vec![],
        params: HashMap::new(),
        tensor_info,
        metadata: HashMap::new(),
        warnings: vec![],
        dim_vars: Default::default(),
        shape_constraints: Default::default(),
        subgraphs,
        tensor_names: HashMap::new(),
            topo_cache: Default::default(),
    };

    let kv = KvCacheLayout::none();
    let opts = LoweringOptions::default();
    let result = lower(&graph, &kv, &opts, &LowerPhase::Forward);
    assert!(
        result.is_ok(),
        "Loop (0-trip) lowering failed: {:?}",
        result.err()
    );
}
