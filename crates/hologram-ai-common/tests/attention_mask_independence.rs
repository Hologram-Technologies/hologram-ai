//! Attention fusion must DROP the external causal mask, not carry it to lowering.
//!
//! hologram's attention kernel builds its own causal mask from the `causal` flag
//! ([`AiOp::GroupedQueryAttention`]), so whatever mask a model's export injects
//! into the `Q@K^T` scores — a lower-triangular constant, or the
//! `ConstantOfShape → Trilu → … → ScatterND` construction Qwen2-family ONNX
//! exports use — is redundant. It must be removed.
//!
//! The imperative `AttentionFusion` pass removed it; commit 6177a15 replaced that
//! pass with a declarative rule that left the terminal `scores@V` MatMul (a second
//! producer of the attention output) in place, so `DeadNodeElimination`'s
//! last-writer-wins producer map kept the whole mask subgraph reachable. It then
//! survived to lowering, where a `ScatterND` that never const-folded aborts the
//! compile with "Scatter must be const-folded at import" — the real Qwen2.5-0.5B
//! regression. That model needs a 2 GB download and only runs in the nightly
//! matrix; this witnesses the same property hermetically, on every CI run.
//!
//! Parametric by construction: the fusion drops the mask by graph SHAPE (the SDPA
//! chain that feeds a softmax'd `Q@K^T` into an output matmul), never by the
//! mask's own dtype, shape, or the exporter's dim names.

use hologram_ai_common::opt::dead_node::DeadNodeElimination;
use hologram_ai_common::opt::pipeline::Pass;
use hologram_ai_common::rules::{pattern_rules::attention_fusion_rules, RulePass};
use hologram_ai_common::{
    shape_from_concrete, AiGraph, AiNode, AiOp, AiParam, DType, SemanticHint, TensorInfo,
};

fn info(dims: &[u64]) -> TensorInfo {
    TensorInfo {
        shape: shape_from_concrete(dims),
        logical_dtype: DType::F32,
        storage_dtype: DType::F32,
        quant: hologram_ai_quant::QuantDescriptor::none(),
        known_i64_values: None,
        semantic: SemanticHint::Unknown,
    }
}

fn empty_graph() -> AiGraph {
    AiGraph {
        name: "sdpa".to_string(),
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

/// Build one heads-first SDPA block whose mask is a `ScatterND` — the exact shape
/// of node feeds the Qwen2 causal mask, and the exact op that aborts lowering if
/// it survives:
///
/// ```text
///   scores = MatMul(Q, Kᵀ)                     [1, H, S, S]
///   mask   = ScatterND(data, indices, updates) [1, 1, S, S]
///   biased = Add(scores, mask)
///   probs  = Softmax(biased, axis=-1)
///   out    = MatMul(probs, V)                  [1, H, S, D]   (graph output)
/// ```
fn build_sdpa_with_scatternd_mask() -> AiGraph {
    let mut g = empty_graph();
    let (h, s, d) = (2u64, 4u64, 8u64);

    // Q / K / V graph inputs, heads-first [1, H, S, D].
    for (tid, name) in [(1u32, "q"), (2, "k"), (3, "v")] {
        g.tensor_info.insert(tid, info(&[1, h, s, d]));
        g.inputs.push(tid);
        g.input_names.push(name.to_string());
    }
    // Mask construction operands (constants; their exact values are irrelevant —
    // the point is that a ScatterND node exists and must be swept).
    for (tid, dims, dt) in [
        (10u32, vec![1u64, 1, s, s], DType::INT64), // data
        (11, vec![1, 1, s, s, 4], DType::INT64),    // indices
        (12, vec![1, 1, s, s], DType::BOOL),        // updates
    ] {
        let mut ti = info(&dims);
        ti.logical_dtype = dt;
        ti.storage_dtype = dt;
        let n: usize = dims.iter().product::<u64>() as usize;
        g.params.insert(
            tid,
            AiParam::Inline {
                data: vec![0u8; n * 8].into(),
                info: ti.clone(),
            },
        );
        g.tensor_info.insert(tid, ti);
    }

    let mut nid = 0u32;
    let mut node = |g: &mut AiGraph, op, ins: Vec<u32>, out, out_info| {
        g.tensor_info.insert(out, out_info);
        g.nodes.push(AiNode::new(nid, op, ins, vec![out]));
        nid += 1;
    };

    // scores = Q @ Kᵀ  (the fusion's root pattern is exactly this MatMul)
    node(&mut g, AiOp::MatMul, vec![1, 2], 20, info(&[1, h, s, s]));
    // mask = ScatterND(data, indices, updates)
    node(
        &mut g,
        AiOp::ScatterND {
            reduce: hologram_ai_common::ScatterReduce::None,
        },
        vec![10, 11, 12],
        21,
        info(&[1, 1, s, s]),
    );
    // biased = Add(scores, mask)
    node(&mut g, AiOp::Add, vec![20, 21], 22, info(&[1, h, s, s]));
    // probs = Softmax(biased)
    node(
        &mut g,
        AiOp::Softmax { axis: -1 },
        vec![22],
        23,
        info(&[1, h, s, s]),
    );
    // out = probs @ V
    node(&mut g, AiOp::MatMul, vec![23, 3], 24, info(&[1, h, s, d]));

    g.outputs.push(24);
    g.output_names.push("output".to_string());
    g
}

fn count<F: Fn(&AiOp) -> bool>(g: &AiGraph, pred: F) -> usize {
    g.nodes.iter().filter(|n| pred(&n.op)).count()
}

#[test]
fn attention_fusion_drops_the_external_scatternd_mask() {
    let g = build_sdpa_with_scatternd_mask();
    assert_eq!(
        count(&g, |op| matches!(op, AiOp::ScatterND { .. })),
        1,
        "precondition: the input graph carries a ScatterND mask"
    );

    let g = RulePass::new("AttentionFusion", attention_fusion_rules())
        .run(g)
        .expect("attention fusion");
    let g = DeadNodeElimination.run(g).expect("dce");

    // The block collapsed to one GroupedQueryAttention (causal), which owns its
    // mask — so the whole external SDPA chain, mask included, is gone.
    assert_eq!(
        count(&g, |op| matches!(op, AiOp::GroupedQueryAttention { .. })),
        1,
        "the Q@K^T…softmax…@V block must fuse to one GroupedQueryAttention"
    );
    assert!(
        count(
            &g,
            |op| matches!(op, AiOp::GroupedQueryAttention { causal, .. } if *causal)
        ) == 1,
        "the fused attention must be causal (it absorbed the mask into its own)"
    );
    assert_eq!(
        count(&g, |op| matches!(op, AiOp::ScatterND { .. })),
        0,
        "REGRESSION: the external ScatterND mask survived fusion — it will reach \
         lowering and abort with \"Scatter must be const-folded at import\" \
         (the Qwen2.5 failure). The declarative attention rewrite must retire the \
         terminal scores@V MatMul so DCE can sweep the mask."
    );
    assert_eq!(
        count(&g, |op| matches!(op, AiOp::Softmax { .. } | AiOp::Add)),
        0,
        "the softmax and mask-add of the fused block must not survive"
    );
}
