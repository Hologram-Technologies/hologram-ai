//! Diagnostic: scan a Qwen2.5-int8 ONNX import for structural anomalies
//! after the standard MVP optimization pipeline.
//!
//! Run with: `HOLOGRAM_AI_QWEN25_INT8=<path/to/model.onnx> cargo test
//! --release -p hologram-ai --test diag_qwen25_int8 -- --nocapture`.
//!
//! Pre-fix (pre-DQL/MMI decomposition): import succeeds but lowering
//! errored with `tensor T5147 referenced before definition` because
//! `DynamicQuantizeLinear` (96 ops, all 3 outputs undefined) and
//! `MatMulInteger` (168 ops, all outputs undefined) routed to
//! `AiOp::Opaque`. This test now passes the post-opt graph silently
//! when all MatMul B-inputs carry rank≥2 shapes (the int8 weights
//! preserved end-to-end through the Cast→Sub→Cast→MatMul chain).

use std::path::Path;

use hologram_ai_common::{AiOp, OptPipeline};

#[test]
fn dump_int8_qwen25_around_t5147() {
    let path = match std::env::var("HOLOGRAM_AI_QWEN25_INT8") {
        Ok(p) => p,
        Err(_) => {
            eprintln!("skipping: set HOLOGRAM_AI_QWEN25_INT8=<path/to/model.onnx>");
            return;
        }
    };
    let imported = hologram_ai_onnx::import_onnx_path(
        Path::new(&path),
        hologram_ai_onnx::OnnxImportOptions::default(),
    )
    .expect("import");
    eprintln!(
        "imported: nodes={}, params={}",
        imported.nodes.len(),
        imported.params.len()
    );

    let pipeline = OptPipeline::mvp();
    let opted = pipeline.run(imported).expect("opt pipeline");
    eprintln!("post-opt: nodes={}", opted.nodes.len());

    // T5147 in the source ONNX is the DynamicQuantizeLinear's `y_scale`
    // output. Before the decomposition it had no producer. After it,
    // a `Div(range, 255)` node produces it.
    let producer: std::collections::HashMap<u32, &hologram_ai_common::AiNode> = opted
        .nodes
        .iter()
        .flat_map(|n| n.outputs.iter().map(move |&o| (o, n)))
        .collect();
    if let Some(p) = producer.get(&5147u32) {
        eprintln!("T5147 producer: node {} op={:?}", p.id, p.op);
    } else if opted.params.contains_key(&5147u32) {
        eprintln!("T5147 is a param (folded)");
    }

    // Catalog any MatMul whose B-input shape is rank<2 against rank≥3 A.
    // Pre-data_prop-shape-fix this matched ~168 MatMuls fed by the
    // MatMulInteger decomposition (DataPropagation flattened the [896,128]
    // int weight to a [114688] 1-D fold). Post-fix this should be zero.
    let mut bad = 0u32;
    for n in &opted.nodes {
        if !matches!(n.op, AiOp::MatMul) {
            continue;
        }
        let b = n.inputs.get(1).copied().unwrap_or(u32::MAX);
        let a = n.inputs.first().copied().unwrap_or(u32::MAX);
        let a_rank = opted.tensor_info.get(&a).map(|i| i.shape.len()).unwrap_or(0);
        let b_rank = opted.tensor_info.get(&b).map(|i| i.shape.len()).unwrap_or(0);
        if a_rank >= 3 && b_rank < 2 {
            bad += 1;
        }
    }
    eprintln!("MatMul with rank<2 B against rank≥3 A: {bad}");

    // Surface any surviving Opaque ops (they would re-introduce the
    // pre-fix "T referenced before definition" failure at lowering).
    let mut opaque: std::collections::HashMap<String, u32> = Default::default();
    for n in &opted.nodes {
        if let AiOp::Opaque { op_type, .. } = &n.op {
            *opaque.entry(op_type.clone()).or_insert(0) += 1;
        }
    }
    if !opaque.is_empty() {
        for (k, v) in &opaque {
            eprintln!("OPAQUE survivor: {k} ({v})");
        }
    }
    assert_eq!(bad, 0, "MatMul shape anomalies regressed");
    assert!(opaque.is_empty(), "Opaque op survivors regressed: {opaque:?}");
}
