//! Post-concretize tensor-shape audit for Qwen2.
//!
//! Runs the FULL pre-lower pipeline (import → opt → concretize → repair),
//! then for every node whose output's concrete element count is < the
//! "expected for the seq=context_len full shape", reports the divergence.
//!
//! This is the V&V witness for "which tensor has the wrong concrete
//! seq dim" — the load-bearing question for the Qwen2 residual-add
//! BufferRef.length=3584 bug. Gated by `HOLOGRAM_AI_LIVE=1` +
//! `HOLOGRAM_AI_QWEN2_ONNX=<path>`.

use std::path::PathBuf;

use hologram_ai_common::ir::shape::DimExpr;
use hologram_ai_common::{AiOp, OptPipeline};

fn live_enabled() -> bool {
    std::env::var("HOLOGRAM_AI_LIVE").as_deref() == Ok("1")
}

fn locate_model() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("HOLOGRAM_AI_QWEN2_ONNX") {
        let p = PathBuf::from(p);
        return p.exists().then_some(p);
    }
    let mut ws = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    ws.pop();
    ws.pop();
    let p = ws.join("models/Qwen2-0.5B-Instruct/model.onnx");
    p.exists().then_some(p)
}

#[ignore = "diagnostic — set HOLOGRAM_AI_LIVE=1 + Qwen2 ONNX path"]
#[test]
fn qwen2_post_concretize_residual_shapes() {
    if !live_enabled() {
        eprintln!("skipping: set HOLOGRAM_AI_LIVE=1");
        return;
    }
    let Some(onnx) = locate_model() else {
        eprintln!("skipping: Qwen2 ONNX not found");
        return;
    };

    // Run the full pre-lower pipeline at the same seq_len the diag
    // harness uses (16) so we observe the same concrete shapes the
    // compiler does.
    let graph =
        hologram_ai_onnx::import_onnx_path(&onnx, hologram_ai_onnx::OnnxImportOptions::default())
            .expect("import");
    let pipeline = OptPipeline::mvp();
    let graph = pipeline.run(graph).expect("opt");
    let (graph, _zeroed) =
        hologram_ai::compiler::concretize_all_dims(graph, Some(16), None).expect("concretize");
    let graph = hologram_ai::compiler::post_concretization_repair(graph).expect("repair");

    eprintln!(
        "DIAG: post-concretize/repair, {} nodes, {} tensors",
        graph.nodes.len(),
        graph.tensor_info.len()
    );

    // Walk every FusedLayerNormResidual node and report the concrete
    // shape of its `residual` input. The kernel call's BufferRef.length
    // = element_count * dtype_width; if residual.length is much smaller
    // than x.length, the shape must have collapsed somewhere.
    let mut count = 0;
    for (idx, node) in graph.nodes.iter().enumerate() {
        if !matches!(node.op, AiOp::FusedLayerNormResidual { .. }) {
            continue;
        }
        if node.inputs.len() < 3 {
            continue;
        }
        let x_tid = node.inputs[0];
        let r_tid = node.inputs[1];
        let w_tid = node.inputs[2];
        let elem = |tid: u32| -> Option<u64> {
            let info = graph.tensor_info.get(&tid)?;
            let mut n = 1u64;
            for d in info.shape.iter() {
                match d.as_concrete() {
                    Some(v) => n = n.checked_mul(v)?,
                    None => return None,
                }
            }
            Some(n)
        };
        let fmt = |tid: u32| -> String {
            match graph.tensor_info.get(&tid) {
                Some(info) => format!("shape={:?} dtype={:?}", info.shape, info.logical_dtype),
                None => "<no tensor_info>".into(),
            }
        };
        eprintln!(
            "FusedLayerNormResidual[{count}] node_idx={idx}\n  \
             x(tid={x_tid}, elem={:?}) {}\n  \
             residual(tid={r_tid}, elem={:?}) {}\n  \
             weight(tid={w_tid}, elem={:?}) {}",
            elem(x_tid),
            fmt(x_tid),
            elem(r_tid),
            fmt(r_tid),
            elem(w_tid),
            fmt(w_tid),
        );
        // Trace the residual's producer node, if any.
        if let Some((p_idx, p_node)) = graph
            .nodes
            .iter()
            .enumerate()
            .find(|(_, n)| n.outputs.contains(&r_tid))
        {
            let p_op_name = format!("{:?}", p_node.op);
            let short: String = p_op_name.chars().take(80).collect();
            eprintln!("  residual producer: node_idx={p_idx} op={short}");
        } else {
            eprintln!(
                "  residual producer: <none> (graph_input={}, param={})",
                graph.inputs.contains(&r_tid),
                graph.params.contains_key(&r_tid)
            );
        }
        count += 1;
        if count >= 4 {
            break;
        }
    }
    if count == 0 {
        // No FusedLayerNormResidual produced — the imperative pipeline
        // probably emitted plain RmsNorm; hologram-compiler fuses at
        // load time. Walk plain RmsNorm chains instead and report any
        // whose Add input has mismatched-shape operands.
        for (idx, node) in graph.nodes.iter().enumerate() {
            let AiOp::RmsNorm { .. } = &node.op else {
                continue;
            };
            if node.inputs.is_empty() {
                continue;
            }
            let sum_tid = node.inputs[0];
            let Some((add_idx, add_node)) = graph
                .nodes
                .iter()
                .enumerate()
                .find(|(_, n)| n.outputs.contains(&sum_tid))
            else {
                continue;
            };
            if !matches!(add_node.op, AiOp::Add) || add_node.inputs.len() != 2 {
                continue;
            }
            let lhs = add_node.inputs[0];
            let rhs = add_node.inputs[1];
            let elem = |tid: u32| -> Option<u64> {
                let info = graph.tensor_info.get(&tid)?;
                let mut n = 1u64;
                for d in info.shape.iter() {
                    let v = d.as_concrete()?;
                    n = n.checked_mul(v)?;
                }
                Some(n)
            };
            let l_e = elem(lhs);
            let r_e = elem(rhs);
            if l_e != r_e {
                eprintln!(
                    "MISMATCH RmsNorm node_idx={idx} Add node_idx={add_idx}\n  \
                     lhs(tid={lhs}, elem={l_e:?}) shape={:?}\n  \
                     rhs(tid={rhs}, elem={r_e:?}) shape={:?}",
                    graph.tensor_info.get(&lhs).map(|i| &i.shape),
                    graph.tensor_info.get(&rhs).map(|i| &i.shape),
                );
                count += 1;
                if count >= 4 {
                    break;
                }
            }
        }
    }
    let _ = DimExpr::Concrete(0); // silence unused import if branches don't fire
}
