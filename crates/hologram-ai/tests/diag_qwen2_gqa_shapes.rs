//! V&V diagnostic: Qwen2 attention slot bug — measure where the K
//! buffer's allocated size diverges from the AttentionCall's expected
//! `kv_heads * seq * head_dim * 4` bytes.
//!
//! Runs the full hologram-ai pipeline up through the optimization +
//! concretization passes, then walks the AiGraph and reports, for every
//! `GroupedQueryAttention` node:
//!   - its declared `(num_heads, num_kv_heads, head_dim)`
//!   - the **declared shape of its K input tensor** (from `tensor_info`)
//!   - the K tensor's expected byte size (= `prod(shape) * dtype_width`)
//!   - the AttentionCall's expected K byte size
//!     (= `batch * num_kv_heads * seq * head_dim * 4`)
//!
//! If the two disagree, hologram-ai's shape inference is the bug. If
//! they agree, the bug is below hologram-ai (at hologram-compiler's
//! slot lifetime / allocation).
//!
//! Gated by `HOLOGRAM_AI_LIVE=1` + `HOLOGRAM_AI_QWEN2_ONNX=<path>` (or
//! the conventional `<workspace>/models/Qwen2-0.5B-Instruct/model.onnx`).

use std::path::PathBuf;

use hologram_ai_common::{AiOp, OptPipeline};
// `Pass::run` is consumed indirectly via `OptPipeline::run`; not needed
// directly.

fn live_enabled() -> bool {
    std::env::var("HOLOGRAM_AI_LIVE").as_deref() == Ok("1")
}

fn locate_model() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("HOLOGRAM_AI_QWEN2_ONNX") {
        let p = PathBuf::from(p);
        return p.exists().then_some(p);
    }
    let ws = {
        let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        p.pop();
        p.pop();
        p
    };
    let p = ws.join("models/Qwen2-0.5B-Instruct/model.onnx");
    p.exists().then_some(p)
}

#[ignore = "diagnostic — set HOLOGRAM_AI_LIVE=1 + Qwen2 ONNX path"]
#[test]
fn qwen2_attention_k_shape_at_gqa_inputs() {
    if !live_enabled() {
        eprintln!("skipping: set HOLOGRAM_AI_LIVE=1");
        return;
    }
    let Some(onnx) = locate_model() else {
        eprintln!("skipping: Qwen2 ONNX not found");
        return;
    };
    eprintln!("DIAG: importing {}", onnx.display());

    // Import → optimize, mirroring the early phases of `ModelCompiler::compile`
    // (we stop short of concretize+lower so we see the symbolic AiGraph).
    let graph =
        hologram_ai_onnx::import_onnx_path(&onnx, hologram_ai_onnx::OnnxImportOptions::default())
            .expect("import");

    let pipeline = OptPipeline::mvp();
    let graph = pipeline.run(graph).expect("optimize");

    eprintln!("DIAG: post-opt graph has {} nodes", graph.nodes.len());

    // Walk every GQA node and dump K's declared shape.
    let mut gqa_count = 0;
    for (idx, node) in graph.nodes.iter().enumerate() {
        let AiOp::GroupedQueryAttention {
            num_heads,
            num_kv_heads,
            head_dim,
            ..
        } = &node.op
        else {
            continue;
        };
        if node.inputs.len() < 3 {
            eprintln!("GQA #{idx}: <3 inputs?!");
            continue;
        }
        let k_tid = node.inputs[1];
        let v_tid = node.inputs[2];
        let q_tid = node.inputs[0];

        let q_info = graph.tensor_info.get(&q_tid);
        let k_info = graph.tensor_info.get(&k_tid);
        let v_info = graph.tensor_info.get(&v_tid);

        let fmt_shape = |info: Option<&hologram_ai_common::TensorInfo>| match info {
            Some(i) => format!("{:?}  dtype={:?}", i.shape, i.logical_dtype),
            None => "<no tensor_info>".to_string(),
        };
        let bytes_of = |info: Option<&hologram_ai_common::TensorInfo>| match info {
            Some(i) => {
                let mut n: u64 = 1;
                for d in i.shape.iter() {
                    match d.as_concrete() {
                        Some(v) => n *= v,
                        None => return None, // symbolic
                    }
                }
                Some(n * 4) // assume f32
            }
            None => None,
        };

        let q_bytes = bytes_of(q_info);
        let k_bytes = bytes_of(k_info);
        let v_bytes = bytes_of(v_info);

        // What the AttentionCall expects (at compile-time we have to
        // assume some seq; we report kv_total at seq=16 — the diag
        // harness's seq_len_override).
        let seq = 16u64;
        let expected_k_bytes = (*num_kv_heads as u64) * seq * (*head_dim as u64) * 4;

        eprintln!(
            "GQA[{gqa_count}] node_idx={idx} num_heads={num_heads} num_kv_heads={num_kv_heads} head_dim={head_dim}\n  \
             Q(tid={q_tid}) shape={} bytes={:?}\n  \
             K(tid={k_tid}) shape={} bytes={:?}  expected_at_seq16={}\n  \
             V(tid={v_tid}) shape={} bytes={:?}",
            fmt_shape(q_info),
            q_bytes,
            fmt_shape(k_info),
            k_bytes,
            expected_k_bytes,
            fmt_shape(v_info),
            v_bytes,
        );

        // Trace K's producer chain — what node emitted this K?
        let k_producer = graph
            .nodes
            .iter()
            .enumerate()
            .find(|(_, n)| n.outputs.contains(&k_tid));
        if let Some((p_idx, p_node)) = k_producer {
            eprintln!(
                "  K producer: node_idx={p_idx} id={} op={:?} inputs={:?}",
                p_node.id, p_node.op, p_node.inputs
            );
        } else {
            eprintln!(
                "  K producer: <none> (K is a graph input? in graph.inputs={})",
                graph.inputs.contains(&k_tid)
            );
        }

        // The GQA's OWN output dtype is what the compiler uses for the
        // AttentionCall.dtype — if this is BOOL the kernel reads each
        // element as 1 byte instead of 4 (and the slot-too-small panic
        // is symptomatic of that mismatch).
        let out_tid = node.outputs.first().copied();
        if let Some(out) = out_tid {
            let info = graph.tensor_info.get(&out);
            eprintln!("  GQA output(tid={out}) shape={}", fmt_shape(info));
        }

        gqa_count += 1;
        if gqa_count >= 3 {
            eprintln!("(stopping after first 3 GQA nodes)");
            break;
        }
    }

    assert!(
        gqa_count > 0,
        "expected at least one GroupedQueryAttention node"
    );
}
