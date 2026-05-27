//! PV-5 — full-weight execution of billion-parameter models.
//!
//! Unlike `perf_contract.rs` (which validates the *compile* path at LLM scale
//! with zero weight bytes), this actually **runs a forward pass with real
//! billion-parameter weights** and confirms the content-addressed reuse
//! contract holds with weights resident. Weights are supplied as graph inputs,
//! so they live once in the session's content-addressed pool — the compile
//! stays O(graph), and peak memory is ~the weight set, not a multiple of it.
//!
//! Gated behind `HOLOGRAM_AI_LARGE=1` because the weight set is large (1B f32 ≈
//! 4 GB). Run with:
//!   HOLOGRAM_AI_LARGE=1 cargo test --release -p hologram-ai \
//!     --test perf_contract_large -- --nocapture --test-threads=1
//!
//! `HOLOGRAM_AI_PARAMS` (default 1_000_000_000) overrides the target parameter
//! count so the same test scales to whatever RAM the host has.

use std::collections::HashMap;
use std::time::Instant;

use hologram_ai::{HoloRunner, ModelCompiler, ModelSource};
use hologram_ai_common::{shape_from_concrete, AiGraph, AiNode, AiOp, DType, TensorInfo};

/// `layers` chained `[d, d]` matmuls; weights are graph inputs (tids 1..=layers)
/// supplied at execution. Param count = `layers · d²`.
fn matmul_stack(d: u64, layers: u64) -> AiGraph {
    let mut nodes = Vec::new();
    let mut tensor_info: HashMap<u32, TensorInfo> = HashMap::new();
    let row = shape_from_concrete(&[1, d]);
    let weight = shape_from_concrete(&[d, d]);
    tensor_info.insert(0, TensorInfo::new(DType::F32, row.clone()));
    let mut inputs = vec![0u32];
    let mut prev = 0u32;
    for i in 0..layers {
        let w = 1 + i as u32;
        let out = 1 + layers as u32 + i as u32;
        tensor_info.insert(w, TensorInfo::new(DType::F32, weight.clone()));
        tensor_info.insert(out, TensorInfo::new(DType::F32, row.clone()));
        inputs.push(w);
        nodes.push(AiNode::new(i as u32, AiOp::MatMul, vec![prev, w], vec![out]));
        prev = out;
    }
    AiGraph {
        name: format!("mm_stack_d{d}_l{layers}"),
        nodes,
        inputs,
        outputs: vec![prev],
        input_names: Vec::new(),
        output_names: Vec::new(),
        params: HashMap::new(),
        tensor_info,
        metadata: HashMap::new(),
        warnings: Vec::new(),
        dim_vars: Default::default(),
        shape_constraints: Default::default(),
        subgraphs: HashMap::new(),
        tensor_names: HashMap::new(),
        topo_cache: Default::default(),
    }
}

#[test]
fn full_weight_billion_param_forward_and_reuse() {
    if std::env::var("HOLOGRAM_AI_LARGE").as_deref() != Ok("1") {
        eprintln!("SKIP: set HOLOGRAM_AI_LARGE=1 to run full-weight billion-param execution");
        return;
    }
    let target: u64 = std::env::var("HOLOGRAM_AI_PARAMS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1_000_000_000);

    // Choose d=8192 (a real LLM hidden size) and enough layers to hit target.
    let d = 8192u64;
    let layers = (target / (d * d)).max(1);
    let params = layers * d * d;
    eprintln!("target {target} params → d={d}, {layers} layers = {params} params ({:.2} GB f32 weights)",
        (params * 4) as f64 / 1e9);

    let graph = matmul_stack(d, layers);
    let t = Instant::now();
    let archive = ModelCompiler::default()
        .compile(ModelSource::AiGraph(graph))
        .expect("compile failed");
    eprintln!("compiled in {:?} → {} archive bytes", t.elapsed(), archive.bytes.len());

    let mut runner = HoloRunner::from_bytes(archive.bytes).expect("load failed");

    // Build *verifiable* weights: every weight is the identity matrix, so the
    // composition of all `layers` matmuls is the identity and the output must
    // equal the input X exactly — `out[j] = Σ_k x[k]·I[k,j] = x[j]`. hologram
    // doesn't know the weights are identity, so it still reads all 3.76 GB and
    // performs the full billion MACs; we just get a known-correct answer to
    // check against. (Matmul *arithmetic* on non-trivial weights is verified by
    // EE-1 against ONNX Runtime; this verifies the billion-param data path.)
    //
    // input order is [X, W_0 .. W_{layers-1}] (graph-input order).
    let sizes = runner.input_byte_sizes();
    let total_gb: f64 = sizes.iter().map(|&n| n as f64).sum::<f64>() / 1e9;
    eprintln!("allocating {} input buffers, {:.2} GB total", sizes.len(), total_gb);

    // Known input X[0, j] = (j mod 13) as f32 — varied and non-zero.
    let x: Vec<f32> = (0..d).map(|j| (j % 13) as f32).collect();
    let mut buffers: Vec<Vec<u8>> = Vec::with_capacity(sizes.len());
    buffers.push(x.iter().flat_map(|v| v.to_le_bytes()).collect());
    for _ in 1..sizes.len() {
        // Identity [d, d] as f32 LE bytes: zeros with 1.0 on the diagonal.
        let mut w = vec![0u8; (d * d * 4) as usize];
        for k in 0..d as usize {
            let off = (k * d as usize + k) * 4;
            w[off..off + 4].copy_from_slice(&1.0f32.to_le_bytes());
        }
        buffers.push(w);
    }

    // Forward pass on real billion-parameter weights.
    let refs: Vec<&[u8]> = buffers.iter().map(|v| v.as_slice()).collect();
    let t = Instant::now();
    let out = runner.execute(&refs).expect("forward execute failed");
    let forward = t.elapsed();
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].bytes.len(), (d * 4) as usize, "output is [1, d] f32");
    eprintln!("forward pass over {params} params: {forward:?}");

    // VERIFY: identity composition ⇒ output == input X (exact in f32: each
    // output element is one weighted term plus zeros).
    let y: Vec<f32> = out[0]
        .bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(y.len(), d as usize, "output element count");
    let mut max_err = 0.0f32;
    for (j, (&yj, &xj)) in y.iter().zip(x.iter()).enumerate() {
        let e = (yj - xj).abs();
        max_err = max_err.max(e);
        assert!(
            e <= 1e-4,
            "billion-param forward is WRONG at element {j}: got {yj}, expected {xj}"
        );
    }
    eprintln!("output VERIFIED correct (identity composition, max |err| = {max_err:.2e})");

    // Content-addressed reuse: intern once, re-run by κ-label (memo hit).
    let labels: Vec<_> = buffers.iter().map(|v| runner.intern_input(v)).collect();
    runner.execute_addressed(&labels).expect("warm");
    let t = Instant::now();
    for _ in 0..20 {
        runner.execute_addressed(&labels).expect("reuse");
    }
    let reuse = t.elapsed() / 20;
    eprintln!("κ-label reuse (memo hit) over {params} params: {reuse:?}");

    assert!(
        reuse < forward,
        "reuse ({reuse:?}) must beat the cold forward ({forward:?}) at billion-param scale"
    );
    eprintln!(
        "PV-5 OK: {params}-param full-weight forward {forward:?}, reuse {reuse:?} ({:.0}× faster)",
        forward.as_secs_f64() / reuse.as_secs_f64().max(1e-9)
    );
}
