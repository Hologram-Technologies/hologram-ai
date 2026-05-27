//! Performance contract (V&V class PV).
//!
//! hologram-ai is bound by the same performance contract as hologram. These
//! tests assert the load-bearing guarantees rather than absolute times (which
//! are machine-dependent):
//!
//! 1. **No arbitrary limit.** LLM-scale model *architectures* — 1B / 3B / 5B /
//!    20B parameters — compile without any hardcoded cap, dimension clamp, or
//!    integer-overflow ceiling (ADR-060: element counts are u64, never
//!    ceilinged at 4 GiB). Weights are graph inputs here, so the compile path
//!    is exercised at full LLM scale without materializing the weight bytes —
//!    the property under test is hologram-ai's *handling of the scale*, not
//!    this CI box's RAM.
//! 2. **Content-addressed reuse is the win.** Re-executing an unchanged graph
//!    on the same inputs hits the κ-label memo and returns in O(1) — far faster
//!    than recomputing — the contract that replaces a mutable KV-cache.
//! 3. **Bounded, weight-size-independent compile.** Compile cost tracks graph
//!    structure, not parameter count.
//!
//! Full-weight *execution* of billion-parameter models is hardware-bound (a 1B
//! f32 model is ~4 GB of weights, 20B is ~80 GB); that is a RAM/IO concern, not
//! an hologram-ai limit, and is left to appropriately-sized hardware. uor-addr
//! already validates the streaming/bounded-carrier addressing of a 531 MB GGUF
//! (see `ma_external_models.rs`).

use std::collections::HashMap;
use std::time::Instant;

use hologram_ai::{HoloRunner, ModelCompiler, ModelSource};
use hologram_ai_common::{shape_from_concrete, AiGraph, AiNode, AiOp, DType, TensorInfo};
use hologram_ai_conformance::ort_runner::onnx_builder;

/// Build a stack of `layers` full `[d, d]` matmuls with weights as graph
/// inputs (no embedded data). Parameter count = `layers · d²` — the dominant
/// term of a transformer's projection weights — so this is an LLM-scale
/// architecture with a compile-time footprint independent of param count.
fn matmul_stack(d: u64, layers: u64) -> AiGraph {
    let mut nodes = Vec::new();
    let mut tensor_info: HashMap<u32, TensorInfo> = HashMap::new();
    let row = shape_from_concrete(&[1, d]);
    let weight = shape_from_concrete(&[d, d]);

    // tid 0 = activation input X[1, d].
    tensor_info.insert(0, TensorInfo::new(DType::F32, row.clone()));
    let mut inputs: Vec<u32> = vec![0];
    let mut prev = 0u32;

    for i in 0..layers {
        let w_tid = 1 + i as u32; // weight input W_i [d, d]
        let out_tid = 1 + layers as u32 + i as u32; // matmul output [1, d]
        tensor_info.insert(w_tid, TensorInfo::new(DType::F32, weight.clone()));
        tensor_info.insert(out_tid, TensorInfo::new(DType::F32, row.clone()));
        inputs.push(w_tid);
        nodes.push(AiNode::new(
            i as u32,
            AiOp::MatMul,
            vec![prev, w_tid],
            vec![out_tid],
        ));
        prev = out_tid;
    }

    AiGraph {
        name: format!("matmul_stack_d{d}_l{layers}"),
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
fn billion_parameter_architectures_compile_without_arbitrary_limits() {
    // (label, d, layers) → param count = layers · d².
    let configs = [
        ("1B", 8192u64, 15u64),  // 15 · 8192²  ≈ 1.01 B
        ("3B", 8192, 45),        // 45 · 8192²  ≈ 3.02 B
        ("5B", 8192, 75),        // 75 · 8192²  ≈ 5.03 B
        ("20B", 16384, 75),      // 75 · 16384² ≈ 20.1 B
    ];
    for (label, d, layers) in configs {
        let params = layers * d * d;
        let graph = matmul_stack(d, layers);
        let t = Instant::now();
        let archive = ModelCompiler::default()
            .compile(ModelSource::AiGraph(graph))
            .unwrap_or_else(|e| panic!("[{label}] {params}-param architecture failed to compile: {e:#}"));
        let dt = t.elapsed();
        assert!(!archive.bytes.is_empty(), "[{label}] empty archive");
        println!(
            "[{label}] {params} params (d={d}, {layers} layers) compiled in {dt:?} → {} archive bytes",
            archive.bytes.len()
        );
    }
}

#[test]
fn compile_cost_is_independent_of_parameter_count() {
    // Two architectures differing ~5× in parameters but identical in graph
    // structure (same layer count) should compile in the same ballpark —
    // compile tracks structure, not weight bytes (weights never materialize).
    let small = matmul_stack(4096, 24); // ~0.4 B
    let large = matmul_stack(9216, 24); // ~2.0 B (≈5× params, same 24 layers)

    let t = Instant::now();
    ModelCompiler::default().compile(ModelSource::AiGraph(small)).unwrap();
    let small_dt = t.elapsed();

    let t = Instant::now();
    ModelCompiler::default().compile(ModelSource::AiGraph(large)).unwrap();
    let large_dt = t.elapsed();

    println!("compile: 0.4B {small_dt:?} vs 2.0B {large_dt:?}");
    // Allow generous slack for a shared CI VM; the point is sub-linear in
    // params (not 5×), proving weight bytes don't enter the compile cost.
    assert!(
        large_dt.as_secs_f64() < small_dt.as_secs_f64() * 3.0 + 0.05,
        "compile scaled with parameter count (0.4B {small_dt:?} vs 2.0B {large_dt:?})"
    );
}

#[test]
fn content_addressed_reuse_beats_recompute() {
    // A runnable model (256³ matmul). Cold = novel inputs → full recompute;
    // reuse = same κ-labels → whole-graph memo hit (O(1), no compute/copy).
    let model = onnx_builder::matmul(256, 256, 256);
    let archive = ModelCompiler::default()
        .compile(ModelSource::OnnxBytes(model))
        .expect("compile");
    let mut runner = HoloRunner::from_bytes(archive.bytes).expect("load");

    let sizes = runner.input_byte_sizes();
    // Non-trivial known input so the reference output is not all-zero: A and B
    // each set to the identity matrix, so Y = A·B = identity.
    let n = 256usize;
    let identity_bytes = {
        let mut v = vec![0u8; n * n * 4];
        for k in 0..n {
            v[(k * n + k) * 4..(k * n + k) * 4 + 4].copy_from_slice(&1.0f32.to_le_bytes());
        }
        v
    };
    let base: Vec<Vec<u8>> = sizes.iter().map(|_| identity_bytes.clone()).collect();

    // Reference output from a byte-level forward (independent of the reuse path).
    let refs0: Vec<&[u8]> = base.iter().map(|v| v.as_slice()).collect();
    let reference: Vec<u8> = runner.execute(&refs0).expect("reference")[0].bytes.clone();
    // I·I = I: diagonal 1.0, off-diagonal 0.0 — confirms the forward is correct.
    for r in 0..n {
        for c in 0..n {
            let off = (r * n + c) * 4;
            let v = f32::from_le_bytes(reference[off..off + 4].try_into().unwrap());
            let want = if r == c { 1.0 } else { 0.0 };
            assert!((v - want).abs() <= 1e-5, "I·I wrong at [{r},{c}]: {v} != {want}");
        }
    }

    // Warm + measure reuse (fixed labels → memo hit), and verify the memo hit
    // returns the *identical* bytes (content-addressing correctness, not just speed).
    let labels: Vec<_> = base.iter().map(|v| runner.intern_input(v)).collect();
    let reuse_labels = runner.execute_addressed(&labels).expect("warm");
    let reused_bytes = runner.resolve(&reuse_labels[0]).expect("resolve reuse output");
    assert_eq!(reused_bytes, reference.as_slice(), "memo hit returned different bytes");
    let reuse = {
        let t = Instant::now();
        for _ in 0..50 {
            runner.execute_addressed(&labels).expect("reuse");
        }
        t.elapsed() / 50
    };

    // Measure cold (novel inputs each call → recompute).
    let cold = {
        let t = Instant::now();
        for i in 0..50u8 {
            let mut ins = base.clone();
            for buf in ins.iter_mut() {
                if let Some(f) = buf.first_mut() {
                    *f = i.wrapping_add(1);
                }
            }
            let refs: Vec<&[u8]> = ins.iter().map(|v| v.as_slice()).collect();
            runner.execute(&refs).expect("cold");
        }
        t.elapsed() / 50
    };

    println!("256³ matmul: cold {cold:?} vs κ-label reuse {reuse:?}");
    // The memo hit must be strictly faster than recompute (the content-
    // addressing contract). Conservative on a shared VM; the real ratio is
    // orders of magnitude.
    assert!(
        reuse < cold,
        "content-addressed reuse ({reuse:?}) was not faster than recompute ({cold:?})"
    );
}

#[test]
fn matmul_sweep_runs_at_every_size() {
    // hologram's 64/128/256/512 sweep — every size compiles AND runs through
    // hologram-ai end to end (no size is special-cased or capped).
    for n in [64usize, 128, 256, 512] {
        let archive = ModelCompiler::default()
            .compile(ModelSource::OnnxBytes(onnx_builder::matmul(n, n, n)))
            .unwrap_or_else(|e| panic!("matmul {n}³ compile failed: {e:#}"));
        let mut runner = HoloRunner::from_bytes(archive.bytes).expect("load");
        let ins: Vec<Vec<u8>> = runner.input_byte_sizes().iter().map(|&b| vec![0u8; b]).collect();
        let refs: Vec<&[u8]> = ins.iter().map(|v| v.as_slice()).collect();
        let out = runner.execute(&refs).unwrap_or_else(|e| panic!("matmul {n}³ execute failed: {e:#}"));
        assert_eq!(out.len(), 1, "matmul {n}³: expected one output");
        assert_eq!(out[0].bytes.len(), n * n * 4, "matmul {n}³: output [{n},{n}] f32");
    }
}
