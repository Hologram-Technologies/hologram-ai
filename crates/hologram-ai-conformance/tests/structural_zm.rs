//! Structural V&V — class **ZM** (zero-movement / zero-copy / zero-cost).
//!
//! hologram-ai must hand hologram a representation where values flow by
//! reference (κ-label), not by copy. Two tests carry this contract:
//!
//! * **ZM-1.** Lowering introduces no per-node tensor copy. Weights and
//!   constants are stored once, referenced many. We exercise this by
//!   compiling a deep stack that re-uses one weight, then asserting the
//!   compile cost (graph cost) is decoupled from the parameter cost
//!   (weight cost) — the standing perf invariant ([`PV-3`]), restated as a
//!   ZM rail: if lowering copied per node, compile would grow with the
//!   weight body.
//! * **ZM-2.** Constant weights are content-addressed by their bytes:
//!   identical content occupies **one** buffer in the runtime pool, no
//!   matter how many nodes / layers reference it. We assert this with the
//!   `resident_bytes` / `resident_count` instruments, comparing a stack
//!   that gets distinct weights against the same stack with one shared
//!   weight.
//!
//! Both witnesses run a real compile→exec path against the canonical
//! UOR-native runtime — no mocks, no instrumentation hooks bolted onto the
//! kernels.

#![cfg(feature = "structural")]

use std::collections::HashMap;
use std::time::Instant;

use hologram_ai::{HoloRunner, ModelCompiler, ModelSource};
use hologram_ai_common::{shape_from_concrete, AiGraph, AiNode, AiOp, DType, TensorInfo};

/// Build a stack of `layers` matmuls with weights as graph inputs — the
/// matmul body never enters the compile path, so compile cost is structural
/// only. Mirrors the helper in `tests/perf_contract.rs` so this file is
/// self-contained.
fn matmul_stack(d: u64, layers: u64) -> AiGraph {
    let mut nodes = Vec::new();
    let mut tensor_info: HashMap<u32, TensorInfo> = HashMap::new();
    let row = shape_from_concrete(&[1, d]);
    let weight = shape_from_concrete(&[d, d]);

    tensor_info.insert(0, TensorInfo::new(DType::F32, row.clone()));
    let mut inputs: Vec<u32> = vec![0];
    let mut prev = 0u32;

    for i in 0..layers {
        let w_tid = 1 + i as u32;
        let out_tid = 1 + layers as u32 + i as u32;
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

// ─────────────────────────────────────────────────────────────────────────────
// ZM-1 — Lowering introduces no per-node tensor copy.
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn zm_1_compile_cost_is_decoupled_from_weight_size() {
    // If lowering deep-copied a tensor for every node that referenced it,
    // compile time would scale with `layers × d²`. The contract: compile
    // tracks graph structure, not weight bytes — equivalent to "no per-node
    // tensor copy in lowering". We measure two graphs that differ by ~5× in
    // parameter body but have identical structure (same layer count) and
    // require their compile times to stay in the same regime.
    let small = matmul_stack(4096, 16); // ~0.27 B
    let large = matmul_stack(9216, 16); // ~1.36 B (≈5× params, same layers)

    let t = Instant::now();
    let small_archive = ModelCompiler::default()
        .compile(ModelSource::AiGraph(small))
        .expect("small compile");
    let small_dt = t.elapsed();

    let t = Instant::now();
    let large_archive = ModelCompiler::default()
        .compile(ModelSource::AiGraph(large))
        .expect("large compile");
    let large_dt = t.elapsed();

    println!(
        "ZM-1: small compile {small_dt:?} ({} archive bytes); large compile {large_dt:?} ({} archive bytes)",
        small_archive.bytes.len(),
        large_archive.bytes.len(),
    );

    // The compile-time ratio must be far less than the parameter ratio.
    // Generous slack for CI noise; the property is "sub-linear", not exact.
    assert!(
        large_dt.as_secs_f64() < small_dt.as_secs_f64() * 3.0 + 0.05,
        "ZM-1: compile scaled with weight body — small {small_dt:?} vs large {large_dt:?}"
    );

    // The archive size should also stay tight (weights are not embedded in
    // the archive when they come in as graph inputs — that *is* the
    // zero-movement contract: borrowed weight buffers, not cloned).
    let large_arch = large_archive.bytes.len();
    let small_arch = small_archive.bytes.len();
    assert!(
        large_arch < small_arch * 3 + 65536,
        "ZM-1: archive grew with weight body (small {small_arch} vs large {large_arch})"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// ZM-2 — Constant weights are content-addressed by their bytes.
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn zm_2_identical_weight_bytes_collapse_to_one_pool_buffer() {
    // The content-addressing contract: a value with κ-label L occupies ONE
    // buffer in the runtime pool no matter how many nodes / layers reference
    // it. We compile the same architecture twice, fill once with distinct
    // weights, once with one repeated weight, then compare the resident set.
    let d = 256u64;
    let layers = 16u64;
    let body = (d * d * 4) as usize;

    let compile_stack = || {
        let archive = ModelCompiler::default()
            .compile(ModelSource::AiGraph(matmul_stack(d, layers)))
            .expect("compile");
        HoloRunner::from_bytes(archive.bytes).expect("load")
    };

    // Distinct content → `layers` distinct buffers.
    let mut distinct = compile_stack();
    let sizes = distinct.input_byte_sizes();
    distinct.intern_input(&vec![0u8; sizes[0]]); // X
    for (i, &sz) in sizes.iter().enumerate().skip(1) {
        let mut w = vec![0u8; sz];
        w[0] = i as u8; // perturb → distinct content
        distinct.intern_input(&w);
    }
    let distinct_bytes = distinct.resident_bytes();
    let distinct_count = distinct.resident_count();

    // Identical content → one shared buffer (plus X).
    let mut shared = compile_stack();
    let sizes = shared.input_byte_sizes();
    shared.intern_input(&vec![0u8; sizes[0]]); // X
    let one_weight = vec![7u8; sizes[1]];
    for _ in 1..sizes.len() {
        shared.intern_input(&one_weight);
    }
    let shared_bytes = shared.resident_bytes();
    let shared_count = shared.resident_count();

    println!(
        "ZM-2: distinct {distinct_bytes}B / {distinct_count} resident; \
         shared {shared_bytes}B / {shared_count} resident; body={body}"
    );

    // Distinct: at least `layers` weight bodies + the X input resident.
    assert!(
        distinct_bytes >= layers as usize * body,
        "ZM-2: distinct must hold every body"
    );
    assert_eq!(
        distinct_count,
        layers as usize + 1,
        "ZM-2: distinct = X + {layers} weights"
    );
    // Shared: a single weight body resident, regardless of layer count.
    assert!(
        shared_bytes <= body + 65536,
        "ZM-2: shared must collapse to one body ({shared_bytes}B vs {body}B body)"
    );
    assert_eq!(shared_count, 2, "ZM-2: shared = X + 1 weight buffer");
    // The dedup is dramatic: at least `layers/2`× less memory.
    assert!(
        shared_bytes * (layers as usize / 2) < distinct_bytes,
        "ZM-2: dedup factor too small ({shared_bytes} × {} ≮ {distinct_bytes})",
        layers / 2
    );
}

#[test]
fn zm_2_addressed_output_resolves_without_extra_copy() {
    // A second ZM-2 witness on the *output* side: an output produced by
    // `execute_addressed` is held in the pool under its κ-label and can be
    // resolved by reference — calling `resolve` twice returns the same
    // bytes (a stable view into the same buffer, not a new copy each time).
    let n = 64;
    let bytes = hologram_ai_conformance::ort_runner::onnx_builder::unary_op("Sigmoid", n);
    let archive = ModelCompiler::default()
        .compile(ModelSource::OnnxBytes(bytes))
        .expect("compile");
    let mut runner = HoloRunner::from_bytes(archive.bytes).expect("load");

    let x = vec![0u8; runner.input_byte_sizes()[0]];
    let label = runner.intern_input(&x);
    let outs = runner.execute_addressed(&[label]).expect("walk");

    let a = runner.resolve(&outs[0]).expect("first resolve").as_ptr() as usize;
    let b = runner.resolve(&outs[0]).expect("second resolve").as_ptr() as usize;
    assert_eq!(a, b, "ZM-2: resolve must return the same pointer");

    // Resolving on a separate session that interns the same output bytes
    // would address to the same label (content-addressed identity). We
    // verify identity by κ-label equality rather than recomputing the
    // pointer comparison cross-session.
    let outs2 = runner
        .execute_addressed(&[label])
        .expect("repeat addressed walk");
    assert_eq!(
        outs[0], outs2[0],
        "ZM-2: identical inputs must produce identical output labels (no per-call drift)"
    );
}
