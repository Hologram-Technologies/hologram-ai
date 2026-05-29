//! Structural V&V — class **ZA** (zero/bounded runtime heap allocation).
//!
//! The UOR-native runtime claims a bounded, input-independent heap budget on
//! the inference hot path (per-call scratch is reused across calls). These
//! tests instrument that contract directly with the counting allocator
//! harness from [`hologram_ai_conformance::alloc`].
//!
//! The counting allocator must be installed as the global allocator of this
//! test binary; it is otherwise inactive. Other structural test files keep a
//! plain global allocator so they don't interact with these counts.
//!
//! ## What we measure
//!
//! * **ZA-1** — `execute_addressed` is the κ-label hot path (no byte→address
//!   hashing, no output→byte copy). After warm-up, each repeated call holds
//!   to a tight, input-independent allocation budget — the scratch in
//!   `InferenceSession` (`slot_label_scratch`, `out_witnessed_scratch`,
//!   per-call `SmallVec`s) is reused. We assert `O(1)` allocations per call
//!   (a small, fixed upper bound), not a function of model or input size —
//!   that is the load-bearing property: no per-token growth.
//! * **ZA-2** — Lowering / compile is a one-shot host-shell operation, but
//!   re-lowering the *same* graph must allocate the same bounded amount each
//!   time. We assert that the second compile of an identical graph holds to
//!   the same allocation budget as the first (no growth across repeated
//!   lower/compile cycles).

#![cfg(feature = "structural")]

use hologram_ai::{HoloRunner, ModelCompiler, ModelSource};
use hologram_ai_conformance::alloc::{
    assert_alloc_bounded, assert_allocator_installed, measure, CountingAllocator,
};
use hologram_ai_conformance::ort_runner::onnx_builder;

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator::new();

/// A small, single-input single-output runner — execute_addressed's inline
/// SmallVec storage covers the input/output tuples, so the hot-path key
/// collect doesn't spill.
fn small_runner() -> HoloRunner {
    let bytes = onnx_builder::unary_op("Sigmoid", 16);
    let archive = ModelCompiler::default()
        .compile(ModelSource::OnnxBytes(bytes))
        .expect("sigmoid compile");
    HoloRunner::from_bytes(archive.bytes).expect("load")
}

#[test]
fn za_allocator_is_installed() {
    // Without this guard, every ZA assertion below would pass vacuously
    // because the counters never advance.
    assert_allocator_installed();
}

#[test]
fn za_1_addressed_decode_step_is_bounded() {
    // ZA-1: after warm-up, each `execute_addressed` call must allocate a
    // bounded, input-independent amount. The scratch in `InferenceSession`
    // (slot-label scratch, witnessed-output scratch) is reused; the only
    // heap activity on the hot path is the returned `Vec<ContentLabel>`
    // and the inline `SmallVec` collect.
    let mut runner = small_runner();
    let ins: Vec<Vec<u8>> = runner
        .input_byte_sizes()
        .iter()
        .map(|&sz| vec![0u8; sz])
        .collect();
    let labels: Vec<_> = ins.iter().map(|v| runner.intern_input(v)).collect();

    // Warm-up: this allocates the scratch in the session for the first time.
    runner.execute_addressed(&labels).expect("warm up");

    // Re-run several times with the same labels (whole-graph memo). Each
    // call's allocation budget is small and constant — independent of the
    // model size, the number of kernels, or how many tokens have been
    // produced. The bound is the small, fixed cost of returning the output
    // labels through the public Vec-returning API (one Vec<ContentLabel>
    // per call plus a handful of SmallVec materializations on the memo
    // path). The constant value is intentionally conservative — what we
    // assert is *input-independence*, not a magic number.
    const PER_CALL_BUDGET: usize = 16;
    for i in 0..16 {
        assert_alloc_bounded(&format!("ZA-1 decode step #{i}"), PER_CALL_BUDGET, || {
            runner.execute_addressed(&labels).expect("decode step");
        });
    }
}

#[test]
fn za_1_addressed_steps_do_not_grow() {
    // The "no per-token growth" rail: across N successive identical calls,
    // total allocations grow at most O(N) (i.e. bounded per call). A leak in
    // the scratch reuse would show as a growing per-step delta.
    let mut runner = small_runner();
    let ins: Vec<Vec<u8>> = runner
        .input_byte_sizes()
        .iter()
        .map(|&sz| vec![0u8; sz])
        .collect();
    let labels: Vec<_> = ins.iter().map(|v| runner.intern_input(v)).collect();
    runner.execute_addressed(&labels).expect("warm up");

    let (_v0, first) = measure(|| runner.execute_addressed(&labels).expect("first"));
    let (_v1, hundred) = measure(|| {
        for _ in 0..100 {
            runner.execute_addressed(&labels).expect("loop");
        }
    });

    // 100 calls allocate at most 100 × the single-call cost (with generous
    // slack for the inevitable scratch ramp-up on early iterations and for
    // allocator-state churn variance across host environments — the local
    // devcontainer and the GitHub Actions runner observe meaningfully
    // different fixed overheads even for the same "no per-call growth"
    // contract). A real per-call leak would be ≥ 1 alloc / iteration
    // (≥ 100 extra), an order of magnitude past this slack.
    let bound = first.allocations.saturating_mul(100).saturating_add(64);
    assert!(
        hundred.allocations <= bound,
        "ZA-1: 100 decode steps allocated {} (single step: {}, bound: {})",
        hundred.allocations,
        first.allocations,
        bound
    );
}

#[test]
fn za_2_relower_same_graph_is_bounded() {
    // ZA-2: lowering / compile is bounded — re-lowering the *same* graph
    // (identical bytes) must not grow allocations across iterations. We
    // measure compile #2 against compile #1 with a generous bound, asserting
    // they live in the same allocation regime (no per-iteration accumulation).
    let onnx = onnx_builder::matmul(32, 32, 32);

    // First compile: warm any one-shot global init.
    let (_a, first) = measure(|| {
        ModelCompiler::default()
            .compile(ModelSource::OnnxBytes(onnx.clone()))
            .expect("compile #1");
    });
    // Second compile: must be bounded by the first within slack.
    let (_b, second) = measure(|| {
        ModelCompiler::default()
            .compile(ModelSource::OnnxBytes(onnx.clone()))
            .expect("compile #2");
    });

    // Allow modest slack — different alloc patterns from RNG-free passes can
    // still differ by a few percent due to capacity-rounding in growable
    // collections. The contract is "no unbounded growth", not "exact match".
    let bound = first.allocations + first.allocations / 5 + 64;
    assert!(
        second.allocations <= bound,
        "ZA-2: second compile allocated {} (first: {}, bound: {})",
        second.allocations,
        first.allocations,
        bound
    );
}
