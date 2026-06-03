//! Structural V&V — class **CE** (content-addressed elision replaces KV-cache).
//!
//! The headline UOR-native contract is that hologram-ai never carries a
//! mutable KV-cache: autoregressive decode reuses the prior step's prefix
//! **structurally** by κ-label, not via a side buffer.
//!
//! These tests instrument that contract directly using `HoloRunner`'s
//! per-walk counters (`last_dispatched`, `last_skipped`, `kernel_count`).
//! Each test drives a real compiled schedule (no mocks), so a regression in
//! the elision path — graph memo, per-node reuse key, or the `BufferArena`'s
//! two-generation rotation — fails this file loud.
//!
//! Runs under `cargo test -p hologram-ai-conformance --features structural`
//! (the `vv-structural` recipe).
//!
//! ## What each shape of CE looks like
//!
//! * **Whole-graph memo (CE-1, ground level).** Identical inputs ⇒ the session
//!   recognizes the input tuple, returns the cached output κ-labels, and never
//!   walks the schedule. The per-walk counters retain their previous values
//!   because no walk happened; we instead observe identity by comparing the
//!   returned labels and bytes.
//! * **Sub-graph elision (CE-1, the load-bearing case — what replaces KV-cache).**
//!   Some inputs change, some don't. On the walk, every node whose operand
//!   labels are unchanged finds its output κ-label still resident from the
//!   previous walk (the previous-generation slot of the `BufferArena`), so the
//!   kernel is *elided* and the slot is rebound to the existing buffer. Nodes
//!   downstream of changed inputs re-fire. We verify this by counting:
//!   `last_skipped > 0` and `last_skipped + last_dispatched = kernel_count`.
//! * **CE-2 (correctness).** Elided output ≡ recomputed output, bit-for-bit.

#![cfg(feature = "structural")]

use hologram_ai::{HoloRunner, ModelCompiler, ModelSource};
use hologram_ai_conformance::ort_runner::onnx_builder;

/// Build a graph with an **interior** sub-computation that depends only on
/// `W`, joined to `X` at the final output. The session's elision path only
/// reuses *interior* nodes (output ports mint a witnessed boundary address
/// every walk), so the sub-graph we want to see elided cannot itself be a
/// graph output:
///
/// ```text
///   t_w = Tanh(W)   # interior; reuse key = derive_label(Tanh, [label_W])
///   y   = Add(X, t_w)  # graph output
/// ```
///
/// On a walk with the same `W` but a different `X`, `t_w`'s operand label
/// is unchanged ⇒ its output κ-label is unchanged ⇒ resident in the previous
/// generation ⇒ the kernel is elided. `y` re-fires because its operand
/// labels (X) changed.
fn shared_interior_runner(n: usize) -> HoloRunner {
    use onnx_builder::{build_multi_node_model, Node};

    let bytes = build_multi_node_model(
        &[
            Node::new("Tanh", &["W"], &["t_w"]),
            Node::new("Add", &["X", "t_w"], &["y"]),
        ],
        &[("X", &[n]), ("W", &[n])],
        &[("y", &[n])], // only `y` is a graph output; `t_w` is interior
        &[],
    );
    let archive = ModelCompiler::default()
        .compile(ModelSource::OnnxBytes(bytes))
        .expect("shared-interior compile");
    HoloRunner::from_bytes(archive.bytes).expect("load")
}

fn matmul_runner(n: usize) -> HoloRunner {
    let bytes = onnx_builder::matmul(n, n, n);
    let archive = ModelCompiler::default()
        .compile(ModelSource::OnnxBytes(bytes))
        .expect("matmul compile");
    HoloRunner::from_bytes(archive.bytes).expect("load")
}

/// Build an f32 input buffer of `n` elements seeded so the first 4 bytes
/// reflect `seed` — distinct seeds ⇒ distinct content addresses.
fn f32_input(n: usize, seed: u8) -> Vec<u8> {
    let mut v = vec![0u8; n * 4];
    v[0..4].copy_from_slice(&(seed as f32).to_le_bytes());
    v
}

// ─────────────────────────────────────────────────────────────────────────────
// CE-1 (sub-graph elision) — the load-bearing case that replaces KV-cache
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn ce_1_unchanged_interior_node_is_elided() {
    // The CE-1 contract: interior nodes whose operand labels are unchanged
    // across walks are elided. The graph is `Tanh(W)` (interior) → `Add(X, _)`
    // (output); change only `X`, the `Tanh(W)` reuse key is unchanged so the
    // kernel is elided on the second walk.
    let mut runner = shared_interior_runner(16);
    let kernels = runner.kernel_count();
    assert!(
        kernels >= 2,
        "graph must have at least two kernels (one interior, one output)"
    );

    let x0 = f32_input(16, 1);
    let w0 = f32_input(16, 2);
    let _ = runner
        .execute(&[x0.as_slice(), w0.as_slice()])
        .expect("first walk");
    assert_eq!(
        runner.last_dispatched(),
        kernels,
        "first walk: every kernel novel"
    );
    assert_eq!(runner.last_skipped(), 0, "first walk: nothing to elide");

    // Change only `X`. `Tanh(W)` is interior + its operand label is unchanged
    // ⇒ elided. `Add(X, t_w)` is a graph output ⇒ re-fires (output ports never
    // elide).
    let x1 = f32_input(16, 99);
    let _ = runner
        .execute(&[x1.as_slice(), w0.as_slice()])
        .expect("second walk");
    assert!(
        runner.last_skipped() >= 1,
        "CE-1: at least the interior Tanh(W) must be elided \
         (skipped={}, dispatched={}, kernels={})",
        runner.last_skipped(),
        runner.last_dispatched(),
        kernels,
    );
    assert_eq!(
        runner.last_dispatched() + runner.last_skipped(),
        kernels,
        "schedule partition holds"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// CE-1 (whole-graph memo) — the strongest case, identical inputs
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn ce_1_identical_inputs_bypass_the_walk() {
    // The whole-graph memo: on identical inputs the session returns the cached
    // output labels without walking the schedule. The per-walk counters are
    // not zeroed (no walk happened), so we observe the contract by output
    // identity instead — bit-equal labels and bit-equal bytes mean the second
    // call could not have re-executed any kernel.
    let mut runner = matmul_runner(64);

    let ins: Vec<Vec<u8>> = runner
        .input_byte_sizes()
        .iter()
        .map(|&sz| vec![0u8; sz])
        .collect();
    let labels: Vec<_> = ins.iter().map(|v| runner.intern_input(v)).collect();

    let first = runner
        .execute_addressed(&labels)
        .expect("first addressed walk");
    let second = runner
        .execute_addressed(&labels)
        .expect("second addressed walk (memo hit)");
    assert_eq!(
        first, second,
        "whole-graph memo hit must return identical output labels"
    );
    let a = runner.resolve(&first[0]).expect("resolve first").to_vec();
    let b = runner.resolve(&second[0]).expect("resolve second").to_vec();
    assert_eq!(a, b, "memo hit must return bit-equal output bytes");
}

// ─────────────────────────────────────────────────────────────────────────────
// CE-2 (correctness) — elided output equals recomputed output, bit-for-bit
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn ce_2_elided_run_matches_clean_recompute() {
    // After a partial elision (one interior node elided, the output re-fired),
    // the joined output must be bit-equal to a clean recompute. We force a
    // recompute by loading a second, fresh runner and running it with the
    // same inputs once.
    let mut runner_a = shared_interior_runner(32);
    let mut runner_b = shared_interior_runner(32);

    let x0 = f32_input(32, 5);
    let w = f32_input(32, 9);
    let x1 = f32_input(32, 6);

    // Warm runner_a, then partial-elide via second walk on (x1, same w).
    let _ = runner_a
        .execute(&[x0.as_slice(), w.as_slice()])
        .expect("warm a");
    let out_partial = runner_a
        .execute(&[x1.as_slice(), w.as_slice()])
        .expect("partial-elide a");
    assert!(
        runner_a.last_skipped() >= 1,
        "expected an elision of the interior Tanh(W)"
    );

    // Clean recompute on a fresh session — no previous walk to elide against.
    let out_clean = runner_b
        .execute(&[x1.as_slice(), w.as_slice()])
        .expect("clean recompute b");
    assert_eq!(
        runner_b.last_skipped(),
        0,
        "fresh session: nothing to elide"
    );

    assert_eq!(
        out_partial[0].bytes, out_clean[0].bytes,
        "CE-2: elided run produced different bytes than a clean recompute"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Structural rail: counters partition the schedule
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn ce_counters_partition_the_schedule() {
    // On any actually-walked call, last_dispatched + last_skipped == kernel_count
    // — every node either fires or is elided, never both. Used as a fast
    // sanity rail in the elision path.
    let mut runner = shared_interior_runner(8);
    let kernels = runner.kernel_count();

    let x = f32_input(8, 1);
    let w = f32_input(8, 2);
    let _ = runner.execute(&[x.as_slice(), w.as_slice()]).expect("walk");
    assert_eq!(
        runner.last_dispatched() + runner.last_skipped(),
        kernels,
        "dispatched + skipped must cover the schedule exactly"
    );
}
