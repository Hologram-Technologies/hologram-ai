//! **Real published-model compile contract (V&V classes EE / PV).**
//!
//! Compiles a *real, authoritative* transformer — the onnx-community
//! `SmolLM2-135M-Instruct-ONNX` export (a Llama-family decoder: RoPE + GQA +
//! RMSNorm + SwiGLU, with a KV-cache `with-past` graph) — all the way through
//! `hologram_compiler::compile`, and asserts two load-bearing guarantees that
//! whole-model graphs (not single ops) are uniquely able to catch:
//!
//!  - **EE (end-to-end completeness).** The model compiles to a `.holo` with no
//!    `CompletenessFailure`. This guards the class of compiler-boundary gaps a
//!    real Llama graph exercises but synthetic single-op fixtures do not — e.g.
//!    the Transpose default-perm, batched/non-axis-0 Slice→Gather, and the ONNX
//!    `Expand`-must-broadcast (HF `.expand(..., -1, ...)`) shape idioms.
//!  - **PV (bounded compile).** Compilation finishes well within a wall-clock
//!    budget. This guards against the model-κ-labeling perf regression (full
//!    uor-addr canonicalization of a 540 MB ONNX on the compile critical path
//!    took *minutes*; it is opt-in / off by default — if that flips back on,
//!    this test trips instead of hanging CI).
//!
//! Why a real model and not a hand-built fixture: per `VERIFICATION.md`, the
//! authority must be one we did **not** author. The model architecture and its
//! exported graph are the authoritative artifact here; we only assert that
//! hologram-ai consumes it correctly.
//!
//! The model is large (≈540 MB) and git-ignored, so this test is **skip-safe**:
//! it runs only when `HOLOGRAM_AI_LIVE=1` *and* the model is present. Point it at
//! the export with `HOLOGRAM_AI_SMOLLM2_ONNX=/path/to/model.onnx`, or drop the
//! onnx-community files under `models/smollm2-135m/` at the workspace root. Build
//! with `--features onnx-spec`.
#![cfg(feature = "onnx-spec")]

use std::path::PathBuf;
use std::time::{Duration, Instant};

use hologram_ai::{HoloRunner, ModelCompiler, ModelSource};

/// Wall-clock ceiling for the compile. Normal compile (κ-labeling off) is a few
/// seconds even in a debug build; the regression this guards (540 MB ONNX
/// canonicalization on the critical path) takes minutes, so a generous ceiling
/// catches it without flaking on a slow/loaded CI box.
const COMPILE_BUDGET: Duration = Duration::from_secs(180);

/// Resolve the authoritative SmolLM2 ONNX path: explicit env override first,
/// then the conventional workspace location. `None` if neither exists.
fn smollm2_onnx_path() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("HOLOGRAM_AI_SMOLLM2_ONNX") {
        let p = PathBuf::from(p);
        return p.exists().then_some(p);
    }
    // CARGO_MANIFEST_DIR = <workspace>/crates/hologram-ai-conformance
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../models/smollm2-135m/model.onnx");
    p.exists().then_some(p)
}

#[test]
fn smollm2_real_model_compiles_within_budget() {
    if std::env::var("HOLOGRAM_AI_LIVE").as_deref() != Ok("1") {
        eprintln!("skipping: set HOLOGRAM_AI_LIVE=1 to run the real-model compile contract");
        return;
    }
    let Some(model_path) = smollm2_onnx_path() else {
        eprintln!(
            "skipping: SmolLM2 ONNX not found. Provide the authoritative onnx-community \
             `SmolLM2-135M-Instruct-ONNX` export via HOLOGRAM_AI_SMOLLM2_ONNX=/path/to/model.onnx \
             or place it at <workspace>/models/smollm2-135m/model.onnx"
        );
        return;
    };

    // Compile at a short sequence length: the property under test is graph
    // structure / completeness / compile cost, none of which depend on the
    // concretized seq, and a short seq keeps activation buffers bounded.
    let compiler = ModelCompiler {
        seq_len_override: Some(64),
        ..Default::default()
    };

    let started = Instant::now();
    let archive = compiler
        .compile(ModelSource::OnnxPath(model_path.clone()))
        .expect("EE: authoritative SmolLM2 must compile with no CompletenessFailure");
    let elapsed = started.elapsed();

    // EE — produced a non-trivial archive (a full decoder is hundreds of nodes).
    assert!(
        archive.stats.node_count > 100,
        "EE: expected a multi-layer decoder graph, got {} nodes",
        archive.stats.node_count
    );
    assert!(!archive.bytes.is_empty(), "EE: empty .holo archive");

    // PV — bounded compile (guards the κ-labeling perf regression).
    assert!(
        elapsed < COMPILE_BUDGET,
        "PV: compile took {elapsed:?} (> {COMPILE_BUDGET:?}) — likely the model-κ-labeling \
         canonicalization regressed back onto the compile critical path"
    );

    // The archive must load back into a session (archive/codec round-trip) —
    // a real decoder has many named KV-cache ports, so this exercises the
    // port/extension decode paths a single-op fixture never reaches.
    let runner =
        HoloRunner::from_bytes(archive.bytes).expect("EE: compiled .holo must load into a session");
    assert!(
        runner.input_count() > 0 && runner.output_count() > 0,
        "EE: loaded session must expose its input/output ports"
    );

    println!(
        "SmolLM2 EE/PV OK — {} nodes, compiled in {elapsed:?} (budget {COMPILE_BUDGET:?}), \
         {} inputs / {} outputs",
        archive.stats.node_count,
        runner.input_count(),
        runner.output_count()
    );
}
