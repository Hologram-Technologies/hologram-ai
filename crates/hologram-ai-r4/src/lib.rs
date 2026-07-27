//! Private adapter between hologram-ai and `uor-r4` (ADR-0001).
//!
//! This crate is the only workspace crate that depends on uor-r4. All
//! current-upstream compatibility code lives behind this narrow module;
//! nothing in the public surface carries a uor-r4 type, so the
//! `hologram-ai` facade is independent of uor-r4's crate layout.
//!
//! # What lives here
//!
//! - [`compile_source_to_bundle`]: verified local HF-style source →
//!   uor-r4 three-stage compile (via `uor-r4-api`) → deterministic
//!   schema v1 inference bundle ([`hologram_ai_bundle`]). R4G1-only
//!   (ADR-0002): the bundle carries the scored graph, signature artifact,
//!   tokenizer (when the compile produced one), and score report — never
//!   `tless_store.bin`, never corpus/cover intermediates. The compile
//!   report is a work-dir diagnostic, not a deployed component, so it is
//!   deliberately omitted from the bundle (ADR-0005: bundles carry only
//!   what the runtime needs; its digest still appears nowhere — quality
//!   provenance is pinned via `quality_report_digest` over the score
//!   report).
//! - [`Engine`]: the loaded R4G1 inference engine behind hologram-ai's
//!   typed status/abstention vocabulary, with an allocation-free
//!   steady-state step path (ADR-0009).
//!
//! # Key adapter decisions
//!
//! - **Engine ownership.** `uor_r4_api::R4Engine` owns every byte it
//!   loads (it parses the borrowed component slices into owned state at
//!   `load`), so [`Engine`] simply wraps it — no self-referential
//!   buffers, no unsafe. The bundle's component slices are only borrowed
//!   for the duration of [`Engine::load`].
//! - **Generation.** [`Engine::generate_into`] delegates the per-step
//!   loop to `uor_r4_api::R4Engine::generate_into` rather than
//!   re-implementing it over `predict_decision`. The EOS decision (which
//!   token ids end a run) and the sliding-window policy are upstream
//!   contract details that are not exported for reuse; duplicating them
//!   here would fork the policy. Consequences, accepted deliberately:
//!   stream events are emitted as a batch when the run finishes (not
//!   per step), and a cancellation token set via
//!   [`Engine::set_cancellation`] is honored at run entry, not between
//!   steps. Single-step [`Engine::predict_next_into`] is fully
//!   allocation-free and cancellation-free by construction.
//! - **Error mapping.** Upstream `LoadError`/`InferenceError`/
//!   `CompileError` map onto the stable `hologram-ai-core` categories;
//!   no upstream error type leaks across the boundary.

#![forbid(unsafe_code)]

pub mod compile;
pub mod engine;

pub use compile::{
    compile_source_to_bundle, CompileOptions, SourceIdentity, COMPILER_REVISION,
    DEFAULT_MAX_RESUMES,
};
pub use engine::{Engine, FinishInfo, PredictOutcome, Prediction};
