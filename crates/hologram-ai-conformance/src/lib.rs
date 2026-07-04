//! Conformance testing harness for hologram float kernels.
//!
//! Validates `dispatch_float()` outputs against reference implementations.
//! Pure-Rust reference implementations cover all ops. ORT cross-validation
//! is available behind the `conformance` feature flag.
//!
//! # Modules
//!
//! - `tolerance` — per-op-category atol/rtol definitions + comparison
//! - `reference` — pure-Rust reference implementations for complex ops
//! - `comparator` — run dispatch_float and compare against expected output
//! - `ort_runner` — ORT session management + ONNX model builder (feature-gated)

pub mod comparator;
pub mod exec_comparator;
pub mod ort_runner;
pub mod reference;
pub mod tolerance;
#[cfg(feature = "conformance")]
pub mod validate_ort;
pub mod witness;

// Structural V&V harness (classes ZA/ZM/CE). Self-contained — no hologram
// dependency — so it is usable the moment the runtime core builds.
#[cfg(feature = "structural")]
pub mod alloc;
