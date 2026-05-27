//! Output comparison utilities for conformance.
//!
//! Per-kernel numeric conformance (class KC) is hologram's responsibility — it
//! validates each `OpKind` against the ONNX op spec / an independent f64
//! reference. hologram-ai's harness instead validates **import** (IM),
//! **lowering** (LW), **quantization** (QZ), and **end-to-end** output (EE)
//! against external authorities (ONNX Runtime / PyTorch / GGML). Those use the
//! shared comparison primitives below and the exec-level comparator
//! (`exec_comparator`), not an op-level dispatch path (the old `dispatch_float`
//! surface no longer exists in the UOR-native runtime).

pub use crate::tolerance::{compare_outputs, tolerance_for, ComparisonResult, Tolerance};
