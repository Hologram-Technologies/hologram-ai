//! Padding operation translators.
//!
//! This module provides translators for ONNX padding operations including:
//! - Pad: Pad tensor with various modes (constant, reflect, edge, wrap)

mod pad_op;

pub use pad_op::PadTranslator;
