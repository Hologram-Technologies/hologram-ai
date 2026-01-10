//! Matrix multiplication operation translators.
//!
//! This module provides translators for ONNX matrix multiplication operations:
//! - MatMul: Standard matrix multiplication
//! - Gemm: General Matrix Multiplication with optional bias

mod matmul_op;
mod gemm;

pub use matmul_op::MatMulTranslator;
pub use gemm::GemmTranslator;
