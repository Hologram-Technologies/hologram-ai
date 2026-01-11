//! Matrix multiplication operation translators.
//!
//! This module provides translators for ONNX matrix multiplication operations:
//! - MatMul: Standard matrix multiplication
//! - Gemm: General Matrix Multiplication with optional bias

mod gemm;
mod matmul_op;

pub use gemm::GemmTranslator;
pub use matmul_op::MatMulTranslator;
