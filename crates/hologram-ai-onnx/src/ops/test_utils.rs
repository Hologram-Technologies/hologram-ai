//! Test utilities for hologram-onnx-ops tests.
//!
//! This module provides helper functions for creating types in tests.

use hologram::ir::types::{ScalarType, TensorType, Type};

/// Create a tensor type with F32 elements and the given shape.
///
/// This is a convenience function for tests to avoid verbose type construction.
///
/// # Examples
///
/// ```rust,ignore
/// use crate::ops::test_utils::f32_tensor;
///
/// let ty = f32_tensor(&[1, 64, 224, 224]);
/// builder.add_input("X", ty);
/// ```
pub fn f32_tensor(dims: &[usize]) -> Type {
    Type::Tensor(TensorType::concrete(ScalarType::F32, dims.to_vec()))
}

/// Create a tensor type with I64 elements and the given shape.
pub fn i64_tensor(dims: &[usize]) -> Type {
    Type::Tensor(TensorType::concrete(ScalarType::I64, dims.to_vec()))
}

/// Create a tensor type with I32 elements and the given shape.
pub fn i32_tensor(dims: &[usize]) -> Type {
    Type::Tensor(TensorType::concrete(ScalarType::I32, dims.to_vec()))
}

/// Create a symbolic tensor type (unknown type to be inferred).
pub fn symbolic_tensor() -> Type {
    Type::Unknown
}
