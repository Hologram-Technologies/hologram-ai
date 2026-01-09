//! Constant operation translators.
//!
//! This module provides translators for ONNX constant operations including:
//! - Constant: Create a constant tensor from attribute
//! - ConstantOfShape: Create a constant tensor of given shape
//! - Shape: Get shape of tensor as a 1D int64 tensor
//! - Identity: Pass-through operation

mod constant_of_shape;
mod constant_op;
mod identity;
mod shape_op;

pub use constant_of_shape::ConstantOfShapeTranslator;
pub use constant_op::ConstantTranslator;
pub use identity::IdentityTranslator;
pub use shape_op::ShapeOpTranslator;
