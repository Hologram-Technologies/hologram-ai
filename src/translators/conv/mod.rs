//! Convolution operation translators.
//!
//! This module provides translators for ONNX convolution operations including:
//! - Conv: Standard 2D convolution with groups support
//! - ConvTranspose: Transposed convolution (deconvolution)

mod conv_op;
mod conv_transpose;

pub use conv_op::ConvTranslator;
pub use conv_transpose::ConvTransposeTranslator;
