//! Activation function translators.
//!
//! This module provides translators for ONNX activation functions including:
//! - ReLU, Sigmoid, Tanh, GELU (basic activations)
//! - Softmax (normalized exponential)
//! - Clip (clamping)
//! - LeakyReLU, ELU, SELU, PReLU (parameterized ReLU variants)
//! - Swish (self-gated activation)
//! - Erf (error function)

mod clip;
mod elu;
mod erf;
mod gelu;
mod leaky_relu;
mod prelu;
mod relu;
mod selu;
mod sigmoid;
mod softmax;
mod swish;
mod tanh;

pub use clip::ClipTranslator;
pub use elu::EluTranslator;
pub use erf::ErfTranslator;
pub use gelu::GeluTranslator;
pub use leaky_relu::LeakyReluTranslator;
pub use prelu::PReluTranslator;
pub use relu::ReluTranslator;
pub use selu::SeluTranslator;
pub use sigmoid::SigmoidTranslator;
pub use softmax::SoftmaxTranslator;
pub use swish::SwishTranslator;
pub use tanh::TanhTranslator;
