//! Activation function translators.
//!
//! This module provides translators for ONNX activation functions including:
//! - ReLU, Sigmoid, Tanh, GELU (basic activations)
//! - Softmax (normalized exponential)
//! - Clip (clamping)
//! - LeakyReLU, ELU, SELU, PReLU (parameterized ReLU variants)
//! - Swish (self-gated activation)
//! - Erf (error function)

mod relu;
mod sigmoid;
mod tanh;
mod gelu;
mod softmax;
mod clip;
mod leaky_relu;
mod elu;
mod selu;
mod prelu;
mod swish;
mod erf;

pub use relu::ReluTranslator;
pub use sigmoid::SigmoidTranslator;
pub use tanh::TanhTranslator;
pub use gelu::GeluTranslator;
pub use softmax::SoftmaxTranslator;
pub use clip::ClipTranslator;
pub use leaky_relu::LeakyReluTranslator;
pub use elu::EluTranslator;
pub use selu::SeluTranslator;
pub use prelu::PReluTranslator;
pub use swish::SwishTranslator;
pub use erf::ErfTranslator;
