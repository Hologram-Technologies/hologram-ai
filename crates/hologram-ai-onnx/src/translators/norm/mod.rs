//! Normalization operation translators.
//!
//! This module provides translators for ONNX normalization operations including:
//! - LayerNormalization: Normalize over last dimensions
//! - BatchNormalization: Normalize over batch dimension
//! - GroupNormalization: Normalize within channel groups
//! - InstanceNormalization: Normalize per instance (spatial dimensions)

mod layer_norm;
mod batch_norm;
mod group_norm;
mod instance_norm;

pub use layer_norm::LayerNormTranslator;
pub use batch_norm::BatchNormTranslator;
pub use group_norm::GroupNormTranslator;
pub use instance_norm::InstanceNormTranslator;
