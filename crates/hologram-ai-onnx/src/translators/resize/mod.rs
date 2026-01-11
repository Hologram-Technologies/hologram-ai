//! Resize and spatial transformation operation translators.
//!
//! This module provides translators for ONNX resize operations including:
//! - Resize: Resize/interpolate operation
//! - Upsample: Legacy upsample operation (deprecated, redirects to Resize)
//! - DepthToSpace: Rearrange depth to spatial dimensions
//! - SpaceToDepth: Rearrange spatial to depth dimensions

mod depth_space;
mod resize_op;
mod upsample;

pub use depth_space::{DepthToSpaceTranslator, SpaceToDepthTranslator};
pub use resize_op::ResizeTranslator;
pub use upsample::UpsampleTranslator;
