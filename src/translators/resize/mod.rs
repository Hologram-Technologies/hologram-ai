//! Resize and spatial transformation operation translators.
//!
//! This module provides translators for ONNX resize operations including:
//! - Resize: Resize/interpolate operation
//! - Upsample: Legacy upsample operation (deprecated, redirects to Resize)
//! - DepthToSpace: Rearrange depth to spatial dimensions
//! - SpaceToDepth: Rearrange spatial to depth dimensions

mod resize_op;
mod upsample;
mod depth_space;

pub use resize_op::ResizeTranslator;
pub use upsample::UpsampleTranslator;
pub use depth_space::{DepthToSpaceTranslator, SpaceToDepthTranslator};
