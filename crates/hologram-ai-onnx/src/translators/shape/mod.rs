//! Shape manipulation operation translators.
//!
//! This module provides translators for ONNX shape manipulation operations:
//! - Reshape: Change tensor dimensions
//! - Transpose: Permute tensor axes
//! - Concat: Concatenate tensors along an axis
//! - Squeeze: Remove dimensions of size 1
//! - Unsqueeze: Add dimensions of size 1
//! - Flatten: Flatten tensor to 2D
//! - Expand: Broadcast tensor to target shape
//! - Split: Split tensor along an axis
//! - Tile: Repeat tensor along each dimension

mod concat;
mod expand;
mod flatten;
mod reshape;
mod split;
mod squeeze;
mod tile;
mod transpose;
mod unsqueeze;

pub use concat::ConcatTranslator;
pub use expand::ExpandTranslator;
pub use flatten::FlattenTranslator;
pub use reshape::ReshapeTranslator;
pub use split::SplitTranslator;
pub use squeeze::SqueezeTranslator;
pub use tile::TileTranslator;
pub use transpose::TransposeTranslator;
pub use unsqueeze::UnsqueezeTranslator;
