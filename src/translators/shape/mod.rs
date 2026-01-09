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

mod reshape;
mod transpose;
mod concat;
mod squeeze;
mod unsqueeze;
mod flatten;
mod expand;
mod split;
mod tile;

pub use reshape::ReshapeTranslator;
pub use transpose::TransposeTranslator;
pub use concat::ConcatTranslator;
pub use squeeze::SqueezeTranslator;
pub use unsqueeze::UnsqueezeTranslator;
pub use flatten::FlattenTranslator;
pub use expand::ExpandTranslator;
pub use split::SplitTranslator;
pub use tile::TileTranslator;
