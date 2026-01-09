//! Indexing operation translators.
//!
//! This module provides translators for ONNX indexing operations including:
//! - Gather: Gather elements along an axis using indices
//! - Slice: Slice tensor along axes
//! - GatherElements: Gather elements at specific indices
//! - ScatterND: Scatter updates into tensor at indices

mod gather;
mod slice;
mod gather_elements;
mod scatter_nd;

pub use gather::GatherTranslator;
pub use slice::SliceTranslator;
pub use gather_elements::GatherElementsTranslator;
pub use scatter_nd::ScatterNDTranslator;
