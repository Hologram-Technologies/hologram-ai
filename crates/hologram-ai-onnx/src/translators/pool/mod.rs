//! Pooling operation translators.
//!
//! This module provides translators for ONNX pooling operations including:
//! - MaxPool: 2D max pooling
//! - AveragePool: 2D average pooling
//! - GlobalAveragePool: Global average pooling
//! - GlobalMaxPool: Global max pooling

mod max_pool;
mod avg_pool;
mod global_pool;

pub use max_pool::MaxPoolTranslator;
pub use avg_pool::AveragePoolTranslator;
pub use global_pool::{GlobalAveragePoolTranslator, GlobalMaxPoolTranslator};
