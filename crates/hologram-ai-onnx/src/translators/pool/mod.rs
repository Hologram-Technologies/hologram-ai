//! Pooling operation translators.
//!
//! This module provides translators for ONNX pooling operations including:
//! - MaxPool: 2D max pooling
//! - AveragePool: 2D average pooling
//! - GlobalAveragePool: Global average pooling
//! - GlobalMaxPool: Global max pooling

mod avg_pool;
mod global_pool;
mod max_pool;

pub use avg_pool::AveragePoolTranslator;
pub use global_pool::{GlobalAveragePoolTranslator, GlobalMaxPoolTranslator};
pub use max_pool::MaxPoolTranslator;
