//! Reduce operation translators.
//!
//! This module provides translators for ONNX reduction operations:
//! - ReduceSum: Sum reduction along specified axes
//! - ReduceMean: Mean reduction along specified axes
//! - ReduceMax: Maximum reduction along specified axes
//! - ReduceMin: Minimum reduction along specified axes
//! - ReduceProd: Product reduction along specified axes

mod reduce_sum;
mod reduce_mean;
mod reduce_max;
mod reduce_min;
mod reduce_prod;

pub use reduce_sum::ReduceSumTranslator;
pub use reduce_mean::ReduceMeanTranslator;
pub use reduce_max::ReduceMaxTranslator;
pub use reduce_min::ReduceMinTranslator;
pub use reduce_prod::ReduceProdTranslator;
