//! Advanced operation translators.
//!
//! This module provides translators for ONNX advanced operations including:
//! - Cast: Type conversion
//! - Range: Generate sequence of numbers
//! - Trilu: Triangular lower/upper extraction
//! - Dropout: Dropout (identity during inference)

mod cast;
mod range;
mod trilu;
mod dropout;

pub use cast::CastTranslator;
pub use range::RangeTranslator;
pub use trilu::TriluTranslator;
pub use dropout::DropoutTranslator;
