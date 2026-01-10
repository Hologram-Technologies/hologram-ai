//! Logical and comparison operation translators.
//!
//! This module provides translators for ONNX logical operations including:
//! - Comparison: Equal, Greater, Less, GreaterOrEqual, LessOrEqual
//! - Boolean: And, Or, Not
//! - Conditional: Where

mod comparison;
mod boolean;
mod where_op;

pub use comparison::{EqualTranslator, GreaterTranslator, LessTranslator, GreaterOrEqualTranslator, LessOrEqualTranslator};
pub use boolean::{AndTranslator, OrTranslator, NotTranslator};
pub use where_op::WhereTranslator;
