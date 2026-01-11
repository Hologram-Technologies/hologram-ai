//! Logical and comparison operation translators.
//!
//! This module provides translators for ONNX logical operations including:
//! - Comparison: Equal, Greater, Less, GreaterOrEqual, LessOrEqual
//! - Boolean: And, Or, Not
//! - Conditional: Where

mod boolean;
mod comparison;
mod where_op;

pub use boolean::{AndTranslator, NotTranslator, OrTranslator};
pub use comparison::{
    EqualTranslator, GreaterOrEqualTranslator, GreaterTranslator, LessOrEqualTranslator,
    LessTranslator,
};
pub use where_op::WhereTranslator;
