//! Error types for embeddable sections.
//!
//! This module re-exports error types from `hologram::bundle` for section embedding.
//! The base `EmbedError` is sufficient for all section operations - JSON errors
//! from config sections are converted to `InvalidData`.

// Re-export from hologram::bundle via traits module
pub use super::traits::{EmbedError, EmbedResult};
