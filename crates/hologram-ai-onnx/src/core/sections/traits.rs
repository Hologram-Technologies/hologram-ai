//! Core traits for embeddable sections.
//!
//! This module re-exports the embeddable section traits from `hologram::bundle`,
//! providing a consistent interface for defining and deserializing bundle sections.
//!
//! # Re-exported Traits
//!
//! - [`EmbeddableSection`]: For types that can be serialized and embedded in bundles
//! - [`FromEmbeddedSection`]: For types that can be deserialized from embedded data
//! - [`CloneableSection`]: Helper trait for cloneable sections
//!
//! These traits are defined in the `hologram` crate and re-exported here for
//! convenience. AI-specific section implementations in this crate implement
//! these traits.

// Re-export traits and types from hologram::bundle
pub use hologram::bundle::{
    CloneableSection, EmbedError, EmbedResult, EmbeddableSection, FromEmbeddedSection,
};
