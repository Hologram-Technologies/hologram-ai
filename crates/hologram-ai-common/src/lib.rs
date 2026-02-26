//! Common utilities and generic transformer builder for hologram-ai.
//!
//! This crate provides shared functionality used by all format-specific crates:
//!
//! - **Error types**: Unified error handling across formats
//! - **Weight handling**: Common weight extraction and embedding
//! - **Serialization**: .holo file format support
//! - **TransformerBuilder**: Generic transformer graph construction from config
//!
//! ## Generic TransformerBuilder
//!
//! The transformer builder constructs hologram IR graphs from configuration
//! parameters, without needing architecture-specific code. All transformer-based
//! LLMs (LLaMA, Mistral, Qwen, etc.) use the same builder with different configs.
//!
//! ```ignore
//! use hologram_ai_common::transformer::{TransformerConfig, GenericTransformerBuilder};
//!
//! let config = TransformerConfig {
//!     num_layers: 32,
//!     hidden_size: 4096,
//!     num_attention_heads: 32,
//!     // ... other params
//! };
//!
//! let builder = GenericTransformerBuilder::new();
//! let ir_graph = builder.build(&config, &weights)?;
//! ```

#![deny(missing_docs)]
#![warn(clippy::all)]

pub mod error;
// TEMPORARILY DISABLED during hologram API migration:
// pub mod metadata;  // depends on hologram-bundle traits not in categorical-x
pub mod serialization; // Minimal stub for serialize_backend_plan_with_header
pub mod weights;

// Transformer config types (pure data structures, no hologram dependencies)
mod transformer_config;
pub use transformer_config::{Activation, FFNType, NormType, RoPEScaling, TransformerConfig};

// TEMPORARILY DISABLED: GenericTransformerBuilder uses old hologram::ir API
// (Shape, Dim, GraphBuilder, etc.) which needs refactoring for new API
// pub mod transformer;

pub use error::{CommonError, Result};
pub use serialization::{
    LayerHeaderData, LayerInfo, WeightStrategy, serialize_backend_plan_with_header,
};
pub use weights::{WeightDtype, WeightMap, WeightTensor};
