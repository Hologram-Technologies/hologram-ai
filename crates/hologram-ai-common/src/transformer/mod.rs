//! Generic transformer builder for LLM architectures.
//!
//! This module provides a configuration-driven approach to building transformer
//! IR graphs. Instead of having separate builders for LLaMA, Mistral, Qwen, etc.,
//! we use a single `GenericTransformerBuilder` that reads config parameters and
//! constructs the appropriate graph.
//!
//! ## Supported Architectures
//!
//! All standard decoder-only transformers are supported:
//! - LLaMA / LLaMA 2 / LLaMA 3
//! - Mistral / Mixtral (standard attention, MoE requires separate support)
//! - Qwen / Qwen2
//! - DeepSeek (standard attention variant)
//! - Phi-2 / Phi-3
//! - Gemma
//!
//! ## Example
//!
//! ```ignore
//! use hologram_ai_common::transformer::{TransformerConfig, GenericTransformerBuilder, NormType, Activation, FFNType};
//!
//! let config = TransformerConfig {
//!     num_layers: 32,
//!     hidden_size: 4096,
//!     num_attention_heads: 32,
//!     num_kv_heads: Some(8),  // GQA with 8 KV heads
//!     intermediate_size: 11008,
//!     vocab_size: 32000,
//!     max_position_embeddings: 4096,
//!     norm_type: NormType::RMSNorm,
//!     norm_eps: 1e-6,
//!     hidden_act: Activation::SiLU,
//!     rope_theta: Some(10000.0),
//!     rope_scaling: None,
//!     ffn_type: FFNType::Gated,
//!     attention_type: AttentionType::Standard,
//!     tie_word_embeddings: false,
//! };
//!
//! let builder = GenericTransformerBuilder::new();
//! let ir_graph = builder.build(&config, &weights)?;
//! ```

mod attention;
mod builder;
mod config;
mod ffn;
mod norm;

pub use attention::AttentionType;
pub use builder::GenericTransformerBuilder;
pub use config::{Activation, FFNType, NormType, RoPEScaling, TransformerConfig};
