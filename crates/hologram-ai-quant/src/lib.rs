//! Quantization primitives for hologram-ai.
//!
//! Provides block-quantized weight formats (Q4_0, Q8_0, etc.) with fast
//! dequantization. No IR dependency — safe to use from any crate.

pub mod scheme;
pub mod q4_0;
pub mod q8_0;

pub use scheme::{QuantDescriptor, QuantScheme, ScaleDtype};
pub use q4_0::{Q4_0Block, Q4_0_BLOCK_SIZE, dequant_q4_0, dequant_q4_0_block};
pub use q8_0::{Q8_0Block, Q8_0_BLOCK_SIZE, dequant_q8_0, dequant_q8_0_block};
