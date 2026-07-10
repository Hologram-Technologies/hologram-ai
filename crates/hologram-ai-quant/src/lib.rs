//! Quantization primitives for hologram-ai.
//!
//! Provides block-quantized weight formats (Q4_0, Q8_0, etc.) with fast
//! dequantization. No IR dependency — safe to use from any crate.
//!
//! `no_std` runtime core (V&V class NS): dequantization runs on-device
//! (wasm / embedded), so this crate is `#![no_std]` + `alloc`.
#![no_std]

extern crate alloc;

pub mod encode;
pub mod q4_0;
pub mod q8_0;
pub mod scheme;

pub use encode::{encode_int8_per_channel, encode_int8_per_channel_omajor};
pub use q4_0::{dequant_q4_0, dequant_q4_0_block, Q4_0Block, Q4_0_BLOCK_SIZE};
pub use q8_0::{dequant_q8_0, dequant_q8_0_block, Q8_0Block, Q8_0_BLOCK_SIZE};
pub use scheme::{QuantDescriptor, QuantScheme, ScaleDtype};
