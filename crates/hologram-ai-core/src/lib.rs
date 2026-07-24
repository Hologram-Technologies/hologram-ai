//! Portable, modality-neutral inference schemas for hologram-ai.
//!
//! `no_std + alloc`: value/operation/capability descriptors, inference
//! requests and completions, resolution status and finish reasons, stable
//! error categories, and canonical (deterministic) encoding rules. No I/O,
//! no engine, no container dependencies.

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

extern crate alloc;
