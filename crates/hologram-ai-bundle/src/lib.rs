//! Deterministic R4 inference bundle codec (schema v1).
//!
//! `no_std + alloc`: role-addressed bundle writer and bounded zero-copy
//! parser with digest verification and capability/processor consistency
//! validation. Treats all input bytes as untrusted.

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

extern crate alloc;
