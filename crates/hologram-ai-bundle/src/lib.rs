//! Deterministic R4 inference bundle codec (schema v1).
//!
//! `no_std + alloc`: role-addressed bundle writer and bounded zero-copy
//! parser with digest verification and capability/processor consistency
//! validation. Treats all input bytes as untrusted.
//!
//! # Wire format (schema v1)
//!
//! A bundle is a single opaque, deterministic, content-addressed blob:
//!
//! ```text
//! offset  size        field
//! 0       4           magic: ASCII "R4IB" ([BUNDLE_MAGIC])
//! 4       2           u16 LE bundle schema version (= [BUNDLE_SCHEMA_VERSION])
//! 6       4           u32 LE manifest length N in bytes
//! 10      N           canonical `InferenceModelManifest` bytes
//!                     (hologram-ai-core canonical encoding)
//! 10+N    remaining   component payloads, concatenated with no padding in
//!                     the exact order of `manifest.artifacts` (canonical
//!                     `ArtifactRole::ALL` order); each component's length
//!                     and BLAKE3 digest come from its `ArtifactDescriptor`
//! ```
//!
//! Determinism rules: no padding, no timestamps, no paths, no maps — the
//! same model inputs always produce byte-identical bundles. The parser
//! rejects trailing bytes, truncated input, duplicate roles, more than
//! [`MAX_COMPONENTS`] components, and bundles larger than
//! [`MAX_BUNDLE_SIZE`], with checked arithmetic on every offset.
//!
//! # Trust model
//!
//! [`Bundle::parse`] validates structure only; it does **not** hash
//! component payloads. Callers must either use [`Bundle::parse_verified`]
//! (the recommended path) or call [`Bundle::verify_digests`] before any
//! engine sees component bytes. [`Bundle::validate`] checks the semantic
//! rules (mandatory roles, capability⇔processor consistency) and is run
//! automatically by [`BundleBuilder::finish`].
//!
//! # Forward compatibility
//!
//! Unknown engine/artifact-format strings are accepted: the uor-r4/R4G1
//! mandatory-role rules then simply do not apply (capability⇔processor
//! consistency is engine-agnostic and still enforced). Unknown bundle
//! schema versions are rejected with [`ErrorCategory::AbiMismatch`].

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

extern crate alloc;

mod builder;
mod parse;
mod validate;

pub use builder::BundleBuilder;
pub use parse::Bundle;

use hologram_ai_core::ArtifactRole;

/// Bundle magic bytes: ASCII `R4IB`.
pub const BUNDLE_MAGIC: [u8; 4] = *b"R4IB";

/// Bundle wire schema version written and accepted by this crate.
pub const BUNDLE_SCHEMA_VERSION: u16 = 1;

/// Fixed header length: magic + schema version + manifest length.
pub const HEADER_LEN: usize = 4 + 2 + 4;

/// Maximum total bundle size in bytes (4 GiB). Enforced as a checked `u64`
/// comparison so 32-bit and 64-bit targets behave identically.
pub const MAX_BUNDLE_SIZE: u64 = 1 << 32;

/// Maximum number of bundle components: at most one per semantic role.
pub const MAX_COMPONENTS: usize = ArtifactRole::ALL.len();
