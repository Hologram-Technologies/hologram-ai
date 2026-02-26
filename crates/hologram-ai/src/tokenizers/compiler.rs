//! Compile tokenizers to hologram IR and .holo files.
//!
//! TEMPORARILY STUBBED during hologram API migration.
//! The tokenizer compilation functionality requires the new hologram IR API
//! (GraphBuilder, Shape, Dim) which is still being migrated.
//!
//! For now, tokenizers work at runtime using pure Rust implementations
//! (see sentencepiece.rs). The compilation to .holo files will be restored
//! once the hologram IR API is stabilized.

use super::TokenizerConfig;
use anyhow::Result;
use std::path::Path;
use tracing::warn;

/// Compile a tokenizer to a .holo file.
///
/// STUB: This function is temporarily disabled during API migration.
/// Tokenizers currently execute at runtime using pure Rust.
pub fn compile_tokenizer_to_holo(_config: &TokenizerConfig, _output_path: &Path) -> Result<()> {
    warn!("Tokenizer compilation is temporarily disabled during API migration");
    anyhow::bail!(
        "Tokenizer compilation is temporarily disabled. \
         Tokenizers work at runtime using pure Rust implementations."
    )
}

/// Compile a tokenizer to a HOLB bundle format.
///
/// STUB: This function is temporarily disabled during API migration.
pub fn compile_tokenizer_to_bundle(_config: &TokenizerConfig, _output_path: &Path) -> Result<()> {
    warn!("Tokenizer bundle compilation is temporarily disabled during API migration");
    anyhow::bail!(
        "Tokenizer bundle compilation is temporarily disabled. \
         Tokenizers work at runtime using pure Rust implementations."
    )
}
