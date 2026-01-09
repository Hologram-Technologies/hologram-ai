//! Compile tokenizers to hologram IR and .holo files.
//!
//! This command takes a tokenizer.json file and compiles it to a .holo file
//! that can be executed via the hologram backend for SIMD-accelerated tokenization.

use anyhow::{Context, Result};
use std::path::Path;
use tracing::info;

/// Compile a tokenizer to a .holo file.
///
/// # Arguments
/// * `vocab_path` - Path to tokenizer.json (Hugging Face format)
/// * `output_path` - Path to save compiled .holo file
/// * `tokenizer_type` - Type of tokenizer (sentencepiece)
/// * `max_length` - Maximum sequence length for padding
/// * `pad_token_id` - ID of padding token
/// * `eos_token_id` - ID of end-of-sequence token
/// * `unk_token_id` - ID of unknown token
///
/// # Example
/// ```no_run
/// use hologram_onnx::cli::compile_tokenizer_command;
/// use std::path::Path;
///
/// compile_tokenizer_command(
///     Path::new("models/t5-small/tokenizer.json"),
///     Path::new("models/t5-small/tokenizer.holo"),
///     "sentencepiece",
///     512,
///     0,
///     1,
///     2,
/// ).unwrap();
/// ```
pub fn compile_tokenizer_command(
    vocab_path: &Path,
    output_path: &Path,
    tokenizer_type: &str,
    max_length: usize,
    pad_token_id: u32,
    eos_token_id: u32,
    unk_token_id: u32,
) -> Result<()> {
    info!("Compiling {} tokenizer", tokenizer_type);
    info!("  Input: {}", vocab_path.display());
    info!("  Output: {}", output_path.display());
    info!("  Max length: {}", max_length);

    // Verify input file exists
    if !vocab_path.exists() {
        anyhow::bail!("Tokenizer file not found: {}", vocab_path.display());
    }

    // Create TokenizerConfig
    let config = crate::tokenizers::TokenizerConfig {
        tokenizer_type: tokenizer_type.to_string(),
        vocab_path: vocab_path.display().to_string(),
        max_length,
        pad_token_id,
        eos_token_id,
        unk_token_id,
        merges_path: None,
        bos_token_id: None,
    };

    // Compile to .holo file
    crate::tokenizers::compile_tokenizer(&config, output_path)
        .with_context(|| format!("Failed to compile tokenizer from {}", vocab_path.display()))?;

    info!("✅ Successfully compiled tokenizer to: {}", output_path.display());
    info!("   You can now use this .holo file in your pipeline configs");

    Ok(())
}

/// Compile tokenizer from a config file.
///
/// This loads tokenizer settings from a TOML config file and compiles
/// the tokenizer to a .holo file.
///
/// # Arguments
/// * `config_path` - Path to tokenizer config TOML file
/// * `output_path` - Path to save compiled .holo file
///
/// # Config Format
/// ```toml
/// [tokenizer]
/// type = "sentencepiece"
/// vocab_path = "models/t5-small/tokenizer.json"
/// max_length = 512
/// pad_token_id = 0
/// eos_token_id = 1
/// unk_token_id = 2
/// ```
pub fn compile_tokenizer_from_config(
    config_path: &Path,
    output_path: Option<&Path>,
) -> Result<()> {
    use crate::config::UnifiedConfig;

    info!("Loading tokenizer config from: {}", config_path.display());

    let config = UnifiedConfig::from_file(config_path)
        .with_context(|| format!("Failed to load config: {}", config_path.display()))?;

    let tokenizer_config = config
        .tokenizer
        .ok_or_else(|| anyhow::anyhow!("No [tokenizer] section in config file"))?;

    // Determine output path
    let output = if let Some(out) = output_path {
        out.to_path_buf()
    } else {
        // Default: same directory as config, tokenizer.holo
        config_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join("tokenizer.holo")
    };

    // Resolve vocab_path relative to config directory
    let config_dir = config_path.parent().unwrap_or_else(|| Path::new("."));
    let vocab_path = config_dir.join(&tokenizer_config.vocab_path);

    info!("Compiling {} tokenizer", tokenizer_config.tokenizer_type);
    info!("  Vocab: {}", vocab_path.display());
    info!("  Output: {}", output.display());

    // Update vocab_path to absolute path
    let mut abs_config = tokenizer_config.clone();
    abs_config.vocab_path = vocab_path.display().to_string();

    // Compile to .holo file
    crate::tokenizers::compile_tokenizer(&abs_config, &output)
        .with_context(|| "Failed to compile tokenizer")?;

    info!("✅ Successfully compiled tokenizer to: {}", output.display());

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_compile_tokenizer() {
        let output = NamedTempFile::new().unwrap();
        let result = compile_tokenizer_command(
            Path::new("models/t5-small/tokenizer.json"),
            output.path(),
            "sentencepiece",
            512,
            0,
            1,
            2,
        );

        assert!(result.is_ok());
    }
}
