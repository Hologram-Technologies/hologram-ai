//! Compile tokenizers to hologram IR and .holo files.
//!
//! Tokenizers are treated as computational graphs that compile to hologram IR.
//! This enables:
//! - SIMD-accelerated vocabulary lookups
//! - Unified execution path (tokenizer.holo + model.holo)
//! - Config-driven compilation and caching
//! - Hardware-agnostic execution via hologram backend

use super::TokenizerConfig;
use anyhow::{Context, Result};
use hologram::ir::{GraphBuilder, Shape, Dim, DType, ConstantData};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Tokenizer vocabulary specification.
#[derive(Debug, Clone)]
pub struct TokenizerVocab {
    /// Token string to ID mapping
    pub token_to_id: HashMap<String, u32>,
    /// ID to token string mapping
    pub id_to_token: HashMap<u32, String>,
    /// Vocabulary size
    pub vocab_size: usize,
}

/// Parse tokenizer.json vocabulary.
///
/// Extracts vocabulary from Hugging Face tokenizer.json format.
pub fn parse_tokenizer_vocab(vocab_path: &Path) -> Result<TokenizerVocab> {
    let content = fs::read_to_string(vocab_path)
        .with_context(|| format!("Failed to read tokenizer file: {}", vocab_path.display()))?;

    let json: serde_json::Value = serde_json::from_str(&content)
        .with_context(|| "Failed to parse tokenizer.json")?;

    // Extract vocab from model.vocab
    // Handle both formats:
    // 1. Object format: {"token": id, ...}
    // 2. Array format: [[token, score], ...] where index is ID
    let vocab = json.get("model")
        .and_then(|m| m.get("vocab"))
        .ok_or_else(|| anyhow::anyhow!("No model.vocab found in tokenizer.json"))?;

    let mut token_to_id = HashMap::new();
    let mut id_to_token = HashMap::new();

    if let Some(vocab_obj) = vocab.as_object() {
        // Object format: {"token": id}
        for (token, id_val) in vocab_obj {
            if let Some(id) = id_val.as_u64() {
                let id = id as u32;
                token_to_id.insert(token.clone(), id);
                id_to_token.insert(id, token.clone());
            }
        }
    } else if let Some(vocab_arr) = vocab.as_array() {
        // Array format: [[token, score], ...] where index is ID
        for (id, entry) in vocab_arr.iter().enumerate() {
            if let Some(pair) = entry.as_array()
                && let Some(token) = pair.first().and_then(|t| t.as_str()) {
                    let id = id as u32;
                    token_to_id.insert(token.to_string(), id);
                    id_to_token.insert(id, token.to_string());
                }
        }
    } else {
        anyhow::bail!("model.vocab must be an object or array");
    }

    // Also check for added_tokens
    if let Some(added_tokens) = json.get("added_tokens").and_then(|v| v.as_array()) {
        for token_obj in added_tokens {
            if let (Some(id), Some(content)) = (
                token_obj.get("id").and_then(|v| v.as_u64()),
                token_obj.get("content").and_then(|v| v.as_str())
            ) {
                let id = id as u32;
                token_to_id.insert(content.to_string(), id);
                id_to_token.insert(id, content.to_string());
            }
        }
    }

    let vocab_size = token_to_id.len();

    tracing::info!("Loaded vocabulary: {} tokens", vocab_size);

    Ok(TokenizerVocab {
        token_to_id,
        id_to_token,
        vocab_size,
    })
}

/// Compile a tokenizer to hologram IR.
///
/// Creates an OperationGraph that performs:
/// 1. Pass-through of pre-tokenized `token_indices` to `input_ids`
/// 2. Attention mask generation via a lookup table (Gather)
///
/// This graph assumes inputs are already tokenized and padded to `max_length`.
/// It produces a correct attention mask without requiring comparison ops.
///
/// # Graph Interface
///
/// Inputs:
/// - `token_indices`: I64 tensor `[batch, max_length]`
///
/// Outputs:
/// - `input_ids`: I64 tensor `[batch, max_length]`
/// - `attention_mask`: F32 tensor `[batch, max_length]` (1.0 for non-pad, 0.0 for pad)
pub fn compile_tokenizer_to_ir(
    config: &TokenizerConfig,
    vocab: &TokenizerVocab,
) -> Result<hologram::ir::OperationGraph> {
    let mut builder = GraphBuilder::new();

    let max_len = config.max_length;
    if max_len == 0 {
        anyhow::bail!("Tokenizer max_length must be > 0");
    }

    let vocab_size = vocab.vocab_size;
    if vocab_size == 0 {
        anyhow::bail!("Tokenizer vocabulary is empty");
    }

    let pad_token_id = config.pad_token_id as usize;
    if pad_token_id >= vocab_size {
        anyhow::bail!(
            "pad_token_id {} is out of range for vocab size {}",
            pad_token_id,
            vocab_size
        );
    }

    // Input: token indices (assuming pre-tokenized for now)
    let input_indices = builder.input(
        "token_indices",
        Shape::new(vec![
            Dim::Static(1),      // batch
            Dim::Static(max_len), // seq_len (padded)
        ]),
        DType::I64,
    );

    // Pass through as input_ids
    let _input_ids = builder.output("input_ids", input_indices)?;

    // Build attention mask via lookup table (0 for pad token, 1 for others)
    let mut mask_table = vec![1.0f32; vocab_size];
    mask_table[pad_token_id] = 0.0;

    let mask_table_node = builder.constant(
        ConstantData::F32(mask_table),
        Shape::new(vec![Dim::Static(vocab_size)]),
    );

    let attention_mask = builder.gather(mask_table_node, input_indices, 0)?;
    builder.output("attention_mask", attention_mask)?;

    tracing::info!(
        "Created tokenizer IR graph (token indices + attention mask) - vocab_size: {}",
        vocab.vocab_size
    );

    Ok(builder.build())
}

/// Compile a tokenizer to a .holo file.
///
/// This creates a hologram IR graph for tokenization and compiles it
/// to a .holo file that can be executed via the hologram backend.
///
/// # Arguments
/// * `config` - Tokenizer configuration
/// * `output_path` - Path to save compiled .holo file
///
/// # Example
/// ```ignore
/// use hologram_ai::tokenizers::TokenizerConfig;
/// use hologram_ai::tokenizers::compiler::compile_tokenizer_to_holo;
/// use std::path::Path;
///
/// let config = TokenizerConfig {
///     tokenizer_type: "sentencepiece".to_string(),
///     vocab_path: "tokenizer.json".to_string(),
///     max_length: 512,
///     pad_token_id: 0,
///     eos_token_id: 1,
///     unk_token_id: 2,
///     ..Default::default()
/// };
///
/// compile_tokenizer_to_holo(&config, Path::new("tokenizer.holo")).unwrap();
/// ```
pub fn compile_tokenizer_to_holo(
    config: &TokenizerConfig,
    output_path: &Path,
) -> Result<()> {
    tracing::info!("Compiling {} tokenizer to .holo format", config.tokenizer_type);

    // Step 1: Parse vocabulary
    let vocab = parse_tokenizer_vocab(Path::new(&config.vocab_path))?;

    // Step 2: Build IR graph
    let ir_graph = compile_tokenizer_to_ir(config, &vocab)?;
    tracing::info!("Created IR graph with {} nodes", ir_graph.node_count());

    // Step 3: Compile to BackendPlan
    let backend_type = hologram::BackendType::Cpu;
    let backend_plan = hologram::compiler::compile_ir(&ir_graph, backend_type)
        .map_err(|e| anyhow::anyhow!("Failed to compile tokenizer IR: {:?}", e))?;

    // Step 4: Serialize to .holo format
    let serializable = backend_plan.to_serializable();
    let plan_bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&serializable)
        .map(|bytes| bytes.to_vec())
        .map_err(|e| anyhow::anyhow!("Failed to serialize BackendPlan: {}", e))?;

    // Prepend magic bytes
    let mut holo_bytes = Vec::with_capacity(4 + plan_bytes.len());
    holo_bytes.extend_from_slice(&hologram::compiler::HOLO_MAGIC);
    holo_bytes.extend_from_slice(&plan_bytes);

    // Step 5: Write to file
    fs::write(output_path, holo_bytes)
        .with_context(|| format!("Failed to write .holo file: {}", output_path.display()))?;

    tracing::info!("Compiled tokenizer saved to: {}", output_path.display());
    Ok(())
}

/// Compile a tokenizer to a HOLB bundle format.
///
/// This creates a unified bundle (.holo file with HOLB magic) that can be
/// combined with other HOLB bundles into a HOLM pipeline bundle.
///
/// # Arguments
///
/// * `config` - Tokenizer configuration
/// * `output_path` - Path to save the compiled .holo bundle
///
/// # Example
///
/// ```ignore
/// use hologram_ai::tokenizers::{TokenizerConfig, compile_tokenizer_to_bundle};
/// use std::path::Path;
///
/// let config = TokenizerConfig {
///     tokenizer_type: "sentencepiece".into(),
///     vocab_path: "tokenizer.json".into(),
///     max_length: 512,
///     pad_token_id: 0,
///     eos_token_id: 1,
///     unk_token_id: 2,
///     ..Default::default()
/// };
///
/// compile_tokenizer_to_bundle(&config, Path::new("tokenizer.holo")).unwrap();
/// ```
pub fn compile_tokenizer_to_bundle(
    config: &TokenizerConfig,
    output_path: &Path,
) -> Result<()> {
    #[cfg(feature = "onnx")]
    use hologram_ai_onnx::core::UnifiedBundleWriter;

    tracing::info!("Compiling {} tokenizer to HOLB bundle format", config.tokenizer_type);

    // Step 1: Parse vocabulary
    let vocab = parse_tokenizer_vocab(Path::new(&config.vocab_path))?;

    // Step 2: Build IR graph
    let ir_graph = compile_tokenizer_to_ir(config, &vocab)?;
    tracing::info!("Created IR graph with {} nodes", ir_graph.node_count());

    // Step 3: Compile to BackendPlan
    let backend_type = hologram::BackendType::Cpu;
    let backend_plan = hologram::compiler::compile_ir(&ir_graph, backend_type)
        .map_err(|e| anyhow::anyhow!("Failed to compile tokenizer IR: {:?}", e))?;

    // Step 4: Serialize to bytes
    let serializable = backend_plan.to_serializable();
    let plan_bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&serializable)
        .map(|bytes| bytes.to_vec())
        .map_err(|e| anyhow::anyhow!("Failed to serialize BackendPlan: {}", e))?;

    // Prepend HOLP magic (graph format, not bundle magic)
    let mut graph_bytes = Vec::with_capacity(4 + plan_bytes.len());
    graph_bytes.extend_from_slice(&hologram::compiler::HOLO_MAGIC);
    graph_bytes.extend_from_slice(&plan_bytes);

    // Step 5: Create HOLB bundle with empty weights
    let mut writer = UnifiedBundleWriter::new();
    writer.set_graph_bytes(graph_bytes);
    writer.set_weights_bytes(Vec::new()); // Tokenizers have no external weights

    let bundle_bytes = writer.finish();

    // Step 6: Write to file
    fs::write(output_path, bundle_bytes)
        .with_context(|| format!("Failed to write HOLB bundle: {}", output_path.display()))?;

    tracing::info!("Compiled tokenizer bundle saved to: {}", output_path.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_compile_tokenizer_to_ir() {
        let mut file = NamedTempFile::new().expect("temp file");
        let tokenizer_json = r#"{
            "model": {
                "type": "unigram",
                "vocab": [
                    ["<pad>", 0.0],
                    ["</s>", -0.1],
                    ["<unk>", -0.2],
                    ["▁hello", -1.0],
                    ["world", -1.1]
                ]
            }
        }"#;
        file.write_all(tokenizer_json.as_bytes()).expect("write tokenizer");

        let config = TokenizerConfig {
            tokenizer_type: "sentencepiece".to_string(),
            vocab_path: file.path().display().to_string(),
            max_length: 8,
            pad_token_id: 0,
            eos_token_id: 1,
            unk_token_id: 2,
            merges_path: None,
            bos_token_id: None,
        };

        let vocab = parse_tokenizer_vocab(file.path()).expect("parse vocab");
        let graph = compile_tokenizer_to_ir(&config, &vocab).expect("compile ir");
        assert_eq!(graph.inputs.len(), 1);
        assert_eq!(graph.outputs.len(), 2);
    }
}
