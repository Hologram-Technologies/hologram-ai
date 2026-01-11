//! BERT model compilation and runtime execution tests.
//!
//! Tests the full ONNX→IR→execution pipeline with a real BERT model.
//!
//! These tests run serially to avoid memory exhaustion when running
//! multiple large model compilations concurrently.

use hologram_ai_onnx::{OnnxCompiler, OnnxConfig};
use serial_test::serial;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

const BERT_MODEL_PATH: &str = "../../models/bert-base/model.onnx";
const BERT_HOLO_PATH: &str = "../../models/bert-base/model.holo";

/// Reduced sequence length for memory-constrained runtime tests.
/// Using 128 instead of 512 reduces attention memory by ~16x.
const RUNTIME_SEQ_LEN: usize = 128;

#[test]
#[serial]
fn test_bert_compilation() {
    // Skip if model doesn't exist
    let model_path = Path::new(env!("CARGO_MANIFEST_DIR")).join(BERT_MODEL_PATH);
    if !model_path.exists() {
        eprintln!("Skipping test: BERT model not found at {:?}", model_path);
        return;
    }

    // Initialize tracing for debug output
    let _ = tracing_subscriber::fmt()
        .with_env_filter("hologram_ai_onnx=debug")
        .try_init();

    // Read the ONNX model
    let onnx_bytes = fs::read(&model_path).expect("Failed to read BERT model");
    println!("Loaded BERT model: {} bytes", onnx_bytes.len());

    // Create compiler with partitioning for large models
    let config = OnnxConfig {
        enable_partitioning: true,
        partition_size: 500,
        ..Default::default()
    };
    let compiler = OnnxCompiler::with_config(config);

    // Attempt compilation
    match compiler.compile(&onnx_bytes) {
        Ok((holo_bytes, weight_bytes)) => {
            println!("Compilation successful!");
            println!("  .holo size: {} bytes", holo_bytes.len());
            println!("  .weights size: {} bytes", weight_bytes.len());
        }
        Err(e) => {
            panic!("BERT compilation failed: {:?}", e);
        }
    }
}

/// Test parsing BERT model structure without full compilation.
#[test]
#[serial]
fn test_bert_parsing() {
    let model_path = Path::new(env!("CARGO_MANIFEST_DIR")).join(BERT_MODEL_PATH);
    if !model_path.exists() {
        eprintln!("Skipping test: BERT model not found at {:?}", model_path);
        return;
    }

    let onnx_bytes = fs::read(&model_path).expect("Failed to read BERT model");

    // Parse and validate
    let model = hologram_ai_onnx::parse_model(&onnx_bytes).expect("Failed to parse BERT model");
    hologram_ai_onnx::validate_model(&model).expect("BERT model validation failed");

    // Print model info
    let opset = hologram_ai_onnx::extract_opset_version(&model);
    println!("BERT model opset version: {}", opset);

    if let Some(graph) = &model.graph {
        println!("Graph name: {:?}", graph.name);
        println!("Inputs: {}", graph.input.len());
        println!("Outputs: {}", graph.output.len());
        println!("Nodes: {}", graph.node.len());
        println!("Initializers: {}", graph.initializer.len());

        // Print input names
        println!("\nInputs:");
        for input in &graph.input {
            println!("  - {}", input.name);
        }

        // Count operation types
        let mut op_counts: std::collections::HashMap<&str, usize> =
            std::collections::HashMap::new();
        for node in &graph.node {
            *op_counts.entry(&node.op_type).or_insert(0) += 1;
        }

        println!("\nOperation types:");
        let mut ops: Vec<_> = op_counts.iter().collect();
        ops.sort_by_key(|(_, count)| std::cmp::Reverse(*count));
        for (op, count) in ops {
            println!("  {}: {}", op, count);
        }
    }
}

/// Test compiling BERT to unified bundle (HOLB) with embedded weights.
#[test]
#[serial]
fn test_bert_compile_to_bundle() {
    let model_path = Path::new(env!("CARGO_MANIFEST_DIR")).join(BERT_MODEL_PATH);
    let holo_path = Path::new(env!("CARGO_MANIFEST_DIR")).join(BERT_HOLO_PATH);

    if !model_path.exists() {
        eprintln!("Skipping test: BERT model not found at {:?}", model_path);
        return;
    }

    // Initialize tracing
    let _ = tracing_subscriber::fmt()
        .with_env_filter("hologram_ai_onnx=info")
        .try_init();

    // Read and compile to unified bundle
    let onnx_bytes = fs::read(&model_path).expect("Failed to read BERT model");
    let compiler = OnnxCompiler::new();

    println!("Compiling BERT to unified bundle (weights embedded)...");
    let bundle_bytes = compiler
        .compile_to_bundle(&onnx_bytes)
        .expect("BERT compilation to bundle failed");

    // Save to file
    fs::write(&holo_path, &bundle_bytes).expect("Failed to write .holo bundle");

    println!("Saved compiled BERT model:");
    println!(
        "  .holo bundle: {} ({} MB)",
        holo_path.display(),
        bundle_bytes.len() / 1_000_000
    );
}

/// Test runtime execution of compiled BERT model.
///
/// Uses reduced sequence length (128 instead of 512) to minimize memory usage.
/// This reduces attention matrix memory by ~16x.
#[test]
#[serial]
#[ignore = "Requires significant memory; run with --ignored flag"]
fn test_bert_runtime_execution() {
    use hologram_ai::runtime::{ModelExecutor, Tensor};

    let holo_path = Path::new(env!("CARGO_MANIFEST_DIR")).join(BERT_HOLO_PATH);

    // Skip if compiled file doesn't exist (run test_bert_compile_to_bundle first)
    if !holo_path.exists() {
        eprintln!(
            "Skipping test: Compiled BERT bundle not found at {:?}",
            holo_path
        );
        eprintln!("  Run test_bert_compile_to_bundle first to create it.");
        return;
    }

    // Initialize tracing
    let _ = tracing_subscriber::fmt()
        .with_env_filter("hologram=info,hologram_ai=info")
        .try_init();

    println!("Loading compiled BERT bundle...");
    let mut executor =
        ModelExecutor::from_holo_file(&holo_path).expect("Failed to load BERT model");

    // BERT inputs: input_ids, attention_mask, token_type_ids
    // Using reduced sequence length to minimize memory usage
    let batch_size = 1;
    let seq_len = RUNTIME_SEQ_LEN;

    // Sample input: "Hello world" tokenized with padding
    // [CLS]=101, Hello=7592, world=2088, [SEP]=102, rest is [PAD]=0
    let mut input_ids: Vec<f32> = vec![101.0, 7592.0, 2088.0, 102.0];
    input_ids.extend(vec![0.0; seq_len - 4]); // Pad to seq_len

    let mut attention_mask: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0];
    attention_mask.extend(vec![0.0; seq_len - 4]); // Mask padding

    let token_type_ids: Vec<f32> = vec![0.0; seq_len];

    let mut inputs = HashMap::new();
    inputs.insert(
        "input_ids".to_string(),
        Tensor::new(input_ids, vec![batch_size, seq_len]),
    );
    inputs.insert(
        "attention_mask".to_string(),
        Tensor::new(attention_mask, vec![batch_size, seq_len]),
    );
    inputs.insert(
        "token_type_ids".to_string(),
        Tensor::new(token_type_ids, vec![batch_size, seq_len]),
    );

    println!("Executing BERT with {} inputs...", inputs.len());
    println!("  input_ids shape: [1, {}]", seq_len);
    println!("  attention_mask shape: [1, {}]", seq_len);
    println!("  token_type_ids shape: [1, {}]", seq_len);
    println!(
        "  (Using reduced seq_len={} for memory efficiency)",
        seq_len
    );

    // Execute
    let outputs = executor.execute(inputs).expect("BERT execution failed");

    println!("\nExecution complete!");
    println!("Number of outputs: {}", outputs.len());

    for (name, tensor) in &outputs {
        println!(
            "  {}: shape={:?}, first 5 values={:?}",
            name,
            tensor.shape,
            &tensor.data[..tensor.data.len().min(5)]
        );
    }

    // Verify we got output
    assert!(!outputs.is_empty(), "Expected at least one output");

    // BERT output should be [batch_size, seq_len, hidden_size] = [1, seq_len, 768]
    if let Some(output) = outputs.get("output").or_else(|| outputs.get("output_0")) {
        println!("\nBERT output shape: {:?}", output.shape);
        println!("Total output elements: {}", output.data.len());

        // Check output is not all zeros (sanity check)
        let non_zero_count = output.data.iter().filter(|&&x| x != 0.0).count();
        println!(
            "Non-zero elements: {} ({:.1}%)",
            non_zero_count,
            100.0 * non_zero_count as f64 / output.data.len() as f64
        );

        assert!(non_zero_count > 0, "Output should not be all zeros");
    }
}

const BERT_VOCAB_PATH: &str = "../../models/bert-base/vocab.txt";
const BERT_EMBEDDED_HOLO_PATH: &str = "../../models/bert-base/model_with_vocab.holo";

/// Test compiling BERT with embedded vocabulary.
///
/// This test creates a V2 bundle with the vocabulary embedded as a section.
#[test]
#[serial]
fn test_bert_compile_with_embedded_vocab() {
    use hologram_ai_onnx::core::{EmbeddedFileConfig, UnifiedBundleReader, VocabularySection};

    let model_path = Path::new(env!("CARGO_MANIFEST_DIR")).join(BERT_MODEL_PATH);
    let vocab_path = Path::new(env!("CARGO_MANIFEST_DIR")).join(BERT_VOCAB_PATH);
    let holo_path = Path::new(env!("CARGO_MANIFEST_DIR")).join(BERT_EMBEDDED_HOLO_PATH);

    if !model_path.exists() || !vocab_path.exists() {
        eprintln!(
            "Skipping test: BERT model or vocab not found at {:?}",
            model_path
        );
        return;
    }

    // Initialize tracing
    let _ = tracing_subscriber::fmt()
        .with_env_filter("hologram_ai_onnx=info")
        .try_init();

    // Create config with embedded vocabulary
    let config = OnnxConfig::new().with_embedded_file(EmbeddedFileConfig::vocabulary(&vocab_path));

    println!("Compiling BERT with embedded vocabulary...");
    println!("  Model: {}", model_path.display());
    println!("  Vocab: {}", vocab_path.display());

    // Read and compile to unified bundle with embedded vocab
    let onnx_bytes = fs::read(&model_path).expect("Failed to read BERT model");
    let compiler = OnnxCompiler::with_config(config);

    let bundle_bytes = compiler
        .compile_to_bundle(&onnx_bytes)
        .expect("BERT compilation with embedded vocab failed");

    // Save to file
    fs::write(&holo_path, &bundle_bytes).expect("Failed to write .holo bundle");

    println!("Saved compiled BERT model with embedded vocabulary:");
    println!(
        "  .holo bundle: {} ({} MB)",
        holo_path.display(),
        bundle_bytes.len() / 1_000_000
    );

    // Verify we can read the vocabulary back from the bundle
    let reader = UnifiedBundleReader::from_bytes(&bundle_bytes)
        .expect("Failed to read bundle for verification");

    println!("\nVerifying embedded sections:");
    println!("  Version: {}", reader.version());
    println!("  Section count: {}", reader.sections().len());

    // List sections
    for section in reader.sections() {
        println!(
            "    - {} ({}): {} bytes",
            section.id, section.content_type, section.size
        );
    }

    // Extract and verify vocabulary
    assert!(
        reader.has_section("vocabulary"),
        "Bundle should have vocabulary section"
    );

    let vocab: VocabularySection = reader
        .get_section()
        .expect("Failed to extract vocabulary section");

    println!("\nExtracted vocabulary:");
    println!("  Vocabulary size: {} tokens", vocab.len());
    println!("  First 5 tokens:");
    for i in 0..5 {
        if let Some(token) = vocab.get_token(i) {
            println!("    {}: '{}'", i, token);
        }
    }

    // Verify some BERT-specific tokens
    assert_eq!(
        vocab.get_token(101),
        Some("[CLS]"),
        "Token 101 should be [CLS]"
    );
    assert_eq!(
        vocab.get_token(102),
        Some("[SEP]"),
        "Token 102 should be [SEP]"
    );
    assert_eq!(
        vocab.get_token(103),
        Some("[MASK]"),
        "Token 103 should be [MASK]"
    );
    assert_eq!(vocab.get_token(0), Some("[PAD]"), "Token 0 should be [PAD]");
    assert_eq!(
        vocab.get_token(100),
        Some("[UNK]"),
        "Token 100 should be [UNK]"
    );

    // Verify reverse lookup
    assert_eq!(vocab.get_id("[CLS]"), Some(101));
    assert_eq!(vocab.get_id("[SEP]"), Some(102));

    println!("\nAll vocabulary verifications passed!");
}

/// Test reading vocabulary from an existing bundle.
#[test]
#[serial]
fn test_bert_read_embedded_vocab() {
    use hologram_ai_onnx::core::{UnifiedBundleReader, VocabularySection};

    let holo_path = Path::new(env!("CARGO_MANIFEST_DIR")).join(BERT_EMBEDDED_HOLO_PATH);

    if !holo_path.exists() {
        eprintln!(
            "Skipping test: Run test_bert_compile_with_embedded_vocab first to create {:?}",
            holo_path
        );
        return;
    }

    // Read the bundle
    let bundle_bytes = fs::read(&holo_path).expect("Failed to read bundle");
    let reader = UnifiedBundleReader::from_bytes(&bundle_bytes).expect("Failed to parse bundle");

    println!("Reading embedded vocabulary from bundle...");
    println!("  Bundle version: {}", reader.version());
    println!("  Sections: {:?}", reader.sections().len());

    // Extract vocabulary
    if reader.has_section("vocabulary") {
        let vocab: VocabularySection = reader.get_section().expect("Failed to extract vocabulary");

        println!("  Vocabulary size: {} tokens", vocab.len());

        // Build reverse map for efficient lookup
        let reverse_map = vocab.build_reverse_map();
        println!("  Built reverse map with {} entries", reverse_map.len());

        // Test some lookups
        if let Some(&id) = reverse_map.get("hello") {
            println!("  'hello' -> token ID {}", id);
        }
        if let Some(&id) = reverse_map.get("world") {
            println!("  'world' -> token ID {}", id);
        }
    } else {
        println!("  No vocabulary section found (V1 bundle)");
    }
}
