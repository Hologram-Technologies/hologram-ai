//! BERT output decoding test - demonstrates masked language model prediction.
//!
//! **NOTE:** This is a heavyweight integration test that requires:
//! - A compiled BERT bundle at models/bert-base/model.holo
//! - Run test_bert_compile_to_bundle first to create it
//!
//! Run with:
//! ```bash
//! cargo test -p hologram-ai-onnx --test bert_decode -- --ignored
//! ```

use std::collections::HashMap;
use std::fs;
use std::path::Path;

const BERT_HOLO_PATH: &str = "../../models/bert-base/model.holo";
const BERT_VOCAB_PATH: &str = "../../models/bert-base/vocab.txt";

/// Load BERT vocabulary from file.
fn load_vocab(path: &Path) -> Vec<String> {
    fs::read_to_string(path)
        .expect("Failed to read vocab file")
        .lines()
        .map(|s| s.to_string())
        .collect()
}

/// Decode token IDs to text.
fn decode_tokens(token_ids: &[usize], vocab: &[String]) -> String {
    token_ids
        .iter()
        .map(|&id| {
            if id < vocab.len() {
                vocab[id].as_str()
            } else {
                "[UNK]"
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
        .replace(" ##", "") // Handle wordpiece
}

/// Get top-k predictions with scores.
fn top_k(logits: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut indexed: Vec<(usize, f32)> = logits.iter().cloned().enumerate().collect();
    indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());
    indexed.into_iter().take(k).collect()
}

#[test]
#[ignore = "Heavyweight: requires compiled BERT model; run with --ignored"]
fn test_bert_mask_prediction() {
    use hologram_ai::runtime::{ModelExecutor, Tensor};

    let holo_path = Path::new(env!("CARGO_MANIFEST_DIR")).join(BERT_HOLO_PATH);
    let vocab_path = Path::new(env!("CARGO_MANIFEST_DIR")).join(BERT_VOCAB_PATH);

    if !holo_path.exists() {
        eprintln!("Skipping: Run test_bert_compile_to_bundle first");
        return;
    }

    // Load vocabulary
    let vocab = load_vocab(&vocab_path);
    println!("Loaded vocabulary: {} tokens", vocab.len());

    // Special tokens
    let cls_id = 101; // [CLS]
    let sep_id = 102; // [SEP]
    let mask_id = 103; // [MASK]
    let pad_id = 0; // [PAD]

    // Load model
    println!("Loading BERT model...");
    let mut executor = ModelExecutor::from_holo_file(&holo_path).expect("Failed to load BERT");

    // Create input: "The [MASK] is very funny."
    // Token IDs for bert-base-uncased:
    // the=1996, is=2003, very=2200, funny=6057, .=1012
    let seq_len = 512;
    let mut input_ids: Vec<f32> = vec![
        cls_id as f32,  // [CLS]
        1996.0,         // the
        mask_id as f32, // [MASK] - position 2, this is what we want to predict
        2003.0,         // is
        2200.0,         // very
        6057.0,         // funny
        1012.0,         // .
        sep_id as f32,  // [SEP]
    ];
    let actual_len = input_ids.len();
    input_ids.extend(vec![pad_id as f32; seq_len - actual_len]);

    let mut attention_mask: Vec<f32> = vec![1.0; actual_len];
    attention_mask.extend(vec![0.0; seq_len - actual_len]);

    let token_type_ids: Vec<f32> = vec![0.0; seq_len];

    // Show input
    let input_tokens: Vec<usize> = input_ids[..actual_len]
        .iter()
        .map(|&x| x as usize)
        .collect();
    println!("\nInput: {}", decode_tokens(&input_tokens, &vocab));
    println!("Input IDs: {:?}", &input_ids[..actual_len]);

    // Prepare inputs
    let mut inputs = HashMap::new();
    inputs.insert(
        "input_ids".to_string(),
        Tensor::new(input_ids, vec![1, seq_len]),
    );
    inputs.insert(
        "attention_mask".to_string(),
        Tensor::new(attention_mask, vec![1, seq_len]),
    );
    inputs.insert(
        "token_type_ids".to_string(),
        Tensor::new(token_type_ids, vec![1, seq_len]),
    );

    // Execute
    println!("\nRunning BERT inference...");
    let outputs = executor.execute(inputs).expect("Execution failed");

    // Get output
    let output = outputs
        .get("output")
        .or_else(|| outputs.get("output_0"))
        .expect("No output found");

    println!("Output shape: {:?}", output.shape);
    println!("Output total elements: {}", output.data.len());

    // The ONNX model outputs [batch, seq_len, vocab_size] = [1, 512, 28996]
    // But executor shows [1, 1, 512, 28996] - there's an extra dimension
    // Total elements = 1 * 512 * 28996 = 14,845,952 (or with extra dim: 1*1*512*28996)
    let vocab_size = 28996; // Model's actual vocab size (not full BERT vocab)
    let mask_position = 2;

    // For shape [1, 1, 512, 28996], the indexing is:
    // offset = d0 * (1*512*28996) + d1 * (512*28996) + d2 * 28996 + d3
    // For position 2: d0=0, d1=0, d2=2, d3=0..28996
    let start = mask_position * vocab_size;
    let end = start + vocab_size;

    println!(
        "Extracting logits for position {} (indices {}..{})",
        mask_position, start, end
    );

    // Try both layouts to see which makes sense
    // Layout A: [batch, seq, vocab] - logits for pos 2 at indices 2*vocab_size .. 3*vocab_size
    // Layout B: [batch, vocab, seq] - logits for pos 2 at indices v*512+2 for each v

    // Layout B: Extract logits assuming [1, vocab_size, seq_len] layout
    let seq_len_model = 512;
    let mut mask_logits_b: Vec<f32> = Vec::with_capacity(vocab_size);
    for v in 0..vocab_size {
        let idx = v * seq_len_model + mask_position;
        if idx < output.data.len() {
            mask_logits_b.push(output.data[idx]);
        }
    }

    println!("\n=== Trying Layout B: [batch, vocab, seq] ===");
    let mask_logits = &mask_logits_b[..];

    // Debug: check logit statistics
    let min_logit = mask_logits.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_logit = mask_logits
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    println!("Logit range: [{:.4}, {:.4}]", min_logit, max_logit);

    // Check what the logit is for expected tokens like "joke" (8257), "man" (2158), "movie" (3185)
    println!("\nLogits for expected tokens:");
    for (name, idx) in [
        ("joke", 8257),
        ("man", 2158),
        ("movie", 3185),
        ("guy", 3124),
        ("story", 2466),
    ] {
        if idx < mask_logits.len() {
            println!("  {}: {:.4}", name, mask_logits[idx]);
        }
    }

    // Get top 10 predictions for [MASK]
    println!("\n=== Top 10 predictions for [MASK] in 'The [MASK] is very funny.' ===\n");
    let top10 = top_k(mask_logits, 10);
    for (rank, (token_id, score)) in top10.iter().enumerate() {
        let token = if *token_id < vocab.len() {
            &vocab[*token_id]
        } else {
            "[UNK]"
        };
        println!(
            "  {}. {:15} (id={:5}) (score: {:.4})",
            rank + 1,
            token,
            token_id,
            score
        );
    }

    // Show the full decoded sentence with top prediction
    let best_token_id = top10[0].0;
    let mut decoded_ids = input_tokens.clone();
    decoded_ids[mask_position] = best_token_id;
    println!("\n=== Decoded sentence with top prediction ===");
    println!("\"{}\"", decode_tokens(&decoded_ids, &vocab));
}
