//! Test T5 full-sequence generation with tokenizer support.
//!
//! Uses the full-sequence decoder model that accepts all generated tokens at once,
//! enabling proper autoregressive generation without KV-caching.

use anyhow::Result;
use hologram_ai::runtime::ModelExecutor;
use hologram_ai::runtime::Tensor;
use hologram_ai::tokenizers::Tokenizer;
use hologram_ai::tokenizers::sentencepiece::SentencePieceTokenizer;
use std::collections::HashMap;
use std::path::Path;

const ENCODER_SEQ_LEN: usize = 512;
const DECODER_SEQ_LEN: usize = 64;
const HIDDEN_DIM: usize = 512;
const VOCAB_SIZE: usize = 32128;

/// Softmax with temperature scaling
fn softmax_with_temperature(logits: &[f32], temperature: f32) -> Vec<f32> {
    let scaled: Vec<f32> = logits.iter().map(|&x| x / temperature).collect();
    let max_val = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_vals: Vec<f32> = scaled.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f32 = exp_vals.iter().sum();
    exp_vals.iter().map(|&x| x / sum).collect()
}

/// Sample from probability distribution (simple random sampling)
fn sample_from_probs(probs: &[f32], top_k: usize) -> u32 {
    // Get top-k indices
    let mut indexed: Vec<(usize, f32)> = probs.iter().cloned().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let top_k_items: Vec<(usize, f32)> = indexed.into_iter().take(top_k).collect();

    // Renormalize top-k probabilities
    let sum: f32 = top_k_items.iter().map(|(_, p)| p).sum();
    let normalized: Vec<(usize, f32)> = top_k_items.iter().map(|(i, p)| (*i, p / sum)).collect();

    // Simple deterministic sampling: pick based on cumulative probability
    // Use a simple hash of current state for pseudo-randomness
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    probs.len().hash(&mut hasher);
    probs
        .iter()
        .take(5)
        .for_each(|p| p.to_bits().hash(&mut hasher));
    let hash = hasher.finish();
    let rand_val = (hash % 1000) as f32 / 1000.0;

    let mut cumsum = 0.0;
    for (idx, prob) in &normalized {
        cumsum += prob;
        if rand_val < cumsum {
            return *idx as u32;
        }
    }

    // Fallback to top token
    normalized[0].0 as u32
}

/// Full-sequence decoder for T5 with temperature sampling.
///
/// Unlike single-token decoding, this passes the full generated sequence to the decoder
/// at each step, allowing the model to see all previously generated tokens.
///
/// Parameters:
/// - temperature: Controls randomness (1.0 = normal, <1.0 = more deterministic, >1.0 = more random)
/// - top_k: Only sample from top K tokens (0 = greedy/argmax)
fn decode_fullseq_with_sampling(
    encoder_output: &[f32],
    decoder: &mut ModelExecutor,
    encoder_attention_mask: &[f32],
    max_tokens: usize,
    eos_token_id: u32,
    temperature: f32,
    top_k: usize,
) -> Result<Vec<u32>> {
    // Start with decoder start token (pad token = 0 for T5)
    let mut generated_ids: Vec<u32> = vec![0];

    for step in 0..max_tokens {
        // Prepare full decoder input (pad to DECODER_SEQ_LEN)
        let current_len = generated_ids.len();
        let mut decoder_input_ids = vec![0.0f32; DECODER_SEQ_LEN];
        for (i, &id) in generated_ids.iter().enumerate() {
            if i < DECODER_SEQ_LEN {
                decoder_input_ids[i] = id as f32;
            }
        }

        // Build inputs
        let mut inputs = HashMap::new();
        inputs.insert(
            "input_ids".to_string(),
            Tensor::new(decoder_input_ids, vec![1, DECODER_SEQ_LEN]),
        );
        inputs.insert(
            "encoder_attention_mask".to_string(),
            Tensor::new(encoder_attention_mask.to_vec(), vec![1, ENCODER_SEQ_LEN]),
        );
        inputs.insert(
            "encoder_hidden_states".to_string(),
            Tensor::new(
                encoder_output.to_vec(),
                vec![1, ENCODER_SEQ_LEN, HIDDEN_DIM],
            ),
        );

        // Run decoder
        let outputs = decoder.execute(inputs)?;

        // Get logits - shape is [1, 64, 32128]
        let logits = outputs
            .get("logits")
            .or_else(|| outputs.get("output_0"))
            .ok_or_else(|| anyhow::anyhow!("No logits output from decoder"))?;

        // Get logits for the last generated position
        // For sequence of length N, we want logits at position N-1
        let last_pos = current_len - 1;
        let logits_start = last_pos * VOCAB_SIZE;
        let logits_end = logits_start + VOCAB_SIZE;

        if logits_end > logits.data.len() {
            println!(
                "Warning: logits position {} out of bounds (total: {})",
                logits_end,
                logits.data.len()
            );
            break;
        }

        let last_logits = &logits.data[logits_start..logits_end];

        // Select next token based on sampling strategy
        let next_token = if top_k == 0 {
            // Greedy: pick argmax
            last_logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx as u32)
                .unwrap_or(0)
        } else {
            // Temperature sampling with top-k
            let probs = softmax_with_temperature(last_logits, temperature);
            sample_from_probs(&probs, top_k)
        };

        // Print progress
        if step < 5 || step % 5 == 0 {
            println!(
                "Step {}: token {} (logit: {:.3})",
                step, next_token, last_logits[next_token as usize]
            );
        }

        // Check for EOS
        if next_token == eos_token_id {
            println!("EOS token generated at step {}", step);
            break;
        }

        // Check for max sequence length
        if generated_ids.len() >= DECODER_SEQ_LEN - 1 {
            println!("Max sequence length reached");
            break;
        }

        generated_ids.push(next_token);
    }

    Ok(generated_ids)
}

/// Greedy decoding (no sampling)
fn greedy_decode_fullseq(
    encoder_output: &[f32],
    decoder: &mut ModelExecutor,
    encoder_attention_mask: &[f32],
    max_tokens: usize,
    eos_token_id: u32,
) -> Result<Vec<u32>> {
    decode_fullseq_with_sampling(
        encoder_output,
        decoder,
        encoder_attention_mask,
        max_tokens,
        eos_token_id,
        1.0,
        0,
    )
}

#[test]
fn test_t5_fullseq_generation() -> Result<()> {
    let encoder_path = Path::new("/workspace/models/t5-small/encoder_new.holb");
    let decoder_path = Path::new("/workspace/models/t5-small/decoder_fullseq.holb");
    let tokenizer_path = Path::new("/workspace/models/t5-small/tokenizer.json");

    if !encoder_path.exists() {
        eprintln!("Skipping test: encoder not found at {:?}", encoder_path);
        return Ok(());
    }
    if !decoder_path.exists() {
        eprintln!("Skipping test: decoder not found at {:?}", decoder_path);
        return Ok(());
    }
    if !tokenizer_path.exists() {
        eprintln!("Skipping test: tokenizer not found at {:?}", tokenizer_path);
        return Ok(());
    }

    println!("=== T5 Full-Sequence Generation Test ===\n");

    // Load tokenizer
    println!("Loading tokenizer...");
    let tokenizer = SentencePieceTokenizer::from_file(tokenizer_path)?;

    // Encode prompt
    let prompt = "translate English to German: Hello, how are you?";
    println!("Prompt: '{}'\n", prompt);

    let input_ids = tokenizer.encode(prompt, ENCODER_SEQ_LEN)?;

    // Find actual prompt length (before padding/EOS)
    // The tokenizer adds EOS and pads - find first pad token after content
    let prompt_len = input_ids
        .iter()
        .take_while(|&&id| id != 0) // Count until first pad
        .count();

    println!(
        "Encoded tokens (first {}): {:?}",
        prompt_len,
        &input_ids[..prompt_len.min(20)]
    );

    // Use tokens directly (already padded)
    let input_ids_padded: Vec<f32> = input_ids.iter().map(|&x| x as f32).collect();

    // Attention mask: 1 for real tokens, 0 for padding
    let mut attention_mask: Vec<f32> = vec![0.0; ENCODER_SEQ_LEN];
    attention_mask[..prompt_len].fill(1.0);

    // Load encoder
    println!("\nLoading encoder...");
    let mut encoder = ModelExecutor::from_holo_file(encoder_path)?;

    // Run encoder
    let mut encoder_inputs = HashMap::new();
    encoder_inputs.insert(
        "input_ids".to_string(),
        Tensor::new(input_ids_padded.clone(), vec![1, ENCODER_SEQ_LEN]),
    );
    encoder_inputs.insert(
        "attention_mask".to_string(),
        Tensor::new(attention_mask.clone(), vec![1, ENCODER_SEQ_LEN]),
    );

    println!("Running encoder...");
    let encoder_outputs = encoder.execute(encoder_inputs)?;

    let encoder_hidden = encoder_outputs
        .values()
        .next()
        .ok_or_else(|| anyhow::anyhow!("No encoder output"))?;

    println!(
        "Encoder output: {} elements, stats: min={:.4}, max={:.4}",
        encoder_hidden.data.len(),
        encoder_hidden
            .data
            .iter()
            .cloned()
            .reduce(f32::min)
            .unwrap_or(0.0),
        encoder_hidden
            .data
            .iter()
            .cloned()
            .reduce(f32::max)
            .unwrap_or(0.0),
    );

    // Load decoder
    println!("\nLoading full-sequence decoder...");
    let mut decoder = ModelExecutor::from_holo_file(decoder_path)?;

    // Generate
    println!("\nGenerating (max 32 tokens)...");
    let generated = greedy_decode_fullseq(
        &encoder_hidden.data,
        &mut decoder,
        &attention_mask,
        32,
        1, // EOS token ID
    )?;

    println!("\n=== Results ===");
    println!("Generated token IDs: {:?}", generated);

    // Decode to text
    let decoded_text = tokenizer.decode(&generated)?;
    println!("\nDecoded text: '{}'", decoded_text);

    // Verify we generated something meaningful
    assert!(
        !generated.is_empty(),
        "Should generate at least start token"
    );
    assert!(
        generated.len() > 1,
        "Should generate at least one token beyond start"
    );

    // Check that tokens are in valid range
    for &token in &generated {
        assert!(
            token < VOCAB_SIZE as u32,
            "Token {} out of vocab range",
            token
        );
    }

    // Check that decoded text is non-empty (after removing start token)
    assert!(
        !decoded_text.trim().is_empty() || generated.len() <= 2,
        "Decoded text should not be empty for multi-token output"
    );

    println!("\n✓ Full-sequence generation completed!");
    Ok(())
}

/// Helper to run T5 generation with a given prompt
fn run_t5_generation(
    prompt: &str,
    encoder_path: &Path,
    decoder_path: &Path,
    tokenizer: &SentencePieceTokenizer,
    max_tokens: usize,
    temperature: f32,
    top_k: usize,
) -> Result<(Vec<u32>, String)> {
    let input_ids = tokenizer.encode(prompt, ENCODER_SEQ_LEN)?;
    let prompt_len = input_ids.iter().take_while(|&&id| id != 0).count();

    let input_ids_padded: Vec<f32> = input_ids.iter().map(|&x| x as f32).collect();
    let mut attention_mask: Vec<f32> = vec![0.0; ENCODER_SEQ_LEN];
    attention_mask[..prompt_len].fill(1.0);

    // Run encoder
    let mut encoder = ModelExecutor::from_holo_file(encoder_path)?;
    let mut encoder_inputs = HashMap::new();
    encoder_inputs.insert(
        "input_ids".to_string(),
        Tensor::new(input_ids_padded, vec![1, ENCODER_SEQ_LEN]),
    );
    encoder_inputs.insert(
        "attention_mask".to_string(),
        Tensor::new(attention_mask.clone(), vec![1, ENCODER_SEQ_LEN]),
    );
    let encoder_outputs = encoder.execute(encoder_inputs)?;
    let encoder_hidden = encoder_outputs
        .values()
        .next()
        .ok_or_else(|| anyhow::anyhow!("No encoder output"))?;

    // Run decoder with sampling
    let mut decoder = ModelExecutor::from_holo_file(decoder_path)?;
    let generated = decode_fullseq_with_sampling(
        &encoder_hidden.data,
        &mut decoder,
        &attention_mask,
        max_tokens,
        1, // EOS
        temperature,
        top_k,
    )?;

    let decoded_text = tokenizer.decode(&generated)?;
    Ok((generated, decoded_text))
}

#[test]
fn test_t5_summarize_with_sampling() -> Result<()> {
    let encoder_path = Path::new("/workspace/models/t5-small/encoder_new.holb");
    let decoder_path = Path::new("/workspace/models/t5-small/decoder_fullseq.holb");
    let tokenizer_path = Path::new("/workspace/models/t5-small/tokenizer.json");

    if !encoder_path.exists() || !decoder_path.exists() || !tokenizer_path.exists() {
        eprintln!("Skipping test: models or tokenizer not found");
        return Ok(());
    }

    println!("=== T5 Summarization with Sampling ===\n");
    let tokenizer = SentencePieceTokenizer::from_file(tokenizer_path)?;

    // A longer text that's more suitable for summarization
    let prompt = "summarize: Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed. It uses algorithms to find patterns in data and make predictions or decisions.";
    println!("Prompt: '{}'\n", prompt);

    let (generated, decoded_text) = run_t5_generation(
        prompt,
        encoder_path,
        decoder_path,
        &tokenizer,
        24,  // max tokens
        0.7, // temperature (lower = more focused)
        40,  // top_k
    )?;

    println!("Generated token IDs: {:?}", generated);
    println!("\n>>> Summary: '{}'\n", decoded_text);

    assert!(generated.len() > 1, "Should generate tokens");
    assert!(
        !decoded_text.trim().is_empty(),
        "Summary should not be empty"
    );

    println!("✓ Summarization with sampling completed!");
    Ok(())
}

#[test]
fn test_t5_question_answering() -> Result<()> {
    let encoder_path = Path::new("/workspace/models/t5-small/encoder_new.holb");
    let decoder_path = Path::new("/workspace/models/t5-small/decoder_fullseq.holb");
    let tokenizer_path = Path::new("/workspace/models/t5-small/tokenizer.json");

    if !encoder_path.exists() || !decoder_path.exists() || !tokenizer_path.exists() {
        eprintln!("Skipping test: models or tokenizer not found");
        return Ok(());
    }

    println!("=== T5 Question Answering ===\n");
    let tokenizer = SentencePieceTokenizer::from_file(tokenizer_path)?;

    // T5 can do question answering with the right prompt format
    let prompt = "question: What is the capital of France? context: France is a country in Western Europe. Paris is the capital and largest city of France.";
    println!("Prompt: '{}'\n", prompt);

    let (generated, decoded_text) = run_t5_generation(
        prompt,
        encoder_path,
        decoder_path,
        &tokenizer,
        16,  // max tokens
        0.5, // lower temperature for factual answers
        20,  // top_k
    )?;

    println!("Generated token IDs: {:?}", generated);
    println!("\n>>> Answer: '{}'\n", decoded_text);

    assert!(generated.len() > 1, "Should generate tokens");

    println!("✓ Question answering completed!");
    Ok(())
}

#[test]
fn test_t5_sentence_completion() -> Result<()> {
    let encoder_path = Path::new("/workspace/models/t5-small/encoder_new.holb");
    let decoder_path = Path::new("/workspace/models/t5-small/decoder_fullseq.holb");
    let tokenizer_path = Path::new("/workspace/models/t5-small/tokenizer.json");

    if !encoder_path.exists() || !decoder_path.exists() || !tokenizer_path.exists() {
        eprintln!("Skipping test: models or tokenizer not found");
        return Ok(());
    }

    println!("=== T5 Sentence Completion ===\n");
    let tokenizer = SentencePieceTokenizer::from_file(tokenizer_path)?;

    // T5 can complete sentences
    let prompt = "complete: The weather today is";
    println!("Prompt: '{}'\n", prompt);

    let (generated, decoded_text) =
        run_t5_generation(prompt, encoder_path, decoder_path, &tokenizer, 16, 0.8, 30)?;

    println!("Generated token IDs: {:?}", generated);
    println!("\n>>> Completion: '{}'\n", decoded_text);

    assert!(generated.len() > 1, "Should generate tokens");

    println!("✓ Sentence completion completed!");
    Ok(())
}
