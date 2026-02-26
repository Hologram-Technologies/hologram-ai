//! Test T5 with a joke prompt to verify end-to-end generation.

use anyhow::Result;
use hologram_ai::runtime::ModelExecutor;
use hologram_ai::runtime::Tensor;
use std::collections::HashMap;
use std::path::Path;

/// Simple greedy decoder for T5
///
/// The decoder model expects input_ids shape [1, 1] - single token at a time.
/// For autoregressive generation, we pass only the last generated token.
fn greedy_decode(
    encoder_output: &[f32],
    decoder: &mut ModelExecutor,
    attention_mask: &[f32],
    max_tokens: usize,
    eos_token_id: u32,
) -> Result<Vec<u32>> {
    let seq_len = 512;
    let hidden_dim = 512;

    // Start with decoder_input_ids = [0] (pad token, T5's decoder start token)
    let mut generated_ids: Vec<u32> = vec![0];

    for step in 0..max_tokens {
        // Get the last token to feed to decoder (autoregressive - single token input)
        let last_token = *generated_ids.last().unwrap_or(&0);
        let decoder_input_ids = vec![last_token as f32];

        // Build inputs for decoder
        // Decoder expects: input_ids [1, 1], encoder_attention_mask [1, 512], encoder_hidden_states [1, 512, 512]
        let mut inputs = HashMap::new();
        inputs.insert(
            "input_ids".to_string(),
            Tensor::new(decoder_input_ids, vec![1, 1]),
        );
        inputs.insert(
            "encoder_attention_mask".to_string(),
            Tensor::new(attention_mask.to_vec(), vec![1, seq_len]),
        );
        inputs.insert(
            "encoder_hidden_states".to_string(),
            Tensor::new(encoder_output.to_vec(), vec![1, seq_len, hidden_dim]),
        );

        // Run decoder
        let outputs = decoder.execute(inputs)?;

        // Get logits (shape: [1, 1, vocab_size])
        // Output order: output_0 = logits (32128 f32), output_1-24 = key/value caches
        let logits = outputs
            .get("logits")
            .or_else(|| outputs.get("output_0"))
            .ok_or_else(|| anyhow::anyhow!("No logits output from decoder"))?;

        // For [1, 1, vocab_size] output, logits are directly the vocabulary scores
        let vocab_size = 32128; // T5 vocab size

        if logits.data.len() < vocab_size {
            println!(
                "Warning: logits too small, got {} elements, expected {}",
                logits.data.len(),
                vocab_size
            );
            break;
        }

        // Take the first vocab_size elements (position 0)
        let last_logits = &logits.data[0..vocab_size];

        // Greedy: pick argmax
        let next_token = last_logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx as u32)
            .unwrap_or(0);

        println!("Step {}: generated token {}", step, next_token);

        // Check for EOS
        if next_token == eos_token_id {
            println!("EOS token generated, stopping");
            break;
        }

        generated_ids.push(next_token);
    }

    Ok(generated_ids)
}

#[test]
fn test_t5_joke_generation() -> Result<()> {
    let encoder_path = Path::new("/workspace/models/t5-small/encoder_new.holb");
    let decoder_path = Path::new("/workspace/models/t5-small/decoder_new.holb");

    if !encoder_path.exists() || !decoder_path.exists() {
        eprintln!("Skipping test: models not found");
        return Ok(());
    }

    println!("=== T5 Joke Generation Test ===");
    println!("Prompt: 'Tell me a joke'");

    // Token IDs for "Tell me a joke" (T5 tokenization)
    // 11736 = "Tell", 140 = "▁me", 3 = "▁a", 9 = "▁", 8989 = "joke", 1 = EOS
    let input_ids: Vec<i64> = vec![11736, 140, 3, 9, 8989, 1];
    let prompt_len = input_ids.len();

    // Pad to 512
    let mut input_ids_padded: Vec<f32> = input_ids.iter().map(|&x| x as f32).collect();
    input_ids_padded.resize(512, 0.0);

    let mut attention_mask: Vec<f32> = vec![1.0; prompt_len];
    attention_mask.resize(512, 0.0);

    // Load encoder
    println!("\nLoading encoder...");
    let mut encoder = ModelExecutor::from_holo_file(encoder_path)?;

    // Run encoder
    let mut encoder_inputs = HashMap::new();
    encoder_inputs.insert(
        "input_ids".to_string(),
        Tensor::new(input_ids_padded.clone(), vec![1, 512]),
    );
    encoder_inputs.insert(
        "attention_mask".to_string(),
        Tensor::new(attention_mask.clone(), vec![1, 512]),
    );

    println!("Running encoder...");
    let encoder_outputs = encoder.execute(encoder_inputs)?;

    let encoder_hidden = encoder_outputs
        .values()
        .next()
        .ok_or_else(|| anyhow::anyhow!("No encoder output"))?;

    println!("Encoder output shape: {:?}", encoder_hidden.shape);
    println!(
        "Encoder output stats: min={:.4}, max={:.4}, mean={:.4}",
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
        encoder_hidden.data.iter().sum::<f32>() / encoder_hidden.data.len() as f32,
    );

    // Load decoder
    println!("\nLoading decoder...");
    let mut decoder = ModelExecutor::from_holo_file(decoder_path)?;

    // Generate tokens
    println!("\nGenerating response (max 20 tokens)...");
    let generated = greedy_decode(
        &encoder_hidden.data,
        &mut decoder,
        &attention_mask,
        20, // max tokens
        1,  // EOS token ID
    )?;

    println!("\n=== Results ===");
    println!("Generated token IDs: {:?}", generated);

    // Decode tokens to text (simplified - just show token IDs)
    // A proper implementation would use the tokenizer
    println!("\nNote: Token decoding requires tokenizer integration.");
    println!("Non-gibberish would show coherent token sequences.");

    // Check that we generated something
    assert!(generated.len() > 1, "Should generate at least one token");

    // Check that tokens are in valid range
    for &token in &generated {
        assert!(token < 32128, "Token {} out of vocab range", token);
    }

    println!("\n✓ Generation completed successfully!");
    Ok(())
}
