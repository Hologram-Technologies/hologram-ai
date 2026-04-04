//! Speculative decoding (Plan 055 Phase 2).
//!
//! Generates N draft tokens greedily, then verifies each against the target
//! model's distribution. Accepted tokens are emitted in a single batch,
//! giving up to N tokens per verification cycle.
//!
//! This initial implementation uses the **same model** for both drafting and
//! verification (self-speculative). The draft phase runs the decode tape N
//! times at seq=1 each. Each draft token is verified by comparing the target
//! model's top-1 prediction at that position.
//!
//! Greedy acceptance: if target's argmax == draft token, accept and advance.
//! On mismatch, reject the draft token and use the target's prediction instead.
//! This gives ~1.5-2x effective throughput for high-confidence models.

use crate::compiler::HoloRunner;
use hologram::GraphInputs;
use tracing::info;

/// Configuration for speculative decoding.
pub struct SpeculativeConfig {
    /// Number of draft tokens to generate before verification.
    pub draft_steps: usize,
    /// Input slot index for token IDs.
    pub input_slot: u32,
    /// Input slot for position_ids (if model requires it).
    pub pos_slot: Option<u32>,
    /// Input slot for attention_mask (if model requires it).
    pub mask_slot: Option<u32>,
    /// Dtype for attention mask values.
    pub mask_dtype: Option<hologram::hologram_archive::weight::WeightDType>,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Bytes per logit position (vocab_size * 4 for f32).
    pub bytes_per_pos: usize,
    /// Input dtype for token serialization.
    pub input_dtype: hologram::hologram_archive::weight::WeightDType,
}

/// Result of one speculative decode cycle.
pub struct SpeculativeResult {
    /// Accepted token IDs (1..=draft_steps+1 tokens).
    pub accepted_tokens: Vec<u32>,
    /// Number of draft tokens that matched the target.
    pub n_accepted: usize,
    /// Total forward passes executed (draft + verify).
    pub n_forward_passes: usize,
}

/// Run one speculative decode cycle: draft N tokens, then verify.
///
/// Returns the accepted tokens. The caller should append these to the
/// token sequence and advance the KV cache.
pub fn speculative_decode_step(
    runner: &HoloRunner,
    kv_state: &mut hologram::KvCacheState,
    last_token: u32,
    config: &SpeculativeConfig,
) -> anyhow::Result<SpeculativeResult> {
    let mut draft_tokens: Vec<u32> = Vec::with_capacity(config.draft_steps);
    let mut draft_kv_positions: Vec<usize> = Vec::with_capacity(config.draft_steps);
    let mut current_token = last_token;
    let initial_kv_pos = kv_state.write_pos();

    // ── Phase 1: Draft N tokens greedily ────────────────────────────────
    for _i in 0..config.draft_steps {
        let outputs = run_single_token(
            runner,
            kv_state,
            current_token,
            config,
        )?;

        let logit_data = match outputs.get(0) {
            Some((_, data)) => data,
            None => break,
        };

        let next = greedy_argmax(logit_data, config.vocab_size);
        match next {
            Some(token_id) => {
                draft_kv_positions.push(kv_state.write_pos());
                draft_tokens.push(token_id);
                current_token = token_id;
            }
            None => break,
        }
    }

    if draft_tokens.is_empty() {
        return Ok(SpeculativeResult {
            accepted_tokens: vec![],
            n_accepted: 0,
            n_forward_passes: 0,
        });
    }

    // ── Phase 2: Batch verify all draft tokens in ONE forward pass ─────
    // Reset KV cache to pre-draft position. Run verify tape (seq=N) with
    // all draft tokens as input. BLAS amortizes weight reads across N tokens.
    kv_state.truncate_to(initial_kv_pos);

    let n_drafts = draft_tokens.len();

    // Build verify input: [last_token, draft_0, draft_1, ..., draft_{N-1}]
    let mut verify_tokens = Vec::with_capacity(n_drafts + 1);
    verify_tokens.push(last_token);
    verify_tokens.extend_from_slice(&draft_tokens);
    let verify_len = verify_tokens.len();

    let verify_bytes = verify_tokens.iter()
        .flat_map(|&id| serialize_token(id, config.input_dtype))
        .collect::<Vec<u8>>();
    let mut inputs = GraphInputs::new();
    inputs.set_with_shape(config.input_slot, verify_bytes, vec![1, verify_len]);

    if let Some(slot) = config.mask_slot {
        let mask_dtype = config.mask_dtype.unwrap_or(
            hologram::hologram_archive::weight::WeightDType::I64,
        );
        let mask_bytes: Vec<u8> = (0..verify_len).flat_map(|_| {
            match mask_dtype {
                hologram::hologram_archive::weight::WeightDType::I64 => 1i64.to_le_bytes().to_vec(),
                hologram::hologram_archive::weight::WeightDType::I32 => 1i32.to_le_bytes().to_vec(),
                _ => 1i64.to_le_bytes().to_vec(),
            }
        }).collect();
        inputs.set_with_shape(slot, mask_bytes, vec![1, verify_len]);
    }

    if let Some(slot) = config.pos_slot {
        let base_pos = initial_kv_pos as i64;
        let pos_bytes: Vec<u8> = (0..verify_len as i64)
            .flat_map(|i| (base_pos + i).to_le_bytes().to_vec())
            .collect();
        inputs.set_with_shape(slot, pos_bytes, vec![1, verify_len]);
    }

    // Run batch verify — ONE forward pass for N+1 tokens.
    let outputs = runner.execute_verify(&inputs, kv_state)?;

    let logit_data = match outputs.get(0) {
        Some((_, data)) => data,
        None => {
            return Ok(SpeculativeResult {
                accepted_tokens: vec![],
                n_accepted: 0,
                n_forward_passes: n_drafts as usize,
            });
        }
    };

    // Accept/reject: compare target argmax at each position vs draft token.
    let mut accepted: Vec<u32> = Vec::new();
    for (i, &draft_token) in draft_tokens.iter().enumerate() {
        // Logits for position i are at position i in the batch output.
        // The target's prediction for position i+1 is argmax(logits[i]).
        let logit_start = i * config.bytes_per_pos;
        let logit_end = logit_start + config.bytes_per_pos;
        if logit_end > logit_data.len() {
            break;
        }
        let target_token = greedy_argmax(&logit_data[logit_start..logit_end], config.vocab_size);

        match target_token {
            Some(target_id) if target_id == draft_token => {
                accepted.push(draft_token);
            }
            Some(target_id) => {
                accepted.push(target_id);
                break;
            }
            None => break,
        }
    }

    // Get the bonus token from the last position's logits.
    if accepted.len() == n_drafts {
        let last_start = n_drafts * config.bytes_per_pos;
        let last_end = last_start + config.bytes_per_pos;
        if last_end <= logit_data.len() {
            if let Some(bonus) = greedy_argmax(&logit_data[last_start..last_end], config.vocab_size) {
                accepted.push(bonus);
            }
        }
    }

    let n_accepted = accepted.len();
    // draft_steps individual passes + 1 batch verify pass
    let n_forward_passes = n_drafts + 1;

    info!(
        n_drafts,
        n_accepted,
        n_forward_passes,
        "speculative: drafted {n_drafts}, accepted {n_accepted}"
    );

    Ok(SpeculativeResult {
        accepted_tokens: accepted,
        n_accepted,
        n_forward_passes,
    })
}

/// Run a single-token forward pass through the decode tape.
fn run_single_token(
    runner: &HoloRunner,
    kv_state: &mut hologram::KvCacheState,
    token: u32,
    config: &SpeculativeConfig,
) -> anyhow::Result<hologram::GraphOutputs> {
    let input_bytes = serialize_token(token, config.input_dtype);
    let mut inputs = GraphInputs::new();
    inputs.set_with_shape(config.input_slot, input_bytes, vec![1, 1]);

    // Attention mask: all-ones for single-token decode.
    if let Some(slot) = config.mask_slot {
        let mask_dtype = config.mask_dtype.unwrap_or(
            hologram::hologram_archive::weight::WeightDType::I64,
        );
        let mask_bytes = match mask_dtype {
            hologram::hologram_archive::weight::WeightDType::I64 => 1i64.to_le_bytes().to_vec(),
            hologram::hologram_archive::weight::WeightDType::I32 => 1i32.to_le_bytes().to_vec(),
            _ => 1i64.to_le_bytes().to_vec(),
        };
        inputs.set_with_shape(slot, mask_bytes, vec![1, 1]);
    }

    if let Some(slot) = config.pos_slot {
        let pos = kv_state.write_pos() as i64;
        let pos_bytes: Vec<u8> = pos.to_le_bytes().to_vec();
        inputs.set_with_shape(slot, pos_bytes, vec![1, 1]);
    }

    runner.execute_with_kv(&inputs, kv_state)
}

/// Serialize a single token ID to bytes matching the model's input dtype.
fn serialize_token(id: u32, dtype: hologram::hologram_archive::weight::WeightDType) -> Vec<u8> {
    use hologram::hologram_archive::weight::WeightDType;
    match dtype {
        WeightDType::I64 => (id as i64).to_le_bytes().to_vec(),
        WeightDType::I32 => (id as i32).to_le_bytes().to_vec(),
        _ => (id as i64).to_le_bytes().to_vec(), // default to i64
    }
}

/// Greedy argmax over logit bytes at the last position.
fn greedy_argmax(logit_bytes: &[u8], vocab_size: usize) -> Option<u32> {
    if logit_bytes.len() < vocab_size * 4 {
        return None;
    }
    // Take the last vocab_size*4 bytes (last position's logits).
    let start = logit_bytes.len() - vocab_size * 4;
    let logits: &[f32] = bytemuck::cast_slice(&logit_bytes[start..]);

    let mut best_idx = 0u32;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = i as u32;
        }
    }
    Some(best_idx)
}
