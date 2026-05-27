//! Autoregressive text generation over a compiled `.holo` causal LM.
//!
//! The UOR-native runtime has no KV-cache: each decode step is one forward
//! `execute` over the whole window, and the repeated prefix is recognized by
//! κ-label and elided inside the session (architecture §5.3) — so generation
//! is a plain loop of forward passes, not a host-managed cache walk.
//!
//! The compiled archive carries **no tokenizer and no tensor names** (the
//! `.holo` section set is closed; a port is identified only by position and
//! dtype). So the tokenizer and generation metadata are supplied by the caller
//! at run time (CLI flags), and the LM contract is taken by convention:
//!
//! - input port 0 — `input_ids`, shape `[1, seq_len]`, integer dtype; its
//!   element count is the fixed sequence length baked at compile time;
//! - output port 0 — `logits`, shape `[1, seq_len, vocab_size]`, f32; so
//!   `element_count / seq_len` is the vocabulary size.
//!
//! A model that doesn't match (≠1 input/output, non-integer ids, non-f32
//! logits, or a logit count not divisible by seq_len) is rejected with a clear
//! error rather than guessed at — generation targets causal LMs specifically;
//! other graphs use the raw `--input` path.

use std::io::Write;

use anyhow::{bail, Context, Result};
use hologram_ai_tokenizer::Tokenizer;

use crate::runner::HoloRunner;

/// Sampling / stopping configuration for one generation request.
#[derive(Debug, Clone)]
pub struct GenConfig {
    /// Maximum number of new tokens to generate.
    pub max_tokens: usize,
    /// Softmax temperature. `0.0` (or negative) ⇒ greedy argmax (deterministic).
    pub temperature: f32,
    /// If set, restrict sampling to the `k` highest-probability tokens.
    pub top_k: Option<usize>,
    /// Stop strings — generation halts once the decoded suffix contains one.
    pub stop: Vec<String>,
    /// End-of-sequence token id. `None` ⇒ use the tokenizer's `eos_token_id()`.
    pub eos: Option<u32>,
    /// RNG seed for temperature sampling (reproducibility). Unused when greedy.
    pub seed: u64,
}

impl Default for GenConfig {
    fn default() -> Self {
        Self {
            max_tokens: 64,
            temperature: 0.0,
            top_k: None,
            stop: Vec::new(),
            eos: None,
            seed: 0x9E3779B97F4A7C15,
        }
    }
}

/// Apply a prompt template (`{prompt}` placeholder) if one is given.
pub fn apply_template(template: Option<&str>, prompt: &str) -> String {
    match template {
        Some(t) if t.contains("{prompt}") => t.replace("{prompt}", prompt),
        Some(t) => format!("{t}{prompt}"),
        None => prompt.to_string(),
    }
}

/// The fixed LM port contract resolved from the compiled archive.
struct LmContract {
    /// Fixed sequence length (input_ids element count).
    seq_len: usize,
    /// input_ids dtype tag (I32 = 4, I64 = 5).
    id_dtype: u8,
    /// Vocabulary size (logits element count / seq_len).
    vocab: usize,
}

impl LmContract {
    fn resolve(runner: &HoloRunner) -> Result<Self> {
        let ins = runner.input_port_info();
        let outs = runner.output_port_info();
        if ins.len() != 1 {
            bail!(
                "generation expects a causal LM with a single `input_ids` input, but the model \
                 has {} inputs; use the raw `--input INDEX:HEX` path for multi-input models",
                ins.len()
            );
        }
        if outs.len() != 1 {
            bail!(
                "generation expects a single `logits` output, but the model has {} outputs",
                outs.len()
            );
        }
        let id_dtype = ins[0].dtype;
        // Integer id dtypes (hologram_backend::cpu::dtype): U8=1, I8=2, I32=4, I64=5.
        if !matches!(id_dtype, 1 | 2 | 4 | 5) {
            bail!(
                "input_ids must be an integer tensor (U8/I8/I32/I64), got dtype tag {id_dtype}"
            );
        }
        // logits must be f32 (tag 8).
        if outs[0].dtype != 8 {
            bail!("logits output must be f32 (dtype tag 8), got tag {}", outs[0].dtype);
        }
        let seq_len = ins[0].element_count;
        if seq_len == 0 {
            bail!("input_ids has zero elements — model was compiled with an empty sequence length");
        }
        let logit_count = outs[0].element_count;
        if !logit_count.is_multiple_of(seq_len) {
            bail!(
                "logits element count {logit_count} is not divisible by seq_len {seq_len}; \
                 the model does not match the [1, seq_len, vocab] causal-LM contract"
            );
        }
        let vocab = logit_count / seq_len;
        Ok(Self { seq_len, id_dtype, vocab })
    }
}

/// Encode a `[1, seq_len]` token-id input buffer: positions `0..tokens.len()`
/// hold the window, the rest are padded with 0. A causal LM's logits at the
/// last real position attend only to `0..pos`, so padding never affects them.
fn encode_ids(tokens: &[u32], seq_len: usize, dtype: u8) -> Vec<u8> {
    let width = match dtype {
        1 | 2 => 1, // U8 / I8
        5 => 8,     // I64
        _ => 4,     // I32 (and default)
    };
    let mut buf = vec![0u8; seq_len * width];
    for (i, &tok) in tokens.iter().take(seq_len).enumerate() {
        let off = i * width;
        match dtype {
            1 | 2 => buf[off] = tok as u8,
            5 => buf[off..off + 8].copy_from_slice(&(tok as i64).to_le_bytes()),
            _ => buf[off..off + 4].copy_from_slice(&(tok as i32).to_le_bytes()),
        }
    }
    buf
}

/// Greedy argmax over a logit row.
fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best as u32
}

/// SplitMix64 → uniform f64 in [0, 1). Self-contained so generation needs no
/// RNG dependency; deterministic given the seed.
fn next_unit(state: &mut u64) -> f64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^= z >> 31;
    // 53-bit mantissa → [0, 1).
    (z >> 11) as f64 / (1u64 << 53) as f64
}

/// Temperature + optional top-k sampling over a logit row.
fn sample(logits: &[f32], temperature: f32, top_k: Option<usize>, rng: &mut u64) -> u32 {
    if temperature <= 0.0 {
        return argmax(logits);
    }
    // Indices sorted by descending logit, truncated to top-k.
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_unstable_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap_or(std::cmp::Ordering::Equal));
    let k = top_k.map(|k| k.clamp(1, idx.len())).unwrap_or(idx.len());
    let keep = &idx[..k];

    // Softmax over the kept logits (shifted by max for numerical stability).
    let max = keep.iter().map(|&i| logits[i] / temperature).fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f64> = keep
        .iter()
        .map(|&i| ((logits[i] / temperature - max) as f64).exp())
        .collect();
    let sum: f64 = exps.iter().sum();

    // Inverse-CDF sample.
    let r = next_unit(rng) * sum;
    let mut acc = 0.0;
    for (n, &i) in keep.iter().enumerate() {
        acc += exps[n];
        if r <= acc {
            return i as u32;
        }
    }
    keep[keep.len() - 1] as u32
}

/// Run autoregressive generation, streaming each decoded delta to `out`.
/// Returns the full generated text (excluding the prompt).
///
/// `prompt` is the already-templated text. Generation stops at `max_tokens`,
/// the eos token, or the first `stop` string in the decoded suffix.
pub fn generate_stream(
    runner: &mut HoloRunner,
    tokenizer: &dyn Tokenizer,
    prompt: &str,
    cfg: &GenConfig,
    out: &mut dyn Write,
) -> Result<String> {
    let lm = LmContract::resolve(runner)?;
    let eos = cfg.eos.unwrap_or_else(|| tokenizer.eos_token_id());

    let prompt_tokens = tokenizer.encode(prompt);
    if prompt_tokens.is_empty() {
        bail!("prompt encoded to zero tokens");
    }
    if prompt_tokens.len() >= lm.seq_len {
        bail!(
            "prompt is {} tokens but the model's fixed sequence length is {}; \
             leave room for at least one generated token",
            prompt_tokens.len(),
            lm.seq_len
        );
    }

    let mut sequence: Vec<u32> = prompt_tokens;
    let mut generated: Vec<u32> = Vec::new();
    let mut emitted = 0usize; // chars of `generated` text already streamed
    let mut rng = cfg.seed;

    for _ in 0..cfg.max_tokens {
        // The window is the last `seq_len` tokens of the running sequence.
        let start = sequence.len().saturating_sub(lm.seq_len);
        let window = &sequence[start..];
        let cur_len = window.len();

        let ids = encode_ids(window, lm.seq_len, lm.id_dtype);
        let outputs = runner.execute(&[&ids]).context("forward pass failed")?;
        let logits: &[f32] = bytemuck::cast_slice(&outputs[0].bytes);

        // Next-token distribution is the logit row at the last real position.
        let pos = cur_len - 1;
        let row = &logits[pos * lm.vocab..(pos + 1) * lm.vocab];
        let next = sample(row, cfg.temperature, cfg.top_k, &mut rng);

        if next == eos {
            break;
        }
        generated.push(next);
        sequence.push(next);

        // Stream the newly-decoded suffix (handles multi-token characters by
        // re-decoding the full generated text and emitting only the delta).
        let text = tokenizer.decode(&generated);
        if let Some(delta) = text.get(emitted..) {
            if !delta.is_empty() {
                out.write_all(delta.as_bytes()).ok();
                out.flush().ok();
                emitted = text.len();
            }
        }

        // Stop strings: halt once the decoded text contains one.
        if cfg.stop.iter().any(|s| !s.is_empty() && text.contains(s.as_str())) {
            break;
        }
    }

    Ok(tokenizer.decode(&generated))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmax_picks_highest() {
        assert_eq!(argmax(&[0.1, 0.9, 0.3]), 1);
        assert_eq!(argmax(&[2.0, -1.0, 1.0]), 0);
    }

    #[test]
    fn template_substitution() {
        assert_eq!(apply_template(Some("<u>{prompt}</u>"), "hi"), "<u>hi</u>");
        assert_eq!(apply_template(Some("PRE: "), "hi"), "PRE: hi");
        assert_eq!(apply_template(None, "hi"), "hi");
    }

    #[test]
    fn greedy_sample_equals_argmax() {
        let mut rng = 1;
        assert_eq!(sample(&[0.1, 5.0, 0.2], 0.0, None, &mut rng), 1);
    }

    #[test]
    fn top_k_one_is_argmax() {
        let mut rng = 42;
        // Even with temperature, top_k=1 forces the highest-logit token.
        assert_eq!(sample(&[1.0, 9.0, 2.0, 3.0], 1.0, Some(1), &mut rng), 1);
    }
}
