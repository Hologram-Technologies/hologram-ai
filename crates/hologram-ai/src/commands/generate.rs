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

/// A standard auxiliary LM input synthesized each step (the model carries it as
/// a named port alongside `input_ids`).
enum AuxKind {
    /// `attention_mask`: 1 for real positions, 0 for padding.
    AttentionMask,
    /// `position_ids`: `0..cur_len`, padding 0.
    PositionIds,
}

struct AuxInput {
    index: usize,
    dtype: u8,
    kind: AuxKind,
}

/// The fixed LM port contract resolved from the compiled archive. Every port is
/// bound **by name** (`input_ids`, `logits`, `attention_mask`, `position_ids`) —
/// archives carry names end to end (importer → lowering → archive), so an
/// unidentifiable port is a hard error, never a positional guess.
struct LmContract {
    n_inputs: usize,
    ids_index: usize,
    id_dtype: u8,
    seq_len: usize,
    logits_index: usize,
    vocab: usize,
    aux: Vec<AuxInput>,
}

impl LmContract {
    fn resolve(runner: &HoloRunner) -> Result<Self> {
        let ins = runner.input_port_info();
        let outs = runner.output_port_info();

        // Bind input_ids and logits strictly by name — no positional guess.
        let ids_index = runner.input_index_by_name("input_ids").ok_or_else(|| {
            anyhow::anyhow!(
                "no input named `input_ids` (the model's input ports are {:?}); generation \
                 requires named ports — recompile so the token input is named `input_ids`",
                ins.iter().map(|p| p.name.as_str()).collect::<Vec<_>>()
            )
        })?;
        let logits_index = runner.output_index_by_name("logits").ok_or_else(|| {
            anyhow::anyhow!(
                "no output named `logits` (the model's output ports are {:?}); generation \
                 requires the logits output to be named `logits`",
                outs.iter().map(|p| p.name.as_str()).collect::<Vec<_>>()
            )
        })?;

        let id_dtype = ins[ids_index].dtype;
        // Integer id dtypes (hologram_backend::cpu::dtype): U8=1, I8=2, I32=4, I64=5.
        if !matches!(id_dtype, 1 | 2 | 4 | 5) {
            bail!("input_ids must be an integer tensor (U8/I8/I32/I64), got dtype tag {id_dtype}");
        }
        if outs[logits_index].dtype != 8 {
            bail!("logits output must be f32 (dtype tag 8), got tag {}", outs[logits_index].dtype);
        }
        let seq_len = ins[ids_index].element_count;
        if seq_len == 0 {
            bail!("input_ids has zero elements — model was compiled with an empty sequence length");
        }
        let logit_count = outs[logits_index].element_count;
        if !logit_count.is_multiple_of(seq_len) {
            bail!(
                "logits element count {logit_count} is not divisible by seq_len {seq_len}; \
                 the model does not match the [1, seq_len, vocab] causal-LM contract"
            );
        }
        let vocab = logit_count / seq_len;

        // Any other input must be a recognized auxiliary we can synthesize —
        // otherwise fail loud (no silent zero-fill of an unknown semantic input).
        let mut aux = Vec::new();
        for (i, p) in ins.iter().enumerate() {
            if i == ids_index {
                continue;
            }
            let kind = match p.name.as_str() {
                "attention_mask" => AuxKind::AttentionMask,
                "position_ids" => AuxKind::PositionIds,
                other => bail!(
                    "generation can't synthesize input[{i}] {other:?}; only input_ids, \
                     attention_mask, and position_ids are auto-filled — supply it via the raw path"
                ),
            };
            aux.push(AuxInput { index: i, dtype: p.dtype, kind });
        }

        Ok(Self { n_inputs: ins.len(), ids_index, id_dtype, seq_len, logits_index, vocab, aux })
    }
}

/// Encode a `[1, seq_len]` input buffer: `vals` fill positions `0..vals.len()`,
/// the rest are padded with 0, in the port's dtype (int or float). A causal LM's
/// logits at the last real position attend only to `0..pos`, so padding never
/// affects them.
fn encode_vals(vals: &[f64], seq_len: usize, dtype: u8) -> Vec<u8> {
    let width = match dtype {
        1 | 2 => 1,     // U8 / I8
        5 | 9 => 8,     // I64 / F64
        _ => 4,         // I32 / F32 (and default)
    };
    let mut buf = vec![0u8; seq_len * width];
    for (i, &v) in vals.iter().take(seq_len).enumerate() {
        let off = i * width;
        match dtype {
            1 | 2 => buf[off] = v as u8,                                   // u8 / i8
            5 => buf[off..off + 8].copy_from_slice(&(v as i64).to_le_bytes()), // i64
            9 => buf[off..off + 8].copy_from_slice(&v.to_le_bytes()),      // f64
            8 => buf[off..off + 4].copy_from_slice(&(v as f32).to_le_bytes()), // f32
            _ => buf[off..off + 4].copy_from_slice(&(v as i32).to_le_bytes()), // i32 (and width-4 default)
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

        // Build every graph input: the token window at `input_ids`, and each
        // recognized auxiliary (attention_mask = 1s, position_ids = 0..cur_len)
        // synthesized at its named port.
        let win: Vec<f64> = window.iter().map(|&t| t as f64).collect();
        let mut bufs: Vec<Vec<u8>> = (0..lm.n_inputs).map(|_| Vec::new()).collect();
        bufs[lm.ids_index] = encode_vals(&win, lm.seq_len, lm.id_dtype);
        for a in &lm.aux {
            let vals: Vec<f64> = match a.kind {
                AuxKind::AttentionMask => vec![1.0; cur_len],
                AuxKind::PositionIds => (0..cur_len).map(|p| p as f64).collect(),
            };
            bufs[a.index] = encode_vals(&vals, lm.seq_len, a.dtype);
        }
        let refs: Vec<&[u8]> = bufs.iter().map(|b| b.as_slice()).collect();
        let outputs = runner.execute(&refs).context("forward pass failed")?;
        let logits: &[f32] = bytemuck::cast_slice(&outputs[lm.logits_index].bytes);

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
