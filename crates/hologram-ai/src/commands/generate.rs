//! Autoregressive text generation over a compiled causal LM.
//!
//! The UOR-native runtime has no KV-cache: each decode step is one forward
//! `execute` over the whole window, and the repeated prefix is recognized by
//! κ-label and elided inside the session (architecture §5.3) — so generation
//! is a plain loop of forward passes, not a host-managed cache walk.
//!
//! Arbitrary length comes from a [`SessionProvider`](crate::engine): the loop
//! asks for a window at least as long as the running sequence, and the provider
//! supplies (growing/recompiling) a session that can run it — up to the model's
//! real context length. So the prompt and the generated continuation are bounded
//! only by the model, never by an archive's baked `seq_len` or a fixed token
//! cap: generation is bounded by the model's stop conditions and the remaining
//! context (journey S4). An explicit `max_tokens` is a caller-imposed cap
//! within that structural bound, never a way past it.
//!
//! The LM contract is taken by convention from the session's named ports:
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
use hologram_ai_tokenizer::{StreamingDecoder, Tokenizer};

use crate::decode::DecodeSession;
use crate::engine::{LmSession, SessionProvider};

/// Sampling / stopping configuration for one generation request.
#[derive(Debug, Clone)]
pub struct GenConfig {
    /// Maximum number of new tokens to generate. `None` (the default) means
    /// the remaining context window — the model's context length minus the
    /// encoded prompt (journey S4: generation is bounded by the model's stop
    /// conditions and the remaining context, never a fixed token cap). An
    /// explicit `Some(n)` is additionally clamped by that same remaining
    /// window; a remaining window of zero is the empty completion, not an
    /// error.
    pub max_tokens: Option<usize>,
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
            max_tokens: None,
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
    /// `last_pos`: the single consumed position `cur_len - 1` (row
    /// `single-position-head` — the head gathers one row).
    PositionIds,
    /// The consumed position (`cur_len - 1`) for the single-position head.
    LastPos,
    /// `past_key_values.*`: an empty (zero-length) past — hologram-ai runs a
    /// with-past decoder export as a full-recompute prefill (no external
    /// KV-cache; reuse is content-addressed κ-label elision). The port is
    /// concretized to a 0-length sequence, so the synthesized buffer is empty.
    EmptyPast,
}

struct AuxInput {
    index: usize,
    dtype: u8,
    /// The port's declared element count — the synthesized buffer's length. Not
    /// every aux is `seq_len`-sized (e.g. an empty past is 0; a `[1, past+1]`
    /// mask shrinks with an empty past).
    element_count: usize,
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
    /// True when the head gathers the consumed position (`last_pos` port).
    single_position: bool,
    vocab: usize,
    aux: Vec<AuxInput>,
}

impl LmContract {
    fn resolve(runner: &dyn LmSession) -> Result<Self> {
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
            bail!(
                "logits output must be f32 (dtype tag 8), got tag {}",
                outs[logits_index].dtype
            );
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
        // A single-position head (a `last_pos` input port) emits ONE logit
        // row; a whole-window head emits `seq_len` rows.
        let single_position = ins.iter().any(|p| p.name == "last_pos");
        let vocab = if single_position {
            logit_count
        } else {
            logit_count / seq_len
        };

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
                "last_pos" => AuxKind::LastPos,
                name if name.starts_with("past_key_values") || name.starts_with("past.") => {
                    AuxKind::EmptyPast
                }
                other => bail!(
                    "generation can't synthesize input[{i}] {other:?}; only input_ids, \
                     attention_mask, position_ids, and (empty) past_key_values are auto-filled — \
                     supply it via the raw path"
                ),
            };
            aux.push(AuxInput {
                index: i,
                dtype: p.dtype,
                element_count: p.element_count,
                kind,
            });
        }

        Ok(Self {
            n_inputs: ins.len(),
            ids_index,
            id_dtype,
            seq_len,
            logits_index,
            vocab,
            single_position,
            aux,
        })
    }
}

/// Encode a `[1, seq_len]` input buffer: `vals` fill positions `0..vals.len()`,
/// the rest are padded with 0, in the port's dtype (int or float). A causal LM's
/// logits at the last real position attend only to `0..pos`, so padding never
/// affects them.
fn encode_vals(vals: &[f64], seq_len: usize, dtype: u8) -> Vec<u8> {
    let width = match dtype {
        1 | 2 => 1, // U8 / I8
        5 | 9 => 8, // I64 / F64
        _ => 4,     // I32 / F32 (and default)
    };
    let mut buf = vec![0u8; seq_len * width];
    for (i, &v) in vals.iter().take(seq_len).enumerate() {
        let off = i * width;
        match dtype {
            1 | 2 => buf[off] = v as u8, // u8 / i8
            5 => buf[off..off + 8].copy_from_slice(&(v as i64).to_le_bytes()), // i64
            9 => buf[off..off + 8].copy_from_slice(&v.to_le_bytes()), // f64
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

/// A per-absolute-position RNG state (SplitMix64 mixing the seed and position).
/// The sampler's draw depends only on `(seed, position)`, NEVER on the decode
/// PATH — so plain and speculative sampled decode draw the identical token at
/// every position (the sampled byte-identity witness), and a run reproduces
/// regardless of how many tokens each pass batched.
fn position_rng(seed: u64, position: u64) -> u64 {
    let mut z = seed ^ position.wrapping_mul(0x9E3779B97F4A7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// [`sample`] seeded by the token's ABSOLUTE position (via [`position_rng`]).
/// Greedy (`temperature <= 0`) is argmax and ignores the seed; sampling draws
/// from the per-position stream, so the token at a position is a pure function
/// of `(logits, seed, position)` — path-independent, which is what lets
/// speculative decode reproduce plain sampled decode exactly.
fn sample_at(
    logits: &[f32],
    temperature: f32,
    top_k: Option<usize>,
    seed: u64,
    position: u64,
) -> u32 {
    let mut rng = position_rng(seed, position);
    sample(logits, temperature, top_k, &mut rng)
}

/// Temperature + optional top-k sampling over a logit row.
///
/// The candidate set is built in the order the inverse-CDF walks it — WITHOUT a
/// full O(vocab·log vocab) sort, which at chat vocab sizes (150K) cost ~5–6 ms per
/// token (`pool-bench.rs`), a vocab-scaling per-token ceiling with no purpose:
/// - **top-k:** the k highest logits are selected in O(vocab) (`select_nth_unstable`),
///   then only those k are sorted descending. The kept set and its order are
///   identical to sorting the whole vocab and truncating, so the sampled token is
///   byte-for-byte unchanged (barring an exact float tie straddling the k-boundary,
///   where a full sort is already arbitrary). O(vocab + k·log k).
/// - **no top-k:** the inverse-CDF over the FULL distribution needs no ordering at
///   all — token i is drawn with probability p_i in any index order — so the sort
///   is skipped entirely and the natural order is walked. Same distribution, O(vocab).
fn sample(logits: &[f32], temperature: f32, top_k: Option<usize>, rng: &mut u64) -> u32 {
    if temperature <= 0.0 {
        return argmax(logits);
    }
    let desc = |a: &usize, b: &usize| {
        logits[*b]
            .partial_cmp(&logits[*a])
            .unwrap_or(std::cmp::Ordering::Equal)
    };
    let keep: Vec<usize> = match top_k {
        Some(k) => {
            let k = k.clamp(1, logits.len());
            let mut idx: Vec<usize> = (0..logits.len()).collect();
            if k < idx.len() {
                idx.select_nth_unstable_by(k - 1, desc); // top-k to the front, O(vocab)
                idx.truncate(k);
            }
            idx.sort_unstable_by(desc); // order only the k kept
            idx
        }
        None => (0..logits.len()).collect(), // full vocab, natural order — no sort
    };
    let keep = &keep[..];

    // Softmax over the kept logits (shifted by max for numerical stability).
    let max = keep
        .iter()
        .map(|&i| logits[i] / temperature)
        .fold(f32::NEG_INFINITY, f32::max);
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

/// Rolling stop-string scanner with exact `decode(generated).contains(stop)`
/// semantics over streamed deltas — without rescanning the accumulated text.
/// A new occurrence must end in text that is new (the delta) or still mutable
/// (the decoder's pending tail), and can start at most `longest stop − 1`
/// bytes earlier — so a bounded suffix of the stable text plus the pending
/// tail is exactly the region a whole-text scan could newly match in.
struct StopScan<'a> {
    stops: &'a [String],
    /// Longest stop string in bytes minus one — how far back into stable text
    /// an occurrence ending in new text can reach.
    keep: usize,
    /// The retained suffix of the stable (already emitted) text.
    tail: String,
}

impl<'a> StopScan<'a> {
    fn new(stops: &'a [String]) -> Self {
        let keep = stops
            .iter()
            .map(|s| s.len())
            .max()
            .unwrap_or(0)
            .saturating_sub(1);
        Self {
            stops,
            keep,
            tail: String::new(),
        }
    }

    /// Absorb the newly emitted `delta`; report whether any stop string
    /// occurs in the running text (`… tail + delta + pending`).
    fn hit(&mut self, delta: &str, pending: &str) -> bool {
        if self.stops.is_empty() {
            return false;
        }
        self.tail.push_str(delta);
        let candidate = format!("{}{}", self.tail, pending);
        let hit = self
            .stops
            .iter()
            .any(|s| !s.is_empty() && candidate.contains(s.as_str()));
        // Trim the front, keeping AT LEAST `keep` bytes (backing up to the
        // nearest char boundary keeps a byte or two more, never fewer).
        if self.tail.len() > self.keep {
            let mut cut = self.tail.len() - self.keep;
            while !self.tail.is_char_boundary(cut) {
                cut -= 1;
            }
            self.tail.drain(..cut);
        }
        hit
    }
}

/// Flush the decoder's held-back tail (an unfinished multi-token character at
/// the very end) into `acc` and the sink — the returned text is exactly the
/// accumulation of the streamed deltas.
fn finish_stream(mut streamer: StreamingDecoder<'_>, acc: &mut String, out: &mut dyn Write) {
    let rest = streamer.finish();
    if !rest.is_empty() {
        out.write_all(rest.as_bytes()).ok();
        out.flush().ok();
        acc.push_str(&rest);
    }
}

/// Run autoregressive generation, streaming each decoded delta to `out`.
/// Returns the full generated text (excluding the prompt).
///
/// `prompt` is the already-templated text. Generation stops at the eos token,
/// the first `stop` string in the decoded suffix, or when the remaining
/// context window (the model's context length minus the encoded prompt) is
/// exhausted; an explicit `max_tokens` caps it earlier, never later.
///
/// The window grows with the sequence via the [`SessionProvider`], so the
/// prompt and continuation are bounded only by the model's context length.
pub fn generate_stream(
    provider: &mut dyn SessionProvider,
    tokenizer: &dyn Tokenizer,
    prompt: &str,
    cfg: &GenConfig,
    out: &mut dyn Write,
) -> Result<String> {
    let eos = cfg.eos.unwrap_or_else(|| tokenizer.eos_token_id());
    let max_window = provider.max_window();

    let prompt_tokens = tokenizer.encode(prompt);
    if prompt_tokens.is_empty() {
        bail!("prompt encoded to zero tokens");
    }
    if prompt_tokens.len() > max_window {
        bail!(
            "prompt is {} tokens but the model's context length is {}; the model cannot attend \
             to a prompt longer than its trained context",
            prompt_tokens.len(),
            max_window
        );
    }

    // The structural bound (journey S4): the context remaining after the
    // prompt. `None` means exactly this bound; an explicit cap is clamped by
    // it. Zero remaining context ends generation immediately with the empty
    // completion — the prompt filled the window, which is not an error.
    let remaining = max_window - prompt_tokens.len();
    let budget = cfg.max_tokens.map_or(remaining, |n| n.min(remaining));

    let mut sequence: Vec<u32> = prompt_tokens;
    let mut generated: Vec<u32> = Vec::new();
    // Incremental detokenization: the decoder re-decodes only its short
    // pending window per token — O(N) total text work where re-decoding
    // `generated` every step was O(N²) — and the returned text is the
    // accumulation of its deltas, byte-identical to the one-shot decode.
    let mut streamer = StreamingDecoder::new(tokenizer);
    let mut stops = StopScan::new(&cfg.stop);
    let mut acc = String::new();

    for _ in 0..budget {
        // The window is the running sequence — the budget keeps it within the
        // model's context. The provider yields a session whose compiled window
        // is at least this long.
        let cur_len = sequence.len();
        debug_assert!(
            cur_len <= max_window,
            "the remaining-context budget must keep the sequence within the model's window"
        );
        let window = sequence.as_slice();

        let runner = provider.session_for(cur_len)?;
        let lm = LmContract::resolve(runner)?;
        debug_assert!(
            lm.seq_len >= cur_len,
            "provider must serve a window >= request"
        );

        // Build every graph input: the token window at `input_ids`, and each
        // recognized auxiliary (attention_mask = 1s, position_ids = 0..cur_len)
        // synthesized at its named port. The window is padded to the session's
        // compiled `seq_len`; a causal LM's logits at the last real position
        // attend only to real positions, so trailing padding never affects them.
        let win: Vec<f64> = window.iter().map(|&t| t as f64).collect();
        let mut bufs: Vec<Vec<u8>> = (0..lm.n_inputs).map(|_| Vec::new()).collect();
        bufs[lm.ids_index] = encode_vals(&win, lm.seq_len, lm.id_dtype);
        for a in &lm.aux {
            bufs[a.index] = match a.kind {
                // Empty past (no external KV-cache) → zero-length buffer.
                AuxKind::EmptyPast => Vec::new(),
                AuxKind::AttentionMask => {
                    encode_vals(&vec![1.0; cur_len], a.element_count, a.dtype)
                }
                AuxKind::PositionIds => {
                    let vals: Vec<f64> = (0..cur_len).map(|p| p as f64).collect();
                    encode_vals(&vals, a.element_count, a.dtype)
                }
                AuxKind::LastPos => encode_vals(&[(cur_len - 1) as f64], a.element_count, a.dtype),
            };
        }
        let refs: Vec<&[u8]> = bufs.iter().map(|b| b.as_slice()).collect();
        let outputs = runner.execute(&refs).context("forward pass failed")?;
        let logits: &[f32] = bytemuck::cast_slice(&outputs[lm.logits_index].bytes);

        // Next-token distribution: the single gathered row, or the row at
        // the last real position of a whole-window head.
        let pos = if lm.single_position { 0 } else { cur_len - 1 };
        let row = &logits[pos * lm.vocab..(pos + 1) * lm.vocab];
        // Position-indexed sampling (see `sample_at`): the draw is a function of
        // the absolute position, not the running RNG stream.
        let next = sample_at(row, cfg.temperature, cfg.top_k, cfg.seed, cur_len as u64);

        if next == eos {
            break;
        }
        generated.push(next);
        sequence.push(next);

        // Stream the newly stable text; a character still split across
        // tokens stays held until complete (never partial garbage).
        let delta = streamer.feed(next);
        if !delta.is_empty() {
            out.write_all(delta.as_bytes()).ok();
            out.flush().ok();
            acc.push_str(&delta);
        }

        // Stop strings: halt once the decoded text contains one.
        if stops.hit(&delta, streamer.pending()) {
            break;
        }
    }

    finish_stream(streamer, &mut acc, out);
    debug_assert_eq!(
        acc,
        tokenizer.decode(&generated),
        "streamed deltas must accumulate to the one-shot decode"
    );
    Ok(acc)
}

/// [`generate_stream`] over the decode-step plan (row `decode-plan`): the
/// prompt prefills as decode steps (each token one single-position pass) and
/// every generated token costs exactly one step — never a window-sized
/// forward. Sampling, streaming, and stop conditions are the same code as the
/// whole-window loop; the plans differ only in how a logit row is produced.
///
/// Cross-turn K/V retention: the session's carried rows are keyed by their
/// realized tokens, so a prompt sharing a prefix with the session's realized
/// sequence (a chat transcript extends its own history) rewinds to the
/// common prefix and prefills ONLY the suffix — a warm turn's cost is its
/// novel tokens, never the transcript.
pub fn generate_stream_decode<S: LmSession>(
    session: &mut DecodeSession<S>,
    tokenizer: &dyn Tokenizer,
    prompt: &str,
    cfg: &GenConfig,
    out: &mut dyn Write,
) -> Result<String> {
    let eos = cfg.eos.unwrap_or_else(|| tokenizer.eos_token_id());
    let max_window = session.context_len() as usize;

    let prompt_tokens = tokenizer.encode(prompt);
    if prompt_tokens.is_empty() {
        bail!("prompt encoded to zero tokens");
    }
    if prompt_tokens.len() > max_window {
        bail!(
            "prompt is {} tokens but the model's context length is {}; the model cannot attend \
             to a prompt longer than its trained context",
            prompt_tokens.len(),
            max_window
        );
    }

    // Rewind to the longest common prefix of the prompt and the session's
    // realized sequence — capped at prompt_len - 1 so the last prompt token
    // always re-steps (its logit row feeds the sampler).
    let common = prompt_tokens
        .iter()
        .zip(session.realized_tokens())
        .take_while(|(&p, &r)| p as i64 == r)
        .count()
        .min(prompt_tokens.len() - 1);
    session.rewind_to(common);

    // Prefill: feed the prompt's novel suffix — chunked when a seeder is
    // installed (row `chunked-prefill`: ceil(suffix/chunk) passes amortize
    // the weight stream), one step per token otherwise. Only the last row
    // feeds the sampler (causality makes the earlier rows byproducts).
    let suffix: Vec<i64> = prompt_tokens[common..].iter().map(|&t| t as i64).collect();
    let mut row = session.feed(&suffix).context("decode prefill failed")?;

    let remaining = max_window - prompt_tokens.len();
    let budget = cfg.max_tokens.map_or(remaining, |n| n.min(remaining));
    let mut generated: Vec<u32> = Vec::new();
    // Incremental detokenization — same O(N)-total delta streaming as
    // [`generate_stream`] (the O(N²) full re-decode per token is gone).
    let mut streamer = StreamingDecoder::new(tokenizer);
    let mut stops = StopScan::new(&cfg.stop);
    let mut acc = String::new();

    for step in 0..budget {
        // Position-indexed sampling: the token at this absolute position is a
        // pure function of (logits, seed, position) — the same law speculative
        // decode samples by, so the two paths agree token-for-token.
        let next = sample_at(
            &row,
            cfg.temperature,
            cfg.top_k,
            cfg.seed,
            session.realized_len() as u64,
        );
        if next == eos {
            break;
        }
        generated.push(next);

        let delta = streamer.feed(next);
        if !delta.is_empty() {
            out.write_all(delta.as_bytes()).ok();
            out.flush().ok();
            acc.push_str(&delta);
        }
        if stops.hit(&delta, streamer.pending()) {
            break;
        }

        // The final sampled token needs no step after it — a step only
        // produces the NEXT row.
        if step + 1 < budget {
            row = session.step(next as i64).context("decode step failed")?;
        }
    }

    finish_stream(streamer, &mut acc, out);
    debug_assert_eq!(
        acc,
        tokenizer.decode(&generated),
        "streamed deltas must accumulate to the one-shot decode"
    );
    Ok(acc)
}

/// Generation over the decode plan with **speculative decode** (row
/// `speculative-decode`): each iteration asks `drafter` for the next tokens,
/// verifies them in ONE `M = K` pass on `verify_runner`, accepts the longest
/// prefix the model itself would produce under the sampler (its K/V spliced from
/// that pass), and commits one correcting bonus token. Byte-identical to
/// [`generate_stream_decode`] at the same temperature and seed — a batched
/// shortcut over the same path — while a well-predicted stretch commits several
/// tokens per two passes. `drafter` is the draft SOURCE — a zero-weight
/// [`prompt-lookup`](crate::speculative::PromptLookupDrafter) or a small
/// [`draft model`](crate::speculative::ModelDrafter); it changes only the
/// acceptance rate, never the output. `max_draft` must not exceed
/// `verify_runner`'s chunk.
#[allow(clippy::too_many_arguments)]
pub fn generate_stream_speculative<S: LmSession>(
    session: &mut DecodeSession<S>,
    verify_runner: &mut S,
    tokenizer: &dyn Tokenizer,
    prompt: &str,
    cfg: &GenConfig,
    drafter: &mut dyn crate::speculative::Drafter,
    max_draft: usize,
    out: &mut dyn Write,
) -> Result<String> {
    let eos = cfg.eos.unwrap_or_else(|| tokenizer.eos_token_id());
    let max_window = session.context_len() as usize;
    // The token rule, shared by the draft-verify acceptance and the plain-step
    // fallback: greedy argmax, or a per-position sample. Because it is a pure
    // function of (logits, absolute position), speculative and plain decode
    // draw the identical token at every position — the batched shortcut is
    // byte-identical at ANY temperature, not just greedy.
    let mut next_token = |logits: &[f32], pos: u64| {
        sample_at(logits, cfg.temperature, cfg.top_k, cfg.seed, pos) as i64
    };

    let prompt_tokens = tokenizer.encode(prompt);
    if prompt_tokens.is_empty() {
        bail!("prompt encoded to zero tokens");
    }
    if prompt_tokens.len() > max_window {
        bail!(
            "prompt is {} tokens but the model's context length is {}",
            prompt_tokens.len(),
            max_window
        );
    }

    // Prefill exactly as the plain decode: rewind to the shared prefix, feed the
    // novel suffix; only the last row feeds the first draft's acceptance.
    let common = prompt_tokens
        .iter()
        .zip(session.realized_tokens())
        .take_while(|(&p, &r)| p as i64 == r)
        .count()
        .min(prompt_tokens.len() - 1);
    session.rewind_to(common);
    let suffix: Vec<i64> = prompt_tokens[common..].iter().map(|&t| t as i64).collect();
    let mut row = session.feed(&suffix).context("decode prefill failed")?;
    // Prime the drafter to the same prompt (a draft model prefills its own K/V;
    // prompt-lookup is a no-op) so its proposals share the target's context.
    let prompt_i64: Vec<i64> = prompt_tokens.iter().map(|&t| t as i64).collect();
    drafter
        .prefill(&prompt_i64)
        .context("drafter prefill failed")?;

    let remaining_window = max_window - prompt_tokens.len();
    let budget = cfg
        .max_tokens
        .map_or(remaining_window, |n| n.min(remaining_window));
    let draft_cap = max_draft.max(1);

    let mut generated: Vec<u32> = Vec::new();
    // Incremental detokenization — same O(N)-total delta streaming as
    // [`generate_stream`] (the O(N²) full re-decode per token is gone).
    let mut streamer = StreamingDecoder::new(tokenizer);
    let mut stops = StopScan::new(&cfg.stop);
    let mut acc = String::new();

    // Emit one token: stream its text delta, honor stop strings and eos, and
    // report whether to end. Shared by the drafted and plain paths so both stop
    // on exactly the same conditions.
    let mut emit = |next: i64, generated: &mut Vec<u32>, acc: &mut String| -> bool {
        if next == eos as i64 {
            return true;
        }
        generated.push(next as u32);
        let delta = streamer.feed(next as u32);
        if !delta.is_empty() {
            out.write_all(delta.as_bytes()).ok();
            out.flush().ok();
            acc.push_str(&delta);
        }
        if stops.hit(&delta, streamer.pending()) {
            return true;
        }
        generated.len() >= budget
    };

    // Speculation is a projection of speed, never of meaning: it retires to
    // plain decode — never worse — the moment it would leave the current bucket
    // (proactively, below) or a verify pass fails. Once retired it stays retired
    // for the turn (the verify runner is not rebuilt at a wider bucket).
    let mut speculate = true;
    while generated.len() < budget {
        let cap = draft_cap.min(budget - generated.len());
        // Speculation must stay STRICTLY within the current bucket. `draft_verify`
        // splices the accepted prefix into FIXED bucket rows `pos..pos+accepted`
        // (it never grows), so a batch that reached the bucket would overflow the
        // K/V buffer; and the bonus step must not regrow the bucket while the
        // verify runner — a full resident plan — is co-resident with the step
        // runner's rebuild of the wider bucket (the wasm 4 GiB over-commit). Both
        // are avoided by retiring the moment the next batch (≤ cap accepted + 1
        // bonus) would reach the bucket: a growth would stale the verify runner
        // anyway, so retire PROACTIVELY and free it here, letting the plain steps
        // below regrow the bucket safely. The output is unchanged — the target's
        // own tokens, byte for byte — only the acceptance boundary moves.
        if speculate && session.realized_len() + cap + 1 > session.geometry().bucket {
            verify_runner.evict_resident();
            speculate = false;
        }
        let draft = if speculate {
            drafter.propose(session.realized_tokens(), cap)?
        } else {
            Vec::new()
        };
        match draft.is_empty() {
            true => {
                // No recurrence (or speculation retired): a plain step under the
                // SAME token rule the verify path uses, at this absolute position.
                let next = next_token(&row, session.realized_len() as u64);
                if emit(next, &mut generated, &mut acc) {
                    break;
                }
                row = session.step(next).context("decode step failed")?;
            }
            false => match session.draft_verify(verify_runner, &row, &draft, &mut next_token) {
                Ok((accepted, bonus)) => {
                    let n_accepted = accepted.len();
                    let mut stop = false;
                    for t in accepted.into_iter().chain(std::iter::once(bonus)) {
                        if emit(t, &mut generated, &mut acc) {
                            stop = true;
                            break;
                        }
                    }
                    if stop {
                        break;
                    }
                    // Commit the bonus's K/V and get the next acceptance row, then
                    // sync the drafter to the same committed sequence.
                    row = session.step(bonus).context("decode step failed")?;
                    drafter
                        .commit(n_accepted, bonus)
                        .context("drafter commit failed")?;
                }
                Err(e) => {
                    // A genuine verify FAILURE — the bucket-growth staleness is
                    // handled proactively above, so this is an execution error or
                    // a real misconfiguration. Retire to plain decode (never
                    // worse) and surface it rather than swallow a silent slowdown.
                    tracing::warn!("speculative verify failed, decoding plainly: {e:#}");
                    speculate = false;
                    // Free the retired verify runner's resident stages — never
                    // used again this turn (no-op on an unbounded budget).
                    verify_runner.evict_resident();
                }
            },
        }
    }
    finish_stream(streamer, &mut acc, out);
    debug_assert_eq!(
        acc,
        tokenizer.decode(&generated),
        "streamed deltas must accumulate to the one-shot decode"
    );
    Ok(acc)
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

    #[test]
    fn top_k_selection_matches_full_sort_byte_for_byte() {
        // The O(vocab) select_nth top-k path must draw the IDENTICAL token to a
        // full O(vocab·log vocab) sort-then-truncate, across k and seed. If the
        // partition kept a different set or order, the inverse-CDF would land on a
        // different token — this fails-without the byte-identity it claims.
        fn full_sort_reference(logits: &[f32], temperature: f32, k: usize, rng: &mut u64) -> u32 {
            let mut idx: Vec<usize> = (0..logits.len()).collect();
            idx.sort_unstable_by(|&a, &b| {
                logits[b]
                    .partial_cmp(&logits[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let keep = &idx[..k.clamp(1, idx.len())];
            let max = keep
                .iter()
                .map(|&i| logits[i] / temperature)
                .fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f64> = keep
                .iter()
                .map(|&i| ((logits[i] / temperature - max) as f64).exp())
                .collect();
            let sum: f64 = exps.iter().sum();
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
        // Distinct sinusoidal logits (no exact ties at any k-boundary).
        let logits: Vec<f32> = (0..3000).map(|i| (i as f32 * 0.013).sin() * 4.0).collect();
        for k in [1usize, 5, 40, 128, 1000, 2999] {
            for seed in [1u64, 7, 42, 12345, 99999] {
                let mut a = seed;
                let mut b = seed;
                assert_eq!(
                    sample(&logits, 0.8, Some(k), &mut a),
                    full_sort_reference(&logits, 0.8, k, &mut b),
                    "top_k={k} seed={seed}"
                );
            }
        }
    }
}
