//! Witnesses for the streaming detokenizer laws (`StreamingDecoder`):
//!
//! 1. **Byte identity** — the concatenation of every `feed` delta plus
//!    `finish` is byte-for-byte `Tokenizer::decode(all_ids)`, across every
//!    decode path this crate ships (SentencePiece/Metaspace with its per-call
//!    leading-space strip, byte-level BPE with its lossy byte recovery, plain
//!    piece-join) and across the repo's real tokenizer fixtures.
//! 2. **No partial garbage** — a character split across several tokens
//!    (byte-fallback `<0xNN>` pieces, byte-level pieces) never streams as a
//!    U+FFFD fragment; it is held until complete.
//! 3. **O(1) amortized work** — the pending window never grows with the
//!    sequence position; total decode work over N tokens is O(N), where the
//!    replaced whole-sequence re-decode was O(N²).

use std::sync::atomic::{AtomicUsize, Ordering};

use hologram_ai_tokenizer::{
    MergeRules, NativeTokenizer, NormalizationConfig, PreTokenizerConfig, SpecialTokens,
    StreamingDecoder, Tokenizer, TokenizerAlgorithm, TokenizerConfig, VocabTable,
};

// ── Harness ──────────────────────────────────────────────────────────────────

/// Stream `ids` one at a time; return (accumulated text, every delta emitted).
fn stream(tok: &dyn Tokenizer, ids: &[u32]) -> (String, Vec<String>) {
    let mut decoder = StreamingDecoder::new(tok);
    let mut acc = String::new();
    let mut deltas = Vec::new();
    for &id in ids {
        let delta = decoder.feed(id);
        // The stop-scan invariant the generation loops rely on: at every
        // point the emitted text plus the pending tail IS the whole decode.
        acc.push_str(&delta);
        assert_eq!(
            format!("{acc}{}", decoder.pending()),
            tok.decode(&ids[..deltas.len() + 1]),
            "emitted + pending must equal the whole decode at every step"
        );
        deltas.push(delta);
    }
    let rest = decoder.finish();
    acc.push_str(&rest);
    deltas.push(rest);
    (acc, deltas)
}

fn assert_identity(tok: &dyn Tokenizer, ids: &[u32], label: &str) {
    let (acc, _) = stream(tok, ids);
    assert_eq!(
        acc,
        tok.decode(ids),
        "{label}: streamed deltas must accumulate to decode(all_ids), ids={ids:?}"
    );
}

/// Texts that stress UTF-8 boundaries: CJK, emoji (incl. ZWJ sequences),
/// combining marks, whitespace runs, code.
fn corpus() -> Vec<&'static str> {
    vec![
        "Hello world",
        "多语言分词器测试",
        "naïve café résumé",
        "🦀🚀 rust to the moon 🌕",
        "family: 👨\u{200d}👩\u{200d}👧\u{200d}👦 end",
        "combining: e\u{301}\u{327} a\u{30a}",
        "  leading, double  and trailing spaces  ",
        "fn main() {\n    println!(\"hi\\n\");\n}",
        "mixed 混合 🎉 text",
    ]
}

fn special_tokens(bos: Option<u32>, eos: u32) -> SpecialTokens {
    SpecialTokens {
        bos_id: bos,
        eos_id: eos,
        pad_id: None,
        unk_id: None,
        additional: Default::default(),
    }
}

// ── Synthetic SentencePiece-style tokenizer (Metaspace + byte fallback) ─────
//
// Vocab: ▁, every printable ASCII char, and all 256 <0xNN> byte tokens — so
// ASCII encodes char-level and anything else byte-falls-back, splitting every
// multibyte character across several tokens (the case the hold logic exists
// for). decode() for this family replaces ▁ with a space and strips ONE
// leading space PER CALL — the segment-vs-whole concatenation hazard.
fn sentencepiece_tok() -> NativeTokenizer {
    let mut tokens: Vec<Vec<u8>> = vec![b"<s>".to_vec(), b"</s>".to_vec()];
    tokens.push("\u{2581}".as_bytes().to_vec());
    for c in 0x20u8..0x7f {
        tokens.push(vec![c]);
    }
    for b in 0u16..256 {
        tokens.push(format!("<0x{b:02X}>").into_bytes());
    }
    NativeTokenizer::from_config(TokenizerConfig {
        algorithm: TokenizerAlgorithm::Bpe {
            vocab: VocabTable::new(tokens),
            merges: MergeRules { merges: vec![] },
        },
        special_tokens: special_tokens(Some(0), 1),
        normalization: NormalizationConfig::None,
        pre_tokenizer: PreTokenizerConfig::Metaspace {
            replacement: '\u{2581}',
            prepend: true,
        },
        byte_fallback: true,
        add_bos: false,
        add_eos: false,
    })
}

#[test]
fn streamed_deltas_accumulate_to_whole_decode_sentencepiece_byte_fallback() {
    let tok = sentencepiece_tok();
    for text in corpus() {
        let ids = tok.encode(text);
        assert!(!ids.is_empty(), "premise: {text:?} must encode");
        assert_identity(&tok, &ids, "sentencepiece");
    }
}

#[test]
fn a_character_split_across_byte_fallback_tokens_never_streams_partially() {
    let tok = sentencepiece_tok();
    // Premise of the whole hold mechanism: the crab emoji is NOT in the vocab,
    // so it byte-falls-back to one token per UTF-8 byte — 4 tokens, 1 char.
    let ids = tok.encode("🦀");
    assert!(
        ids.len() >= 4,
        "premise: a rare emoji must split across byte-fallback tokens, got {ids:?}"
    );
    let (acc, deltas) = stream(&tok, &ids);
    assert_eq!(acc, tok.decode(&ids), "byte identity");
    for d in &deltas {
        assert!(
            !d.contains('\u{FFFD}'),
            "a partial character must be held, never streamed as U+FFFD: {deltas:?}"
        );
    }
    // The character arrives whole, in one delta.
    assert!(
        deltas.iter().any(|d| d.contains('🦀')),
        "the completed character must eventually stream: {deltas:?}"
    );
}

#[test]
fn adversarial_id_streams_match_whole_decode() {
    let tok = sentencepiece_tok();
    let id_of = |s: &str| tok.token_to_id(s).unwrap();
    let meta = id_of("\u{2581}");
    let (bos, eos) = (0u32, 1u32);
    let byte = |b: u8| id_of(&format!("<0x{b:02X}>"));

    let streams: Vec<Vec<u32>> = vec![
        // Lone metaspace tokens — the per-call leading-space strip eats the
        // first one of every decode call; only the overlap diff gets it right.
        vec![meta, meta, meta, id_of("x"), meta, id_of("y")],
        // bos/eos are filtered by decode wherever they appear — including as
        // the whole overlap-adjacent tail.
        vec![id_of("a"), bos, meta, id_of("b"), eos, id_of("c")],
        // Invalid byte followed by ASCII: the U+FFFD is final, not pending.
        vec![byte(0xE2), id_of("A")],
        // A literal U+FFFD character (EF BF BD) — valid text that merely
        // LOOKS like the lossy marker.
        vec![byte(0xEF), byte(0xBF), byte(0xBD), id_of("k")],
        // Incomplete character at end-of-stream: finish() must flush it
        // exactly as the one-shot decode renders it.
        vec![id_of("q"), byte(0xE2), byte(0x96)],
        // Consecutive multibyte characters back to back.
        "🦀🦀🦀"
            .chars()
            .flat_map(|_| [0xF0, 0x9F, 0xA6, 0x80])
            .map(byte)
            .collect(),
    ];
    for ids in streams {
        assert_identity(&tok, &ids, "adversarial");
    }
}

// ── Synthetic byte-level BPE (GPT-2 / Qwen style) ────────────────────────────

/// The GPT-2 byte→unicode map (printable ranges map to themselves, the rest
/// to U+0100+n) — mirrors `bpe.rs`, which keeps it private.
fn gpt2_byte_char(b: u8) -> char {
    let mut n = 0u32;
    for x in 0u16..=255 {
        match x as u32 {
            33..=126 | 161..=172 | 174..=255 => {
                if x as u8 == b {
                    return char::from_u32(x as u32).unwrap();
                }
            }
            _ => {
                if x as u8 == b {
                    return char::from_u32(256 + n).unwrap();
                }
                n += 1;
            }
        }
    }
    unreachable!()
}

fn byte_level_tok() -> NativeTokenizer {
    let tokens: Vec<Vec<u8>> = (0u16..256)
        .map(|b| gpt2_byte_char(b as u8).to_string().into_bytes())
        .collect();
    NativeTokenizer::from_config(TokenizerConfig {
        algorithm: TokenizerAlgorithm::Bpe {
            vocab: VocabTable::new(tokens),
            merges: MergeRules { merges: vec![] },
        },
        special_tokens: special_tokens(None, u32::MAX),
        normalization: NormalizationConfig::None,
        pre_tokenizer: PreTokenizerConfig::ByteLevel { regex: None },
        byte_fallback: false,
        add_bos: false,
        add_eos: false,
    })
}

#[test]
fn streamed_deltas_accumulate_to_whole_decode_byte_level_bpe() {
    let tok = byte_level_tok();
    for text in corpus() {
        let ids = tok.encode(text);
        assert!(!ids.is_empty(), "premise: {text:?} must encode");
        // Premise: every non-ASCII character spans several one-byte tokens.
        if !text.is_ascii() {
            assert!(ids.len() > text.chars().count());
        }
        assert_identity(&tok, &ids, "byte-level");
    }
    // Raw byte soup (deterministic sweep): arbitrary — including invalid —
    // UTF-8 must still match the one-shot lossy decode.
    let ids: Vec<u32> = (0..512u32)
        .map(|i| (i.wrapping_mul(97) + 13) % 256)
        .collect();
    assert_identity(&tok, &ids, "byte-level raw sweep");
}

// ── Plain piece-join decoder (Unigram backend) ───────────────────────────────

#[test]
fn streamed_deltas_accumulate_to_whole_decode_join_decoder() {
    let pieces: Vec<Vec<u8>> = ["Hel", "lo", ", ", "wor", "ld", "!", " ", "🦀", "多语"]
        .iter()
        .map(|s| s.as_bytes().to_vec())
        .collect();
    let n = pieces.len() as u32;
    let tok = NativeTokenizer::from_config(TokenizerConfig {
        algorithm: TokenizerAlgorithm::Unigram {
            scores: vec![0.0; pieces.len()],
            vocab: VocabTable::new(pieces),
        },
        special_tokens: special_tokens(None, u32::MAX),
        normalization: NormalizationConfig::None,
        pre_tokenizer: PreTokenizerConfig::None,
        byte_fallback: false,
        add_bos: false,
        add_eos: false,
    });
    let ids: Vec<u32> = (0..64u32).map(|i| (i * 5 + 2) % n).collect();
    assert_identity(&tok, &ids, "join");
}

// ── Real fixture tokenizers ──────────────────────────────────────────────────

#[test]
fn fixture_tokenizers_stream_byte_identical() {
    let root = {
        let mut p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        p.pop();
        p.pop();
        p
    };
    // The committed oracle fixture always runs; the model-dir tokenizers run
    // when present (same skip convention as `native.rs` tests).
    let fixtures = [
        ("oracle", "oracles/fixture/tokenizer.json", true),
        (
            "qwen2.5",
            "models/Qwen2.5-0.5B-Instruct/tokenizer.json",
            false,
        ),
        (
            "smollm2-instruct",
            "models/SmolLM2-135M-Instruct/tokenizer.json",
            false,
        ),
        ("smollm2", "models/smollm2-135m/tokenizer.json", false),
    ];
    for (label, rel, required) in fixtures {
        let path = root.join(rel);
        if !path.exists() {
            assert!(!required, "committed fixture missing: {rel}");
            eprintln!("skipping {label}: {rel} not found");
            continue;
        }
        let tok = NativeTokenizer::from_tokenizer_json(&path).unwrap();
        for text in corpus() {
            let ids = tok.encode(text);
            if ids.is_empty() {
                continue;
            }
            assert_identity(&tok, &ids, label);
        }
        // Deterministic raw-id sweep across the vocab — specials, byte
        // pieces, and unknown-locale tokens included.
        let vocab = tok.vocab_size() as u32;
        let ids: Vec<u32> = (0..300u32)
            .map(|i| (i.wrapping_mul(7919) + 31) % vocab)
            .collect();
        assert_identity(&tok, &ids, label);
    }
}

// ── Boundedness: O(1) amortized per token, O(N) total ────────────────────────

/// Counts decode work (token ids passed to `decode`) — the observable that
/// separates O(N) streaming from the O(N²) whole-sequence re-decode.
struct CountingTok<'a> {
    inner: &'a dyn Tokenizer,
    work: AtomicUsize,
    max_call: AtomicUsize,
}

impl Tokenizer for CountingTok<'_> {
    fn encode(&self, text: &str) -> Vec<u32> {
        self.inner.encode(text)
    }
    fn decode(&self, tokens: &[u32]) -> String {
        self.work.fetch_add(tokens.len(), Ordering::Relaxed);
        self.max_call.fetch_max(tokens.len(), Ordering::Relaxed);
        self.inner.decode(tokens)
    }
    fn eos_token_id(&self) -> u32 {
        self.inner.eos_token_id()
    }
    fn bos_token_id(&self) -> Option<u32> {
        self.inner.bos_token_id()
    }
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }
    fn id_to_token(&self, id: u32) -> Option<&str> {
        self.inner.id_to_token(id)
    }
    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }
}

#[test]
fn pending_window_and_total_work_stay_o_n_over_10k_tokens() {
    let inner = sentencepiece_tok();
    // A realistic mixed stream (words, CJK, emoji) cycled to 10k tokens.
    let unit = inner.encode("The quick 棕色 fox 🦊 jumps over the lazy 狗 dog. ");
    let n = 10_000usize;
    let ids: Vec<u32> = unit.iter().copied().cycle().take(n).collect();

    let tok = CountingTok {
        inner: &inner,
        work: AtomicUsize::new(0),
        max_call: AtomicUsize::new(0),
    };
    let mut decoder = StreamingDecoder::new(&tok);
    let mut acc = String::new();
    for &id in &ids {
        acc.push_str(&decoder.feed(id));
    }
    acc.push_str(&decoder.finish());

    let work = tok.work.load(Ordering::Relaxed);
    let max_call = tok.max_call.load(Ordering::Relaxed);
    // O(N) with a generous constant: the replaced whole-sequence re-decode
    // does ~N²/2 = 50,000,000 here — three orders of magnitude over this
    // bound. A regression to O(position) work cannot pass.
    assert!(
        work <= 8 * n + 64,
        "total decode work must be O(N): {work} token-decodes over {n} tokens"
    );
    // The pending window is bounded by how long a character stays incomplete,
    // never by the sequence position.
    assert!(
        max_call <= 32,
        "the decode window must never grow with the sequence: max {max_call}"
    );
    // And the identity law still holds over the long run.
    assert_eq!(acc, inner.decode(&ids), "byte identity over 10k tokens");
}
