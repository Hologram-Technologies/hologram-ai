//! Conformance V&V — class **TK** (tokenization vs. reference tokenizer).
//!
//! hologram-ai must produce token IDs identical to the model's published
//! reference tokenizer; otherwise generation is correct math on the wrong
//! token sequence. We validate against the **HuggingFace `tokenizers`**
//! crate, which is the canonical Rust port of `tokenizer.json`.
//!
//! Both classes:
//!
//! * **TK-1.** For a curated, representative corpus, hologram-ai's encode
//!   produces the exact token-ID sequence the HF reference produces.
//! * **TK-2.** Decode(encode(x)) == x byte-for-byte on the round-trippable
//!   subset (Unicode-clean text — round-tripping arbitrary bytes through any
//!   tokenizer can vary by special-token handling).
//!
//! Gated on `HOLOGRAM_AI_LIVE=1` + the SmolLM2 tokenizer being present
//! locally (`models/smollm2-135m/tokenizer.json` or
//! `HOLOGRAM_AI_SMOLLM2_TOKENIZER`). The reference path uses HF tokenizers
//! directly — no Python, no network.

#![cfg(feature = "conformance")]

use std::path::PathBuf;

use hologram_ai_tokenizer::{NativeTokenizer, Tokenizer};
use tokenizers::Tokenizer as HfTokenizer;

/// Locate the SmolLM2 tokenizer or skip the test. Honors
/// `HOLOGRAM_AI_SMOLLM2_TOKENIZER` (an explicit path) before falling back to
/// `<workspace>/models/smollm2-135m/tokenizer.json`.
fn locate_tokenizer() -> Option<PathBuf> {
    if let Ok(env_path) = std::env::var("HOLOGRAM_AI_SMOLLM2_TOKENIZER") {
        let p = PathBuf::from(env_path);
        return p.exists().then_some(p);
    }
    // `CARGO_MANIFEST_DIR` is hologram-ai-conformance — climb up to workspace.
    let mut ws = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    ws.pop(); // crates/
    ws.pop(); // <workspace>
    let p = ws.join("models/smollm2-135m/tokenizer.json");
    p.exists().then_some(p)
}

fn live_enabled() -> bool {
    std::env::var("HOLOGRAM_AI_LIVE").is_ok()
}

/// The **strict** corpus: hologram-ai's encode must match the HF reference
/// token-for-token on every entry. This is the load-bearing TK-1 surface —
/// everyday generation inputs (English text, punctuation, Unicode, code,
/// URLs). If anything here regresses the file fails loud.
const STRICT_CORPUS: &[&str] = &[
    "Hello, world!",
    "The capital of France is Paris.",
    "The sun rises in the east.",
    "Once upon a time, there was a curious bunny.",
    "It's a beautiful day, isn't it?",
    "Numbers: 0, 1, 42, 3.14, -7, 1e-9.",
    "Punctuation? Semicolons; colons: dashes-and-underscores_.",
    "Mixed case: BeginNing, MiddleEnD, ENDing.",
    "Café résumé naïve coöperate piñata", // unicode/diacritics
    "新年快乐",                           // CJK
    "हिन्दी",                              // Devanagari
    "🌅 sun rises in the 🌄",             // emoji
    "Multiple\nnewlines\nin\nthe\ntext.",
    "Tabs\tand\tspaces.",
    "Repeated repeated repeated words words words.",
    "URL: https://example.com/path?q=1&v=2#frag",
    "Code: `let x: u32 = 0;` and `fn main() { }`",
];

/// The **edge** corpus: inputs with known divergence from the HF reference,
/// tracked for follow-up. Currently: doubled leading whitespace — HF
/// pre-tokenizes the leading space sequence into a different prefix token
/// than hologram-ai-tokenizer's BPE. Recorded as a soft-warn so a future
/// fix in `hologram-ai-tokenizer::bpe` lands with this list as the witness.
const EDGE_CORPUS_KNOWN_DIVERGENT: &[&str] = &["  leading and trailing whitespace  "];

fn load_native(path: &std::path::Path) -> NativeTokenizer {
    NativeTokenizer::from_tokenizer_json(path).expect("load native tokenizer")
}

fn load_hf(path: &std::path::Path) -> HfTokenizer {
    HfTokenizer::from_file(path).expect("load HF tokenizer")
}

#[test]
fn tk_1_encode_matches_hf_reference() {
    if !live_enabled() {
        eprintln!("SKIP TK-1: set HOLOGRAM_AI_LIVE=1 to run");
        return;
    }
    let Some(path) = locate_tokenizer() else {
        eprintln!("SKIP TK-1: tokenizer.json not found (set HOLOGRAM_AI_SMOLLM2_TOKENIZER or copy to models/smollm2-135m/)");
        return;
    };
    let native = load_native(&path);
    let hf = load_hf(&path);

    let encode_one = |text: &str| -> (Vec<u32>, Vec<u32>) {
        let got = native.encode(text);
        let enc = hf.encode(text, false).expect("HF encode");
        let want: Vec<u32> = enc.get_ids().to_vec();
        let trimmed: Vec<u32> = if native.bos_token_id() == got.first().copied() {
            got[1..].to_vec()
        } else {
            got
        };
        (trimmed, want)
    };

    // Strict corpus: a mismatch fails the test.
    let mut strict_mismatches = Vec::<(String, Vec<u32>, Vec<u32>)>::new();
    for &text in STRICT_CORPUS {
        let (got, want) = encode_one(text);
        if got != want {
            strict_mismatches.push((text.to_string(), got, want));
        }
    }
    if !strict_mismatches.is_empty() {
        for (text, got, want) in &strict_mismatches {
            eprintln!("TK-1 STRICT mismatch on {text:?}:");
            eprintln!("  got:  {got:?}");
            eprintln!("  want: {want:?}");
        }
        panic!(
            "TK-1: {}/{} strict-corpus strings disagree with the HF reference",
            strict_mismatches.len(),
            STRICT_CORPUS.len()
        );
    }

    // Edge corpus: a recorded gap; assert it is still present (i.e. the
    // documented behaviour). If it suddenly matches HF, the divergence was
    // fixed — move the entry to STRICT_CORPUS and update CONFORMANCE.md.
    for &text in EDGE_CORPUS_KNOWN_DIVERGENT {
        let (got, want) = encode_one(text);
        if got == want {
            panic!("TK-1: edge case {text:?} now matches HF — promote it to STRICT_CORPUS");
        }
        eprintln!("TK-1 known divergence on {text:?} (got {got:?} vs HF {want:?})");
    }
}

#[test]
fn tk_2_round_trip_decode_equals_input() {
    if !live_enabled() {
        eprintln!("SKIP TK-2: set HOLOGRAM_AI_LIVE=1 to run");
        return;
    }
    let Some(path) = locate_tokenizer() else {
        eprintln!("SKIP TK-2: tokenizer.json not found");
        return;
    };
    let native = load_native(&path);

    // The "round-trippable" subset: text whose decode-of-encode loses no
    // characters. Whitespace handling and special-token folding can both
    // affect this; we exclude leading/trailing whitespace and tab/newline-
    // heavy inputs from the round-trip oracle (still encoded above for TK-1).
    let trippable: Vec<&str> = STRICT_CORPUS
        .iter()
        .copied()
        .filter(|t| !t.starts_with(' ') && !t.ends_with(' '))
        .filter(|t| !t.contains('\n') && !t.contains('\t'))
        .collect();

    let mut bad = Vec::<(String, String)>::new();
    for &text in &trippable {
        let ids = native.encode(text);
        let back = native.decode(&ids);
        // SentencePiece / BPE round-trip usually preserves text up to leading-
        // space normalisation. Compare trimmed of a single leading space.
        let normalized_back = back.strip_prefix(' ').unwrap_or(&back).to_string();
        let normalized_text = text.strip_prefix(' ').unwrap_or(text).to_string();
        if normalized_back != normalized_text {
            bad.push((text.to_string(), back));
        }
    }
    if !bad.is_empty() {
        for (text, got) in &bad {
            eprintln!("TK-2 round-trip failed:");
            eprintln!("  input: {text:?}");
            eprintln!("  back:  {got:?}");
        }
        panic!(
            "TK-2: {}/{} round-trippable strings failed",
            bad.len(),
            trippable.len()
        );
    }
}
