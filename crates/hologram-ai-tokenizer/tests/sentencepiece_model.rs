//! Fails-without witnesses for the SentencePiece `tokenizer.model` loader
//! (content-sniffed by `NativeTokenizer::from_tokenizer_json_bytes`) and for
//! the loud `.tiktoken` refusal. Every ModelProto fixture is written
//! byte-by-byte with a minimal protobuf field writer, so each law is pinned
//! to the wire format itself — no fixture files, no fitting to a model.

use hologram_ai_tokenizer::{NativeTokenizer, Tokenizer};

// ── minimal protobuf field writer ───────────────────────────────────────────

fn varint(mut v: u64, out: &mut Vec<u8>) {
    loop {
        let b = (v & 0x7F) as u8;
        v >>= 7;
        if v == 0 {
            out.push(b);
            break;
        }
        out.push(b | 0x80);
    }
}

/// Wire type 0 (varint) field.
fn varint_field(field: u32, v: u64, out: &mut Vec<u8>) {
    varint(u64::from(field) << 3, out);
    varint(v, out);
}

/// Wire type 2 (length-delimited) field.
fn bytes_field(field: u32, data: &[u8], out: &mut Vec<u8>) {
    varint((u64::from(field) << 3) | 2, out);
    varint(data.len() as u64, out);
    out.extend_from_slice(data);
}

/// Wire type 5 (fixed32) field.
fn float_field(field: u32, v: f32, out: &mut Vec<u8>) {
    varint((u64::from(field) << 3) | 5, out);
    out.extend_from_slice(&v.to_le_bytes());
}

// ── ModelProto builders (sentencepiece_model.proto field numbers) ───────────

// SentencePiece.Type values.
const NORMAL: u64 = 1;
const UNKNOWN: u64 = 2;
const CONTROL: u64 = 3;
const BYTE: u64 = 6;

// TrainerSpec.ModelType values.
const UNIGRAM: u64 = 1;
const BPE: u64 = 2;

/// SentencePiece message: piece = 1 (string), score = 2 (float), type = 3.
fn piece(text: &str, score: f32, ty: u64) -> Vec<u8> {
    let mut p = Vec::new();
    bytes_field(1, text.as_bytes(), &mut p);
    float_field(2, score, &mut p);
    varint_field(3, ty, &mut p);
    p
}

/// TrainerSpec message: model_type = 3, byte_fallback = 35,
/// unk_id/bos_id/eos_id = 40/41/42 (fixed here at 0/1/2 like Llama-family).
fn trainer(model_type: u64, byte_fallback: bool) -> Vec<u8> {
    let mut t = Vec::new();
    varint_field(3, model_type, &mut t);
    if byte_fallback {
        varint_field(35, 1, &mut t);
    }
    varint_field(40, 0, &mut t);
    varint_field(41, 1, &mut t);
    varint_field(42, 2, &mut t);
    t
}

/// NormalizerSpec message: name = 1, precompiled_charsmap = 2,
/// add_dummy_prefix = 3, remove_extra_whitespaces = 4, escape_whitespaces = 5.
fn normalizer(
    add_dummy_prefix: bool,
    remove_extra_whitespaces: bool,
    escape_whitespaces: bool,
    charsmap: Option<&[u8]>,
) -> Vec<u8> {
    let mut n = Vec::new();
    bytes_field(1, b"identity", &mut n);
    if let Some(cm) = charsmap {
        bytes_field(2, cm, &mut n);
    }
    varint_field(3, u64::from(add_dummy_prefix), &mut n);
    varint_field(4, u64::from(remove_extra_whitespaces), &mut n);
    varint_field(5, u64::from(escape_whitespaces), &mut n);
    n
}

/// ModelProto message: pieces = 1 (repeated), trainer_spec = 2,
/// normalizer_spec = 3 (`None` = absent, so proto2 defaults apply: dummy
/// prefix ON, whitespace collapse ON, ▁-escaping ON).
fn model_proto(pieces: &[Vec<u8>], trainer_spec: &[u8], normalizer_spec: Option<&[u8]>) -> Vec<u8> {
    let mut out = Vec::new();
    for p in pieces {
        bytes_field(1, p, &mut out);
    }
    bytes_field(2, trainer_spec, &mut out);
    if let Some(n) = normalizer_spec {
        bytes_field(3, n, &mut out);
    }
    out
}

/// Ids 0/1/2 — the trainer's unk/bos/eos.
fn specials() -> Vec<Vec<u8>> {
    vec![
        piece("<unk>", 0.0, UNKNOWN),
        piece("<s>", 0.0, CONTROL),
        piece("</s>", 0.0, CONTROL),
    ]
}

/// `expect_err` without requiring `Debug` on `NativeTokenizer`.
fn load_err(bytes: &[u8], why_must_fail: &str) -> String {
    match NativeTokenizer::from_tokenizer_json_bytes(bytes) {
        Ok(_) => panic!("{why_must_fail}"),
        Err(e) => e.to_string(),
    }
}

// ── witnesses ────────────────────────────────────────────────────────────────

/// Law: a UNIGRAM tokenizer.model segments by Viterbi (maximum total score,
/// not greedy-longest and not file order) and decode inverts it exactly.
///
/// Hand-computed: "hello world" normalizes to "▁hello▁world";
/// ▁hell(-0.5) + o(-0.5) = -1.0 beats the longer ▁hello at -3.0, then
/// ▁world(-1.0). Greedy-longest (or score-blind) segmentation would pick
/// ▁hello and fail this test.
#[test]
fn unigram_viterbi_picks_max_score_segmentation_and_decode_inverts() {
    let mut pieces = specials();
    pieces.push(piece("▁hello", -3.0, NORMAL)); // 3
    pieces.push(piece("▁hell", -0.5, NORMAL)); // 4
    pieces.push(piece("o", -0.5, NORMAL)); // 5
    pieces.push(piece("▁world", -1.0, NORMAL)); // 6
    let proto = model_proto(&pieces, &trainer(UNIGRAM, false), None);

    let tok = NativeTokenizer::from_tokenizer_json_bytes(&proto).expect("unigram model loads");
    assert_eq!(tok.bos_token_id(), Some(1), "bos from trainer_spec.bos_id");
    assert_eq!(tok.eos_token_id(), 2, "eos from trainer_spec.eos_id");
    let ids = tok.encode("hello world");
    assert_eq!(ids, vec![1, 4, 5, 6], "bos + ▁hell + o + ▁world");
    assert_eq!(tok.decode(&ids), "hello world");
}

/// Law: SP-BPE merges the adjacent pair whose CONCATENATION has the highest
/// piece score — never the pair whose merged piece merely comes first in the
/// file. "abc" with ab(id 6, score -3) and bc(id 7, score -1): bc merges
/// first, leaving [a, bc]; a file-order / id-order merge would give [ab, c].
#[test]
fn sp_bpe_merges_by_score_priority_not_file_order() {
    let mut pieces = specials();
    pieces.push(piece("a", -10.0, NORMAL)); // 3
    pieces.push(piece("b", -10.0, NORMAL)); // 4
    pieces.push(piece("c", -10.0, NORMAL)); // 5
    pieces.push(piece("ab", -3.0, NORMAL)); // 6
    pieces.push(piece("bc", -1.0, NORMAL)); // 7
    let norm = normalizer(false, false, true, None);
    let proto = model_proto(&pieces, &trainer(BPE, false), Some(&norm));

    let tok = NativeTokenizer::from_tokenizer_json_bytes(&proto).expect("sp-bpe model loads");
    let ids = tok.encode("abc");
    assert_eq!(
        ids,
        vec![1, 3, 7],
        "bos + a + bc: the -1 merge must win over the -3 merge"
    );
    assert_eq!(tok.decode(&ids), "abc");
}

/// Law: add_dummy_prefix + escape_whitespaces produce ▁-prefixed pieces;
/// the same vocab with add_dummy_prefix switched off must not prepend.
#[test]
fn dummy_prefix_and_whitespace_escaping_follow_normalizer_flags() {
    let mut pieces = specials();
    pieces.push(piece("▁hello", -1.0, NORMAL)); // 3
    pieces.push(piece("▁world", -1.0, NORMAL)); // 4
    pieces.push(piece("hello", -1.5, NORMAL)); // 5
    pieces.push(piece("▁", -5.0, NORMAL)); // 6

    // normalizer_spec absent → proto2 default add_dummy_prefix = true.
    let with_prefix = model_proto(&pieces, &trainer(UNIGRAM, false), None);
    let tok = NativeTokenizer::from_tokenizer_json_bytes(&with_prefix).unwrap();
    assert_eq!(
        tok.encode("hello world"),
        vec![1, 3, 4],
        "▁hello ▁world: the dummy prefix makes the first word ▁-formed"
    );

    let norm = normalizer(false, true, true, None);
    let no_prefix = model_proto(&pieces, &trainer(UNIGRAM, false), Some(&norm));
    let tok = NativeTokenizer::from_tokenizer_json_bytes(&no_prefix).unwrap();
    assert_eq!(
        tok.encode("hello world"),
        vec![1, 5, 4],
        "hello ▁world: no dummy prefix on the first word"
    );
}

/// Law: remove_extra_whitespaces (proto2 default ON) trims and collapses
/// runs of spaces before segmentation; switched off, every space survives
/// as its own ▁.
#[test]
fn remove_extra_whitespaces_collapses_runs_when_set() {
    let mut pieces = specials();
    pieces.push(piece("▁hello", -1.0, NORMAL)); // 3
    pieces.push(piece("▁world", -1.0, NORMAL)); // 4
    pieces.push(piece("▁", -5.0, NORMAL)); // 5

    let collapsing = model_proto(&pieces, &trainer(UNIGRAM, false), None);
    let tok = NativeTokenizer::from_tokenizer_json_bytes(&collapsing).unwrap();
    assert_eq!(
        tok.encode("  hello   world  "),
        vec![1, 3, 4],
        "extra whitespace trims + collapses away"
    );

    let norm = normalizer(true, false, true, None);
    let preserving = model_proto(&pieces, &trainer(UNIGRAM, false), Some(&norm));
    let tok = NativeTokenizer::from_tokenizer_json_bytes(&preserving).unwrap();
    assert_eq!(
        tok.encode("  hello   world  "),
        vec![1, 5, 5, 3, 5, 5, 4, 5, 5],
        "▁▁▁hello▁▁▁world▁▁: with the flag off every space survives"
    );
}

/// Law: byte_fallback decomposes a character with no piece into its UTF-8
/// `<0xNN>` BYTE pieces, and decode reassembles the original bytes.
#[test]
fn byte_fallback_decomposes_unknown_char_and_decode_reassembles() {
    let mut pieces = specials();
    pieces.push(piece("▁", -1.0, NORMAL)); // 3
    pieces.push(piece("<0xC3>", 0.0, BYTE)); // 4
    pieces.push(piece("<0xA9>", 0.0, BYTE)); // 5
    let proto = model_proto(&pieces, &trainer(UNIGRAM, true), None);

    let tok = NativeTokenizer::from_tokenizer_json_bytes(&proto).unwrap();
    // 'é' = UTF-8 C3 A9, not in the vocab → its two byte pieces.
    let ids = tok.encode("é");
    assert_eq!(ids, vec![1, 3, 4, 5], "bos + ▁ + <0xC3> + <0xA9>");
    assert_eq!(tok.decode(&ids), "é", "byte pieces reassemble to UTF-8");
}

/// Law: a non-empty precompiled_charsmap is refused BY NAME — applying it
/// needs the Darts trie, and silently skipping it would tokenize wrong.
#[test]
fn precompiled_charsmap_is_refused_by_name() {
    let pieces = specials();
    let norm = normalizer(true, true, true, Some(&[1, 2, 3, 4]));
    let proto = model_proto(&pieces, &trainer(UNIGRAM, false), Some(&norm));

    let err = load_err(&proto, "charsmap must refuse, not silently skip");
    assert!(
        err.contains("precompiled_charsmap"),
        "refusal must name the field: {err}"
    );
}

/// Law: SentencePiece model types with no runtime encoder (WORD, CHAR) are
/// refused naming the type, never silently mapped onto a different encoder.
#[test]
fn word_and_char_model_types_refused_by_name() {
    for (model_type, name) in [(3u64, "WORD"), (4u64, "CHAR")] {
        let proto = model_proto(&specials(), &trainer(model_type, false), None);
        let err = load_err(&proto, "unsupported model_type must refuse");
        assert!(
            err.contains(name),
            "refusal must name model_type {name}: {err}"
        );
    }
}

/// Law: `.tiktoken` rank data is refused loudly, naming the format and the
/// missing pre-tokenization regex — the file underdetermines the tokenizer.
#[test]
fn tiktoken_ranks_refused_naming_format_and_missing_regex() {
    let err = load_err(b"IQ== 0\nIg== 1\nIw== 2\n", "tiktoken ranks must refuse");
    assert!(err.contains(".tiktoken"), "refusal names the format: {err}");
    assert!(
        err.contains("pre-tokenization regex"),
        "refusal names the missing information: {err}"
    );
}

/// Law: malformed protobuf is a loud error, never a panic and never a
/// half-parsed tokenizer.
#[test]
fn malformed_model_proto_errors_loud_not_panic() {
    let mut pieces = specials();
    pieces.push(piece("▁hello", -1.0, NORMAL));
    let good = model_proto(&pieces, &trainer(UNIGRAM, false), None);

    // Cut inside the trailing trainer_spec: overruns the input.
    assert!(NativeTokenizer::from_tokenizer_json_bytes(&good[..good.len() - 1]).is_err());
    // Field key promising bytes that never come.
    assert!(NativeTokenizer::from_tokenizer_json_bytes(&[0x0A]).is_err());
    // Not protobuf at all.
    assert!(NativeTokenizer::from_tokenizer_json_bytes(&[0xFF; 16]).is_err());
    assert!(NativeTokenizer::from_tokenizer_json_bytes(&[]).is_err());
}

/// Law: JSON bytes (optionally BOM/whitespace-prefixed) still route to the
/// tokenizer.json path after content sniffing.
#[test]
fn json_bytes_still_route_to_tokenizer_json_path() {
    let json = r#"{
        "model": {
            "type": "BPE",
            "vocab": {"h": 0, "e": 1, "l": 2, "o": 3, "ll": 4},
            "merges": ["l l"]
        }
    }"#;
    let tok = NativeTokenizer::from_tokenizer_json_bytes(json.as_bytes()).unwrap();
    assert_eq!(tok.encode("hello"), vec![0, 1, 4, 3]);

    let mut prefixed = vec![0xEF, 0xBB, 0xBF];
    prefixed.extend_from_slice(b" \n\t");
    prefixed.extend_from_slice(json.as_bytes());
    let tok = NativeTokenizer::from_tokenizer_json_bytes(&prefixed).unwrap();
    assert_eq!(tok.encode("hello"), vec![0, 1, 4, 3]);
}

/// Law: a repo that ships only `tokenizer.model` loads through the
/// conventional `tokenizer.json` path (the loader reads the SentencePiece
/// sibling); with neither file present the error names both.
#[test]
fn tokenizer_model_sibling_loads_when_tokenizer_json_absent() {
    let mut pieces = specials();
    pieces.push(piece("▁hi", -1.0, NORMAL)); // 3
    let proto = model_proto(&pieces, &trainer(UNIGRAM, false), None);

    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("tokenizer.model"), &proto).unwrap();
    let tok = NativeTokenizer::from_tokenizer_json(&dir.path().join("tokenizer.json"))
        .expect("the .model sibling must load when the .json is absent");
    assert_eq!(tok.encode("hi"), vec![1, 3]);

    let empty = tempfile::tempdir().unwrap();
    let err = match NativeTokenizer::from_tokenizer_json(&empty.path().join("tokenizer.json")) {
        Ok(_) => panic!("neither file present must fail"),
        Err(e) => e.to_string(),
    };
    assert!(
        err.contains("tokenizer.model"),
        "error names the sibling too: {err}"
    );
}
