//! Witnesses for the generation loop's incremental detokenization:
//!
//! 1. The loop streams DELTAS — each sink write is exactly the next slice of
//!    the returned text (a cumulative-snapshot regression would re-write
//!    already-emitted text and fail), and their concatenation IS the returned
//!    text, byte for byte.
//! 2. The loop's total tokenizer-decode work over a T-token generation is
//!    O(T) — the replaced per-token whole-sequence re-decode was O(T²) and
//!    exceeds the asserted bound by an order of magnitude.
//! 3. A stop string spanning a delta boundary still halts generation — the
//!    rolling stop scan keeps exactly the tail a whole-text scan could newly
//!    match in.
//!
//! Harness: the synthetic successor LM from `generation_synthetic.rs` (greedy
//! next token from `a` is always `a+1`) — the real compile+execute path, no
//! downloads.

use std::collections::HashMap;
use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};

use hologram_ai::commands::generate::{generate_stream, GenConfig};
use hologram_ai::{FixedSession, HoloRunner, ModelCompiler, ModelSource};
use hologram_ai_common::{shape_from_concrete, AiGraph, AiNode, AiOp, AiParam, DType, TensorInfo};
use hologram_ai_tokenizer::Tokenizer;

/// Mock tokenizer over base-10 integers: "5 6" ↔ `[5, 6]`.
struct IntTok {
    vocab: usize,
    eos: u32,
}

impl Tokenizer for IntTok {
    fn encode(&self, text: &str) -> Vec<u32> {
        text.split_whitespace()
            .filter_map(|w| w.parse().ok())
            .collect()
    }
    fn decode(&self, tokens: &[u32]) -> String {
        tokens
            .iter()
            .map(|t| t.to_string())
            .collect::<Vec<_>>()
            .join(" ")
    }
    fn eos_token_id(&self) -> u32 {
        self.eos
    }
    fn bos_token_id(&self) -> Option<u32> {
        None
    }
    fn vocab_size(&self) -> usize {
        self.vocab
    }
    fn id_to_token(&self, _id: u32) -> Option<&str> {
        None
    }
    fn token_to_id(&self, _token: &str) -> Option<u32> {
        None
    }
}

/// Counts decode work (token ids passed to `decode`) — the observable that
/// separates the O(T) loop from the old O(T²) whole-sequence re-decode.
struct CountingTok {
    inner: IntTok,
    work: AtomicUsize,
}

impl Tokenizer for CountingTok {
    fn encode(&self, text: &str) -> Vec<u32> {
        self.inner.encode(text)
    }
    fn decode(&self, tokens: &[u32]) -> String {
        self.work.fetch_add(tokens.len(), Ordering::Relaxed);
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

/// A sink that records every individual write.
#[derive(Default)]
struct RecordingSink {
    writes: Vec<Vec<u8>>,
}

impl Write for RecordingSink {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.writes.push(buf.to_vec());
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// `Gather(W[V,V], input_ids[1,S], axis=0) → logits[1,S,V]` with
/// `W[t, (t+1) mod V] = 1` — the "successor" LM (see `generation_synthetic.rs`).
fn successor_lm(seq_len: usize, vocab: usize) -> AiGraph {
    let ids = 0u32;
    let w = 1u32;
    let logits = 2u32;

    let mut tensor_info: HashMap<u32, TensorInfo> = HashMap::new();
    tensor_info.insert(
        ids,
        TensorInfo::new(DType::INT64, shape_from_concrete(&[1, seq_len as u64])),
    );
    tensor_info.insert(
        w,
        TensorInfo::new(
            DType::F32,
            shape_from_concrete(&[vocab as u64, vocab as u64]),
        ),
    );
    tensor_info.insert(
        logits,
        TensorInfo::new(
            DType::F32,
            shape_from_concrete(&[1, seq_len as u64, vocab as u64]),
        ),
    );

    let mut w_bytes = vec![0u8; vocab * vocab * 4];
    for t in 0..vocab {
        let col = (t + 1) % vocab;
        let off = (t * vocab + col) * 4;
        w_bytes[off..off + 4].copy_from_slice(&1.0f32.to_le_bytes());
    }

    let mut params = HashMap::new();
    params.insert(w, AiParam::inline(w_bytes, tensor_info[&w].clone()));

    AiGraph {
        name: "successor_lm".into(),
        nodes: vec![AiNode::new(
            0,
            AiOp::Gather { axis: 0 },
            vec![w, ids],
            vec![logits],
        )],
        inputs: vec![ids],
        outputs: vec![logits],
        input_names: vec!["input_ids".into()],
        output_names: vec!["logits".into()],
        params,
        tensor_info,
        metadata: HashMap::new(),
        warnings: Vec::new(),
        dim_vars: Default::default(),
        shape_constraints: Default::default(),
        subgraphs: HashMap::new(),
        tensor_names: HashMap::new(),
        topo_cache: Default::default(),
    }
}

fn compile_runner(seq_len: usize, vocab: usize) -> FixedSession {
    let archive = ModelCompiler::default()
        .compile(ModelSource::AiGraph(successor_lm(seq_len, vocab)))
        .expect("compile synthetic LM");
    let runner = HoloRunner::from_bytes(archive.bytes).expect("load synthetic LM");
    FixedSession::new(runner)
}

#[test]
fn every_write_is_the_next_slice_of_the_returned_text_never_a_snapshot() {
    let (seq_len, vocab) = (32usize, 200usize);
    let mut runner = compile_runner(seq_len, vocab);
    let tok = IntTok { vocab, eos: 999 };

    let cfg = GenConfig {
        max_tokens: Some(20),
        temperature: 0.0,
        ..Default::default()
    };
    let mut sink = RecordingSink::default();
    let out = generate_stream(&mut runner, &tok, "100", &cfg, &mut sink).unwrap();
    assert_eq!(out, tok.decode(&(101..=120).collect::<Vec<_>>()));

    // Concatenation of the writes IS the returned text…
    let streamed: Vec<u8> = sink.writes.iter().flatten().copied().collect();
    assert_eq!(
        String::from_utf8(streamed).unwrap(),
        out,
        "streamed writes must concatenate to the returned text"
    );
    // …and each write is exactly the NEXT slice: a cumulative snapshot would
    // re-write already-emitted text and break the offset walk.
    let mut off = 0usize;
    for w in &sink.writes {
        assert_eq!(
            &out.as_bytes()[off..off + w.len()],
            &w[..],
            "write at byte {off} is not the next delta"
        );
        off += w.len();
    }
    assert_eq!(off, out.len());
}

#[test]
fn loop_decode_work_is_o_t_where_the_full_redecode_was_o_t_squared() {
    let (seq_len, vocab) = (256usize, 300usize);
    let mut runner = compile_runner(seq_len, vocab);
    let t = 200usize;
    let tok = CountingTok {
        inner: IntTok { vocab, eos: 9999 },
        work: AtomicUsize::new(0),
    };

    let cfg = GenConfig {
        max_tokens: Some(t),
        temperature: 0.0,
        ..Default::default()
    };
    let mut sink = Vec::new();
    let out = generate_stream(&mut runner, &tok, "1", &cfg, &mut sink).unwrap();
    assert_eq!(String::from_utf8(sink).unwrap(), out);

    // Streaming decode + one whole-sequence check at the end is ≤ ~4·T; the
    // old per-token `decode(&generated)` alone is T·(T+1)/2 = 20,100 — an
    // order of magnitude over this bound. O(T²) cannot come back unseen.
    let work = tok.work.load(Ordering::Relaxed);
    assert!(
        work <= 8 * t + 64,
        "generation loop decode work must be O(T): {work} token-decodes for T={t}"
    );
}

#[test]
fn stop_string_spanning_a_delta_boundary_still_halts() {
    let (seq_len, vocab) = (16usize, 32usize);
    let mut runner = compile_runner(seq_len, vocab);
    let tok = IntTok { vocab, eos: 99 };

    // "7 8" only exists ACROSS two deltas ("7" then " 8") — a scan of each
    // delta alone would never see it; halting proves the rolling tail works.
    let cfg = GenConfig {
        max_tokens: Some(10),
        temperature: 0.0,
        stop: vec!["7 8".into()],
        ..Default::default()
    };
    let mut sink = Vec::new();
    let out = generate_stream(&mut runner, &tok, "5", &cfg, &mut sink).unwrap();
    assert_eq!(out, "6 7 8", "generation must halt on the spanning stop");
    assert_eq!(String::from_utf8(sink).unwrap(), out);
}
