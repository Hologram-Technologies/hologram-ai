//! Generation-loop conformance on a synthetic causal LM — no model downloads.
//!
//! Builds a tiny but *exactly predictable* "language model": an embedding-style
//! `Gather` whose weight row `t` has its single maximum at column `(t+1) mod V`.
//! So the model's greedy next-token is always "the next integer": from a prompt
//! token `a` it must generate `a+1, a+2, …` (mod V). That gives a closed-form
//! expected sequence to check the whole loop against — encode → forward →
//! argmax/sample → detokenize → stop — through the real compile+execute path,
//! with no tokenizer files or network.

use std::collections::HashMap;

use hologram_ai::commands::generate::{generate_stream, GenConfig};
use hologram_ai::{FixedSession, HoloRunner, ModelCompiler, ModelSource};
use hologram_ai_common::{shape_from_concrete, AiGraph, AiNode, AiOp, AiParam, DType, TensorInfo};
use hologram_ai_tokenizer::Tokenizer;

/// Mock tokenizer over base-10 integers: "5 6" ↔ `[5, 6]`. Lets the test drive
/// `generate_stream` directly with a fully deterministic vocab.
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

/// `Gather(W[V,V], input_ids[1,S], axis=0) → logits[1,S,V]`, with
/// `W[t, (t+1) mod V] = 1` (else 0). argmax over each position's logits is
/// therefore `(token + 1) mod V` — the "successor" LM.
fn successor_lm(seq_len: usize, vocab: usize) -> AiGraph {
    let ids = 0u32;
    let w = 1u32;
    let logits = 2u32;

    let mut tensor_info: HashMap<u32, TensorInfo> = HashMap::new();
    // Token ids as INT64 — the real-LM dtype. Embedding lowers to the
    // first-class `OpKind::Gather` (no one-hot, no int→float cast), so the vocab
    // is unconstrained (no i8 ≤ 127 limit).
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

    // W[t, (t+1) % V] = 1.0
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
        // Named ports — generation binds input_ids/logits by name (no positional
        // guessing), exactly as a real ONNX import does.
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
fn greedy_generation_predicts_successor_sequence() {
    // Vocab 200 (> 127) with int64 ids — exercises the real Gather embedding at
    // token values impossible under the old i8 ≤ 127 workaround.
    let (seq_len, vocab) = (8usize, 200usize);
    let mut runner = compile_runner(seq_len, vocab);
    let tok = IntTok { vocab, eos: 999 }; // eos out of range — never sampled

    let cfg = GenConfig {
        max_tokens: 5,
        temperature: 0.0,
        ..Default::default()
    };
    let mut sink = Vec::new();
    let out = generate_stream(&mut runner, &tok, "150", &cfg, &mut sink).unwrap();

    // From token 150, greedy successor LM must emit 151..155.
    assert_eq!(out, "151 152 153 154 155", "greedy decode mismatch");
    // Streamed bytes equal the final text (streamed incrementally as deltas).
    assert_eq!(String::from_utf8(sink).unwrap(), "151 152 153 154 155");
}

#[test]
fn eos_token_halts_generation() {
    let (seq_len, vocab) = (8usize, 32usize);
    let mut runner = compile_runner(seq_len, vocab);
    // eos = 8: from prompt 5 → 6,7, then 8 is eos and generation stops before it.
    let tok = IntTok { vocab, eos: 8 };

    let cfg = GenConfig {
        max_tokens: 10,
        temperature: 0.0,
        ..Default::default()
    };
    let mut sink = Vec::new();
    let out = generate_stream(&mut runner, &tok, "5", &cfg, &mut sink).unwrap();
    assert_eq!(out, "6 7", "eos must halt before emitting the eos token");
}

#[test]
fn stop_string_halts_generation() {
    let (seq_len, vocab) = (8usize, 32usize);
    let mut runner = compile_runner(seq_len, vocab);
    let tok = IntTok { vocab, eos: 99 };

    // Stop once the decoded suffix contains "9": 5 → 6,7,8,9 then halt.
    let cfg = GenConfig {
        max_tokens: 20,
        temperature: 0.0,
        stop: vec!["9".into()],
        ..Default::default()
    };
    let mut sink = Vec::new();
    let out = generate_stream(&mut runner, &tok, "5", &cfg, &mut sink).unwrap();
    assert_eq!(out, "6 7 8 9", "generation must stop at the stop string");
}

#[test]
fn temperature_sampling_with_top_k_one_is_deterministic_successor() {
    let (seq_len, vocab) = (8usize, 32usize);
    let mut runner = compile_runner(seq_len, vocab);
    let tok = IntTok { vocab, eos: 99 };

    // top_k=1 collapses any temperature to the argmax → same successor sequence.
    let cfg = GenConfig {
        max_tokens: 4,
        temperature: 1.5,
        top_k: Some(1),
        seed: 12345,
        ..Default::default()
    };
    let mut sink = Vec::new();
    let out = generate_stream(&mut runner, &tok, "10", &cfg, &mut sink).unwrap();
    assert_eq!(out, "11 12 13 14");
}

#[test]
fn prompt_equal_to_context_slides_not_errors() {
    // A prompt exactly as long as the window is valid: predict from the last
    // position, then the window slides (the model's genuine finite context).
    let (seq_len, vocab) = (4usize, 16usize);
    let mut runner = compile_runner(seq_len, vocab);
    let tok = IntTok { vocab, eos: 99 };
    let cfg = GenConfig {
        max_tokens: 3,
        temperature: 0.0,
        ..Default::default()
    };
    let mut sink = Vec::new();
    // 4 prompt tokens == window. Successor LM: from "…4" → 5, then window slides
    // to [2,3,4,5] → 6, → 7. No error, deterministic successor sequence.
    let out = generate_stream(&mut runner, &tok, "1 2 3 4", &cfg, &mut sink).unwrap();
    assert_eq!(
        out, "5 6 7",
        "prompt == window must slide and predict successors"
    );
}

#[test]
fn rejects_prompt_longer_than_context() {
    let (seq_len, vocab) = (4usize, 16usize);
    let mut runner = compile_runner(seq_len, vocab);
    let tok = IntTok { vocab, eos: 99 };
    let cfg = GenConfig::default();
    let mut sink = Vec::new();
    // 5 prompt tokens > window of 4: the model cannot attend to it → clear
    // error, no panic. (A growable session would instead recompile a larger
    // window; a fixed window cannot.)
    let err = generate_stream(&mut runner, &tok, "1 2 3 4 5", &cfg, &mut sink).unwrap_err();
    assert!(
        err.to_string().contains("context length"),
        "expected a context-length error, got: {err}"
    );
}
