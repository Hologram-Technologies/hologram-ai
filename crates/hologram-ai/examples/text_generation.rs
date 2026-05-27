//! Text generation, end-to-end, with **verifiable** output.
//!
//! Run: `cargo run -p hologram-ai --example text_generation`
//!
//! This example drives the *real* hologram-ai pipeline —
//! `ModelCompiler` → `.holo` → `HoloRunner` → `generate_stream` — exactly the
//! path the `run --prompt` CLI uses, but on a tiny, fully-deterministic model so
//! the output can be checked against a closed-form expectation (no downloads, no
//! tokenizer files, no GPU).
//!
//! The model is a "successor LM": an embedding `Gather` whose weight row `t` has
//! its single maximum at column `(t+1) mod V`. Greedy decoding from a token `a`
//! must therefore emit `a+1, a+2, …` — a verifiable stand-in for a language
//! model that exercises the entire generation loop (encode → forward → argmax →
//! detokenize → stop) through the canonical compile + execute path.
//!
//! For real pretrained LLMs the same code applies, but the model must be a
//! **no-past** forward export (`input_ids[1,S] → logits[1,S,V]`): hologram-ai
//! replaces the mutable KV-cache with content-addressed κ-label elision, so it
//! recomputes the growing token window each step and reuses the unchanged prefix
//! by content address. With-past *decode-step* exports (input_ids[1,1] + past
//! KV) do not fit this loop — see `examples/tinyllama.toml`.

use std::collections::HashMap;

use hologram_ai::commands::generate::{generate_stream, GenConfig};
use hologram_ai::{HoloRunner, ModelCompiler, ModelSource};
use hologram_ai_common::{shape_from_concrete, AiGraph, AiNode, AiOp, AiParam, DType, TensorInfo};
use hologram_ai_tokenizer::Tokenizer;

/// Whitespace-integer tokenizer: `"150 7"` ↔ `[150, 7]`. Stands in for a real
/// BPE tokenizer so the example is self-contained and the vocab is transparent.
struct IntTokenizer {
    vocab: usize,
    eos: u32,
}

impl Tokenizer for IntTokenizer {
    fn encode(&self, text: &str) -> Vec<u32> {
        text.split_whitespace().filter_map(|w| w.parse().ok()).collect()
    }
    fn decode(&self, tokens: &[u32]) -> String {
        tokens.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(" ")
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

/// `Gather(W[V,V], input_ids[1,S], axis=0) → logits[1,S,V]` with
/// `W[t, (t+1) mod V] = 1`. argmax over each position is `(token + 1) mod V`.
fn successor_lm(seq_len: usize, vocab: usize) -> AiGraph {
    let (ids, w, logits) = (0u32, 1u32, 2u32);
    let mut tensor_info: HashMap<u32, TensorInfo> = HashMap::new();
    tensor_info.insert(ids, TensorInfo::new(DType::INT64, shape_from_concrete(&[1, seq_len as u64])));
    tensor_info.insert(w, TensorInfo::new(DType::F32, shape_from_concrete(&[vocab as u64, vocab as u64])));
    tensor_info.insert(
        logits,
        TensorInfo::new(DType::F32, shape_from_concrete(&[1, seq_len as u64, vocab as u64])),
    );

    let mut w_bytes = vec![0u8; vocab * vocab * 4];
    for t in 0..vocab {
        let off = (t * vocab + (t + 1) % vocab) * 4;
        w_bytes[off..off + 4].copy_from_slice(&1.0f32.to_le_bytes());
    }
    let mut params = HashMap::new();
    params.insert(w, AiParam::inline(w_bytes, tensor_info[&w].clone()));

    AiGraph {
        name: "successor_lm".into(),
        nodes: vec![AiNode::new(0, AiOp::Gather { axis: 0 }, vec![w, ids], vec![logits])],
        inputs: vec![ids],
        outputs: vec![logits],
        // Named ports — generation binds `input_ids`/`logits` by name, exactly
        // as a real ONNX import does.
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

fn main() -> anyhow::Result<()> {
    let (seq_len, vocab) = (16usize, 256usize);

    // ── Compile: AiGraph → canonical hologram graph → .holo archive ──────────
    let archive = ModelCompiler::default().compile(ModelSource::AiGraph(successor_lm(seq_len, vocab)))?;
    println!(
        "compiled successor-LM → {} nodes, {} archive bytes",
        archive.stats.node_count,
        archive.bytes.len()
    );

    // ── Load + generate (greedy, temperature 0) ──────────────────────────────
    let mut runner = HoloRunner::from_bytes(archive.bytes)?;
    let tok = IntTokenizer { vocab, eos: vocab as u32 - 1 };
    let cfg = GenConfig { max_tokens: 8, temperature: 0.0, ..Default::default() };

    let prompt = "100";
    let mut sink = Vec::new();
    let out = generate_stream(&mut runner, &tok, prompt, &cfg, &mut sink)?;
    println!("prompt   : {prompt}");
    println!("generated: {out}");

    // ── Verify against the closed-form expectation ───────────────────────────
    let expected = "101 102 103 104 105 106 107 108";
    assert_eq!(out, expected, "greedy successor decode must be deterministic");
    println!("verified : output equals the closed-form successor sequence ✓");
    Ok(())
}
