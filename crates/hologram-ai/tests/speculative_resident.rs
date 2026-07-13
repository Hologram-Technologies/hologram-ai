//! Witnesses for FOLDED resident speculative decode (row `speculative-decode`,
//! ADR-0019): the pending token is committed as the first row of each verify
//! batch, so the verify runner alone executes during speculation and carries
//! the resident K/V truth across batches — no per-batch step on the session
//! runner, no per-batch sync → re-hash → commit-copy → re-ingest traversals.
//!
//! Laws witnessed here:
//! 1. BYTE-IDENTITY — a folded speculative run (accept-all, accept-partial,
//!    reject-at-once, empty-draft plain fallback, re-entry) commits exactly
//!    the tokens plain stepping commits under the same rule, and the session
//!    continues plain-stepping bit-identically afterwards (the carried truth
//!    handed back by `end_speculation` is the real state, not a stale copy).
//! 2. PROTOCOL LOUDNESS — while the verify runner carries the truth, `step`/
//!    `feed`/`verify` refuse loud naming `end_speculation`; handing the truth
//!    back through a runner that never carried it refuses loud too.
//!
//! Same deterministic fixture family as `parametric_reference.rs`.

use hologram_ai::runner::HoloRunner;
use hologram_ai::{DecodeSession, ModelCompiler, ModelSource, RopeSpec};
use hologram_ai_common::{shape_from_concrete, AiGraph, AiParam, DType, TensorInfo};
use std::collections::HashMap;

const HIDDEN: usize = 64;
const LAYERS: usize = 2;
const HEADS: usize = 4;
const KV_HEADS: usize = 2;
const HEAD_DIM: usize = 16;
const VOCAB: usize = 512;
const INTER: usize = 128;
const EPS: f32 = 1e-6;
const THETA: f64 = 10000.0;
const WINDOW: usize = 64;

fn config_json() -> serde_json::Value {
    serde_json::json!({
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": HIDDEN, "intermediate_size": INTER, "num_hidden_layers": LAYERS,
        "num_attention_heads": HEADS, "num_key_value_heads": KV_HEADS, "vocab_size": VOCAB,
        "rms_norm_eps": EPS, "rope_theta": THETA, "max_position_embeddings": WINDOW,
        "tie_word_embeddings": false, "torch_dtype": "float32",
        "bos_token_id": 1, "eos_token_id": 2, "model_type": "llama"
    })
}

fn manifest() -> Vec<(String, Vec<u64>)> {
    let (h, i, v) = (HIDDEN as u64, INTER as u64, VOCAB as u64);
    let kv = (KV_HEADS * HEAD_DIM) as u64;
    let mut m: Vec<(String, Vec<u64>)> = vec![("model.embed_tokens.weight".into(), vec![v, h])];
    for l in 0..LAYERS {
        let p = format!("model.layers.{l}");
        m.push((format!("{p}.input_layernorm.weight"), vec![h]));
        m.push((format!("{p}.self_attn.q_proj.weight"), vec![h, h]));
        m.push((format!("{p}.self_attn.k_proj.weight"), vec![kv, h]));
        m.push((format!("{p}.self_attn.v_proj.weight"), vec![kv, h]));
        m.push((format!("{p}.self_attn.o_proj.weight"), vec![h, h]));
        m.push((format!("{p}.post_attention_layernorm.weight"), vec![h]));
        m.push((format!("{p}.mlp.gate_proj.weight"), vec![i, h]));
        m.push((format!("{p}.mlp.up_proj.weight"), vec![i, h]));
        m.push((format!("{p}.mlp.down_proj.weight"), vec![h, i]));
    }
    m.push(("model.norm.weight".into(), vec![h]));
    m.push(("lm_head.weight".into(), vec![v, h]));
    m
}

fn bytes_for(name: &str, dims: &[u64]) -> Vec<u8> {
    let n: u64 = dims.iter().product();
    let norm = name.contains("layernorm") || name.ends_with(".norm.weight");
    (0..n)
        .flat_map(|k| {
            let v: f32 = if norm {
                1.0
            } else {
                ((k % 13) as f32 - 6.0) * 0.01
            };
            v.to_le_bytes()
        })
        .collect()
}

fn inject_params(graph: &mut AiGraph) {
    let mut name_to_id: HashMap<String, u32> = HashMap::new();
    for (id, name) in &graph.tensor_names {
        name_to_id.insert(name.clone(), *id);
    }
    for (name, dims) in &manifest() {
        let id = *name_to_id.get(name).expect("manifest tensor in graph");
        let info = TensorInfo::new(DType::F32, shape_from_concrete(dims));
        graph.tensor_info.insert(id, info.clone());
        graph
            .params
            .insert(id, AiParam::inline(bytes_for(name, dims), info));
    }
}

fn compile(graph: AiGraph) -> Vec<u8> {
    ModelCompiler::default()
        .compile(ModelSource::AiGraph(graph))
        .expect("compile")
        .bytes
}

fn keys_and_dtypes() -> (Vec<String>, Vec<DType>) {
    let keys: Vec<String> = manifest().into_iter().map(|(n, _)| n).collect();
    let dtypes = vec![DType::F32; keys.len()];
    (keys, dtypes)
}

fn decode_session(bucket: u64) -> DecodeSession<HoloRunner> {
    let (keys, dtypes) = keys_and_dtypes();
    let mut graph =
        hologram_ai_safetensors::parametric::build_parametric_decode_graph_from_manifest(
            &config_json(),
            &keys,
            &dtypes,
            bucket,
        )
        .expect("decode graph builds");
    inject_params(&mut graph);
    let runner = HoloRunner::from_bytes(compile(graph)).expect("decode loads");
    DecodeSession::new(runner, RopeSpec::plain(THETA as f32), WINDOW as u64)
        .expect("decode session opens")
}

fn verify_runner(bucket: u64, chunk: u64) -> HoloRunner {
    let (keys, dtypes) = keys_and_dtypes();
    let mut graph =
        hologram_ai_safetensors::parametric::build_parametric_verify_graph_from_manifest(
            &config_json(),
            &keys,
            &dtypes,
            bucket,
            chunk,
        )
        .expect("verify graph builds");
    inject_params(&mut graph);
    HoloRunner::from_bytes(compile(graph)).expect("verify archive loads")
}

/// Greedy rule shared by both drives — a pure function of the logits.
fn argmax(row: &[f32]) -> i64 {
    row.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i as i64)
        .unwrap()
}

const PROMPT: [i64; 4] = [3, 141, 59, 26];

/// Plain-stepped oracle: prompt then `n` greedy tokens, one step each.
fn plain_tokens(n: usize) -> Vec<i64> {
    let mut session = decode_session(32);
    let mut row = session.feed(&PROMPT).expect("prefill");
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let next = argmax(&row);
        out.push(next);
        row = session.step(next).expect("plain step");
    }
    out
}

#[test]
fn folded_speculation_commits_exactly_the_plain_tokens_and_hands_truth_back() {
    // The scripted regimes cover the first ~5 batches; the remaining
    // iterations run out of plans and fall back to plain steps INSIDE the
    // same protocol — so the tail also witnesses that decode continues
    // bit-identically after the truth is handed back.
    let n = 13usize;
    let oracle = plain_tokens(n);

    // Folded drive, mirroring the generation loop's protocol: pending decided
    // by the same rule, drafts scripted to exercise accept-all (the oracle's
    // own continuation), accept-partial (correct prefix then a wrong token),
    // reject-at-once (wrong first token), and the empty-draft plain fallback.
    let chunk = 4u64;
    let mut session = decode_session(32);
    let mut verify = verify_runner(32, chunk);
    let mut row = session.feed(&PROMPT).expect("prefill");
    let mut next_token = |logits: &[f32], _pos: u64| argmax(logits);

    let mut emitted: Vec<i64> = Vec::new();
    let mut pending: Option<i64> = None;
    // Scripted draft plans, consumed in order. `None` = plain fallback this
    // round; `Some(f)` builds a draft from the oracle continuation at the
    // draft's start index.
    #[allow(clippy::type_complexity)]
    let plans: Vec<Option<Box<dyn Fn(&[i64]) -> Vec<i64>>>> = vec![
        // accept-all: the oracle's own next 3 tokens.
        Some(Box::new(|tail: &[i64]| tail[..3].to_vec())),
        // accept-partial: 1 correct then a deliberately wrong token.
        Some(Box::new(|tail: &[i64]| {
            vec![tail[0], (tail[1] + 1) % VOCAB as i64]
        })),
        // reject-at-once: a wrong first token.
        Some(Box::new(|tail: &[i64]| vec![(tail[0] + 1) % VOCAB as i64])),
        // empty: plain fallback (regime switch out of resident speculation).
        None,
        // re-entry: accept-all again after the plain interlude.
        Some(Box::new(|tail: &[i64]| tail[..2.min(tail.len())].to_vec())),
    ];

    let mut plan_iter = plans.into_iter();
    while emitted.len() < n {
        if pending.is_none() {
            let next = next_token(&row, 0);
            emitted.push(next);
            if emitted.len() >= n {
                break;
            }
            pending = Some(next);
        }
        let plan = plan_iter.next().unwrap_or(None);
        match plan {
            Some(build) => {
                let p = pending.take().expect("decided above");
                // The oracle continuation AFTER the tokens emitted so far.
                let tail = &oracle[emitted.len()..];
                let draft = build(tail);
                let (accepted, bonus) = session
                    .speculate(&mut verify, p, &draft, &mut next_token)
                    .expect("folded batch");
                for t in accepted.iter().chain(std::iter::once(&bonus)) {
                    emitted.push(*t);
                    if emitted.len() >= n {
                        break;
                    }
                }
                if emitted.len() >= n {
                    break;
                }
                pending = Some(bonus);
            }
            None => {
                if let Some(p) = pending.take() {
                    session
                        .end_speculation(&mut verify)
                        .expect("truth hand-back");
                    row = session.step(p).expect("plain commit of pending");
                }
                let next = next_token(&row, 0);
                emitted.push(next);
                if emitted.len() >= n {
                    break;
                }
                row = session.step(next).expect("plain step");
            }
        }
    }
    session.end_speculation(&mut verify).expect("turn-end sync");

    assert_eq!(
        &emitted[..n],
        &oracle[..],
        "folded speculation must commit exactly the plain-stepped tokens"
    );
}

#[test]
fn passes_refuse_loud_while_the_verify_runner_carries_the_truth() {
    let mut session = decode_session(32);
    let mut verify = verify_runner(32, 4);
    let row = session.feed(&PROMPT).expect("prefill");
    let mut next_token = |logits: &[f32], _pos: u64| argmax(logits);

    let pending = argmax(&row);
    session
        .speculate(&mut verify, pending, &[7, 8], &mut next_token)
        .expect("folded batch");

    let err = session
        .step(1)
        .expect_err("step must refuse during speculation");
    assert!(
        err.to_string().contains("end_speculation"),
        "step names the hand-back protocol: {err}"
    );
    let err = session
        .feed(&[1, 2])
        .expect_err("feed must refuse during speculation");
    assert!(
        err.to_string().contains("end_speculation"),
        "feed names the hand-back protocol: {err}"
    );

    // A runner that never carried the truth cannot hand it back.
    let mut stranger = verify_runner(32, 4);
    let err = session
        .end_speculation(&mut stranger)
        .expect_err("a non-carrying runner must refuse");
    assert!(
        err.to_string().contains("not the runner"),
        "end_speculation names the identity failure: {err}"
    );
    // The failed hand-back poisoned the truth (the stranger gave nothing);
    // a reset heals it and plain decode resumes.
    session.reset();
    session
        .feed(&PROMPT)
        .expect("reset session decodes plainly");
}

#[test]
fn session_decode_holds_pool_allocation_flat_across_steps_and_batches() {
    // Session-level confinement (the runner-level witness lives in
    // v090_resident_kv_contract.rs): once the resident carry is warm, neither
    // plain steps nor folded speculative batches grow the runner pools — the
    // ring write moves in place and the leases are recycled, so a long
    // generation is O(bucket) resident, never O(steps).
    let mut session = decode_session(32);
    let mut verify = verify_runner(32, 4);
    let mut row = session.feed(&PROMPT).expect("prefill");
    let mut next_token = |logits: &[f32], _pos: u64| argmax(logits);

    // Warm-up: two steps let every pool reach steady state.
    for _ in 0..2 {
        let next = argmax(&row);
        row = session.step(next).expect("warm-up step");
    }
    let steady = session.runner().pool_allocated_bytes();
    for _ in 0..6 {
        let next = argmax(&row);
        row = session.step(next).expect("steady step");
        assert_eq!(
            session.runner().pool_allocated_bytes(),
            steady,
            "a plain step must not grow the step runner's pools"
        );
    }

    // Folded speculative batches confine the SAME way on the verify runner.
    // The ingest batch and the first CARRIED batch may still grow pools
    // (label-bound scratch differs from byte-bound scratch — warm-up, same as
    // the runner-level witness); the steady tail is the law.
    let pending = argmax(&row);
    session
        .speculate(&mut verify, pending, &[5, 6], &mut next_token)
        .expect("first folded batch (ingest)");
    for w in 0..2 {
        session
            .speculate(&mut verify, 5 + w, &[6 + w], &mut next_token)
            .expect("carried warm-up batch");
    }
    let vsteady = verify.pool_allocated_bytes();
    for t in 0..6 {
        session
            .speculate(&mut verify, 7 + t, &[8 + t], &mut next_token)
            .expect("carried folded batch");
        assert_eq!(
            verify.pool_allocated_bytes(),
            vsteady,
            "a carried folded batch must not grow the verify runner's pools"
        );
        assert_eq!(
            verify.leased_count(),
            2 * session.geometry().layers,
            "exactly the per-layer K/V labels stay leased between batches"
        );
    }
    session.end_speculation(&mut verify).expect("turn-end sync");
}
