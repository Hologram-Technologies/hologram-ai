//! The model-driven Gherkin runner (`[[test]] bdd`, harness = false).
//!
//! Loads the typed conceptual model (`hologram_ai_model::Model`), selects the
//! dictionary rows executed by this runner (executor == rust), and runs
//! exactly the feature files whose `@lane:` tag matches the requested lane:
//!
//! * `HOLOGRAM_AI_BDD_LANE=default` (the default) — pure/local/network
//!   witnesses; builds and runs with no cargo features.
//! * `HOLOGRAM_AI_BDD_LANE=ort` — rows needing the ONNX Runtime dylib; build
//!   with `--features conformance`.
//! * `HOLOGRAM_AI_BDD_LANE=model` — rows needing the pinned real model on
//!   disk; build with `--features conformance`.
//! * `HOLOGRAM_AI_BDD_LANE=target` — the measured probes (`tier = target`,
//!   `@target`-tagged): steps execute real probes and print measurements;
//!   wired non-gating in CI.
//!
//! The runner fails on any skipped, undefined, or failed step and on any
//! selected feature that did not run. There are no conditional skip stubs:
//! every registered step is real, and ORT-dependent step bodies compiled
//! without `--features conformance` fail loud instead of skipping.

use std::collections::{BTreeSet, HashMap};
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use cucumber::{given, then, when, World};
use hologram_ai::materialize::{
    kappa_of, kappa_requirements, materialize_archive, DirKappaStore, KappaStore,
};
use hologram_ai::runner::HoloRunner;
use hologram_ai::{ModelCompiler, ModelSource};
use hologram_ai_common::{shape_from_concrete, AiGraph, AiNode, AiOp, AiParam, DType, TensorInfo};
use hologram_ai_conformance::witness::{parse_streamed_header, split_safetensors};
use hologram_ai_core::domain::Kappa;
use hologram_ai_core::{
    reduce, AiEvent, AiView, InferenceOutput, InferenceParams, InferenceProvenance,
    InferenceRequest, ModelManifest, Prompt, RunnerKind, RunnerManifest,
};
use hologram_ai_model::{Executor, Model, Tier, UseCaseExpects};
use hologram_ai_quant::{dequant_q4_0, dequant_q8_0};
use hologram_ai_safetensors::parametric::{
    build_parametric_graph_from_manifest, selected_family, supported_families,
};
use hologram_ai_tokenizer::{NativeTokenizer, Tokenizer};
use serde::Deserialize;

// The live-HF acquisition helper (shared with its own test target). Some of
// its surface is only used by that target, hence the dead-code allowance.
#[allow(dead_code)]
#[path = "fetch_helper.rs"]
mod fetch_helper;

// ─────────────────────────── shared plumbing ────────────────────────────────

/// The loaded conceptual model, shared by every step that reads registries.
fn model() -> &'static Model {
    static MODEL: OnceLock<Model> = OnceLock::new();
    MODEL.get_or_init(|| Model::load().expect("the conceptual model must load and validate"))
}

/// The repository root.
fn root() -> PathBuf {
    hologram_ai_model::workspace_root()
}

/// A self-cleaning unique temp directory used as a κ-store.
struct StoreDir {
    path: PathBuf,
}

impl StoreDir {
    fn new(tag: &str) -> Self {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "hai-bdd-{tag}-{}-{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&path).expect("creating κ-store temp dir");
        Self { path }
    }
}

impl Drop for StoreDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

fn le_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().expect("4-byte chunk")))
        .collect()
}

fn i64s_le(vals: &[i64]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn f32s_le(vals: &[f32]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn argmax(v: &[f32]) -> usize {
    let mut best = 0;
    for (i, x) in v.iter().enumerate() {
        if *x > v[best] {
            best = i;
        }
    }
    best
}

/// Construct an `AiGraph` with the standard empty auxiliary tables.
#[allow(clippy::too_many_arguments)]
fn ai_graph(
    name: &str,
    nodes: Vec<AiNode>,
    inputs: Vec<u32>,
    outputs: Vec<u32>,
    input_names: Vec<String>,
    output_names: Vec<String>,
    params: HashMap<u32, AiParam>,
    tensor_info: HashMap<u32, TensorInfo>,
) -> AiGraph {
    AiGraph {
        name: name.into(),
        nodes,
        inputs,
        outputs,
        input_names,
        output_names,
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

fn ti(dt: DType, dims: &[u64]) -> TensorInfo {
    TensorInfo::new(dt, shape_from_concrete(dims))
}

fn compile_graph(graph: AiGraph) -> Vec<u8> {
    ModelCompiler::default()
        .compile(ModelSource::AiGraph(graph))
        .expect("compiling the in-test graph")
        .bytes
}

// ────────────────────── the matmul materialization witness ──────────────────

/// The κ-materialization witness kit: one 4×4 matmul weight, its κ, the
/// k-form archive requiring it, and the same graph compiled inline.
struct MatWitness {
    weight: Vec<u8>,
    kappa: String,
    kform: Vec<u8>,
    inline_holo: Vec<u8>,
}

/// `y[1,4] = x[1,4] · w[4,4]` with `w` supplied as `param` — the same witness
/// graph as `hologram-ai/tests/materialize_e2e.rs`.
fn matmul_graph(param: AiParam) -> AiGraph {
    let (x, w, y) = (0u32, 1u32, 2u32);
    let mut tinfo = HashMap::new();
    tinfo.insert(x, ti(DType::F32, &[1, 4]));
    tinfo.insert(w, ti(DType::F32, &[4, 4]));
    tinfo.insert(y, ti(DType::F32, &[1, 4]));
    let mut params = HashMap::new();
    params.insert(w, param);
    ai_graph(
        "bdd-materialize",
        vec![AiNode::new(0, AiOp::MatMul, vec![x, w], vec![y])],
        vec![x],
        vec![y],
        vec!["x".into()],
        vec!["y".into()],
        params,
        tinfo,
    )
}

fn mat_witness() -> MatWitness {
    let weight: Vec<u8> = f32s_le(&(0..16).map(|i| (i as f32) * 0.25 - 1.5).collect::<Vec<_>>());
    let kappa = kappa_of(&weight);
    let kform = compile_graph(matmul_graph(AiParam::External {
        kappa: kappa.clone(),
        info: ti(DType::F32, &[4, 4]),
    }));
    let inline_holo = compile_graph(matmul_graph(AiParam::inline(
        weight.clone(),
        ti(DType::F32, &[4, 4]),
    )));
    MatWitness {
        weight,
        kappa,
        kform,
        inline_holo,
    }
}

fn matmul_input() -> Vec<u8> {
    f32s_le(&[1.0, -2.0, 3.0, 0.5])
}

// ─────────────────────────── the tiny successor LM ──────────────────────────

/// A tiny compiled causal LM whose logits are a known function of the input
/// tokens: per position `i`, `logits_i = W[tok_i] + 0.25` where row `t` of `W`
/// argmaxes to `(t + 1) % vocab` — greedy decode emits the successor
/// sequence. Each position is a separate graph input, and each gather is an
/// *interior* node (the session elides interior nodes by κ-residency; output
/// ports re-fire every walk), so consecutive decode steps leave the
/// unchanged prefix cone resident — the decode-elision witness surface.
struct TinyLm {
    holo: Vec<u8>,
    seq: usize,
    vocab: usize,
    compile_time: Duration,
}

fn tiny_lm() -> TinyLm {
    let (seq, vocab) = (40u32, 8u32);
    let w_id = 0u32;
    let bias_id = 1u32;
    let mut tinfo = HashMap::new();
    tinfo.insert(w_id, ti(DType::F32, &[vocab as u64, vocab as u64]));
    tinfo.insert(bias_id, ti(DType::F32, &[1, vocab as u64]));
    let mut w_bytes = vec![0u8; (vocab * vocab) as usize * 4];
    for r in 0..vocab as usize {
        let col = (r + 1) % vocab as usize;
        let at = (r * vocab as usize + col) * 4;
        w_bytes[at..at + 4].copy_from_slice(&1.0f32.to_le_bytes());
    }
    let mut params = HashMap::new();
    params.insert(w_id, AiParam::inline(w_bytes, tinfo[&w_id].clone()));
    params.insert(
        bias_id,
        AiParam::inline(
            f32s_le(&vec![0.25f32; vocab as usize]),
            tinfo[&bias_id].clone(),
        ),
    );

    let mut nodes = Vec::new();
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();
    let mut input_names = Vec::new();
    let mut output_names = Vec::new();
    for i in 0..seq {
        let tok = 2 + i;
        let gathered = 2 + seq + i;
        let logits = 2 + 2 * seq + i;
        tinfo.insert(tok, ti(DType::INT64, &[1]));
        tinfo.insert(gathered, ti(DType::F32, &[1, vocab as u64]));
        tinfo.insert(logits, ti(DType::F32, &[1, vocab as u64]));
        nodes.push(AiNode::new(
            2 * i,
            AiOp::Gather { axis: 0 },
            vec![w_id, tok],
            vec![gathered],
        ));
        nodes.push(AiNode::new(
            2 * i + 1,
            AiOp::Add,
            vec![gathered, bias_id],
            vec![logits],
        ));
        inputs.push(tok);
        outputs.push(logits);
        input_names.push(format!("tok_{i}"));
        output_names.push(format!("logits_{i}"));
    }
    let graph = ai_graph(
        "bdd-tiny-lm",
        nodes,
        inputs,
        outputs,
        input_names,
        output_names,
        params,
        tinfo,
    );
    let started = Instant::now();
    let holo = compile_graph(graph);
    TinyLm {
        holo,
        seq: seq as usize,
        vocab: vocab as usize,
        compile_time: started.elapsed(),
    }
}

/// Run `steps` greedy decode steps over the tiny LM from `prompt`, returning
/// the generated tokens and the per-step (dispatched, skipped) counters.
fn greedy_decode(lm: &TinyLm, prompt: &[i64], steps: usize) -> (Vec<i64>, Vec<(usize, usize)>) {
    assert!(
        prompt.len() + steps <= lm.seq,
        "prompt ({}) + steps ({steps}) exceed the compiled window ({})",
        prompt.len(),
        lm.seq
    );
    let mut runner = HoloRunner::from_bytes(lm.holo.clone()).expect("loading the tiny LM");
    let mut toks = prompt.to_vec();
    let mut generated = Vec::new();
    let mut counters = Vec::new();
    for _ in 0..steps {
        let bufs: Vec<Vec<u8>> = (0..lm.seq)
            .map(|i| i64s_le(&[toks.get(i).copied().unwrap_or(0)]))
            .collect();
        let refs: Vec<&[u8]> = bufs.iter().map(Vec::as_slice).collect();
        let outs = runner.execute(&refs).expect("tiny LM decode step");
        counters.push((runner.last_dispatched(), runner.last_skipped()));
        let logits = le_f32(&outs[toks.len() - 1].bytes);
        assert_eq!(logits.len(), lm.vocab, "one logits row per position");
        let next = argmax(&logits) as i64;
        generated.push(next);
        toks.push(next);
    }
    (generated, counters)
}

// ─────────────────────────── app-domain fixtures ────────────────────────────

fn kap(label: &str) -> Kappa {
    Kappa(holospaces::address(label.as_bytes()))
}

fn request(tag: &str) -> InferenceRequest {
    InferenceRequest {
        request_kappa: kap(&format!("request-{tag}")),
        app_kappa: Some(kap("app")),
        model_kappa: kap("model-manifest"),
        runner_kappa: kap("runner-manifest"),
        prompt: Prompt {
            prompt_kappa: kap(&format!("prompt-{tag}")),
            text: format!("prompt {tag}"),
        },
        params: InferenceParams {
            params_kappa: Some(kap("params")),
            max_output_tokens: Some(16),
            temperature_milli: Some(0),
            stop_sequences: vec!["</s>".to_string()],
        },
    }
}

fn ev_registered(name: &str) -> AiEvent {
    AiEvent::ModelRegistered {
        event_kappa: kap(&format!("event-register-{name}")),
        manifest: ModelManifest {
            model_kappa: kap(&format!("model-{name}")),
            archive_kappa: kap(&format!("archive-{name}")),
            name: name.to_string(),
            description: Some("bdd fixture model".to_string()),
        },
    }
}

fn ev_submitted(tag: &str) -> AiEvent {
    AiEvent::PromptSubmitted {
        event_kappa: kap(&format!("event-submit-{tag}")),
        request: Box::new(request(tag)),
    }
}

fn ev_started(tag: &str) -> AiEvent {
    AiEvent::InferenceStarted {
        event_kappa: kap(&format!("event-start-{tag}")),
        request_kappa: kap(&format!("request-{tag}")),
        model_kappa: kap("model-manifest"),
        runner: RunnerManifest {
            runner_kappa: kap("runner-manifest"),
            name: "bdd-echo".to_string(),
            kind: RunnerKind::TestEcho,
        },
        worker_kappa: kap("worker"),
    }
}

fn ev_completed(tag: &str) -> AiEvent {
    AiEvent::InferenceCompleted {
        event_kappa: kap(&format!("event-complete-{tag}")),
        output: Box::new(InferenceOutput {
            request_kappa: kap(&format!("request-{tag}")),
            output_kappa: kap(&format!("output-{tag}")),
            content: format!("echo: prompt {tag}"),
            provenance: InferenceProvenance {
                request_kappa: kap(&format!("request-{tag}")),
                input_event_kappa: kap(&format!("event-submit-{tag}")),
                prompt_kappa: kap(&format!("prompt-{tag}")),
                model_kappa: kap("model-manifest"),
                runner_kappa: kap("runner-manifest"),
                worker_kappa: kap("worker"),
                params_kappa: Some(kap("params")),
                output_kappa: kap(&format!("output-{tag}")),
            },
        }),
    }
}

fn ev_failed(tag: &str) -> AiEvent {
    AiEvent::InferenceFailed {
        event_kappa: kap(&format!("event-fail-{tag}")),
        request_kappa: kap(&format!("request-{tag}")),
        model_kappa: kap("model-manifest"),
        runner_kappa: kap("runner-manifest"),
        worker_kappa: kap("worker"),
        error: format!("deliberate bdd failure for {tag}"),
    }
}

// ───────────────────── handshake-tiny parametric fixtures ───────────────────

/// Arbitrary-instance quantities not recorded in `usecases.toml` expects.
/// They parameterize the *tiny* instance only — nothing canonical.
const TINY_INTERMEDIATE: u64 = 128;
const TINY_CONTEXT: u64 = 64;

/// config.json for the handshake-tiny use-case, from the model registry.
fn handshake_config(family: &str, e: &UseCaseExpects, tied: bool) -> serde_json::Value {
    serde_json::json!({
        "architectures": [family],
        "hidden_size": e.hidden_size,
        "num_hidden_layers": e.num_hidden_layers,
        "num_attention_heads": e.num_attention_heads,
        "num_key_value_heads": e.num_key_value_heads,
        "vocab_size": e.vocab_size,
        "intermediate_size": TINY_INTERMEDIATE,
        "rope_theta": e.rope_theta,
        "rms_norm_eps": e.rms_norm_eps,
        "max_position_embeddings": TINY_CONTEXT,
        "tie_word_embeddings": tied,
    })
}

/// The Llama-family tensor manifest (name, shape) for the given quantities.
fn decoder_manifest(e: &UseCaseExpects, tied: bool, qkv_bias: bool) -> Vec<(String, Vec<u64>)> {
    let hidden = e.hidden_size;
    let head_dim = e.hidden_size / e.num_attention_heads;
    let q_out = e.num_attention_heads * head_dim;
    let kv_out = e.num_key_value_heads * head_dim;
    let mut m = vec![
        (
            "model.embed_tokens.weight".to_string(),
            vec![e.vocab_size, hidden],
        ),
        ("model.norm.weight".to_string(), vec![hidden]),
    ];
    if !tied {
        m.push(("lm_head.weight".to_string(), vec![e.vocab_size, hidden]));
    }
    for l in 0..e.num_hidden_layers {
        let base = format!("model.layers.{l}");
        m.push((format!("{base}.input_layernorm.weight"), vec![hidden]));
        m.push((
            format!("{base}.post_attention_layernorm.weight"),
            vec![hidden],
        ));
        m.push((
            format!("{base}.self_attn.q_proj.weight"),
            vec![q_out, hidden],
        ));
        m.push((
            format!("{base}.self_attn.k_proj.weight"),
            vec![kv_out, hidden],
        ));
        m.push((
            format!("{base}.self_attn.v_proj.weight"),
            vec![kv_out, hidden],
        ));
        m.push((
            format!("{base}.self_attn.o_proj.weight"),
            vec![hidden, q_out],
        ));
        m.push((
            format!("{base}.mlp.gate_proj.weight"),
            vec![TINY_INTERMEDIATE, hidden],
        ));
        m.push((
            format!("{base}.mlp.up_proj.weight"),
            vec![TINY_INTERMEDIATE, hidden],
        ));
        m.push((
            format!("{base}.mlp.down_proj.weight"),
            vec![hidden, TINY_INTERMEDIATE],
        ));
        if qkv_bias {
            m.push((format!("{base}.self_attn.q_proj.bias"), vec![q_out]));
            m.push((format!("{base}.self_attn.k_proj.bias"), vec![kv_out]));
            m.push((format!("{base}.self_attn.v_proj.bias"), vec![kv_out]));
        }
    }
    m
}

/// Deterministic seeded f32 weights for a named tensor: an xorshift64* stream
/// seeded from the tensor name's blake3, mapped to small values (norm weights
/// sit near 1.0 for numerical sanity). Reproducible by construction.
fn seeded_weights(name: &str, count: usize) -> Vec<f32> {
    let digest = blake3::hash(name.as_bytes());
    let mut state = u64::from_le_bytes(
        digest.as_bytes()[..8]
            .try_into()
            .expect("blake3 digest has 32 bytes"),
    ) | 1;
    let norm_like = name.contains("layernorm") || name.ends_with("norm.weight");
    (0..count)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let unit = (state >> 11) as f32 / (1u64 << 53) as f32; // [0, 1)
            let centered = unit * 2.0 - 1.0;
            if norm_like {
                1.0 + centered * 0.01
            } else {
                centered * 0.05
            }
        })
        .collect()
}

// ─────────────────────────── tokenizer corpus (TK) ──────────────────────────

/// The representative encode corpus (mirrors the strict TK-1 surface).
const TOKENIZER_CORPUS: &[&str] = &[
    "Hello, world!",
    "The capital of France is Paris.",
    "The sun rises in the east.",
    "Once upon a time, there was a curious bunny.",
    "It's a beautiful day, isn't it?",
    "Numbers: 0, 1, 42, 3.14, -7, 1e-9.",
    "Punctuation? Semicolons; colons: dashes-and-underscores_.",
    "Mixed case: BeginNing, MiddleEnD, ENDing.",
    "Café résumé naïve coöperate piñata",
    "新年快乐",
    "हिन्दी",
    "🌅 sun rises in the 🌄",
    "Multiple\nnewlines\nin\nthe\ntext.",
    "Tabs\tand\tspaces.",
    "Repeated repeated repeated words words words.",
    "URL: https://example.com/path?q=1&v=2#frag",
    "Code: `let x: u32 = 0;` and `fn main() { }`",
];

/// Locate the pinned model's tokenizer.json, fetching it from the pinned
/// revision recorded in the oracle registry when it is not on disk.
async fn locate_or_fetch_tokenizer() -> PathBuf {
    // Both tokenizer scenarios may run concurrently; the OnceCell serializes
    // the (at most one) network fetch and shares the resolved path.
    static TOKENIZER: tokio::sync::OnceCell<PathBuf> = tokio::sync::OnceCell::const_new();
    TOKENIZER
        .get_or_init(|| async {
            if let Ok(p) = std::env::var("HOLOGRAM_AI_SMOLLM2_TOKENIZER") {
                let p = PathBuf::from(p);
                assert!(
                    p.exists(),
                    "HOLOGRAM_AI_SMOLLM2_TOKENIZER points at {p:?} which does not exist"
                );
                return p;
            }
            let local = root().join("models/smollm2-135m/tokenizer.json");
            if local.exists() {
                return local;
            }
            let oracle = model()
                .oracle("smollm2-135m")
                .expect("the smollm2-135m oracle is registered");
            let repo = oracle
                .source
                .strip_prefix("https://huggingface.co/")
                .expect("the smollm2-135m oracle source is an HF repo url");
            let cache = root().join("target/bdd-cache");
            std::fs::create_dir_all(&cache).expect("creating target/bdd-cache");
            let dest = cache.join(format!("smollm2-{}-tokenizer.json", &oracle.pin[..12]));
            if !dest.exists() {
                let url = format!(
                    "https://huggingface.co/{repo}/resolve/{}/tokenizer.json",
                    oracle.pin
                );
                let resp = reqwest::get(&url)
                    .await
                    .unwrap_or_else(|e| panic!("fetching {url}: {e}"));
                assert!(
                    resp.status().is_success(),
                    "fetching {url}: HTTP {}",
                    resp.status()
                );
                let bytes = resp.bytes().await.expect("reading tokenizer.json body");
                // Write-then-rename so a concurrent reader never sees a
                // partial file.
                let part = dest.with_extension(format!("part-{}", std::process::id()));
                std::fs::write(&part, &bytes).expect("writing the cached tokenizer.json");
                std::fs::rename(&part, &dest).expect("publishing the cached tokenizer.json");
            }
            dest
        })
        .await
        .clone()
}

// ─────────────────────────── workflow-text helpers ──────────────────────────

/// The lines of the top-level `key:` block (exclusive of the key line).
fn top_block<'t>(text: &'t str, key: &str) -> Option<Vec<&'t str>> {
    let mut lines = text.lines();
    lines.find(|l| l.trim_end() == format!("{key}:"))?;
    let mut block = Vec::new();
    for line in lines {
        if !line.trim().is_empty() && !line.starts_with(' ') && !line.starts_with('#') {
            break;
        }
        block.push(line);
    }
    Some(block)
}

/// Whether the workflow's `on:` block declares a `push:` trigger whose
/// `branches:` include `branch`.
fn push_triggers_branch(text: &str, branch: &str) -> bool {
    let Some(on) = top_block(text, "on") else {
        return false;
    };
    let mut in_push = false;
    for line in on {
        let trimmed = line.trim();
        let indent = line.len() - line.trim_start().len();
        if trimmed == "push:" {
            in_push = true;
            continue;
        }
        if in_push && indent <= 2 && !trimmed.is_empty() && trimmed != "push:" && indent < 4 {
            in_push = false;
        }
        if in_push && trimmed.starts_with("branches") && trimmed.contains(branch) {
            return true;
        }
    }
    false
}

/// The `needs:` list of the job that publishes to Pages (the job whose body
/// uses `deploy-pages`, falling back to a job literally named deploy/publish).
fn publish_job_needs(text: &str) -> Vec<String> {
    let Some(jobs) = top_block(text, "jobs") else {
        return Vec::new();
    };
    // Split the jobs block into (name, body) chunks at 2-space-indented keys.
    let mut found: Vec<(String, Vec<&str>)> = Vec::new();
    for line in jobs {
        let indent = line.len() - line.trim_start().len();
        let trimmed = line.trim();
        if indent == 2 && trimmed.ends_with(':') && !trimmed.starts_with('#') {
            found.push((trimmed.trim_end_matches(':').to_string(), Vec::new()));
        } else if let Some((_, body)) = found.last_mut() {
            body.push(line);
        }
    }
    let publisher = found
        .iter()
        .find(|(_, body)| body.iter().any(|l| l.contains("deploy-pages")))
        .or_else(|| found.iter().find(|(n, _)| n == "deploy" || n == "publish"));
    let Some((_, body)) = publisher else {
        return Vec::new();
    };
    let mut needs = Vec::new();
    let mut in_needs_list = false;
    for line in body {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("needs:") {
            let rest = rest.trim();
            if rest.is_empty() {
                in_needs_list = true;
            } else if let Some(list) = rest.strip_prefix('[') {
                needs.extend(
                    list.trim_end_matches(']')
                        .split(',')
                        .map(|s| s.trim().to_string())
                        .filter(|s| !s.is_empty()),
                );
            } else {
                needs.push(rest.to_string());
            }
            continue;
        }
        if in_needs_list {
            if let Some(item) = trimmed.strip_prefix("- ") {
                needs.push(item.trim().to_string());
            } else if !trimmed.is_empty() {
                in_needs_list = false;
            }
        }
    }
    needs
}

// ────────────────────────────── the World ───────────────────────────────────

#[derive(Deserialize)]
struct KatCase {
    input_len: usize,
    hash: String,
}

#[derive(Deserialize)]
struct KatFile {
    cases: Vec<KatCase>,
}

/// One KAT input with its expected (truncated to 32-byte) hex digest.
struct Kat {
    input: Vec<u8>,
    digest: String,
}

#[derive(Deserialize)]
struct GoldenVector {
    name: String,
    block_bytes: Vec<u8>,
    expected: Vec<f64>,
}

/// The (inline, materialized) execution outputs of the matmul witness.
type ExecPair = (Vec<Vec<u8>>, Vec<Vec<u8>>);

/// Streamed manifest metadata (config.json + names/κ/shapes/dtypes).
struct StreamedMeta {
    config_json: String,
    keys: Vec<String>,
    kappas: Vec<String>,
    shapes: Vec<Vec<u64>>,
    dtypes: Vec<DType>,
}

#[derive(Default, cucumber::World)]
struct BddWorld {
    // S0 — addressing
    kats: Vec<Kat>,
    kat_labels: Vec<(String, String, String, String)>, // expected, ours, substrate, reference
    chunk_pairs: Vec<(String, String, String)>,        // incremental, one-shot, expected
    store: Option<StoreDir>,
    stored_kappa: Option<String>,
    resolve_outcome: Option<(String, Result<Vec<u8>, String>)>, // κ, bytes-or-error
    mat: Option<MatWitness>,
    mat_outcome: Option<Result<Vec<u8>, String>>,
    exec_pair: Option<ExecPair>,
    // S1 — acquisition
    repo: String,
    manifest: Option<Result<Vec<String>, String>>,
    st_file: Vec<u8>,
    st_parsed: Vec<hologram_ai_conformance::witness::StreamedTensorMeta>,
    // S2 — compilation
    graph_fixture: Option<(serde_json::Value, Vec<String>, UseCaseExpects)>,
    graph: Option<AiGraph>,
    graph_err: Option<String>,
    streamed: Option<StreamedMeta>,
    /// Weight-byte audit of the pinned-revision metadata walk (dictionary
    /// row `family-registry-support`): must be zero — headers only.
    streamed_weight_bytes: Option<u64>,
    archive: Option<Vec<u8>>,
    goldens: Vec<GoldenVector>,
    dequant: Vec<(String, Vec<f32>, Vec<f64>)>,
    fixture: Option<String>,
    holo_out: Vec<f32>,
    ort_out: Vec<f32>,
    // S2 — parametricity
    usecase_store: Option<StoreDir>,
    materialized: Option<Vec<u8>>,
    forward_out: Option<Vec<Vec<u8>>>,
    // S2 — coverage probe
    probe_families: Vec<String>,
    probe_results: Vec<(String, bool)>, // family, registry knows it
    // S3 — execution parity
    smollm2: Option<(PathBuf, PathBuf)>, // model.onnx, tokenizer.json
    parity_logits: Option<(Vec<f32>, Vec<f32>)>, // hologram, ORT (last position)
    parity_next: Option<(usize, usize)>,
    continuations: Option<(String, String)>,
    // S3 — tokenizer parity
    tok_path: Option<PathBuf>,
    tok_encoded: Vec<(String, Vec<u32>, Vec<u32>)>, // text, ours, reference
    tok_round: Vec<(String, String)>,               // text, decoded-back
    // S3 — structural witnesses
    witness_run: Option<(String, bool, String)>, // name, green, combined output
    // S4 — generation / elision / performance
    lm: Option<TinyLm>,
    runs: Vec<Vec<i64>>,
    counters: Vec<(usize, usize)>,
    perf_report: Option<String>,
    // S4 — app-domain events
    streams: Vec<Vec<AiEvent>>,
    views: Vec<AiView>,
    registered_manifest: Option<ModelManifest>,
    // S4 — deployment gate
    workflow: Option<String>,
}

impl fmt::Debug for BddWorld {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BddWorld")
            .field("kats", &self.kats.len())
            .field("repo", &self.repo)
            .field("fixture", &self.fixture)
            .field("streams", &self.streams.len())
            .field("witness_run", &self.witness_run.as_ref().map(|w| &w.0))
            .finish_non_exhaustive()
    }
}

// ─────────────────────────── S0 — κ-addressing ──────────────────────────────

/// The repeating 0..=250 byte pattern of the official BLAKE3 vectors.
fn kat_input(len: usize) -> Vec<u8> {
    (0..len).map(|i| (i % 251) as u8).collect()
}

#[given("the official BLAKE3 test vectors")]
async fn given_blake3_kats(w: &mut BddWorld) {
    let path = root().join("oracles/blake3/test_vectors.json");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
    let file: KatFile = serde_json::from_str(&text).expect("parsing the BLAKE3 KAT file");
    assert!(!file.cases.is_empty(), "the KAT file must carry cases");
    w.kats = file
        .cases
        .into_iter()
        .map(|c| {
            assert!(c.hash.len() >= 64, "KAT hash carries at least 32 bytes hex");
            Kat {
                input: kat_input(c.input_len),
                digest: c.hash[..64].to_string(),
            }
        })
        .collect();
}

#[when("every KAT input is κ-labeled by the pipeline, the substrate, and the reference hasher")]
async fn when_kats_labeled(w: &mut BddWorld) {
    w.kat_labels = w
        .kats
        .iter()
        .map(|kat| {
            let expected = format!("blake3:{}", kat.digest);
            let ours = kappa_of(&kat.input);
            let substrate = holospaces::address(&kat.input).as_str().to_string();
            let reference = format!("blake3:{}", blake3::hash(&kat.input).to_hex());
            (expected, ours, substrate, reference)
        })
        .collect();
}

#[then("every κ-label equals `blake3:` followed by the KAT digest")]
async fn then_kats_match(w: &mut BddWorld) {
    assert!(!w.kat_labels.is_empty(), "no KATs were labeled");
    for (i, (expected, ours, _, _)) in w.kat_labels.iter().enumerate() {
        assert_eq!(ours, expected, "KAT case {i}: κ diverges from the oracle");
    }
}

#[then("the pipeline, the substrate, and the reference hasher agree on every label")]
async fn then_kat_surfaces_agree(w: &mut BddWorld) {
    for (i, (_, ours, substrate, reference)) in w.kat_labels.iter().enumerate() {
        assert_eq!(
            ours, substrate,
            "KAT case {i}: holospaces::address diverges"
        );
        assert_eq!(ours, reference, "KAT case {i}: blake3 reference diverges");
    }
}

#[when(expr = "every KAT input is κ-hashed incrementally in chunks of {word} bytes")]
async fn when_kats_chunked(w: &mut BddWorld, chunk: String) {
    assert!(!w.kats.is_empty(), "the KATs must be loaded first");
    w.chunk_pairs = w
        .kats
        .iter()
        .map(|kat| {
            let size = match chunk.as_str() {
                "whole" => kat.input.len().max(1),
                n => n.parse::<usize>().expect("a numeric chunk size"),
            };
            let mut hasher = blake3::Hasher::new();
            for piece in kat.input.chunks(size.max(1)) {
                hasher.update(piece);
            }
            let incremental = format!("blake3:{}", hasher.finalize().to_hex());
            let one_shot = kappa_of(&kat.input);
            let expected = format!("blake3:{}", kat.digest);
            (incremental, one_shot, expected)
        })
        .collect();
}

#[then("every incremental digest equals the one-shot κ of the whole input")]
async fn then_chunked_matches(w: &mut BddWorld) {
    assert!(
        !w.chunk_pairs.is_empty(),
        "no chunked digests were computed"
    );
    for (i, (incremental, one_shot, expected)) in w.chunk_pairs.iter().enumerate() {
        assert_eq!(
            incremental, one_shot,
            "KAT case {i}: chunking changed the κ"
        );
        assert_eq!(
            one_shot, expected,
            "KAT case {i}: κ diverges from the oracle"
        );
    }
}

// ──────────────────── S0 — content-verified resolution ──────────────────────

#[given("an empty κ-store directory")]
async fn given_empty_store(w: &mut BddWorld) {
    w.store = Some(StoreDir::new("store"));
}

#[when("bytes are persisted under their derived κ")]
async fn when_bytes_persisted(w: &mut BddWorld) {
    let store = w.store.as_ref().expect("a κ-store directory");
    let bytes = b"content-verified resolution witness bytes";
    let kappa = DirKappaStore::new(&store.path)
        .insert(bytes)
        .expect("persisting bytes under their κ");
    assert_eq!(kappa, kappa_of(bytes), "insert derives the content κ");
    w.stored_kappa = Some(kappa);
}

#[then("resolving that κ returns bytes that re-hash to the same κ")]
async fn then_resolution_verifies(w: &mut BddWorld) {
    let store = w.store.as_ref().expect("a κ-store directory");
    let kappa = w.stored_kappa.as_ref().expect("a persisted κ");
    let bytes = DirKappaStore::new(&store.path)
        .resolve(kappa)
        .expect("resolving a persisted κ");
    assert_eq!(
        &kappa_of(&bytes),
        kappa,
        "resolved content must reproduce its κ"
    );
}

#[when("a κ that was never persisted is resolved")]
async fn when_missing_resolved(w: &mut BddWorld) {
    let store = w.store.as_ref().expect("a κ-store directory");
    let kappa = kappa_of(b"never persisted anywhere");
    let outcome = DirKappaStore::new(&store.path)
        .resolve(&kappa)
        .map_err(|e| format!("{e:#}"));
    w.resolve_outcome = Some((kappa, outcome));
}

#[then("the resolution fails naming that κ")]
async fn then_missing_fails_naming(w: &mut BddWorld) {
    let (kappa, outcome) = w.resolve_outcome.as_ref().expect("a resolution attempt");
    let err = outcome
        .as_ref()
        .expect_err("resolving a missing κ must fail");
    assert!(err.contains(kappa), "the error must name the κ: {err}");
}

#[given("a κ-store holding corrupt bytes under a known κ")]
async fn given_corrupt_store(w: &mut BddWorld) {
    let mat = mat_witness();
    let store = StoreDir::new("corrupt");
    let mut wrong = mat.weight.clone();
    wrong[0] ^= 0xFF;
    std::fs::write(store.path.join(format!("{}.bin", mat.kappa)), &wrong)
        .expect("planting corrupt content");
    w.mat = Some(mat);
    w.store = Some(store);
}

#[when("a k-form archive requiring that κ is materialized against the store")]
async fn when_kform_materialized_against_store(w: &mut BddWorld) {
    materialize_attempt(w);
}

fn materialize_attempt(w: &mut BddWorld) {
    let mat = w.mat.as_ref().expect("the matmul witness");
    let store = w.store.as_ref().expect("a κ-store directory");
    let mut dir_store = DirKappaStore::new(&store.path);
    w.mat_outcome =
        Some(materialize_archive(&mat.kform, &mut dir_store).map_err(|e| format!("{e:#}")));
}

#[then("materialization fails the integrity check naming the expected κ")]
async fn then_integrity_failure(w: &mut BddWorld) {
    let mat = w.mat.as_ref().expect("the matmul witness");
    let outcome = w.mat_outcome.as_ref().expect("a materialization attempt");
    let err = outcome
        .as_ref()
        .expect_err("corrupt content must not materialize");
    assert!(
        err.contains("integrity"),
        "the error must be the κ integrity failure: {err}"
    );
    assert!(err.contains(&mat.kappa), "the error must name the κ: {err}");
}

// ──────────────────── S1 — HuggingFace model resolution ─────────────────────

/// Companion assets per the dictionary row (config/tokenizer/generation).
const COMPANIONS: &[&str] = &["config.json", "tokenizer.json", "generation_config.json"];

async fn resolve_manifest(repo: &str) -> Result<Vec<String>, String> {
    let url = format!("https://huggingface.co/api/models/{repo}");
    let resp = reqwest::get(&url)
        .await
        .map_err(|e| format!("resolving `{repo}`: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!(
            "resolving `{repo}` via the Hub API failed: HTTP {}",
            resp.status()
        ));
    }
    let info: serde_json::Value = resp
        .json()
        .await
        .map_err(|e| format!("resolving `{repo}`: malformed Hub response: {e}"))?;
    let files = info
        .get("siblings")
        .and_then(|s| s.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|f| f.get("rfilename").and_then(|n| n.as_str()))
                .map(str::to_string)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    if files.is_empty() {
        return Err(format!("resolving `{repo}`: the Hub returned no files"));
    }
    Ok(files)
}

#[given(expr = "the HuggingFace repository {string}")]
async fn given_repo(w: &mut BddWorld, repo: String) {
    w.repo = repo;
}

#[when("the file manifest is resolved via the Hub API")]
async fn when_manifest_resolved(w: &mut BddWorld) {
    let files = resolve_manifest(&w.repo)
        .await
        .unwrap_or_else(|e| panic!("{e}"));
    w.manifest = Some(Ok(files));
}

#[when("the file manifest resolution is attempted")]
async fn when_manifest_attempted(w: &mut BddWorld) {
    w.manifest = Some(resolve_manifest(&w.repo).await);
}

fn manifest_files(w: &BddWorld) -> &[String] {
    w.manifest
        .as_ref()
        .expect("a resolved manifest")
        .as_ref()
        .expect("the manifest resolved successfully")
}

#[then("the manifest classifies at least one safetensors shard")]
async fn then_manifest_has_shards(w: &mut BddWorld) {
    let shards: Vec<&String> = manifest_files(w)
        .iter()
        .filter(|f| f.ends_with(".safetensors"))
        .collect();
    assert!(
        !shards.is_empty(),
        "expected at least one safetensors shard in the manifest"
    );
}

#[then(expr = "the manifest classifies {string} and {string} as companions")]
async fn then_manifest_has_companions(w: &mut BddWorld, a: String, b: String) {
    let files = manifest_files(w);
    for name in [&a, &b] {
        assert!(
            COMPANIONS.contains(&name.as_str()),
            "`{name}` is not a companion asset by classification"
        );
        assert!(
            files.iter().any(|f| f == name),
            "`{name}` is missing from the resolved manifest"
        );
    }
}

#[then("no file is classified as both shard and companion")]
async fn then_classification_disjoint(w: &mut BddWorld) {
    for f in manifest_files(w) {
        let shard = f.ends_with(".safetensors");
        let companion = COMPANIONS.contains(&f.as_str());
        assert!(
            !(shard && companion),
            "`{f}` classified as both shard and companion"
        );
    }
}

#[then("the resolution fails naming the repository")]
async fn then_resolution_fails(w: &mut BddWorld) {
    let outcome = w.manifest.as_ref().expect("a resolution attempt");
    let err = outcome
        .as_ref()
        .expect_err("an unknown repository must fail to resolve");
    assert!(
        err.contains(&w.repo),
        "the error must name the repository `{}`: {err}",
        w.repo
    );
}

// ──────────────────── S1 — safetensors header streaming ─────────────────────

#[given("a multi-tensor safetensors file serialized by the reference crate")]
async fn given_reference_safetensors(w: &mut BddWorld) {
    let alpha: Vec<u8> = f32s_le(&(0..6).map(|i| i as f32 * 0.5 - 1.25).collect::<Vec<_>>());
    let beta: Vec<u8> = i64s_le(&[1, -2, 3, -4]);
    let gamma: Vec<u8> = f32s_le(&(0..8).map(|i| (i as f32).sin()).collect::<Vec<_>>());
    let views = vec![
        (
            "alpha.weight",
            safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![2, 3], &alpha)
                .expect("alpha view"),
        ),
        (
            "beta.ids",
            safetensors::tensor::TensorView::new(safetensors::Dtype::I64, vec![4], &beta)
                .expect("beta view"),
        ),
        (
            "gamma.table",
            safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![2, 2, 2], &gamma)
                .expect("gamma view"),
        ),
    ];
    let metadata = Some(HashMap::from([(
        "format".to_string(),
        "bdd-reference".to_string(),
    )]));
    w.st_file = safetensors::serialize(views, &metadata)
        .expect("the reference crate serializes the fixture");
}

#[when("the file is stream-parsed by the import path")]
async fn when_safetensors_stream_parsed(w: &mut BddWorld) {
    let (header, _) = split_safetensors(&w.st_file).expect("well-formed length prefix");
    w.st_parsed = parse_streamed_header(header).expect("the streaming header parse succeeds");
}

#[then("every tensor name, dtype, shape, and data range matches the reference crate's view")]
async fn then_safetensors_identical(w: &mut BddWorld) {
    let reference =
        safetensors::SafeTensors::deserialize(&w.st_file).expect("the reference crate view");
    let (_, data) = split_safetensors(&w.st_file).expect("well-formed length prefix");
    assert_eq!(
        w.st_parsed.len(),
        reference.tensors().len(),
        "tensor count diverges from the reference view"
    );
    for entry in &w.st_parsed {
        let view = reference
            .tensor(&entry.name)
            .unwrap_or_else(|e| panic!("`{}` missing from the reference view: {e}", entry.name));
        let expected_dtype =
            hologram_ai_conformance::witness::safetensors_dtype(&format!("{:?}", view.dtype()));
        assert_eq!(
            entry.dtype, expected_dtype,
            "`{}`: dtype diverges from the reference",
            entry.name
        );
        let ref_shape: Vec<u64> = view.shape().iter().map(|&d| d as u64).collect();
        assert_eq!(
            entry.shape, ref_shape,
            "`{}`: shape diverges from the reference",
            entry.name
        );
        let (begin, end) = entry.data_offsets;
        let ours = data
            .get(begin as usize..end as usize)
            .unwrap_or_else(|| panic!("`{}`: data range out of bounds", entry.name));
        assert_eq!(
            ours,
            view.data(),
            "`{}`: data bytes diverge from the reference",
            entry.name
        );
    }
}

// ─────────────────────── S2 — the parametric graph ──────────────────────────

fn handshake_expects() -> (String, UseCaseExpects) {
    let uc = model()
        .usecase("handshake-tiny")
        .expect("the handshake-tiny use-case is registered");
    (uc.family.clone(), uc.expects)
}

#[given("a Llama-family config with the handshake-tiny quantities and an untied manifest")]
async fn given_untied_fixture(w: &mut BddWorld) {
    let (family, e) = handshake_expects();
    let config = handshake_config(&family, &e, false);
    let keys = decoder_manifest(&e, false, false)
        .into_iter()
        .map(|(k, _)| k)
        .collect();
    w.graph_fixture = Some((config, keys, e));
}

#[given("a Llama-family config with the handshake-tiny quantities and a tied manifest")]
async fn given_tied_fixture(w: &mut BddWorld) {
    let (family, e) = handshake_expects();
    let config = handshake_config(&family, &e, true);
    let keys = decoder_manifest(&e, true, false)
        .into_iter()
        .map(|(k, _)| k)
        .collect();
    w.graph_fixture = Some((config, keys, e));
}

#[given(expr = "a config naming the architecture family {string}")]
async fn given_family_fixture(w: &mut BddWorld, family: String) {
    let (_, e) = handshake_expects();
    let config = handshake_config(&family, &e, false);
    let keys = decoder_manifest(&e, false, false)
        .into_iter()
        .map(|(k, _)| k)
        .collect();
    w.graph_fixture = Some((config, keys, e));
}

#[when("the parametric graph is built from config and manifest alone")]
async fn when_graph_built(w: &mut BddWorld) {
    let (config, keys, _) = w.graph_fixture.as_ref().expect("a graph fixture");
    let dtypes = vec![DType::F32; keys.len()];
    w.graph = Some(
        build_parametric_graph_from_manifest(config, keys, &dtypes, None)
            .expect("the parametric graph builds from config + manifest"),
    );
}

#[when("the parametric graph build is attempted")]
async fn when_graph_attempted(w: &mut BddWorld) {
    let (config, keys, _) = w.graph_fixture.as_ref().expect("a graph fixture");
    let dtypes = vec![DType::F32; keys.len()];
    match build_parametric_graph_from_manifest(config, keys, &dtypes, None) {
        Ok(g) => {
            w.graph = Some(g);
            w.graph_err = None;
        }
        Err(e) => w.graph_err = Some(format!("{e:#}")),
    }
}

fn built_graph(w: &BddWorld) -> &AiGraph {
    w.graph.as_ref().expect("a built parametric graph")
}

#[then("every RmsNorm epsilon equals the config's rms_norm_eps")]
async fn then_eps_from_config(w: &mut BddWorld) {
    let (_, _, e) = w.graph_fixture.as_ref().expect("a graph fixture");
    let expected = e.rms_norm_eps as f32;
    let eps: Vec<f32> = built_graph(w)
        .nodes
        .iter()
        .filter_map(|n| match n.op {
            AiOp::RmsNorm { epsilon } => Some(epsilon),
            _ => None,
        })
        .collect();
    let want = (e.num_hidden_layers * 2 + 1) as usize;
    assert_eq!(eps.len(), want, "two norms per layer plus the final norm");
    for (i, got) in eps.iter().enumerate() {
        assert!(
            (got - expected).abs() < 1e-12,
            "RmsNorm {i}: epsilon {got} diverges from config {expected}"
        );
    }
}

#[then("every attention node carries the config's heads, KV heads, head dim, and rope_theta")]
async fn then_attention_from_config(w: &mut BddWorld) {
    let (_, _, e) = w.graph_fixture.as_ref().expect("a graph fixture");
    let head_dim = (e.hidden_size / e.num_attention_heads) as u32;
    let gqa: Vec<(u32, u32, u32, f32)> = built_graph(w)
        .nodes
        .iter()
        .filter_map(|n| match n.op {
            AiOp::GroupedQueryAttention {
                num_heads,
                num_kv_heads,
                head_dim,
                rope_base,
                ..
            } => Some((num_heads, num_kv_heads, head_dim, rope_base)),
            _ => None,
        })
        .collect();
    assert_eq!(
        gqa.len(),
        e.num_hidden_layers as usize,
        "one attention node per layer"
    );
    for (i, (h, kv, d, theta)) in gqa.iter().enumerate() {
        assert_eq!(*h, e.num_attention_heads as u32, "layer {i}: heads");
        assert_eq!(*kv, e.num_key_value_heads as u32, "layer {i}: KV heads");
        assert_eq!(*d, head_dim, "layer {i}: head dim");
        assert!(
            (theta - e.rope_theta as f32).abs() < 1e-3,
            "layer {i}: rope_theta {theta} diverges from config {}",
            e.rope_theta
        );
    }
}

fn graph_tensor_id(graph: &AiGraph, name: &str) -> Option<u32> {
    graph
        .tensor_names
        .iter()
        .find(|(_, n)| n.as_str() == name)
        .map(|(id, _)| *id)
}

#[then("the graph declares a separate lm_head weight")]
async fn then_untied_head(w: &mut BddWorld) {
    assert!(
        graph_tensor_id(built_graph(w), "lm_head.weight").is_some(),
        "an untied model must declare lm_head.weight"
    );
}

#[then("the graph declares no separate lm_head weight")]
async fn then_tied_head(w: &mut BddWorld) {
    assert!(
        graph_tensor_id(built_graph(w), "lm_head.weight").is_none(),
        "a tied model must not declare lm_head.weight"
    );
}

#[then("the embedding weight feeds both the token gather and the head projection")]
async fn then_embed_feeds_both(w: &mut BddWorld) {
    let graph = built_graph(w);
    let embed = graph_tensor_id(graph, "model.embed_tokens.weight")
        .expect("the embedding weight is declared");
    let consumers: Vec<&AiOp> = graph
        .nodes
        .iter()
        .filter(|n| n.inputs.contains(&embed))
        .map(|n| &n.op)
        .collect();
    assert!(
        consumers.iter().any(|op| matches!(op, AiOp::Gather { .. })),
        "the embedding weight must feed the token gather"
    );
    assert!(
        consumers
            .iter()
            .any(|op| matches!(op, AiOp::Transpose { .. })),
        "the tied embedding weight must feed the head projection transpose"
    );
}

#[then("the graph metadata carries the model's own context length")]
async fn then_context_from_config(w: &mut BddWorld) {
    let got = match built_graph(w).metadata.get("context_length") {
        Some(hologram_ai_common::MetaValue::Int(i)) => *i,
        other => panic!("context_length metadata missing or mistyped: {other:?}"),
    };
    assert_eq!(
        got, TINY_CONTEXT as i64,
        "context length must be the config's max_position_embeddings"
    );
}

#[then(expr = "the build fails naming {string} and the supported families")]
async fn then_family_fails(w: &mut BddWorld, family: String) {
    let err = w
        .graph_err
        .as_ref()
        .expect("an unsupported family must fail the build");
    assert!(
        err.contains(&family),
        "the error must name `{family}`: {err}"
    );
    assert!(
        err.contains("supported families"),
        "the error must name the supported set: {err}"
    );
}

// ────────────────── S2 — streamed weightless compilation ────────────────────

#[given(expr = "the streamed metadata of {string} from the Hub")]
async fn given_streamed_metadata(w: &mut BddWorld, repo: String) {
    let (config_json, keys, kappas, shapes, dtypes) =
        fetch_helper::fetch_authoritative_metadata(&repo).await;
    assert!(!keys.is_empty(), "the Hub manifest must name tensors");
    w.streamed = Some(StreamedMeta {
        config_json,
        keys,
        kappas,
        shapes,
        dtypes,
    });
}

// ─────────────── S2 — family-registry support (external authority) ──────────

/// The pinned-revision variant of the streamed-metadata Given: the
/// family-registry witness holds every registered family to its published
/// authority *at the oracle pin*, auditing that no weight bytes flow.
#[given(expr = "the streamed metadata of {string} at revision {string} from the Hub")]
async fn given_streamed_metadata_at(w: &mut BddWorld, repo: String, revision: String) {
    let m = fetch_helper::fetch_authoritative_metadata_at(&repo, &revision).await;
    assert!(!m.keys.is_empty(), "the Hub manifest must name tensors");
    w.streamed_weight_bytes = Some(m.weight_bytes_fetched);
    w.streamed = Some(StreamedMeta {
        config_json: m.config_json,
        keys: m.keys,
        kappas: m.kappas,
        shapes: m.shapes,
        dtypes: m.dtypes,
    });
}

#[then(expr = "the selected family is {string}")]
async fn then_selected_family(w: &mut BddWorld, family: String) {
    let m = w.streamed.as_ref().expect("streamed metadata");
    let config: serde_json::Value =
        serde_json::from_str(&m.config_json).expect("parsing the authority's config.json");
    let selected =
        selected_family(&config).expect("the registry selects a family for the authority's config");
    assert_eq!(
        selected, family,
        "registry selection diverges from the outline row"
    );
    assert!(
        supported_families().contains(&selected),
        "`{selected}` must be listed by supported_families()"
    );
    println!("[family-registry] {selected}: selected for the pinned authority");
}

#[then("no weight bytes were fetched")]
async fn then_no_weight_bytes(w: &mut BddWorld) {
    let bytes = w
        .streamed_weight_bytes
        .expect("the pinned-revision walk recorded its weight-byte audit");
    assert_eq!(
        bytes, 0,
        "the metadata walk must fetch zero weight bytes, got {bytes}"
    );
    println!("[family-registry] weight bytes fetched: {bytes}");
}

#[when("the manifest is compiled without weights")]
async fn when_weightless_compiled(w: &mut BddWorld) {
    let m = w.streamed.as_ref().expect("streamed metadata");
    let source = ModelSource::SafetensorsStreamed {
        config_json: m.config_json.clone(),
        keys: m.keys.clone(),
        kappas: m.kappas.clone(),
        shapes: m.shapes.clone(),
        dtypes: m.dtypes.clone(),
    };
    let prepared = ModelCompiler::default()
        .prepare(source)
        .expect("preparing the streamed manifest");
    let archive = prepared
        .compile_at(Some(64), Default::default())
        .expect("the weightless compile succeeds");
    w.archive = Some(archive.bytes);
}

#[then("the archive carries a kappa_map")]
async fn then_archive_has_kappa_map(w: &mut BddWorld) {
    let archive = w.archive.as_ref().expect("a compiled archive");
    let reqs = kappa_requirements(archive).expect("the κ-map parses");
    assert!(!reqs.is_empty(), "the k-form archive must carry a κ-map");
}

#[then("the kappa_map names every manifest weight tensor exactly once")]
async fn then_kappa_map_complete(w: &mut BddWorld) {
    let m = w.streamed.as_ref().expect("streamed metadata");
    let archive = w.archive.as_ref().expect("a compiled archive");
    let reqs = kappa_requirements(archive).expect("the κ-map parses");
    let mut got: Vec<&str> = reqs.iter().map(|r| r.kappa.as_str()).collect();
    got.sort_unstable();
    let dupes: Vec<&str> = got
        .windows(2)
        .filter(|p| p[0] == p[1])
        .map(|p| p[0])
        .collect();
    assert!(dupes.is_empty(), "κ named more than once: {dupes:?}");
    let got_set: BTreeSet<&str> = got.iter().copied().collect();
    let want_set: BTreeSet<&str> = m.kappas.iter().map(String::as_str).collect();
    let missing: Vec<&&str> = want_set.difference(&got_set).collect();
    let extra: Vec<&&str> = got_set.difference(&want_set).collect();
    assert!(
        missing.is_empty() && extra.is_empty(),
        "κ-map does not cover the manifest exactly once — missing: {missing:?}, extra: {extra:?}"
    );
}

// ───────────────────────── S2 — quant golden vectors ────────────────────────

#[given(expr = "the committed golden vectors {string}")]
async fn given_goldens(w: &mut BddWorld, file: String) {
    let path = root().join("oracles/quant").join(&file);
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
    w.goldens =
        serde_json::from_str(&text).unwrap_or_else(|e| panic!("parsing {}: {e}", path.display()));
    assert!(!w.goldens.is_empty(), "no golden vectors in {file}");
}

#[when(expr = "every block is dequantized with the {word} kernel")]
async fn when_dequantized(w: &mut BddWorld, scheme: String) {
    let kernel: fn(&[u8]) -> Vec<f32> = match scheme.as_str() {
        "Q4_0" => dequant_q4_0,
        "Q8_0" => dequant_q8_0,
        other => panic!("unknown quant scheme `{other}`"),
    };
    w.dequant = w
        .goldens
        .iter()
        .map(|v| (v.name.clone(), kernel(&v.block_bytes), v.expected.clone()))
        .collect();
}

#[then(expr = "every element matches the reference within {float}")]
async fn then_dequant_matches(w: &mut BddWorld, tolerance: f32) {
    assert!(!w.dequant.is_empty(), "nothing was dequantized");
    for (name, ours, expected) in &w.dequant {
        assert_eq!(ours.len(), expected.len(), "{name}: length mismatch");
        for (i, (got, want)) in ours.iter().zip(expected.iter()).enumerate() {
            let want = *want as f32;
            assert!(
                (got - want).abs() < tolerance,
                "{name}: index {i}: ours {got} vs golden {want} (tolerance {tolerance})"
            );
        }
    }
}

// ─────────────────────────── S2 — parametricity ─────────────────────────────

#[given("the handshake-tiny use-case from the model registry")]
async fn given_handshake_usecase(w: &mut BddWorld) {
    let uc = model()
        .usecase("handshake-tiny")
        .expect("the handshake-tiny use-case is registered");
    assert!(
        !uc.canonical,
        "handshake-tiny must be an arbitrary (non-canonical) instance"
    );
    let e = uc.expects;
    let manifest = decoder_manifest(&e, e.tie_word_embeddings, false);
    let config = handshake_config(&uc.family, &e, e.tie_word_embeddings);
    let (keys, shapes): (Vec<String>, Vec<Vec<u64>>) = manifest.into_iter().unzip();
    let dtypes = vec![DType::F32; keys.len()];
    w.streamed = Some(StreamedMeta {
        config_json: config.to_string(),
        kappas: Vec::new(), // filled once the seeded weights exist
        keys,
        shapes,
        dtypes,
    });
}

#[given("deterministic seeded weights for its manifest in a κ-store")]
async fn given_seeded_weights(w: &mut BddWorld) {
    let store = StoreDir::new("parametricity");
    let dir = DirKappaStore::new(&store.path);
    let m = w.streamed.as_mut().expect("the handshake-tiny manifest");
    let mut kappas = Vec::with_capacity(m.keys.len());
    for (key, shape) in m.keys.iter().zip(m.shapes.iter()) {
        let count = shape.iter().product::<u64>() as usize;
        let bytes = f32s_le(&seeded_weights(key, count));
        kappas.push(dir.insert(&bytes).expect("persisting a seeded weight"));
    }
    m.kappas = kappas;
    w.usecase_store = Some(store);
}

#[when("the manifest is compiled without weights and materialized against the store")]
async fn when_pipeline_materialized(w: &mut BddWorld) {
    let m = w.streamed.as_ref().expect("the handshake-tiny manifest");
    let source = ModelSource::SafetensorsStreamed {
        config_json: m.config_json.clone(),
        keys: m.keys.clone(),
        kappas: m.kappas.clone(),
        shapes: m.shapes.clone(),
        dtypes: m.dtypes.clone(),
    };
    let prepared = ModelCompiler::default()
        .prepare(source)
        .expect("preparing the handshake-tiny manifest");
    let archive = prepared
        .compile_at(Some(4), Default::default())
        .expect("the weightless compile succeeds");
    let store = w.usecase_store.as_ref().expect("the seeded κ-store");
    let mut dir = DirKappaStore::new(&store.path);
    w.materialized = Some(
        materialize_archive(&archive.bytes, &mut dir)
            .expect("materialization against the seeded store succeeds"),
    );
}

fn forward_pass(materialized: &[u8]) -> Vec<Vec<u8>> {
    let mut runner =
        HoloRunner::from_bytes(materialized.to_vec()).expect("loading the materialized archive");
    let ids = i64s_le(&[1, 2, 3, 4]);
    let outs = runner.execute(&[&ids]).expect("the forward pass executes");
    outs.into_iter().map(|o| o.bytes).collect()
}

#[then("the materialized session executes a forward pass")]
async fn then_forward_pass(w: &mut BddWorld) {
    let materialized = w.materialized.as_ref().expect("a materialized archive");
    let outs = forward_pass(materialized);
    assert!(!outs.is_empty(), "the session must produce outputs");
    let logits = le_f32(&outs[0]);
    assert!(!logits.is_empty(), "the logits must be non-empty");
    assert!(
        logits.iter().all(|v| v.is_finite()),
        "the logits must be finite"
    );
    w.forward_out = Some(outs);
}

#[then("a second materialized session executes to byte-identical output")]
async fn then_forward_deterministic(w: &mut BddWorld) {
    let materialized = w.materialized.as_ref().expect("a materialized archive");
    let first = w.forward_out.as_ref().expect("the first forward pass");
    let second = forward_pass(materialized);
    assert_eq!(
        first, &second,
        "two independent materialized sessions must agree byte-for-byte"
    );
}

// ──────────────── S2 — architecture coverage probe (target) ─────────────────

#[given("a fixed list of common HuggingFace architecture families")]
async fn given_probe_families(w: &mut BddWorld) {
    w.probe_families = [
        "LlamaForCausalLM",
        "Qwen2ForCausalLM",
        "MistralForCausalLM",
        "Gemma2ForCausalLM",
        "Phi3ForCausalLM",
        "GPT2LMHeadModel",
        "GPTNeoXForCausalLM",
        "MixtralForCausalLM",
        "Qwen3ForCausalLM",
        "BertForMaskedLM",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
}

#[when("each family is probed against the parametric registry")]
async fn when_families_probed(w: &mut BddWorld) {
    let (_, e) = handshake_expects();
    // Probe with a bias-carrying manifest so bias-structural families
    // (e.g. Qwen2) can build; the probe measures *registry selection*.
    let keys: Vec<String> = decoder_manifest(&e, false, true)
        .into_iter()
        .map(|(k, _)| k)
        .collect();
    let dtypes = vec![DType::F32; keys.len()];
    w.probe_results = w
        .probe_families
        .iter()
        .map(|family| {
            let config = handshake_config(family, &e, false);
            let known = match build_parametric_graph_from_manifest(&config, &keys, &dtypes, None) {
                Ok(_) => true,
                Err(e) => !format!("{e:#}").contains("unsupported architecture family"),
            };
            println!(
                "[coverage] {family}: {}",
                if known { "supported" } else { "unsupported" }
            );
            (family.clone(), known)
        })
        .collect();
}

#[then("the supported and unsupported counts are reported for every probed family")]
async fn then_probe_reported(w: &mut BddWorld) {
    assert_eq!(
        w.probe_results.len(),
        w.probe_families.len(),
        "every family in the list must be probed"
    );
    let supported = w.probe_results.iter().filter(|(_, s)| *s).count();
    let unsupported = w.probe_results.len() - supported;
    println!(
        "[coverage] measured: {supported} supported / {unsupported} unsupported of {} probed families",
        w.probe_results.len()
    );
}

// ────────────────────────── S2 — ONNX compile parity ────────────────────────

#[given(expr = "the external authoritative ONNX fixture {string}")]
async fn given_onnx_fixture(w: &mut BddWorld, name: String) {
    w.fixture = Some(name);
}

#[when("the fixture is compiled and executed by hologram and by ONNX Runtime")]
async fn when_fixture_parity(w: &mut BddWorld) {
    #[cfg(feature = "conformance")]
    ort_gated::fixture_parity(w);
    #[cfg(not(feature = "conformance"))]
    {
        let _ = w;
        panic!(
            "the onnx-compile-parity steps execute ONNX Runtime — rebuild with \
             `--features conformance` (the ort lane always does)"
        );
    }
}

#[then("the outputs match ONNX Runtime within tolerance")]
async fn then_fixture_matches_ort(w: &mut BddWorld) {
    assert!(
        !w.holo_out.is_empty() && !w.ort_out.is_empty(),
        "both engines must have produced output"
    );
    assert_eq!(w.holo_out.len(), w.ort_out.len(), "output length mismatch");
    let mut max_diff = 0f32;
    for (i, (h, r)) in w.holo_out.iter().zip(w.ort_out.iter()).enumerate() {
        let diff = (h - r).abs();
        max_diff = max_diff.max(diff);
        let tol = 1e-2 + 2e-3 * r.abs();
        assert!(
            diff <= tol,
            "element {i}: hologram {h} vs ORT {r} (|diff| {diff} > tol {tol})"
        );
    }
    println!("[onnx-parity] max |diff| vs ORT: {max_diff:e}");
}

// ─────────────────────── S3 — κ-materialization ─────────────────────────────

#[given("a matmul graph whose weight is available as bytes")]
async fn given_mat_witness(w: &mut BddWorld) {
    w.mat = Some(mat_witness());
}

#[given("a κ-store holding the weight under its κ")]
async fn given_store_with_weight(w: &mut BddWorld) {
    let mat = w.mat.as_ref().expect("the matmul witness");
    let store = StoreDir::new("materialize");
    let kappa = DirKappaStore::new(&store.path)
        .insert(&mat.weight)
        .expect("persisting the weight");
    assert_eq!(kappa, mat.kappa, "the store derives the weight's κ");
    w.store = Some(store);
}

#[given("a κ-store holding corrupt bytes under the weight's κ")]
async fn given_store_with_corrupt_weight(w: &mut BddWorld) {
    let mat = w.mat.as_ref().expect("the matmul witness");
    let store = StoreDir::new("materialize-corrupt");
    let mut wrong = mat.weight.clone();
    wrong[0] ^= 0xFF;
    std::fs::write(store.path.join(format!("{}.bin", mat.kappa)), &wrong)
        .expect("planting corrupt content");
    w.store = Some(store);
}

#[when("the k-form archive is materialized and executed next to the inline archive")]
async fn when_materialized_and_executed(w: &mut BddWorld) {
    materialize_attempt(w);
    let mat = w.mat.as_ref().expect("the matmul witness");
    let materialized = w
        .mat_outcome
        .as_ref()
        .expect("a materialization attempt")
        .as_ref()
        .expect("materialization succeeds against the populated store")
        .clone();
    let x = matmul_input();
    let mut inline_runner =
        HoloRunner::from_bytes(mat.inline_holo.clone()).expect("the inline archive loads");
    let inline_out: Vec<Vec<u8>> = inline_runner
        .execute(&[&x])
        .expect("the inline archive executes")
        .into_iter()
        .map(|o| o.bytes)
        .collect();
    let mut mat_runner =
        HoloRunner::from_bytes(materialized).expect("the materialized archive loads");
    let mat_out: Vec<Vec<u8>> = mat_runner
        .execute(&[&x])
        .expect("the materialized archive executes")
        .into_iter()
        .map(|o| o.bytes)
        .collect();
    w.exec_pair = Some((inline_out, mat_out));
}

#[then("the k-form archive declares exactly the weight's κ as its one requirement")]
async fn then_one_requirement(w: &mut BddWorld) {
    let mat = w.mat.as_ref().expect("the matmul witness");
    let reqs = kappa_requirements(&mat.kform).expect("the κ-map parses");
    assert_eq!(reqs.len(), 1, "one external weight → one requirement");
    assert_eq!(
        reqs[0].kappa, mat.kappa,
        "the requirement names the weight's κ"
    );
    assert!(
        kappa_requirements(&mat.inline_holo)
            .expect("the inline archive parses")
            .is_empty(),
        "an inline archive carries no κ-map"
    );
}

#[then("both executions produce byte-identical non-trivial output")]
async fn then_byte_identical(w: &mut BddWorld) {
    let (inline_out, mat_out) = w.exec_pair.as_ref().expect("both executions ran");
    assert_eq!(
        inline_out, mat_out,
        "materialized execution must be byte-identical"
    );
    let y = le_f32(&mat_out[0]);
    assert!(
        y.iter().any(|v| v.abs() > 1e-6),
        "output must reflect real weights, got {y:?}"
    );
}

#[when("materialization of the k-form archive is attempted")]
async fn when_materialization_attempted(w: &mut BddWorld) {
    materialize_attempt(w);
}

#[then("materialization fails naming the weight's κ")]
async fn then_missing_kappa_named(w: &mut BddWorld) {
    let mat = w.mat.as_ref().expect("the matmul witness");
    let outcome = w.mat_outcome.as_ref().expect("a materialization attempt");
    let err = outcome
        .as_ref()
        .expect_err("an empty store cannot materialize");
    assert!(err.contains(&mat.kappa), "the error must name the κ: {err}");
}

// ───────────────────────── S3 — execution parity ────────────────────────────

#[given("the pinned SmolLM2 export on disk")]
async fn given_smollm2(w: &mut BddWorld) {
    let onnx = if let Ok(p) = std::env::var("HOLOGRAM_AI_SMOLLM2_ONNX") {
        PathBuf::from(p)
    } else {
        root().join("models/smollm2-135m/model.onnx")
    };
    assert!(
        onnx.exists(),
        "the model lane requires the pinned SmolLM2 export: place it at \
         <workspace>/models/smollm2-135m/model.onnx or set HOLOGRAM_AI_SMOLLM2_ONNX"
    );
    let tokenizer = onnx.with_file_name("tokenizer.json");
    assert!(
        tokenizer.exists(),
        "tokenizer.json must sit next to the model at {tokenizer:?}"
    );
    w.smollm2 = Some((onnx, tokenizer));
}

#[when(expr = "the prompt {string} is executed by hologram and by ONNX Runtime")]
async fn when_prefill_parity(w: &mut BddWorld, prompt: String) {
    #[cfg(feature = "conformance")]
    ort_gated::smollm2_prefill_parity(w, &prompt);
    #[cfg(not(feature = "conformance"))]
    {
        let _ = (w, prompt);
        panic!(
            "the execution-parity steps execute ONNX Runtime — rebuild with \
             `--features conformance` (the model lane always does)"
        );
    }
}

#[then("the last-position logits agree within tolerance")]
async fn then_logits_agree(w: &mut BddWorld) {
    let (holo, ort) = w
        .parity_logits
        .as_ref()
        .expect("both engines produced logits");
    assert_eq!(holo.len(), ort.len(), "vocab width mismatch");
    let mut max_diff = 0f32;
    for (i, (h, r)) in holo.iter().zip(ort.iter()).enumerate() {
        let diff = (h - r).abs();
        max_diff = max_diff.max(diff);
        let tol = 2e-2 + 2e-3 * r.abs();
        assert!(
            diff <= tol,
            "logit {i}: hologram {h} vs ORT {r} (|diff| {diff} > tol {tol})"
        );
    }
    println!("[execution-parity] max |logit diff| vs ORT: {max_diff:e}");
}

#[then("both engines agree on the greedy next token")]
async fn then_next_token_agrees(w: &mut BddWorld) {
    let (ours, theirs) = w.parity_next.expect("both engines picked a next token");
    assert_eq!(ours, theirs, "greedy next-token divergence");
    println!("[execution-parity] greedy next token: {ours}");
}

#[when(expr = "both engines greedily decode {int} tokens from {string}")]
async fn when_greedy_parity(w: &mut BddWorld, n: usize, prompt: String) {
    #[cfg(feature = "conformance")]
    ort_gated::smollm2_greedy_parity(w, &prompt, n);
    #[cfg(not(feature = "conformance"))]
    {
        let _ = (w, n, prompt);
        panic!(
            "the execution-parity steps execute ONNX Runtime — rebuild with \
             `--features conformance` (the model lane always does)"
        );
    }
}

#[then("the decoded continuations are identical")]
async fn then_continuations_identical(w: &mut BddWorld) {
    let (ours, theirs) = w.continuations.as_ref().expect("both engines decoded");
    assert_eq!(ours, theirs, "greedy continuation divergence");
    println!("[execution-parity] continuation: {ours:?}");
}

// ───────────────────────── S3 — tokenizer parity ────────────────────────────

#[given("the pinned model's published tokenizer.json")]
async fn given_tokenizer(w: &mut BddWorld) {
    w.tok_path = Some(locate_or_fetch_tokenizer().await);
}

#[when("the representative corpus is encoded by our tokenizer and the reference")]
async fn when_corpus_encoded(w: &mut BddWorld) {
    let path = w.tok_path.as_ref().expect("a tokenizer path");
    let native = NativeTokenizer::from_tokenizer_json(path).expect("loading our tokenizer");
    let reference =
        tokenizers::Tokenizer::from_file(path).expect("loading the reference tokenizer");
    w.tok_encoded = TOKENIZER_CORPUS
        .iter()
        .map(|&text| {
            let got = native.encode(text);
            let got = if native.bos_token_id() == got.first().copied() {
                got[1..].to_vec()
            } else {
                got
            };
            let want = reference
                .encode(text, false)
                .expect("reference encode")
                .get_ids()
                .to_vec();
            (text.to_string(), got, want)
        })
        .collect();
}

#[then("every corpus entry encodes to the reference token ids")]
async fn then_encode_matches(w: &mut BddWorld) {
    assert!(!w.tok_encoded.is_empty(), "nothing was encoded");
    let mismatches: Vec<&(String, Vec<u32>, Vec<u32>)> = w
        .tok_encoded
        .iter()
        .filter(|(_, got, want)| got != want)
        .collect();
    for (text, got, want) in &mismatches {
        eprintln!("tokenizer mismatch on {text:?}:\n  got:  {got:?}\n  want: {want:?}");
    }
    assert!(
        mismatches.is_empty(),
        "{}/{} corpus entries disagree with the HF reference",
        mismatches.len(),
        w.tok_encoded.len()
    );
}

#[when("the round-trippable corpus is encoded and decoded by our tokenizer")]
async fn when_corpus_round_tripped(w: &mut BddWorld) {
    let path = w.tok_path.as_ref().expect("a tokenizer path");
    let native = NativeTokenizer::from_tokenizer_json(path).expect("loading our tokenizer");
    w.tok_round = TOKENIZER_CORPUS
        .iter()
        .filter(|t| !t.starts_with(' ') && !t.ends_with(' '))
        .filter(|t| !t.contains('\n') && !t.contains('\t'))
        .map(|&text| {
            let ids = native.encode(text);
            (text.to_string(), native.decode(&ids))
        })
        .collect();
}

#[then("every entry round-trips to its input text")]
async fn then_round_trip(w: &mut BddWorld) {
    assert!(!w.tok_round.is_empty(), "nothing was round-tripped");
    for (text, back) in &w.tok_round {
        // BPE round-trips preserve text up to a single leading-space
        // normalization (the SentencePiece convention).
        let back_norm = back.strip_prefix(' ').unwrap_or(back);
        assert_eq!(
            back_norm, text,
            "round-trip diverged: {text:?} decoded back as {back:?}"
        );
    }
}

// ──────────────────── S3 — structural witness subprocesses ──────────────────

/// Per-witness wall-clock ceiling: covers a cold `--release` build of the
/// structural feature set plus the run, with generous slack for a loaded box.
const WITNESS_TIMEOUT: Duration = Duration::from_secs(3600);

const STRUCTURAL_WITNESSES: &[&str] = &[
    "structural_za",
    "structural_zm",
    "structural_ce",
    "structural_cf",
    "structural_lw",
    "structural_im",
];

#[when(expr = "the structural witness {string} runs in an isolated release process")]
async fn when_structural_witness(w: &mut BddWorld, name: String) {
    assert!(
        STRUCTURAL_WITNESSES.contains(&name.as_str()),
        "unknown structural witness `{name}`"
    );
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string());
    let mut cmd = tokio::process::Command::new(cargo);
    cmd.current_dir(root()).args([
        "test",
        "-p",
        "hologram-ai-conformance",
        "--release",
        "--features",
        "structural",
        "--test",
        &name,
        "--",
        "--nocapture",
    ]);
    let output = tokio::time::timeout(WITNESS_TIMEOUT, cmd.output())
        .await
        .unwrap_or_else(|_| {
            panic!(
                "structural witness `{name}` timed out after {}s",
                WITNESS_TIMEOUT.as_secs()
            )
        })
        .unwrap_or_else(|e| panic!("spawning cargo for `{name}`: {e}"));
    let combined = format!(
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    w.witness_run = Some((name, output.status.success(), combined));
}

#[then("the witness process exits green")]
async fn then_witness_green(w: &mut BddWorld) {
    let (name, green, output) = w.witness_run.as_ref().expect("the witness process ran");
    assert!(*green, "structural witness `{name}` failed:\n{output}");
    println!("[structural] {name}: green");
}

// ────────────────── S4 — generation loop / decode elision ───────────────────

#[given("a tiny compiled language model whose logits depend on the input tokens")]
async fn given_tiny_lm(w: &mut BddWorld) {
    w.lm = Some(tiny_lm());
}

#[when(expr = "greedy decoding runs for {int} steps twice from the same prompt")]
async fn when_greedy_twice(w: &mut BddWorld, steps: usize) {
    let lm = w.lm.as_ref().expect("the tiny LM");
    let (first, _) = greedy_decode(lm, &[1], steps);
    let (second, _) = greedy_decode(lm, &[1], steps);
    w.runs = vec![first, second];
}

#[then(expr = "each run emits {int} tokens")]
async fn then_runs_emit(w: &mut BddWorld, steps: usize) {
    assert!(!w.runs.is_empty(), "no runs recorded");
    for (i, run) in w.runs.iter().enumerate() {
        assert_eq!(run.len(), steps, "run {i} emitted {} tokens", run.len());
    }
}

#[then("both runs emit the identical token sequence")]
async fn then_runs_identical(w: &mut BddWorld) {
    assert_eq!(w.runs.len(), 2, "two runs are required");
    assert_eq!(
        w.runs[0], w.runs[1],
        "greedy decoding must be deterministic"
    );
}

#[then("the emitted tokens follow the model's deterministic successor table")]
async fn then_successor_sequence(w: &mut BddWorld) {
    let lm = w.lm.as_ref().expect("the tiny LM");
    let run = &w.runs[0];
    let mut prev = 1i64; // the prompt token
    for (i, &tok) in run.iter().enumerate() {
        let want = (prev + 1) % lm.vocab as i64;
        assert_eq!(
            tok,
            want,
            "step {}: expected successor {want}, got {tok}",
            i + 1
        );
        prev = tok;
    }
}

#[when(expr = "greedy decoding runs for {int} steps reporting dispatch counters")]
async fn when_greedy_with_counters(w: &mut BddWorld, steps: usize) {
    let lm = w.lm.as_ref().expect("the tiny LM");
    let (run, counters) = greedy_decode(lm, &[1], steps);
    w.runs = vec![run];
    w.counters = counters;
}

#[then("step 1 dispatches at least one kernel")]
async fn then_first_step_dispatches(w: &mut BddWorld) {
    let (dispatched, _) = *w.counters.first().expect("step counters recorded");
    assert!(
        dispatched > 0,
        "the first decode step must dispatch kernels"
    );
}

#[then("every step from 2 on reports skipped dispatches for the unchanged prefix")]
async fn then_later_steps_skip(w: &mut BddWorld) {
    assert!(
        w.counters.len() >= 2,
        "at least two decode steps are required"
    );
    for (i, (_, skipped)) in w.counters.iter().enumerate().skip(1) {
        assert!(
            *skipped > 0,
            "decode step {}: expected elided dispatches for the unchanged prefix, got 0",
            i + 1
        );
    }
}

#[then("the per-step dispatched and skipped counts are printed")]
async fn then_counters_printed(w: &mut BddWorld) {
    assert!(!w.counters.is_empty(), "no counters recorded");
    for (i, (dispatched, skipped)) in w.counters.iter().enumerate() {
        println!(
            "[decode-elision] step {}: dispatched {dispatched}, skipped {skipped}",
            i + 1
        );
    }
}

// ───────────────────── S4 — performance probe (target) ──────────────────────

#[when(expr = "{int} decode steps are timed")]
async fn when_decode_timed(w: &mut BddWorld, steps: usize) {
    let lm = w.lm.as_ref().expect("the tiny LM");
    let started = Instant::now();
    let (run, counters) = greedy_decode(lm, &[1], steps);
    let elapsed = started.elapsed();
    let tok_per_s = run.len() as f64 / elapsed.as_secs_f64();
    let (dispatched, skipped) = counters
        .iter()
        .fold((0usize, 0usize), |(d, s), (dd, ss)| (d + dd, s + ss));
    w.perf_report = Some(format!(
        "[performance] compile {:.1?}; {} steps in {:.1?} → {tok_per_s:.0} tok/s; \
         dispatched {dispatched}, skipped {skipped}",
        lm.compile_time,
        run.len(),
        elapsed
    ));
    w.counters = counters;
}

#[then("the compile time, tokens per second, and reuse counters are reported")]
async fn then_perf_reported(w: &mut BddWorld) {
    let report = w.perf_report.as_ref().expect("the probe measured");
    println!("{report}");
}

// ───────────────────────── S4 — app-domain events ───────────────────────────

#[given("an event stream covering registration, submission, start, completion, and failure")]
async fn given_full_stream(w: &mut BddWorld) {
    let stream = vec![
        ev_registered("bdd-model"),
        ev_submitted("a"),
        ev_started("a"),
        ev_completed("a"),
        ev_submitted("b"),
        ev_failed("b"),
    ];
    w.streams = vec![stream.clone(), stream];
}

#[given("two interleavings of the same events on two independent requests")]
async fn given_interleavings(w: &mut BddWorld) {
    let first = vec![
        ev_submitted("a"),
        ev_started("a"),
        ev_completed("a"),
        ev_submitted("b"),
        ev_failed("b"),
    ];
    let second = vec![
        ev_submitted("a"),
        ev_submitted("b"),
        ev_started("a"),
        ev_failed("b"),
        ev_completed("a"),
    ];
    w.streams = vec![first, second];
}

#[when("every stream is reduced")]
async fn when_streams_reduced(w: &mut BddWorld) {
    w.views = w.streams.iter().map(|s| reduce(s)).collect();
}

#[then("all reductions project the identical view")]
async fn then_views_identical(w: &mut BddWorld) {
    assert!(w.views.len() >= 2, "at least two reductions are required");
    for (i, view) in w.views.iter().enumerate().skip(1) {
        assert_eq!(&w.views[0], view, "reduction {i} diverged from the first");
    }
}

#[given("a stream where a request fails and then completes")]
async fn given_fail_then_complete(w: &mut BddWorld) {
    w.streams = vec![vec![ev_submitted("a"), ev_failed("a"), ev_completed("a")]];
}

#[when("the stream is reduced")]
async fn when_stream_reduced(w: &mut BddWorld) {
    w.views = w.streams.iter().map(|s| reduce(s)).collect();
}

#[then("the request is completed, not failed")]
async fn then_completed_wins(w: &mut BddWorld) {
    let view = w.views.first().expect("a reduced view");
    let key = kap("request-a");
    assert!(
        view.completed_jobs.contains_key(&key),
        "the later completion event must win"
    );
    assert!(
        view.failed_jobs.is_empty(),
        "the earlier failure must be superseded"
    );
}

#[given("a model manifest carrying a κ-label")]
async fn given_manifest_with_kappa(w: &mut BddWorld) {
    w.registered_manifest = Some(ModelManifest {
        model_kappa: kap("bdd-manifest-model"),
        archive_kappa: kap("bdd-manifest-archive"),
        name: "bdd-registered-model".to_string(),
        description: Some("κ-preservation witness".to_string()),
    });
}

#[when("the manifest is registered and the stream is reduced")]
async fn when_manifest_registered(w: &mut BddWorld) {
    let manifest = w.registered_manifest.clone().expect("a manifest fixture");
    let stream = vec![AiEvent::ModelRegistered {
        event_kappa: kap("event-register-bdd"),
        manifest,
    }];
    w.views = vec![reduce(&stream)];
    w.streams = vec![stream];
}

#[then("the view holds the manifest under its κ unchanged")]
async fn then_manifest_preserved(w: &mut BddWorld) {
    let manifest = w.registered_manifest.as_ref().expect("a manifest fixture");
    let view = w.views.first().expect("a reduced view");
    let projected = view
        .models
        .get(&manifest.model_kappa)
        .expect("the manifest is projected under its κ");
    assert_eq!(
        projected, manifest,
        "the projection must preserve the manifest"
    );
}

// ─────────────────────────── S4 — deployment gate ───────────────────────────

#[given("the Pages deployment workflow")]
async fn given_pages_workflow(w: &mut BddWorld) {
    let path = root().join(".github/workflows/pages.yml");
    w.workflow = Some(
        std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("reading {}: {e}", path.display())),
    );
}

#[then("the workflow triggers on pushes to the default branch")]
async fn then_workflow_triggers(w: &mut BddWorld) {
    let text = w.workflow.as_ref().expect("the workflow text");
    assert!(
        push_triggers_branch(text, "main"),
        "pages.yml must trigger on pushes to the default branch"
    );
}

#[then(expr = "the publish job requires the {string} job")]
async fn then_publish_needs(w: &mut BddWorld, job: String) {
    let text = w.workflow.as_ref().expect("the workflow text");
    let needs = publish_job_needs(text);
    assert!(
        needs.iter().any(|n| n == &job),
        "the Pages publish job must declare `needs: {job}` — its needs are {needs:?}"
    );
}

// ───────────────────── ORT-backed step bodies (gated) ────────────────────────

#[cfg(feature = "conformance")]
mod ort_gated {
    use super::*;
    use hologram_ai::commands::generate::{generate_stream, GenConfig};
    use hologram_ai::GrowableSession;
    use hologram_ai_conformance::ort_runner::fixtures;
    use hologram_ai_conformance::ort_runner::runner::{
        run_onnx_file_typed, run_onnx_typed, OrtInputTyped,
    };

    /// Backend dtype tags (`hologram_backend::cpu::dtype` encoding).
    const DTYPE_I64: u8 = 5;

    pub fn fixture_parity(w: &mut BddWorld) {
        let name = w.fixture.as_ref().expect("a fixture name");
        let bytes = fixtures::load_or_panic(name);
        let (seq, hidden) = (4usize, 32usize);
        let x: Vec<f32> = (0..seq * hidden)
            .map(|i| ((i * 7 % 13) as f32 - 6.0) * 0.1)
            .collect();

        let archive = ModelCompiler {
            seq_len_override: Some(seq as u64),
            ..Default::default()
        }
        .compile(ModelSource::OnnxBytes {
            model_bytes: bytes.clone(),
            external_data: None,
        })
        .expect("compiling the fixture");
        let mut runner = HoloRunner::from_bytes(archive.bytes).expect("loading the fixture");
        let outs = runner
            .execute(&[&f32s_le(&x)])
            .expect("executing the fixture");
        w.holo_out = le_f32(&outs[0].bytes);

        let ort_out = run_onnx_typed(
            &bytes,
            vec![OrtInputTyped::F32 {
                name: "X".into(),
                shape: vec![seq, hidden],
                data: x,
            }],
        )
        .expect("ORT executes the fixture");
        w.ort_out = ort_out
            .first()
            .expect("ORT produced an output")
            .data
            .clone();
    }

    /// Encode `prompt` with the model's own tokenizer.
    fn encode_prompt(tokenizer: &Path, prompt: &str) -> (NativeTokenizer, Vec<i64>) {
        let tok = NativeTokenizer::from_tokenizer_json(tokenizer).expect("loading the tokenizer");
        let ids: Vec<i64> = tok.encode(prompt).iter().map(|&t| t as i64).collect();
        assert!(!ids.is_empty(), "the prompt must encode to tokens");
        (tok, ids)
    }

    /// Build the ORT input set mirroring the compiled graph's own ports:
    /// integer ports carry the token window / mask / positions, `past.*`
    /// ports carry empty tensors at their compiled (zero-length) shapes.
    fn ort_inputs_for(runner: &HoloRunner, ids: &[i64]) -> Vec<OrtInputTyped> {
        let seq = ids.len();
        runner
            .input_port_info()
            .iter()
            .map(|p| match p.name.as_str() {
                "input_ids" => OrtInputTyped::I64 {
                    name: p.name.clone(),
                    shape: vec![1, seq],
                    data: ids.to_vec(),
                },
                "attention_mask" => OrtInputTyped::I64 {
                    name: p.name.clone(),
                    shape: vec![1, seq],
                    data: vec![1; seq],
                },
                "position_ids" => OrtInputTyped::I64 {
                    name: p.name.clone(),
                    shape: vec![1, seq],
                    data: (0..seq as i64).collect(),
                },
                n if n.starts_with("past_key_values") || n.starts_with("past.") => {
                    assert!(
                        !p.shape.is_empty(),
                        "past port `{n}` carries no compiled shape"
                    );
                    OrtInputTyped::F32 {
                        name: p.name.clone(),
                        shape: p.shape.clone(),
                        data: Vec::new(),
                    }
                }
                other => panic!("unexpected model input port `{other}`"),
            })
            .collect()
    }

    /// One ORT forward pass; returns the last-position logits row.
    fn ort_last_logits(onnx: &Path, runner: &HoloRunner, ids: &[i64]) -> Vec<f32> {
        let outs = run_onnx_file_typed(onnx, ort_inputs_for(runner, ids))
            .expect("ORT executes the pinned model");
        let logits = outs
            .iter()
            .find(|t| t.name == "logits")
            .expect("ORT exposes a `logits` output");
        let vocab = *logits.shape.last().expect("logits carry a vocab dim");
        let last = logits.data.len() - vocab;
        logits.data[last..].to_vec()
    }

    pub fn smollm2_prefill_parity(w: &mut BddWorld, prompt: &str) {
        let (onnx, tokenizer) = w.smollm2.clone().expect("the pinned model paths");
        let (_tok, ids) = encode_prompt(&tokenizer, prompt);
        let seq = ids.len();

        let archive = ModelCompiler {
            seq_len_override: Some(seq as u64),
            ..Default::default()
        }
        .compile(ModelSource::OnnxPath(onnx.clone()))
        .expect("compiling the pinned model");
        let mut runner = HoloRunner::from_bytes(archive.bytes).expect("loading the pinned model");

        // hologram: fill every port by name (token window + empty past).
        let ports = runner.input_port_info();
        let sizes = runner.input_byte_sizes();
        let bufs: Vec<Vec<u8>> = ports
            .iter()
            .zip(sizes.iter())
            .map(|(p, &size)| {
                let buf = match p.name.as_str() {
                    "input_ids" => i64s_le(&ids),
                    "attention_mask" => {
                        assert_eq!(p.dtype, DTYPE_I64, "attention_mask dtype");
                        i64s_le(&vec![1; seq])
                    }
                    "position_ids" => i64s_le(&(0..seq as i64).collect::<Vec<_>>()),
                    n if n.starts_with("past_key_values") || n.starts_with("past.") => Vec::new(),
                    other => panic!("unexpected model input port `{other}`"),
                };
                assert_eq!(buf.len(), size, "port `{}` byte size", p.name);
                buf
            })
            .collect();
        let refs: Vec<&[u8]> = bufs.iter().map(Vec::as_slice).collect();
        let outs = runner.execute(&refs).expect("the prefill executes");
        let logits_idx = runner
            .output_index_by_name("logits")
            .expect("a `logits` output port");
        let holo = le_f32(&outs[logits_idx].bytes);
        let vocab = holo.len() / seq;
        let holo_last = holo[(seq - 1) * vocab..].to_vec();

        let ort_last = ort_last_logits(&onnx, &runner, &ids);
        w.parity_next = Some((argmax(&holo_last), argmax(&ort_last)));
        w.parity_logits = Some((holo_last, ort_last));
    }

    pub fn smollm2_greedy_parity(w: &mut BddWorld, prompt: &str, n: usize) {
        let (onnx, tokenizer) = w.smollm2.clone().expect("the pinned model paths");
        let (tok, prompt_ids) = encode_prompt(&tokenizer, prompt);

        // hologram: the production generation loop, greedy.
        let prepared = ModelCompiler::default()
            .prepare(ModelSource::OnnxPath(onnx.clone()))
            .expect("preparing the pinned model");
        let mut provider = GrowableSession::new(prepared);
        let cfg = GenConfig {
            max_tokens: n,
            temperature: 0.0,
            ..Default::default()
        };
        let mut sink = Vec::new();
        let ours = generate_stream(&mut provider, &tok, prompt, &cfg, &mut sink)
            .expect("hologram generates");

        // ORT: an explicit greedy full-recompute loop over the same export.
        // Port names/shapes are snapshotted from one compiled window.
        let snapshot = ModelCompiler {
            seq_len_override: Some(prompt_ids.len() as u64),
            ..Default::default()
        }
        .compile(ModelSource::OnnxPath(onnx.clone()))
        .expect("compiling the port snapshot");
        let runner = HoloRunner::from_bytes(snapshot.bytes).expect("loading the port snapshot");
        let mut ids = prompt_ids;
        let mut generated: Vec<u32> = Vec::new();
        let eos = tok.eos_token_id();
        for _ in 0..n {
            let last = ort_last_logits(&onnx, &runner, &ids);
            let next = argmax(&last) as u32;
            if next == eos {
                break;
            }
            generated.push(next);
            ids.push(next as i64);
        }
        let theirs = tok.decode(&generated);
        w.continuations = Some((ours, theirs));
    }
}

// ────────────────────────────── the runner ──────────────────────────────────

/// The `@lane:` tag of a feature file's tag line.
fn lane_of(feature_path: &Path) -> String {
    let text = std::fs::read_to_string(feature_path)
        .unwrap_or_else(|e| panic!("reading {}: {e}", feature_path.display()));
    text.lines()
        .flat_map(|l| l.split_whitespace())
        .find_map(|tag| tag.strip_prefix("@lane:"))
        .unwrap_or_else(|| {
            panic!(
                "{} carries no @lane: tag — every rust feature declares its lane",
                feature_path.display()
            )
        })
        .to_string()
}

fn canonical(p: &Path) -> PathBuf {
    p.canonicalize().unwrap_or_else(|_| p.to_path_buf())
}

#[tokio::main]
async fn main() {
    let lane = std::env::var("HOLOGRAM_AI_BDD_LANE").unwrap_or_else(|_| "default".to_string());
    let root = root();
    let model = Model::load().expect("the conceptual model must load and validate");

    // Model-driven selection: rust rows, by tier and by the feature's own
    // declared lane. `target` runs the measured probes instead of the suites.
    let selected: BTreeSet<PathBuf> = model
        .rows
        .iter()
        .filter(|r| r.executor == Executor::Rust)
        .filter(|r| {
            let path = root.join(&r.feature);
            match lane.as_str() {
                "target" => r.tier == Tier::Target,
                lane => r.tier == Tier::Suite && lane_of(&path) == lane,
            }
        })
        .map(|r| canonical(&root.join(&r.feature)))
        .collect();
    assert!(
        !selected.is_empty(),
        "no rust features select lane `{lane}` — valid lanes: default, ort, model, target"
    );
    println!(
        "bdd lane `{lane}`: running {} feature(s):{}",
        selected.len(),
        selected
            .iter()
            .map(|p| format!("\n  {}", p.display()))
            .collect::<String>()
    );

    let admitted: Arc<Mutex<BTreeSet<PathBuf>>> = Arc::new(Mutex::new(BTreeSet::new()));
    let admitted_in_filter = Arc::clone(&admitted);
    let filter_selected = selected.clone();

    let writer = BddWorld::cucumber()
        .fail_on_skipped()
        // The runner is env-driven (HOLOGRAM_AI_BDD_LANE); an explicit default
        // CLI keeps stray harness flags (e.g. a `--nocapture` passed through
        // `cargo test -- --nocapture`) from failing cucumber's arg parser.
        .with_cli(cucumber::cli::Opts::<_, _, _, cucumber::cli::Empty>::default())
        .filter_run(root.join("features/suites"), move |feature, _, _| {
            let Some(path) = feature.path.as_ref() else {
                return false;
            };
            let path = canonical(path);
            if filter_selected.contains(&path) {
                admitted_in_filter
                    .lock()
                    .expect("the admitted set is never poisoned")
                    .insert(path);
                true
            } else {
                false
            }
        })
        .await;

    // The gate: no failures, no skipped steps, no undefined steps (undefined
    // steps surface as skipped and fail_on_skipped converts them to failures),
    // no parsing errors, and every selected feature actually ran.
    use cucumber::writer::Stats as _;
    let (passed, skipped, failed, parsing, hooks) = (
        writer.passed_steps(),
        writer.skipped_steps(),
        writer.failed_steps(),
        writer.parsing_errors(),
        writer.hook_errors(),
    );
    assert_eq!(parsing, 0, "gherkin parsing errors in features/suites");
    assert_eq!(hooks, 0, "scenario hook errors");
    assert_eq!(failed, 0, "{failed} step(s) failed in lane `{lane}`");
    assert_eq!(
        skipped, 0,
        "{skipped} step(s) skipped in lane `{lane}` — a gating suite defers no work"
    );
    assert!(passed > 0, "lane `{lane}` ran no steps");

    let ran = admitted
        .lock()
        .expect("the admitted set is never poisoned")
        .clone();
    let missing: Vec<&PathBuf> = selected.difference(&ran).collect();
    assert!(
        missing.is_empty(),
        "selected features never ran: {missing:?}"
    );
    println!(
        "bdd lane `{lane}`: {} feature(s), {passed} step(s) passed, 0 skipped, 0 failed",
        selected.len()
    );
}
