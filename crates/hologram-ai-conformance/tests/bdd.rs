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
use hologram_ai::staged::StagedRunner;
use hologram_ai::{LmSession, ModelCompiler, ModelSource};
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
    build_parametric_graph_from_manifest, build_parametric_stage_graphs, selected_family,
    supported_families,
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
        range: None,
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

/// A (tensor name, shape) k-form manifest for the deterministic-compile witness.
type DetManifest = Vec<(String, Vec<u64>)>;

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
    // S2 — deterministic compile (row `deterministic-compile`)
    det_fixture: Option<(serde_json::Value, DetManifest)>,
    det_mono: Option<Vec<String>>,
    det_staged: Option<Vec<Vec<String>>>,
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
    // S3 — staged (windowed) execution
    staged_store: Option<StoreDir>,
    staged_partition: Option<(BTreeSet<String>, Vec<BTreeSet<String>>)>,
    staged_exec: Option<StagedExec>,
    staged_completions: Option<(String, String)>,
    // S3 — staged window growth (row `staged-window-growth`)
    staged_windows: Option<Vec<usize>>,
    staged_growth_err: Option<String>,
    // S3 — stage residency cache (row `stage-residency-cache`)
    residency_run: Option<ResidencyRun>,
    // S3 — total algebraic path (row `total-algebraic-path`, open/measured)
    float_dispatch_tally: Option<(usize, usize)>,
    // S3 — session verified-κ (row `session-verified-kappa`)
    verified_second_pass: Option<anyhow::Result<Vec<u8>>>,
    verified_fresh_err: Option<String>,
    verified_corrupt_kappa: Option<String>,
    // S3 — chunked head (row `chunked-head`)
    chunked: Option<ChunkedKit>,
    chunked_logits: Option<(Vec<u8>, Vec<u8>)>,
    chunked_completion: Option<(String, u64)>,
    chunked_traffic: Option<ChunkedTraffic>,
    // S4 — performance contract (row `performance-contract`)
    perf: Option<PerfReport>,
    perf_kit: Option<PerfKit>,
    // S3 — quantized transit (row `quantized-transit`)
    quant: Option<QuantKit>,
    quant_derive: Option<(usize, u64, u64)>,
    quant_completions: Option<(String, String)>,
    quant_evicted: Option<(String, TrafficJournal)>,
    // S3 — idle derivation (row `idle-derivation`)
    idle_state: Option<IdleDeriveState>,
    // S3 — admission margin (row `stage-residency-cache`)
    admission_margins: Option<Vec<u64>>,
    admission_refused_materializations: Option<(u64, usize, usize)>,
    // S3 — derived-artifact closure (row `derived-artifact-kappa`)
    derived_completions: Option<(String, String)>,
    derived_hits: Option<u64>,
    derived_root: Option<StoreDir>,
    // S3 — saturation residency (row `saturation-residency`)
    unpin_result: Option<anyhow::Result<Vec<u8>>>,
    unpin_corrupt_kappa: Option<String>,
    unpin_store: Option<StoreDir>,
    unpin_other_entries: Option<usize>,
    // S3 — bounded embedding / fused Phi3 (row `bounded-embedding`)
    bounded_embed: Option<BoundedEmbed>,
    bounded_embed_compiled: Option<Vec<Vec<u8>>>,
    fused_phi3_store: Option<StoreDir>,
    fused_phi3_exec: Option<(Vec<u8>, Vec<u8>)>,
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

// ────────────────── S2 — deterministic compilation ──────────────────────────

/// How many times the deterministic-compile witness recompiles the fixture.
/// Content addressing demands one stable κ; a HashMap-seeded emission order
/// would surface as a divergent κ within a handful of repeats.
const DET_COMPILE_RUNS: usize = 8;

/// Bind each manifest tensor to a content-verified External κ (the κ of its own
/// name — arbitrary but reproducible), the weightless k-form the downloader
/// persists. Only names the graph declares are bound.
fn bind_external_kappas(graph: &mut AiGraph, manifest: &[(String, Vec<u64>)]) {
    let name_to_id: HashMap<String, u32> = graph
        .tensor_names
        .iter()
        .map(|(id, name)| (name.clone(), *id))
        .collect();
    for (name, dims) in manifest {
        let Some(&id) = name_to_id.get(name) else {
            continue;
        };
        let info = TensorInfo::new(DType::F32, shape_from_concrete(dims));
        graph.tensor_info.insert(id, info.clone());
        graph.params.insert(
            id,
            AiParam::External {
                kappa: kappa_of(name.as_bytes()),
                info,
                range: None,
            },
        );
    }
}

#[given("the handshake-tiny config and its Llama k-form manifest")]
async fn given_det_fixture(w: &mut BddWorld) {
    let (family, e) = handshake_expects();
    let config = handshake_config(&family, &e, false);
    let manifest = decoder_manifest(&e, false, false);
    w.det_fixture = Some((config, manifest));
}

fn det_fixture(w: &BddWorld) -> &(serde_json::Value, DetManifest) {
    w.det_fixture
        .as_ref()
        .expect("a deterministic-compile fixture")
}

#[when("the manifest is compiled to a monolithic k-form archive several times")]
async fn when_det_monolithic(w: &mut BddWorld) {
    let (config, manifest) = det_fixture(w);
    let keys: Vec<String> = manifest.iter().map(|(n, _)| n.clone()).collect();
    let dtypes = vec![DType::F32; keys.len()];
    let kappas = (0..DET_COMPILE_RUNS)
        .map(|_| {
            let mut graph = build_parametric_graph_from_manifest(config, &keys, &dtypes, None)
                .expect("the k-form graph builds from config + manifest");
            bind_external_kappas(&mut graph, manifest);
            let archive = ModelCompiler::default()
                .compile(ModelSource::AiGraph(graph))
                .expect("the k-form graph compiles");
            kappa_of(&archive.bytes)
        })
        .collect();
    w.det_mono = Some(kappas);
}

#[when("the manifest is compiled to staged k-form archives several times")]
async fn when_det_staged(w: &mut BddWorld) {
    let (config, manifest) = det_fixture(w);
    let keys: Vec<String> = manifest.iter().map(|(n, _)| n.clone()).collect();
    let kappas: Vec<String> = keys.iter().map(|n| kappa_of(n.as_bytes())).collect();
    let shapes: Vec<Vec<u64>> = manifest.iter().map(|(_, d)| d.clone()).collect();
    let dtypes = vec![DType::F32; keys.len()];
    let config_json = config.to_string();
    let per_stage = std::num::NonZeroU64::new(1).expect("one layer per stage is non-zero");
    let runs = (0..DET_COMPILE_RUNS)
        .map(|_| {
            let stages = hologram_ai::staged::compile_stages(
                &config_json,
                &keys,
                &kappas,
                &shapes,
                &dtypes,
                None,
                per_stage,
            )
            .expect("the staged k-form partition compiles");
            stages.iter().map(|s| kappa_of(s)).collect::<Vec<String>>()
        })
        .collect();
    w.det_staged = Some(runs);
}

#[then("every monolithic archive is byte-identical, a single stable κ")]
async fn then_det_monolithic_stable(w: &mut BddWorld) {
    let kappas = w.det_mono.as_ref().expect("the monolithic compiles ran");
    let distinct: BTreeSet<&String> = kappas.iter().collect();
    assert_eq!(
        distinct.len(),
        1,
        "the monolithic k-form compile is nondeterministic: {} distinct κ across {} runs: {:?}",
        distinct.len(),
        kappas.len(),
        distinct
    );
}

#[then("every stage archive is byte-identical to its first compile, a single stable κ per stage")]
async fn then_det_staged_stable(w: &mut BddWorld) {
    let runs = w.det_staged.as_ref().expect("the staged compiles ran");
    let first = runs.first().expect("at least one staged run");
    assert!(
        first.len() >= 2,
        "a staged partition has at least an embedding and a head stage, got {}",
        first.len()
    );
    for (r, run) in runs.iter().enumerate() {
        assert_eq!(
            run, first,
            "staged run {r} diverges from the first — the staged k-form compile is nondeterministic"
        );
    }
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

// ───────────────── S3 — staged (windowed) execution over k ──────────────────

/// Quantities of the deterministic tiny decoder fixture — the
/// `hologram-ai/tests/parametric_reference.rs` weights pattern (norm weights
/// 1.0, everything else cycling `((k % 13) - 6) · 0.01`).
const STG_HIDDEN: u64 = 64;
const STG_LAYERS: u64 = 2;
const STG_KV_DIM: u64 = 32; // 2 KV heads × head_dim 16
const STG_VOCAB: u64 = 512;
const STG_INTER: u64 = 128;
const STG_WINDOW: u64 = 128;
const STG_TOKENS: [i64; 6] = [3, 141, 59, 26, 5, 35];

fn staged_config(tied: bool) -> serde_json::Value {
    serde_json::json!({
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": STG_HIDDEN, "intermediate_size": STG_INTER,
        "num_hidden_layers": STG_LAYERS, "num_attention_heads": 4,
        "num_key_value_heads": 2, "vocab_size": STG_VOCAB,
        "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
        "max_position_embeddings": STG_WINDOW,
        "tie_word_embeddings": tied,
    })
}

fn staged_manifest(tied: bool) -> Vec<(String, Vec<u64>)> {
    let (h, i, v, kv) = (STG_HIDDEN, STG_INTER, STG_VOCAB, STG_KV_DIM);
    let mut m: Vec<(String, Vec<u64>)> = vec![("model.embed_tokens.weight".into(), vec![v, h])];
    for l in 0..STG_LAYERS {
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
    if !tied {
        m.push(("lm_head.weight".into(), vec![v, h]));
    }
    m
}

fn staged_tensor_bytes(name: &str, dims: &[u64]) -> Vec<u8> {
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

/// The compiled staged-execution kit: the fixture manifest with its κs, the
/// monolithic k-form archive, and the one-layer-per-stage k-form archives.
/// Deterministic (κs are content addresses of deterministic bytes), so it is
/// compiled once and shared across scenarios.
struct StagedKit {
    keys: Vec<String>,
    kappas: Vec<String>,
    monolithic: Vec<u8>,
    stages: Vec<Vec<u8>>,
    total_weight_bytes: u64,
}

fn build_staged_kit(tied: bool) -> StagedKit {
    let manifest = staged_manifest(tied);
    let (keys, shapes): (Vec<String>, Vec<Vec<u64>>) = manifest.into_iter().unzip();
    let kappas: Vec<String> = keys
        .iter()
        .zip(&shapes)
        .map(|(name, dims)| kappa_of(&staged_tensor_bytes(name, dims)))
        .collect();
    let dtypes = vec![DType::F32; keys.len()];
    let config = staged_config(tied);
    let total_weight_bytes = keys
        .iter()
        .zip(&shapes)
        .map(|(name, dims)| staged_tensor_bytes(name, dims).len() as u64)
        .sum();

    // Monolithic k-form: the parametric graph with External κ params — the
    // same binding the streamed compile performs.
    let mut graph = build_parametric_graph_from_manifest(&config, &keys, &dtypes, None)
        .expect("the monolithic fixture graph builds");
    let name_to_id: HashMap<String, u32> = graph
        .tensor_names
        .iter()
        .map(|(id, name)| (name.clone(), *id))
        .collect();
    for (i, key) in keys.iter().enumerate() {
        let id = *name_to_id.get(key).expect("manifest tensor in the graph");
        let info = ti(DType::F32, &shapes[i]);
        graph.tensor_info.insert(id, info.clone());
        graph.params.insert(
            id,
            AiParam::External {
                kappa: kappas[i].clone(),
                info,
                range: None,
            },
        );
    }
    let monolithic = compile_graph(graph);

    let stages = hologram_ai::staged::compile_stages(
        &config.to_string(),
        &keys,
        &kappas,
        &shapes,
        &dtypes,
        None,
        std::num::NonZeroU64::new(1).expect("1 is non-zero"),
    )
    .expect("the staged fixture compiles");

    StagedKit {
        keys,
        kappas,
        monolithic,
        stages,
        total_weight_bytes,
    }
}

fn staged_kit() -> &'static StagedKit {
    static KIT: OnceLock<StagedKit> = OnceLock::new();
    KIT.get_or_init(|| build_staged_kit(false))
}

fn staged_tied_kit() -> &'static StagedKit {
    static KIT: OnceLock<StagedKit> = OnceLock::new();
    KIT.get_or_init(|| build_staged_kit(true))
}

/// The κ set a k-form archive requires.
fn kappa_set(archive: &[u8]) -> BTreeSet<String> {
    kappa_requirements(archive)
        .expect("the κ-map parses")
        .into_iter()
        .map(|r| r.kappa)
        .collect()
}

/// The stage indices expected to consume a fixture tensor, by name, in the
/// one-layer-per-stage partition: embedding → stage 0; layer `l` → stage
/// `1 + l`; final norm and head → the head stage. A tied embedding is
/// consumed by BOTH stage 0 and the head stage — one κ-store blob bound by
/// two stage κ-maps (k-form sharing, not duplication).
fn expected_stages_for(name: &str, tied: bool, stage_count: usize) -> BTreeSet<usize> {
    let head = stage_count - 1;
    if name == "model.embed_tokens.weight" {
        if tied {
            return BTreeSet::from([0, head]);
        }
        return BTreeSet::from([0]);
    }
    if name == "model.norm.weight" || name == "lm_head.weight" {
        return BTreeSet::from([head]);
    }
    let l = extract_layer_idx(name).expect("a layer tensor names its layer");
    BTreeSet::from([1 + l])
}

fn extract_layer_idx(key: &str) -> Option<usize> {
    let mut parts = key.split('.');
    parts
        .by_ref()
        .find(|p| *p == "layers")
        .and_then(|_| parts.next())
        .and_then(|p| p.parse().ok())
}

/// The (monolithic logits, staged logits, per-stage weight bytes, peak) of
/// one token-window execution.
struct StagedExec {
    mono_logits: Vec<u8>,
    staged_logits: Vec<u8>,
    stage_weight_bytes: Vec<u64>,
    peak_weight_bytes: u64,
}

/// A fresh κ-store directory holding the fixture weights.
fn staged_store(tag: &str) -> StoreDir {
    let store = StoreDir::new(tag);
    let dir = DirKappaStore::new(&store.path);
    for (name, dims) in staged_manifest(false) {
        dir.insert(&staged_tensor_bytes(&name, &dims))
            .expect("persisting a fixture weight");
    }
    store
}

/// The fixed-window `input_ids` buffer: the fixture tokens left-aligned,
/// zero-padded to the compiled window (causal attention makes the padding
/// irrelevant at real positions).
fn staged_window_ids() -> Vec<u8> {
    let mut ids = vec![0i64; STG_WINDOW as usize];
    ids[..STG_TOKENS.len()].copy_from_slice(&STG_TOKENS);
    i64s_le(&ids)
}

/// Base-10 integer tokenizer over the fixture vocabulary — lets the
/// generation scenario drive `generate_stream` deterministically with no
/// tokenizer files (the `generation_synthetic` pattern).
struct DecimalTok;

impl Tokenizer for DecimalTok {
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
        STG_VOCAB as u32 + 1 // out of vocabulary — never sampled
    }
    fn bos_token_id(&self) -> Option<u32> {
        None
    }
    fn vocab_size(&self) -> usize {
        STG_VOCAB as usize
    }
    fn id_to_token(&self, _id: u32) -> Option<&str> {
        None
    }
    fn token_to_id(&self, _token: &str) -> Option<u32> {
        None
    }
}

#[given("the deterministic tiny decoder fixture with its weights in a κ-store")]
async fn given_staged_fixture(w: &mut BddWorld) {
    w.staged_store = Some(staged_store("staged"));
}

#[when("the fixture is compiled monolithically and as one-layer stages")]
async fn when_staged_compiled(w: &mut BddWorld) {
    let kit = staged_kit();
    w.staged_partition = Some((
        kappa_set(&kit.monolithic),
        kit.stages.iter().map(|s| kappa_set(s)).collect(),
    ));
}

#[then("the union of the stage κ-maps equals the monolithic κ-map's tensor set")]
async fn then_staged_partition_covers(w: &mut BddWorld) {
    let (mono, stages) = w.staged_partition.as_ref().expect("the compiled partition");
    let union: BTreeSet<String> = stages.iter().flatten().cloned().collect();
    assert_eq!(
        &union, mono,
        "the stage κ-maps must cover exactly the monolithic κ-map's tensor set"
    );
    assert!(
        stages.iter().all(|s| !s.is_empty()),
        "every stage binds at least one weight"
    );
    println!(
        "[staged-execution] partition: {} stages, {} κs total (monolithic {})",
        stages.len(),
        union.len(),
        mono.len()
    );
}

#[then("each weight κ appears in exactly the stages that consume it")]
async fn then_staged_partition_exact(w: &mut BddWorld) {
    let (_, stages) = w.staged_partition.as_ref().expect("the compiled partition");
    let kit = staged_kit();
    assert_stage_bindings_exact(&kit.keys, &kit.kappas, stages, false);
}

/// Assert that every fixture κ is bound by exactly the stages that consume
/// it. Expectations aggregate **per κ**, not per name: κ is a CONTENT
/// address, so distinct tensors with identical bytes (the fixture's
/// embedding and lm_head share the deterministic pattern) collapse to one
/// κ-store blob bound by every stage consuming any of those tensors — the
/// same k-form sharing the tied-embedding assertion documents.
fn assert_stage_bindings_exact(
    keys: &[String],
    kappas: &[String],
    stages: &[BTreeSet<String>],
    tied: bool,
) {
    let mut expected: HashMap<&str, (Vec<&str>, BTreeSet<usize>)> = HashMap::new();
    for (name, kappa) in keys.iter().zip(kappas) {
        let entry = expected.entry(kappa).or_default();
        entry.0.push(name);
        entry
            .1
            .extend(expected_stages_for(name, tied, stages.len()));
    }
    for (kappa, (names, want)) in &expected {
        let got: BTreeSet<usize> = stages
            .iter()
            .enumerate()
            .filter(|(_, set)| set.contains(*kappa))
            .map(|(i, _)| i)
            .collect();
        assert_eq!(
            &got, want,
            "{names:?} must be bound by exactly the stages that consume them"
        );
        if names.len() > 1 {
            println!(
                "[staged-execution] {names:?} share one κ (content dedup) bound by stages {got:?}"
            );
        }
    }
}

#[then("a tied fixture shares the embedding κ between the embedding and head stages")]
async fn then_staged_tied_shares_embedding(_w: &mut BddWorld) {
    let kit = staged_tied_kit();
    let stages: Vec<BTreeSet<String>> = kit.stages.iter().map(|s| kappa_set(s)).collect();
    let union: BTreeSet<String> = stages.iter().flatten().cloned().collect();
    assert_eq!(
        union,
        kappa_set(&kit.monolithic),
        "the tied partition must still cover the monolithic κ-map exactly"
    );
    assert_stage_bindings_exact(&kit.keys, &kit.kappas, &stages, true);
    let embed_kappa = &kit.kappas[0];
    assert!(
        stages[0].contains(embed_kappa) && stages[stages.len() - 1].contains(embed_kappa),
        "the tied embedding κ must be bound by both the embedding and head stages"
    );
    println!(
        "[staged-execution] tied embedding κ shared by stages 0 and {}: one κ-store blob, two bindings",
        stages.len() - 1
    );
}

#[when("the same token window is executed monolithically and through the staged runner")]
async fn when_staged_executed(w: &mut BddWorld) {
    let kit = staged_kit();
    let store = w.staged_store.as_ref().expect("the fixture κ-store");
    let ids = staged_window_ids();

    let mut dir = DirKappaStore::new(&store.path);
    let mono = materialize_archive(&kit.monolithic, &mut dir)
        .expect("the monolithic archive materializes");
    let mut runner = HoloRunner::from_bytes(mono).expect("the monolithic archive loads");
    let mono_out = runner
        .execute(&[&ids])
        .expect("the monolithic pass executes");
    let mono_logits = mono_out
        .into_iter()
        .next()
        .expect("the monolithic pass produces logits")
        .bytes;

    let mut staged = StagedRunner::from_archives(
        kit.stages.clone(),
        Box::new(DirKappaStore::new(&store.path)),
    )
    .expect("the staged runner builds");
    let staged_out = LmSession::execute(&mut staged, &[&ids]).expect("the staged pass executes");
    let staged_logits = staged_out
        .into_iter()
        .next()
        .expect("the staged pass produces logits")
        .bytes;

    w.staged_exec = Some(StagedExec {
        mono_logits,
        staged_logits,
        stage_weight_bytes: staged.stage_weight_bytes().to_vec(),
        peak_weight_bytes: staged.peak_resident_weight_bytes(),
    });
}

#[then("the staged logits are byte-identical to the monolithic logits")]
async fn then_staged_logits_equal(w: &mut BddWorld) {
    let exec = w.staged_exec.as_ref().expect("both executions ran");
    assert_eq!(
        exec.mono_logits.len(),
        exec.staged_logits.len(),
        "logits sizes must agree"
    );
    assert_eq!(
        exec.mono_logits, exec.staged_logits,
        "staged execution must reproduce the monolithic logits byte-for-byte"
    );
    let logits = le_f32(&exec.staged_logits);
    assert!(
        logits.iter().all(|v| v.is_finite()) && logits.iter().any(|v| v.abs() > 1e-6),
        "the logits must be finite and non-trivial"
    );
    println!(
        "[staged-execution] byte-identical logits: {} bytes ({} f32 elements)",
        exec.staged_logits.len(),
        logits.len()
    );
}

#[then("the peak resident weight bytes are at most the largest stage's weight bytes")]
async fn then_staged_peak_bounded(w: &mut BddWorld) {
    let exec = w.staged_exec.as_ref().expect("the staged execution ran");
    assert!(
        exec.stage_weight_bytes.iter().all(|&b| b > 0),
        "every stage materialized real weights: {:?}",
        exec.stage_weight_bytes
    );
    let largest = exec
        .stage_weight_bytes
        .iter()
        .copied()
        .max()
        .expect("at least one stage");
    assert!(
        exec.peak_weight_bytes <= largest,
        "peak resident weight bytes {} must not exceed the largest stage's {largest}",
        exec.peak_weight_bytes
    );
    println!(
        "[staged-execution] stage weight bytes {:?}, peak {} ≤ largest stage {largest}",
        exec.stage_weight_bytes, exec.peak_weight_bytes
    );
}

#[then("the largest stage's weight bytes are strictly less than the model's total weight bytes")]
async fn then_staged_window_below_model(w: &mut BddWorld) {
    let exec = w.staged_exec.as_ref().expect("the staged execution ran");
    let kit = staged_kit();
    assert!(
        exec.stage_weight_bytes.len() > 1,
        "the bound is meaningful only for a multi-stage split"
    );
    let largest = exec
        .stage_weight_bytes
        .iter()
        .copied()
        .max()
        .expect("at least one stage");
    assert!(
        largest < kit.total_weight_bytes,
        "the window ({largest} bytes) must be strictly below the model ({} bytes)",
        kit.total_weight_bytes
    );
    println!(
        "[staged-execution] window {largest} bytes < model {} bytes ({} stages)",
        kit.total_weight_bytes,
        exec.stage_weight_bytes.len()
    );
}

#[when(
    "the same greedy completion is generated through the staged runner and the monolithic session"
)]
async fn when_staged_generated(w: &mut BddWorld) {
    use hologram_ai::commands::generate::{generate_stream, GenConfig};

    let kit = staged_kit();
    let store = w.staged_store.as_ref().expect("the fixture κ-store");
    let prompt = "3 141 59 26 5";
    let cfg = GenConfig {
        max_tokens: Some(8),
        temperature: 0.0,
        ..Default::default()
    };

    let mut dir = DirKappaStore::new(&store.path);
    let mono = materialize_archive(&kit.monolithic, &mut dir)
        .expect("the monolithic archive materializes");
    let mut fixed =
        hologram_ai::FixedSession::new(HoloRunner::from_bytes(mono).expect("the archive loads"));
    let mut sink = Vec::new();
    let mono_text = generate_stream(&mut fixed, &DecimalTok, prompt, &cfg, &mut sink)
        .expect("monolithic generation completes");

    let mut staged = StagedRunner::from_archives(
        kit.stages.clone(),
        Box::new(DirKappaStore::new(&store.path)),
    )
    .expect("the staged runner builds");
    let mut sink = Vec::new();
    let staged_text = generate_stream(&mut staged, &DecimalTok, prompt, &cfg, &mut sink)
        .expect("staged generation completes");

    w.staged_completions = Some((mono_text, staged_text));
}

#[then("both completions are identical and non-empty")]
async fn then_staged_completions_equal(w: &mut BddWorld) {
    let (mono, staged) = w.staged_completions.as_ref().expect("both generations ran");
    assert!(
        !mono.is_empty(),
        "the monolithic completion must be non-empty"
    );
    assert_eq!(
        mono, staged,
        "the staged completion must equal the monolithic completion"
    );
    println!("[staged-execution] greedy completion (both engines): {mono:?}");
}

// ─────────── S3 — staged window growth (row `staged-window-growth`) ──────────

/// A growable staged session over the fixture manifest + κ-store, with a
/// window observer recording every bucket it compiles.
fn growable_staged_session(
    store_path: &std::path::Path,
    windows: std::sync::Arc<std::sync::Mutex<Vec<usize>>>,
) -> hologram_ai::staged::GrowableStagedSession {
    let manifest = staged_manifest(false);
    let (keys, shapes): (Vec<String>, Vec<Vec<u64>>) = manifest.into_iter().unzip();
    let kappas: Vec<String> = keys
        .iter()
        .zip(&shapes)
        .map(|(name, dims)| kappa_of(&staged_tensor_bytes(name, dims)))
        .collect();
    let dtypes = vec![DType::F32; keys.len()];
    let mut session = hologram_ai::staged::GrowableStagedSession::new(
        staged_config(false).to_string(),
        keys,
        kappas,
        shapes,
        dtypes,
        None,
        std::num::NonZeroU64::new(1).expect("1 is non-zero"),
        Box::new(DirKappaStore::new(store_path)),
    )
    .expect("the growable staged session builds");
    session.set_window_observer(Box::new(move |w, _resolved| {
        windows.lock().expect("lock").push(w)
    }));
    session
}

/// Greedy generation through a growable staged session; returns
/// (completion, windows compiled, prompt token count).
fn generate_growable(store_path: &std::path::Path, prompt: &str) -> (String, Vec<usize>, usize) {
    use hologram_ai::commands::generate::{generate_stream, GenConfig};
    let windows = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let mut session = growable_staged_session(store_path, std::sync::Arc::clone(&windows));
    let cfg = GenConfig {
        max_tokens: Some(8),
        temperature: 0.0,
        ..Default::default()
    };
    let mut sink = Vec::new();
    let text = generate_stream(&mut session, &DecimalTok, prompt, &cfg, &mut sink)
        .expect("growable staged generation completes");
    let prompt_tokens = DecimalTok.encode(prompt).len();
    let windows = windows.lock().expect("lock").clone();
    (text, windows, prompt_tokens)
}

#[when("a short prompt is generated through the growable staged session")]
async fn when_growable_short(w: &mut BddWorld) {
    let store = w.staged_store.as_ref().expect("the fixture κ-store");
    let (_, windows, _) = generate_growable(&store.path, "3 141 59 26 5");
    w.staged_windows = Some(windows);
}

#[then("the served window is the smallest geometric bucket holding the sequence")]
async fn then_growable_bucket(w: &mut BddWorld) {
    let windows = w.staged_windows.as_ref().expect("the growth log");
    // 5 prompt tokens + 8 generated stay inside the first bucket (64):
    // exactly one window is ever compiled, and it is the geometric bucket.
    let expect = hologram_ai::engine::geometric_window(5, STG_WINDOW as usize);
    assert_eq!(
        windows,
        &vec![expect],
        "a short prompt must compile exactly the sequence-sized bucket"
    );
    println!("[staged-window-growth] short prompt served by window {expect}");
}

#[then("the served window is smaller than the model's context length")]
async fn then_growable_below_context(w: &mut BddWorld) {
    let windows = w.staged_windows.as_ref().expect("the growth log");
    assert!(
        windows.iter().all(|&win| win < STG_WINDOW as usize),
        "the sequence-sized window ({windows:?}) must be below the model context {STG_WINDOW}"
    );
}

#[when("generation pushes the sequence across a window bucket boundary")]
async fn when_growable_crossing(w: &mut BddWorld) {
    let store = w.staged_store.as_ref().expect("the fixture κ-store");
    // A 60-token prompt + 8 generated tokens crosses the 64-token bucket.
    let prompt: String = (0..60)
        .map(|i| (i % 9 + 1).to_string())
        .collect::<Vec<_>>()
        .join(" ");
    let (_, windows, prompt_tokens) = generate_growable(&store.path, &prompt);
    assert_eq!(
        prompt_tokens, 60,
        "the crossing prompt must encode to 60 tokens"
    );
    w.staged_windows = Some(windows);
}

#[then("the session recompiles the stages exactly once per crossed bucket")]
async fn then_growable_once_per_bucket(w: &mut BddWorld) {
    let windows = w.staged_windows.as_ref().expect("the growth log");
    assert_eq!(
        windows,
        &vec![64, 128],
        "60 prompt + 8 generated tokens cross exactly one bucket boundary (64 → 128)"
    );
    println!("[staged-window-growth] buckets compiled: {windows:?}");
}

#[then("the window never exceeds the model's context length")]
async fn then_growable_capped(w: &mut BddWorld) {
    let windows = w.staged_windows.as_ref().expect("the growth log");
    assert!(
        windows.iter().all(|&win| win <= STG_WINDOW as usize),
        "no bucket ({windows:?}) may exceed the model context {STG_WINDOW}"
    );
}

#[when(
    "the same greedy completion is generated through the growable staged session and the fixed-window staged runner"
)]
async fn when_growable_parity(w: &mut BddWorld) {
    use hologram_ai::commands::generate::{generate_stream, GenConfig};
    let kit = staged_kit();
    let store = w.staged_store.as_ref().expect("the fixture κ-store");
    let prompt = "3 141 59 26 5";
    let cfg = GenConfig {
        max_tokens: Some(8),
        temperature: 0.0,
        ..Default::default()
    };

    let mut fixed = StagedRunner::from_archives(
        kit.stages.clone(),
        Box::new(DirKappaStore::new(&store.path)),
    )
    .expect("the fixed-window staged runner builds");
    let mut sink = Vec::new();
    let fixed_text = generate_stream(&mut fixed, &DecimalTok, prompt, &cfg, &mut sink)
        .expect("fixed-window staged generation completes");

    let (grown_text, _, _) = generate_growable(&store.path, prompt);
    w.staged_completions = Some((fixed_text, grown_text));
}

#[when("a growable staged session is asked for a window past the model's context")]
async fn when_growable_overflow(w: &mut BddWorld) {
    use hologram_ai::engine::SessionProvider;
    let store = w.staged_store.as_ref().expect("the fixture κ-store");
    let windows = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let mut session = growable_staged_session(&store.path, windows);
    let err = session
        .session_for(STG_WINDOW as usize + 1)
        .err()
        .expect("a window past the model's context must be refused")
        .to_string();
    w.staged_growth_err = Some(err);
}

#[then("the refusal names the requested window and the model's context length")]
async fn then_growable_overflow_named(w: &mut BddWorld) {
    let err = w.staged_growth_err.as_ref().expect("the refusal");
    let want = (STG_WINDOW + 1).to_string();
    let ctx = STG_WINDOW.to_string();
    assert!(
        err.contains(&want) && err.contains(&ctx),
        "the refusal must name the requested window ({want}) and the context ({ctx}): {err}"
    );
    println!("[staged-window-growth] loud refusal: {err}");
}

// ─────────── S3 — stage residency cache (row `stage-residency-cache`) ────────

/// One instrumented staged generation: the completion plus the runner's
/// bandwidth/residency measurements.
struct ResidencyRun {
    completion: String,
    materializations: u64,
    peak_resident_bytes: u64,
    stage_count: usize,
    passes: usize,
    budget: u64,
    largest_stage_bytes: u64,
}

/// Greedy 8-token generation over the fixture stages at `budget` residency.
fn run_with_residency(store_path: &std::path::Path, budget: u64) -> ResidencyRun {
    use hologram_ai::commands::generate::{generate_stream, GenConfig};
    let kit = staged_kit();
    let mut runner =
        StagedRunner::from_archives(kit.stages.clone(), Box::new(DirKappaStore::new(store_path)))
            .expect("the staged runner builds");
    runner.set_residency_budget(budget);
    let cfg = GenConfig {
        max_tokens: Some(8),
        temperature: 0.0,
        ..Default::default()
    };
    let mut sink = Vec::new();
    let completion = generate_stream(&mut runner, &DecimalTok, "3 141 59 26 5", &cfg, &mut sink)
        .expect("staged generation completes");
    let passes = DecimalTok.encode(&completion).len();
    let largest_stage_bytes = runner
        .stage_weight_bytes()
        .iter()
        .copied()
        .max()
        .expect("at least one stage");
    ResidencyRun {
        completion,
        materializations: runner.materialization_count(),
        peak_resident_bytes: runner.peak_resident_weight_bytes(),
        stage_count: runner.stage_count(),
        passes,
        budget,
        largest_stage_bytes,
    }
}

#[when("a completion is generated with a residency budget that holds the whole model")]
async fn when_residency_full(w: &mut BddWorld) {
    let store = w.staged_store.as_ref().expect("the fixture κ-store");
    let kit = staged_kit();
    w.residency_run = Some(run_with_residency(&store.path, kit.total_weight_bytes * 2));
}

#[then("each stage materialized exactly once across the whole generation")]
async fn then_residency_once(w: &mut BddWorld) {
    let run = w.residency_run.as_ref().expect("the instrumented run");
    assert!(run.passes > 1, "the witness needs a multi-pass generation");
    assert_eq!(
        run.materializations, run.stage_count as u64,
        "within the budget, κ-store bandwidth is once per stage per window          ({} passes ran)",
        run.passes
    );
    println!(
        "[stage-residency-cache] {} stages materialized once across {} passes",
        run.stage_count, run.passes
    );
}

#[when("a completion is generated with a zero residency budget")]
async fn when_residency_zero(w: &mut BddWorld) {
    let store = w.staged_store.as_ref().expect("the fixture κ-store");
    w.residency_run = Some(run_with_residency(&store.path, 0));
}

#[then("every forward pass rematerialized every stage")]
async fn then_residency_strict(w: &mut BddWorld) {
    let run = w.residency_run.as_ref().expect("the instrumented run");
    assert!(run.passes > 1, "the witness needs a multi-pass generation");
    assert_eq!(
        run.materializations,
        (run.stage_count * run.passes) as u64,
        "a zero budget must rematerialize every stage every pass"
    );
}

#[then("the strict window's peak residency stays within one stage")]
async fn then_residency_zero_window(w: &mut BddWorld) {
    let run = w.residency_run.as_ref().expect("the instrumented run");
    assert!(
        run.peak_resident_bytes <= run.largest_stage_bytes,
        "strict windowing must keep peak residency ({}) within one stage ({})",
        run.peak_resident_bytes,
        run.largest_stage_bytes
    );
}

#[when("a completion is generated with a residency budget of two stages")]
async fn when_residency_partial(w: &mut BddWorld) {
    let store = w.staged_store.as_ref().expect("the fixture κ-store");
    // Measure real stage sizes with a strict run, then budget exactly two.
    let probe = run_with_residency(&store.path, 0);
    let budget = probe.largest_stage_bytes * 2;
    w.residency_run = Some(run_with_residency(&store.path, budget));
}

#[then("the peak resident weight bytes never exceed the budget or the single-stage floor")]
async fn then_residency_bounded(w: &mut BddWorld) {
    let run = w.residency_run.as_ref().expect("the instrumented run");
    let bound = run.budget.max(run.largest_stage_bytes);
    assert!(
        run.peak_resident_bytes <= bound,
        "peak residency ({}) must stay within max(budget {}, single stage {})",
        run.peak_resident_bytes,
        run.budget,
        run.largest_stage_bytes
    );
    assert!(
        run.materializations < (run.stage_count * run.passes) as u64,
        "a partial budget must save at least some rematerialization"
    );
    println!(
        "[stage-residency-cache] budget {} → {} materializations over {} passes × {} stages",
        run.budget, run.materializations, run.passes, run.stage_count
    );
}

#[when("the same greedy completion is generated with and without a residency budget")]
async fn when_residency_parity(w: &mut BddWorld) {
    let store = w.staged_store.as_ref().expect("the fixture κ-store");
    let kit = staged_kit();
    let strict = run_with_residency(&store.path, 0);
    let cached = run_with_residency(&store.path, kit.total_weight_bytes * 2);
    w.staged_completions = Some((strict.completion, cached.completion));
}

// ──────── S3 — total algebraic path (row `total-algebraic-path`, open) ───────

#[when("the fixture decoder's lowered kernel dtypes are tallied")]
async fn when_float_tally(w: &mut BddWorld) {
    let manifest = staged_manifest(false);
    let (keys, _): (Vec<String>, Vec<Vec<u64>>) = manifest.into_iter().unzip();
    let dtypes = vec![DType::F32; keys.len()];
    let graph = build_parametric_graph_from_manifest(&staged_config(false), &keys, &dtypes, None)
        .expect("the fixture decoder graph builds");
    // Classify each node by its first output's dtype: the substrate's
    // dispatch selects kernels by dtype tag, float first, so a float-dtyped
    // node is a runtime float dispatch today.
    let mut float_nodes = 0usize;
    let mut total = 0usize;
    for node in &graph.nodes {
        let Some(out) = node.outputs.first() else {
            continue;
        };
        let Some(info) = graph.tensor_info.get(out) else {
            continue;
        };
        total += 1;
        if matches!(
            info.logical_dtype,
            DType::F32 | DType::F16 | DType::BF16 | DType::F64
        ) {
            float_nodes += 1;
        }
    }
    w.float_dispatch_tally = Some((float_nodes, total));
}

#[then("the float-dispatched fraction is reported, never asserted")]
async fn then_float_tally(w: &mut BddWorld) {
    let (float_nodes, total) = w.float_dispatch_tally.expect("the tally ran");
    assert!(total > 0, "the tally must cover a real graph");
    println!(
        "[total-algebraic-path] {}/{} kernels float-dtyped ({:.0}%) — the substrate \
         dispatches these to native IEEE-754 kernels at runtime; the row flips to \
         build at 0% with gate-time reference parity per (op, tier)",
        float_nodes,
        total,
        100.0 * float_nodes as f64 / total as f64
    );
}

// ── S3 — cross-turn residency (row `stage-residency-cache`, turns scenario) ──

#[when("two completions are generated over one warm session within the budget")]
async fn when_cross_turn(w: &mut BddWorld) {
    use hologram_ai::commands::generate::{generate_stream, GenConfig};
    let store = w.staged_store.as_ref().expect("the fixture κ-store");
    let kit = staged_kit();
    let windows = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let mut session = growable_staged_session(&store.path, windows);
    session.set_residency_budget(kit.total_weight_bytes * 2);
    let cfg = GenConfig {
        max_tokens: Some(4),
        temperature: 0.0,
        ..Default::default()
    };
    let mut sink = Vec::new();
    generate_stream(&mut session, &DecimalTok, "3 141 59", &cfg, &mut sink)
        .expect("turn one completes");
    let after_turn_one = session.materialization_count();
    let mut sink = Vec::new();
    generate_stream(&mut session, &DecimalTok, "3 141 59 26 5", &cfg, &mut sink)
        .expect("turn two completes");
    let after_turn_two = session.materialization_count();
    w.residency_run = Some(ResidencyRun {
        completion: String::new(),
        materializations: after_turn_two - after_turn_one,
        peak_resident_bytes: 0,
        stage_count: session.stage_count(),
        passes: 2,
        budget: kit.total_weight_bytes * 2,
        largest_stage_bytes: 0,
    });
}

#[then("the second completion adds no stage materializations")]
async fn then_cross_turn(w: &mut BddWorld) {
    let run = w.residency_run.as_ref().expect("both turns ran");
    assert_eq!(
        run.materializations, 0,
        "a warm session's resident set must carry across turns — κ-store \
         bandwidth is per window, never per turn"
    );
    println!(
        "[stage-residency-cache] turn two over {} resident stages: zero rematerializations",
        run.stage_count
    );
}

// ─────────────── S3 — chunked head (row `chunked-head`) ──────────────────────

/// A fixture whose vocabulary outweighs a layer stage, forcing the head to
/// partition into κ-range chunk stages.
const CH_VOCAB: u64 = 4096;

fn chunked_config() -> serde_json::Value {
    let mut config = staged_config(false);
    config["vocab_size"] = serde_json::json!(CH_VOCAB);
    config
}

/// Fixture weights with NO duplicate head rows: the shared generator's
/// period-13 pattern makes vocab rows bit-identical 13 apart, and exact
/// argmax ties then break on kernel reduction-order drift. A tiny
/// index-linear term keeps every element (and so every row) distinct while
/// preserving magnitudes.
fn chunked_tensor_bytes(name: &str, dims: &[u64]) -> Vec<u8> {
    let n: u64 = dims.iter().product();
    let norm = name.contains("layernorm") || name.ends_with(".norm.weight");
    (0..n)
        .flat_map(|k| {
            let v: f32 = if norm {
                1.0
            } else {
                ((k % 13) as f32 - 6.0) * 0.01 + (k as f32) * 1e-7
            };
            v.to_le_bytes()
        })
        .collect()
}

fn chunked_manifest() -> Vec<(String, Vec<u64>)> {
    staged_manifest(false)
        .into_iter()
        .map(|(name, dims)| {
            if name == "model.embed_tokens.weight" || name == "lm_head.weight" {
                (name, vec![CH_VOCAB, STG_HIDDEN])
            } else {
                (name, dims)
            }
        })
        .collect()
}

struct ChunkedKit {
    monolithic: Vec<u8>,
    stages: Vec<Vec<u8>>,
    store: StoreDir,
    keys: Vec<String>,
    kappas: Vec<String>,
    shapes: Vec<Vec<u64>>,
}

fn chunked_kit() -> ChunkedKit {
    let manifest = chunked_manifest();
    let (keys, shapes): (Vec<String>, Vec<Vec<u64>>) = manifest.into_iter().unzip();
    let kappas: Vec<String> = keys
        .iter()
        .zip(&shapes)
        .map(|(name, dims)| kappa_of(&chunked_tensor_bytes(name, dims)))
        .collect();
    let dtypes = vec![DType::F32; keys.len()];
    let config = chunked_config();

    let store = StoreDir::new("chunked-head");
    let dir = DirKappaStore::new(&store.path);
    for (name, dims) in chunked_manifest() {
        dir.insert(&chunked_tensor_bytes(&name, &dims))
            .expect("persisting a fixture weight");
    }

    // Monolithic k-form (whole head), the parity oracle.
    let mut graph = build_parametric_graph_from_manifest(&config, &keys, &dtypes, None)
        .expect("the wide-vocab fixture graph builds");
    let name_to_id: HashMap<String, u32> = graph
        .tensor_names
        .iter()
        .map(|(id, name)| (name.clone(), *id))
        .collect();
    for (i, key) in keys.iter().enumerate() {
        let id = *name_to_id.get(key).expect("manifest tensor in the graph");
        let info = ti(DType::F32, &shapes[i]);
        graph.tensor_info.insert(id, info.clone());
        graph.params.insert(
            id,
            AiParam::External {
                kappa: kappas[i].clone(),
                info,
                range: None,
            },
        );
    }
    let monolithic = compile_graph(graph);

    let stages = hologram_ai::staged::compile_stages(
        &config.to_string(),
        &keys,
        &kappas,
        &shapes,
        &dtypes,
        None,
        std::num::NonZeroU64::new(1).expect("1 is non-zero"),
    )
    .expect("the chunked staged fixture compiles");

    ChunkedKit {
        monolithic,
        stages,
        store,
        keys,
        kappas,
        shapes,
    }
}

#[given("a wide-vocabulary decoder fixture with its weights in a κ-store")]
async fn given_chunked_fixture(w: &mut BddWorld) {
    w.chunked = Some(chunked_kit());
}

#[when("the wide-vocabulary fixture is compiled as stages")]
async fn when_chunked_compiled(_w: &mut BddWorld) {
    // Compilation happened in the Given; the Thens inspect it.
}

#[then("the head partitions into multiple chunk stages bound by κ-ranges")]
async fn then_chunked_partition(w: &mut BddWorld) {
    let kit = w.chunked.as_ref().expect("the chunked kit");
    // Base pipeline: embedding + 2 one-layer stages; anything beyond is head
    // chunks — there must be more than one.
    let head_stages = kit.stages.len() - 3;
    assert!(
        head_stages > 1,
        "a vocabulary heavier than a layer stage must chunk (got {head_stages} head stage(s))"
    );
    // Every chunk binds the head tensor via a RANGE of the same κ; the
    // ranges tile the tensor exactly.
    let head_kappa = kit
        .keys
        .iter()
        .zip(&kit.kappas)
        .find(|(k, _)| *k == "model.embed_tokens.weight")
        .map(|(_, kappa)| kappa.clone())
        .expect("the head source is in the manifest");
    let mut covered = 0u64;
    let mut ranged = 0usize;
    for archive in &kit.stages[3..] {
        for req in kappa_requirements(archive).expect("the κ-map parses") {
            if req.kappa == head_kappa {
                if let Some((offset, len)) = req.range {
                    assert_eq!(offset, covered, "chunk ranges tile in order");
                    covered += len;
                    ranged += 1;
                }
            }
        }
    }
    assert_eq!(ranged, head_stages, "every chunk stage binds one κ-range");
    assert_eq!(
        covered,
        CH_VOCAB * STG_HIDDEN * 4,
        "the chunk ranges tile the whole head tensor exactly"
    );
    println!("[chunked-head] {head_stages} chunk stages tile the head κ by ranges");
}

#[then("every chunk stage stays within the layer-stage granularity")]
async fn then_chunked_granularity(w: &mut BddWorld) {
    let kit = w.chunked.as_ref().expect("the chunked kit");
    let size_of: HashMap<String, u64> = kit
        .keys
        .iter()
        .zip(&kit.shapes)
        .zip(&kit.kappas)
        .map(|((_, dims), kappa)| (kappa.clone(), dims.iter().product::<u64>() * 4))
        .collect();
    let stage_bytes = |archive: &Vec<u8>| -> u64 {
        kappa_requirements(archive)
            .expect("the κ-map parses")
            .iter()
            .map(|r| match r.range {
                Some((_, len)) => len,
                None => size_of.get(&r.kappa).copied().unwrap_or(0),
            })
            .sum()
    };
    let layer_max = kit.stages[1..3]
        .iter()
        .map(stage_bytes)
        .max()
        .expect("layer stages");
    for (i, archive) in kit.stages[3..].iter().enumerate() {
        let bytes = stage_bytes(archive);
        assert!(
            bytes <= layer_max + STG_HIDDEN * 4,
            "chunk stage {i} ({bytes} B) must stay within layer granularity ({layer_max} B)"
        );
    }
}

#[when("the same token window runs through the chunked stages and the monolithic archive")]
async fn when_chunked_parity(w: &mut BddWorld) {
    let kit = w.chunked.as_ref().expect("the chunked kit");
    let ids = staged_window_ids();

    let mut dir = DirKappaStore::new(&kit.store.path);
    let mono = materialize_archive(&kit.monolithic, &mut dir)
        .expect("the monolithic archive materializes");
    let mut mono_runner = HoloRunner::from_bytes(mono).expect("the archive loads");
    let refs: Vec<&[u8]> = vec![&ids];
    let mono_out = mono_runner
        .execute(&refs)
        .expect("the monolithic pass runs");
    let mono_logits = mono_out[mono_runner
        .output_index_by_name("logits")
        .expect("a logits port")]
    .bytes
    .clone();

    let mut staged = StagedRunner::from_archives(
        kit.stages.clone(),
        Box::new(DirKappaStore::new(&kit.store.path)),
    )
    .expect("the chunked staged runner builds");
    use hologram_ai::engine::LmSession;
    let staged_out = staged.execute(&[&ids]).expect("the chunked pass runs");
    let idx = staged
        .output_index_by_name("logits")
        .expect("a logits port");
    let staged_logits = staged_out[idx].bytes.clone();

    w.chunked_logits = Some((mono_logits, staged_logits));
}

#[then("the chunked logits match the monolithic logits within reduction-order tolerance")]
async fn then_chunked_tolerance(w: &mut BddWorld) {
    let (mono, chunked) = w.chunked_logits.as_ref().expect("both passes ran");
    assert_eq!(mono.len(), chunked.len(), "logit buffers agree in size");
    let m: Vec<f32> = le_f32(mono);
    let c: Vec<f32> = le_f32(chunked);
    // The substrate's matmul reduction tiling varies with output width;
    // the measured cross-width drift is ≤ 4e-7 (1–2 ulp at logit scale).
    let max_abs = m
        .iter()
        .zip(&c)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(
        max_abs <= 1e-6,
        "chunked-vs-whole drift must stay at reduction-order scale, got {max_abs:e}"
    );
    println!(
        "[chunked-head] {} logits agree within {max_abs:e} (kernel reduction-order tolerance)",
        m.len()
    );
}

#[then("the greedy choice at every position is identical")]
async fn then_chunked_argmax(w: &mut BddWorld) {
    let (mono, chunked) = w.chunked_logits.as_ref().expect("both passes ran");
    let m = le_f32(mono);
    let c = le_f32(chunked);
    let vocab = CH_VOCAB as usize;
    for (pos, (mr, cr)) in m.chunks(vocab).zip(c.chunks(vocab)).enumerate() {
        assert_eq!(
            argmax(mr),
            argmax(cr),
            "greedy decode must be invariant to the head partition (position {pos})"
        );
    }
    println!("[chunked-head] greedy choice identical at every position");
}

#[when("a greedy completion is generated through the chunked staged session")]
async fn when_chunked_generation(w: &mut BddWorld) {
    use hologram_ai::commands::generate::{generate_stream, GenConfig};
    let kit = w.chunked.as_ref().expect("the chunked kit");
    let mut runner = StagedRunner::from_archives(
        kit.stages.clone(),
        Box::new(DirKappaStore::new(&kit.store.path)),
    )
    .expect("the chunked staged runner builds");
    let cfg = GenConfig {
        max_tokens: Some(6),
        temperature: 0.0,
        ..Default::default()
    };
    let mut sink = Vec::new();
    let text = generate_stream(&mut runner, &DecimalTok, "3 141 59 26 5", &cfg, &mut sink)
        .expect("chunked staged generation completes");

    // The monolithic completion over the same store — the parity oracle.
    let mut dir = DirKappaStore::new(&kit.store.path);
    let mono = materialize_archive(&kit.monolithic, &mut dir)
        .expect("the monolithic archive materializes");
    let mut fixed =
        hologram_ai::FixedSession::new(HoloRunner::from_bytes(mono).expect("the archive loads"));
    let mut sink = Vec::new();
    let mono_text = generate_stream(&mut fixed, &DecimalTok, "3 141 59 26 5", &cfg, &mut sink)
        .expect("monolithic generation completes");
    assert_eq!(
        text, mono_text,
        "chunked greedy generation must equal the monolithic completion"
    );
    let ranged: u64 = kit.stages[3..]
        .iter()
        .map(|a| {
            kappa_requirements(a)
                .expect("the κ-map parses")
                .iter()
                .filter(|r| r.range.is_some())
                .count() as u64
        })
        .sum();
    w.chunked_completion = Some((text, ranged));
}

#[then("the chunked completion equals the monolithic completion and every chunk resolved through its κ-range")]
async fn then_chunked_generation(w: &mut BddWorld) {
    let (text, ranged) = w.chunked_completion.as_ref().expect("the generation ran");
    assert!(!text.is_empty(), "the chunked pipeline must generate");
    assert!(*ranged > 1, "the chunks resolved via κ-ranges");
    println!("[chunked-head] completion {text:?} over {ranged} κ-range bindings");
}

/// The per-resolution traffic journal of the resolve_range witness: every
/// store touch as `(κ, bytes moved, ranged)`, split at the pass boundary.
struct ChunkedTraffic {
    pass1: Vec<(String, u64, bool)>,
    pass2: Vec<(String, u64, bool)>,
    head_kappa: String,
    head_bytes: u64,
    chunk_stages: usize,
}

/// One store touch per entry: `(κ, bytes moved, rode resolve_range)`.
type TrafficJournal = Vec<(String, u64, bool)>;

/// A κ-store adapter that journals each resolution's moved bytes and whether
/// it rode `resolve_range` — the I/O instrument of the resolve_range witness.
struct JournalingStore {
    inner: DirKappaStore,
    log: std::rc::Rc<std::cell::RefCell<TrafficJournal>>,
}

impl KappaStore for JournalingStore {
    fn resolve(&mut self, kappa: &str) -> anyhow::Result<Vec<u8>> {
        let bytes = self.inner.resolve(kappa)?;
        self.log
            .borrow_mut()
            .push((kappa.to_string(), bytes.len() as u64, false));
        Ok(bytes)
    }

    fn invalidate(&mut self, kappa: &str) {
        self.inner.invalidate(kappa);
    }

    fn resolve_range(&mut self, kappa: &str, offset: u64, len: u64) -> anyhow::Result<Vec<u8>> {
        let bytes = self.inner.resolve_range(kappa, offset, len)?;
        self.log
            .borrow_mut()
            .push((kappa.to_string(), bytes.len() as u64, true));
        Ok(bytes)
    }
}

#[when("the chunked stages execute twice in one session")]
async fn when_chunked_twice(w: &mut BddWorld) {
    use hologram_ai::engine::LmSession;
    let kit = w.chunked.as_ref().expect("the chunked kit");
    let ids = staged_window_ids();
    let log = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
    let store = JournalingStore {
        inner: DirKappaStore::new(&kit.store.path),
        log: std::rc::Rc::clone(&log),
    };
    // Residency budget 0: every pass rematerializes every stage — the
    // rematerialization whose traffic this witness measures.
    let mut staged = StagedRunner::from_archives(kit.stages.clone(), Box::new(store))
        .expect("the chunked staged runner builds");
    staged.execute(&[&ids]).expect("the first pass runs");
    let pass1 = std::mem::take(&mut *log.borrow_mut());
    staged.execute(&[&ids]).expect("the second pass runs");
    let pass2 = std::mem::take(&mut *log.borrow_mut());

    let head_idx = kit
        .keys
        .iter()
        .position(|k| k == "lm_head.weight")
        .expect("the untied fixture has a head tensor");
    w.chunked_traffic = Some(ChunkedTraffic {
        pass1,
        pass2,
        head_kappa: kit.kappas[head_idx].clone(),
        head_bytes: kit.shapes[head_idx].iter().product::<u64>() * 4,
        chunk_stages: kit.stages.len() - 3,
    });
}

#[then("every ranged touch of the verified head κ moves only its slice and whole transits stay at one per pass")]
async fn then_chunked_range_traffic(w: &mut BddWorld) {
    let t = w.chunked_traffic.as_ref().expect("both passes ran");
    // The fixture's embedding and head share one κ (identical content), so
    // each pass carries exactly ONE whole transit of it: the embedding
    // stage's own whole-tensor binding. Pass 1's is also the verification
    // read; after it, every ranged binding — including pass 1's own chunk
    // stages — moves only its slice through resolve_range.
    for (pass, name) in [(&t.pass1, "first"), (&t.pass2, "second")] {
        let head: Vec<&(String, u64, bool)> =
            pass.iter().filter(|(k, _, _)| *k == t.head_kappa).collect();
        let whole = head.iter().filter(|(_, _, ranged)| !ranged).count();
        assert_eq!(
            whole, 1,
            "the {name} pass must move the head κ whole exactly once (the embedding's \
             whole-tensor binding); verified ranged bindings never re-transit whole"
        );
        let ranged: Vec<u64> = head
            .iter()
            .filter(|(_, _, ranged)| *ranged)
            .map(|(_, b, _)| *b)
            .collect();
        assert_eq!(
            ranged.len(),
            t.chunk_stages,
            "every chunk stage of the {name} pass resolves through its κ-range"
        );
        assert!(
            ranged.iter().all(|b| *b < t.head_bytes),
            "no ranged read of the {name} pass moves the whole tensor"
        );
        assert_eq!(
            ranged.iter().sum::<u64>(),
            t.head_bytes,
            "the {name} pass's ranged reads tile the tensor exactly"
        );
    }
    let old = (1 + t.chunk_stages as u64) * t.head_bytes;
    let now: u64 = t
        .pass2
        .iter()
        .filter(|(k, _, _)| *k == t.head_kappa)
        .map(|(_, b, _)| *b)
        .sum();
    println!(
        "[chunked-head] verified rematerialization moves {now} B of the head κ \
         (whole-resolve-and-slice would move {old} B)"
    );
}

// ─────────── S4 — performance contract (row `performance-contract`) ──────────

/// One attributed decode step: seconds, kernels dispatched, kernels elided,
/// cumulative stage materializations after the step.
struct PerfStep {
    secs: f64,
    dispatched: u64,
    skipped: u64,
    materializations: u64,
}

struct PerfReport {
    /// Calibrated sequential stream bandwidth, bytes/second.
    bandwidth: f64,
    /// Raw weight bytes the pipeline streams per forward pass.
    weight_bytes: u64,
    window: usize,
    stages: usize,
    steps: Vec<PerfStep>,
}

struct PerfKit {
    store: StoreDir,
    stages: Vec<Vec<u8>>,
}

#[given("a staged decoder fixture in a κ-store for the performance probe")]
async fn given_perf_fixture(w: &mut BddWorld) {
    let manifest = staged_manifest(false);
    let (keys, shapes): (Vec<String>, Vec<Vec<u64>>) = manifest.into_iter().unzip();
    let kappas: Vec<String> = keys
        .iter()
        .zip(&shapes)
        .map(|(name, dims)| kappa_of(&staged_tensor_bytes(name, dims)))
        .collect();
    let dtypes = vec![DType::F32; keys.len()];
    let store = StoreDir::new("perf-contract");
    let dir = DirKappaStore::new(&store.path);
    for (name, dims) in keys.iter().zip(&shapes) {
        dir.insert(&staged_tensor_bytes(name, dims))
            .expect("persisting a fixture weight");
    }
    let stages = hologram_ai::staged::compile_stages(
        &staged_config(false).to_string(),
        &keys,
        &kappas,
        &shapes,
        &dtypes,
        None,
        std::num::NonZeroU64::new(1).expect("1 is non-zero"),
    )
    .expect("the staged fixture compiles");
    w.perf_kit = Some(PerfKit { store, stages });
}

/// Calibrate sequential stream bandwidth: the best of three timed passes
/// summing a 64 MB buffer in u64 words — the floor's unit, measured in this
/// environment, per the Benchmark section of the resource model.
fn calibrate_stream_bandwidth() -> f64 {
    const BYTES: usize = 64 << 20;
    let buf: Vec<u64> = (0..BYTES / 8).map(|i| i as u64).collect();
    let mut best = f64::INFINITY;
    for _ in 0..3 {
        let t = std::time::Instant::now();
        let mut acc = 0u64;
        for &v in &buf {
            acc = acc.wrapping_add(v);
        }
        std::hint::black_box(acc);
        let secs = t.elapsed().as_secs_f64();
        if secs < best {
            best = secs;
        }
    }
    BYTES as f64 / best
}

#[when(
    expr = "the environment stream bandwidth is calibrated and {int} staged decode steps are timed"
)]
async fn when_perf_staged_decode(w: &mut BddWorld, steps: usize) {
    use hologram_ai::engine::LmSession;
    let kit = w.perf_kit.as_ref().expect("the perf kit");
    let bandwidth = calibrate_stream_bandwidth();

    let mut runner = StagedRunner::from_archives(
        kit.stages.clone(),
        Box::new(DirKappaStore::new(&kit.store.path)),
    )
    .expect("the staged runner builds");
    runner.set_residency_budget(u64::MAX);
    let window = runner.window();
    let stages = runner.stage_count();

    let mut toks: Vec<i64> = vec![1];
    let mut recorded = Vec::new();
    for _ in 0..steps {
        let ids: Vec<u8> = (0..window)
            .map(|i| toks.get(i).copied().unwrap_or(0))
            .flat_map(|t| t.to_le_bytes())
            .collect();
        let t = std::time::Instant::now();
        let outs = runner.execute(&[&ids]).expect("a staged decode step");
        let secs = t.elapsed().as_secs_f64();
        let idx = runner
            .output_index_by_name("logits")
            .expect("a logits port");
        let logits = le_f32(&outs[idx].bytes);
        let vocab = logits.len() / window;
        let row = &logits[(toks.len() - 1) * vocab..toks.len() * vocab];
        toks.push(argmax(row) as i64);
        recorded.push(PerfStep {
            secs,
            dispatched: runner.last_dispatched(),
            skipped: runner.last_skipped(),
            materializations: runner.materialization_count(),
        });
    }
    let weight_bytes: u64 = runner.stage_weight_bytes().iter().sum();
    w.perf = Some(PerfReport {
        bandwidth,
        weight_bytes,
        window,
        stages,
        steps: recorded,
    });
}

#[then("the per-token decode ratio and its attribution are reported")]
async fn then_perf_report(w: &mut BddWorld) {
    let r = w.perf.as_ref().expect("the perf probe ran");
    // Structural checks only — ratios are REPORTED, never asserted.
    assert!(!r.steps.is_empty(), "steps were timed");
    for (i, s) in r.steps.iter().enumerate() {
        assert!(
            s.dispatched + s.skipped > 0,
            "step {i}: the walk reports its kernels"
        );
    }
    let first_mat = r.steps[0].materializations;
    assert_eq!(
        r.steps.last().expect("steps").materializations,
        first_mat,
        "residency holds: no rematerialization after the first pass"
    );
    let floor = r.weight_bytes as f64 / r.bandwidth;
    println!(
        "[performance-contract] calibrated stream bandwidth {:.2} GB/s; staged fixture: \
         {} stages, window {}, {:.2} MB weights/pass; decode floor {:.3} ms/token",
        r.bandwidth / 1e9,
        r.stages,
        r.window,
        r.weight_bytes as f64 / 1e6,
        floor * 1e3
    );
    for (i, s) in r.steps.iter().enumerate() {
        println!(
            "[performance-contract] step {}: {:.1} ms/token, ratio {:.1}x floor, \
             dispatched {}, elided {} ({:.0}%), materializations {}",
            i + 1,
            s.secs * 1e3,
            s.secs / floor,
            s.dispatched,
            s.skipped,
            100.0 * s.skipped as f64 / (s.dispatched + s.skipped) as f64,
            s.materializations
        );
    }
}

// ──────────── S3 — quantized transit (row `quantized-transit`) ───────────────

/// Fixture weights with a NAME-dependent term so no two tensors share
/// content (the shared generator gives the tied-dims embedding and head one
/// κ, which would couple the wide-eviction assertion to the Gather path).
fn quant_tensor_bytes(name: &str, dims: &[u64]) -> Vec<u8> {
    let n: u64 = dims.iter().product();
    let norm = name.contains("layernorm") || name.ends_with(".norm.weight");
    // FNV-1a over the name: no two tensors share content (a weak salt let
    // two projections collide onto one κ and collapsed the quant map).
    let salt = (name.bytes().fold(0xcbf29ce484222325u64, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x100000001b3)
    }) % 251) as f32;
    (0..n)
        .flat_map(|k| {
            let v: f32 = if norm {
                1.0
            } else {
                ((k % 13) as f32 - 6.0) * 0.01 + (k as f32) * 1e-7 + salt * 1e-4
            };
            v.to_le_bytes()
        })
        .collect()
}

struct QuantKit {
    store: StoreDir,
    keys: Vec<String>,
    kappas: Vec<String>,
    shapes: Vec<Vec<u64>>,
    dtypes: Vec<DType>,
    config: serde_json::Value,
    /// wide κ → (artifact κ, out, in) for every rank-2 projection weight
    /// (the embedding stays wide: its Gather consumes the whole table).
    map: hologram_ai::quantized::QuantMap,
}

#[given("a decoder fixture with quantizable projection weights in a κ-store")]
async fn given_quant_fixture(w: &mut BddWorld) {
    let manifest = staged_manifest(false);
    let (keys, shapes): (Vec<String>, Vec<Vec<u64>>) = manifest.into_iter().unzip();
    let kappas: Vec<String> = keys
        .iter()
        .zip(&shapes)
        .map(|(name, dims)| kappa_of(&quant_tensor_bytes(name, dims)))
        .collect();
    let dtypes = vec![DType::F32; keys.len()];

    let store = StoreDir::new("quantized-transit");
    let mut dir = DirKappaStore::new(&store.path);
    for (name, dims) in keys.iter().zip(&shapes) {
        dir.insert(&quant_tensor_bytes(name, dims))
            .expect("persisting a fixture weight");
    }

    // Crystallize the quantized derivation of every projection weight —
    // the derived form enters the κ-store as ordinary content.
    let mut map = hologram_ai::quantized::QuantMap::new();
    let mut eligible = 0usize;
    for ((name, dims), kappa) in keys.iter().zip(&shapes).zip(&kappas) {
        if dims.len() != 2 || name == "model.embed_tokens.weight" {
            continue;
        }
        eligible += 1;
        let entry = hologram_ai::quantized::crystallize_quantized(
            &mut dir,
            kappa,
            DType::F32,
            dims[0],
            dims[1],
        )
        .expect("the quantized derivation crystallizes");
        map.insert(kappa.clone(), entry);
    }
    assert!(
        eligible > 1,
        "the fixture has projection weights to quantize"
    );
    assert_eq!(
        map.len(),
        eligible,
        "every projection weight has distinct content — one κ each"
    );

    w.quant = Some(QuantKit {
        store,
        keys,
        kappas,
        shapes,
        dtypes,
        config: staged_config(false),
        map,
    });
}

#[when("the projection weights derive their quantized artifacts twice")]
async fn when_quant_derive_twice(w: &mut BddWorld) {
    let kit = w.quant.as_ref().expect("the quant kit");
    let mut dir = DirKappaStore::new(&kit.store.path);
    let (mut wide_total, mut quant_total) = (0u64, 0u64);
    for (wide_kappa, (artifact_kappa, out, inf)) in &kit.map {
        let wide = dir.resolve(wide_kappa).expect("the wide form resolves");
        let a = hologram_ai::quantized::derive_quantized_artifact(&wide, DType::F32, *out, *inf)
            .expect("the derivation runs");
        let b = hologram_ai::quantized::derive_quantized_artifact(&wide, DType::F32, *out, *inf)
            .expect("the re-derivation runs");
        assert_eq!(a, b, "derivation must be bit-deterministic");
        assert_eq!(
            &kappa_of(&a),
            artifact_kappa,
            "re-derivation must reproduce the crystallized artifact κ"
        );
        assert!(
            a.len() < wide.len(),
            "the artifact must be strictly smaller than its wide form"
        );
        wide_total += wide.len() as u64;
        quant_total += a.len() as u64;
    }
    w.quant_derive = Some((kit.map.len(), wide_total, quant_total));
}

#[then("re-derivation reproduces each artifact κ bit-identically and every artifact is strictly smaller than its wide form")]
async fn then_quant_deterministic(w: &mut BddWorld) {
    let (count, wide, quant) = w.quant_derive.expect("the derivations ran");
    println!(
        "[quantized-transit] {count} artifacts, {quant} B derived from {wide} B wide \
         ({:.2}x smaller), κs reproduced bit-identically",
        wide as f64 / quant as f64
    );
}

fn quant_stages(kit: &QuantKit) -> Vec<Vec<u8>> {
    hologram_ai::staged::compile_stages_with(
        &kit.config.to_string(),
        &kit.keys,
        &kit.kappas,
        &kit.shapes,
        &kit.dtypes,
        None,
        std::num::NonZeroU64::new(1).expect("1 is non-zero"),
        Some(&kit.map),
    )
    .expect("the quantized staged fixture compiles")
}

#[when("the quantized stages and the quantized monolithic archive generate from the same prompt")]
async fn when_quant_parity(w: &mut BddWorld) {
    use hologram_ai::commands::generate::{generate_stream, GenConfig};
    let kit = w.quant.as_ref().expect("the quant kit");
    let cfg = GenConfig {
        max_tokens: Some(6),
        temperature: 0.0,
        ..Default::default()
    };

    let mut staged = StagedRunner::from_archives(
        quant_stages(kit),
        Box::new(DirKappaStore::new(&kit.store.path)),
    )
    .expect("the quantized staged runner builds");
    let mut sink = Vec::new();
    let staged_text = generate_stream(&mut staged, &DecimalTok, "3 141 59 26 5", &cfg, &mut sink)
        .expect("quantized staged generation completes");

    // The monolithic oracle: the same manifest, the same κ-bindings, the
    // same quantize pass — one archive.
    let mut graph = build_parametric_graph_from_manifest(&kit.config, &kit.keys, &kit.dtypes, None)
        .expect("the fixture graph builds");
    let name_to_id: HashMap<String, u32> = graph
        .tensor_names
        .iter()
        .map(|(id, name)| (name.clone(), *id))
        .collect();
    for (i, key) in kit.keys.iter().enumerate() {
        let id = *name_to_id.get(key).expect("manifest tensor in the graph");
        let info = ti(DType::F32, &kit.shapes[i]);
        graph.tensor_info.insert(id, info.clone());
        graph.params.insert(
            id,
            AiParam::External {
                kappa: kit.kappas[i].clone(),
                info,
                range: None,
            },
        );
    }
    let rewritten =
        hologram_ai_common::lower::quantize_external_matmul_weights(&mut graph, &kit.map)
            .expect("the quantize pass runs");
    assert_eq!(
        rewritten,
        kit.map.len(),
        "every crystallized projection rewrites onto its artifact"
    );
    let mono = compile_graph(graph);
    let mut dir = DirKappaStore::new(&kit.store.path);
    let mono = materialize_archive(&mono, &mut dir).expect("the monolithic archive materializes");
    let mut fixed =
        hologram_ai::FixedSession::new(HoloRunner::from_bytes(mono).expect("the archive loads"));
    let mut sink = Vec::new();
    let mono_text = generate_stream(&mut fixed, &DecimalTok, "3 141 59 26 5", &cfg, &mut sink)
        .expect("quantized monolithic generation completes");

    w.quant_completions = Some((staged_text, mono_text));
}

#[then("both quantized completions are identical and non-empty")]
async fn then_quant_parity(w: &mut BddWorld) {
    let (staged, mono) = w.quant_completions.as_ref().expect("both passes ran");
    assert!(!staged.is_empty(), "the quantized pipeline must generate");
    assert_eq!(
        staged, mono,
        "staged and monolithic execution must agree on the quantized tier"
    );
    println!("[quantized-transit] staged == monolithic on the quantized tier: {staged:?}");
}

#[when("the quantized staged session generates with the wide projection blobs evicted")]
async fn when_quant_evicted(w: &mut BddWorld) {
    use hologram_ai::commands::generate::{generate_stream, GenConfig};
    let kit = w.quant.as_ref().expect("the quant kit");
    let stages = quant_stages(kit);

    // The saturation decision: crystallized derivations make the wide blobs
    // gas-phase; evict them all.
    let mut dir = DirKappaStore::new(&kit.store.path);
    for wide_kappa in kit.map.keys() {
        dir.invalidate(wide_kappa);
    }

    let log = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
    let store = JournalingStore {
        inner: DirKappaStore::new(&kit.store.path),
        log: std::rc::Rc::clone(&log),
    };
    let mut staged = StagedRunner::from_archives(stages, Box::new(store))
        .expect("the quantized staged runner builds");
    let cfg = GenConfig {
        max_tokens: Some(4),
        temperature: 0.0,
        ..Default::default()
    };
    let mut sink = Vec::new();
    let text = generate_stream(&mut staged, &DecimalTok, "3 141 59", &cfg, &mut sink)
        .expect("generation completes without the wide forms");
    let journal = std::mem::take(&mut *log.borrow_mut());
    w.quant_evicted = Some((text, journal));
}

#[then("the completion still produces and no wide projection κ is ever resolved")]
async fn then_quant_gas_phase(w: &mut BddWorld) {
    let kit = w.quant.as_ref().expect("the quant kit");
    let (text, journal) = w.quant_evicted.as_ref().expect("the generation ran");
    assert!(
        !text.is_empty(),
        "the pipeline must generate on the quantized tier"
    );
    for (kappa, _, _) in journal {
        assert!(
            !kit.map.contains_key(kappa),
            "a gas-phase wide κ must never be resolved (`{kappa}` moved)"
        );
    }
    let quant_kappas: std::collections::HashSet<&String> =
        kit.map.values().map(|(k, _, _)| k).collect();
    let (mut whole, mut ranged) = (0u64, 0u64);
    for (kappa, bytes, was_ranged) in journal {
        if quant_kappas.contains(kappa) {
            if *was_ranged {
                ranged += bytes;
            } else {
                whole += bytes;
            }
        }
    }
    assert!(whole > 0, "each artifact verifies whole at first touch");
    assert!(
        ranged > 0,
        "the scales bind as a range of the verified artifact (resolve_range)"
    );
    println!(
        "[quantized-transit] wide gas-phase: completion {text:?}; artifact traffic \
         {whole} B whole (first-touch verification) + {ranged} B ranged"
    );
}

// ──────────── S3 — idle derivation (row `idle-derivation`) ───────────────────

struct IdleDeriveState {
    session: hologram_ai::staged::GrowableStagedSession,
    windows: std::sync::Arc<std::sync::Mutex<Vec<(usize, bool)>>>,
    prederived: Option<usize>,
    window_before: usize,
    materializations_before: u64,
    materializations_after_prederive: u64,
    crossed: Option<(usize, bool)>,
}

#[when("a turn completes and the session pre-derives the next window bucket")]
async fn when_idle_prederive(w: &mut BddWorld) {
    use hologram_ai::commands::generate::{generate_stream, GenConfig};
    let store = w.staged_store.as_ref().expect("the fixture κ-store");
    let derived = StoreDir::new("idle-derive");
    let windows = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let mut session = {
        // Rebuild the growable helper inline to observe (window, resolved).
        let manifest = staged_manifest(false);
        let (keys, shapes): (Vec<String>, Vec<Vec<u64>>) = manifest.into_iter().unzip();
        let kappas: Vec<String> = keys
            .iter()
            .zip(&shapes)
            .map(|(name, dims)| kappa_of(&staged_tensor_bytes(name, dims)))
            .collect();
        let dtypes = vec![DType::F32; keys.len()];
        hologram_ai::staged::GrowableStagedSession::new(
            staged_config(false).to_string(),
            keys,
            kappas,
            shapes,
            dtypes,
            None,
            std::num::NonZeroU64::new(1).expect("1 is non-zero"),
            Box::new(DirKappaStore::new(&store.path)),
        )
        .expect("the growable staged session builds")
    };
    {
        let windows = std::sync::Arc::clone(&windows);
        session.set_window_observer(Box::new(move |win, resolved| {
            windows.lock().expect("lock").push((win, resolved));
        }));
    }
    session.set_derived_store(Box::new(hologram_ai::staged::DirDerivedStore::new(
        &derived.path,
    )));
    let cfg = GenConfig {
        max_tokens: Some(4),
        temperature: 0.0,
        ..Default::default()
    };
    let mut sink = Vec::new();
    generate_stream(&mut session, &DecimalTok, "3 141 59", &cfg, &mut sink)
        .expect("the turn completes");
    let window_before = windows
        .lock()
        .expect("lock")
        .last()
        .expect("a window built")
        .0;
    let materializations_before = session.materialization_count();
    let prederived = session
        .prederive_next_window()
        .expect("idle pre-derivation is inert on failure, not an error here");
    let materializations_after_prederive = session.materialization_count();
    w.derived_root = Some(derived);
    w.idle_state = Some(IdleDeriveState {
        session,
        windows,
        prederived,
        window_before,
        materializations_before,
        materializations_after_prederive,
        crossed: None,
    });
}

#[then("the pre-derivation moved no weights and left the resident window untouched")]
async fn then_idle_inert(w: &mut BddWorld) {
    let st = w.idle_state.as_ref().expect("the idle run");
    assert_eq!(
        st.prederived,
        Some((st.window_before * 2).min(STG_WINDOW as usize)),
        "the next geometric bucket is the entailed speculation"
    );
    assert_eq!(
        st.materializations_before, st.materializations_after_prederive,
        "speculation moves no weights"
    );
    let windows = st.windows.lock().expect("lock");
    assert_eq!(
        windows.len(),
        1,
        "the resident window never rebuilt during speculation: {windows:?}"
    );
    println!(
        "[idle-derivation] pre-derived the {}-token bucket off the token path",
        st.prederived.expect("prederived")
    );
}

#[when("a following turn crosses the window bucket boundary")]
async fn when_idle_crossing(w: &mut BddWorld) {
    use hologram_ai::commands::generate::{generate_stream, GenConfig};
    let st = w.idle_state.as_mut().expect("the idle run");
    let prompt: String = (0..60)
        .map(|i| (i % 9 + 1).to_string())
        .collect::<Vec<_>>()
        .join(" ");
    let cfg = GenConfig {
        max_tokens: Some(8),
        temperature: 0.0,
        ..Default::default()
    };
    let mut sink = Vec::new();
    generate_stream(&mut st.session, &DecimalTok, &prompt, &cfg, &mut sink)
        .expect("the crossing turn completes");
    st.crossed = st
        .windows
        .lock()
        .expect("lock")
        .iter()
        .copied()
        .find(|(win, _)| *win > st.window_before);
}

#[then("the crossing resolves the window from the derived store instead of compiling")]
async fn then_idle_resolved(w: &mut BddWorld) {
    let st = w.idle_state.as_ref().expect("the idle run");
    let (window, resolved) = st.crossed.expect("the boundary was crossed");
    assert_eq!(
        Some(window),
        st.prederived,
        "the crossing hit the speculated bucket"
    );
    assert!(
        resolved,
        "the pre-derived window must RESOLVE on crossing, not recompile"
    );
    assert!(
        st.session.derived_hits() >= 1,
        "the derived store served the speculation"
    );
    println!("[idle-derivation] crossing resolved the {window}-token bucket from derived κ");
}

// ─────── S3 — admission margin (row `stage-residency-cache`, margin) ─────────

#[when("a completion is generated under a margin-recording admission probe")]
async fn when_margin_probe(w: &mut BddWorld) {
    use hologram_ai::commands::generate::{generate_stream, GenConfig};
    let store = w.staged_store.as_ref().expect("the fixture κ-store");
    let kit = staged_kit();
    let cfg = GenConfig {
        max_tokens: Some(4),
        temperature: 0.0,
        ..Default::default()
    };

    // Accepting probe: record the margin each admission carries.
    let margins = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let windows = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let mut session = growable_staged_session(&store.path, windows);
    session.set_residency_budget(kit.total_weight_bytes * 2);
    {
        let margins = std::sync::Arc::clone(&margins);
        session.set_admission_probe(std::rc::Rc::new(move |margin| {
            margins.lock().expect("lock").push(margin);
            true
        }));
    }
    let mut sink = Vec::new();
    generate_stream(&mut session, &DecimalTok, "3 141 59 26 5", &cfg, &mut sink)
        .expect("probed generation completes");
    w.admission_margins = Some(margins.lock().expect("lock").clone());

    // Refusing probe: admission never granted ⇒ strict windowing.
    let windows = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let mut strict = growable_staged_session(&store.path, windows);
    strict.set_residency_budget(kit.total_weight_bytes * 2);
    strict.set_admission_probe(std::rc::Rc::new(|_| false));
    let mut sink = Vec::new();
    let text = generate_stream(&mut strict, &DecimalTok, "3 141 59 26 5", &cfg, &mut sink)
        .expect("refused generation completes");
    let passes = DecimalTok.encode(&text).len();
    w.admission_refused_materializations =
        Some((strict.materialization_count(), strict.stage_count(), passes));
}

#[then("every admission carried the largest stage's transient bound as its margin")]
async fn then_margin_value(w: &mut BddWorld) {
    let margins = w.admission_margins.as_ref().expect("the probe recorded");
    assert!(!margins.is_empty(), "admissions were consulted");
    let kit = staged_kit();
    // The expected margin: 4× the largest stage's raw weight bytes, computed
    // independently from each stage archive's κ-map and the fixture tensors.
    let size_of: std::collections::HashMap<String, u64> = staged_manifest(false)
        .into_iter()
        .map(|(name, dims)| {
            let bytes = staged_tensor_bytes(&name, &dims);
            (kappa_of(&bytes), bytes.len() as u64)
        })
        .collect();
    // Per-constant (NOT per-distinct-κ): two constants sharing content each
    // materialize their own copy, so the transient counts both.
    let largest: u64 = kit
        .stages
        .iter()
        .map(|archive| {
            kappa_requirements(archive)
                .expect("the κ-map parses")
                .iter()
                .map(|r| size_of.get(&r.kappa).copied().unwrap_or(0))
                .sum::<u64>()
        })
        .max()
        .expect("stages exist");
    // The bound is 3×raw bytes + 8 bytes per ELEMENT (two full F32
    // execution images: the widened panel and the kernel's pre-transposed
    // scratch); the fixture is F32, so elements = raw/4 and the bound is
    // 5×raw.
    let bound = largest * 3 + (largest / 4) * 8;
    assert!(
        margins.iter().all(|&m| m == bound),
        "the margin is the model's own transient bound ({bound}), got {margins:?}"
    );
    println!(
        "[stage-residency-cache] admission margin = {bound} bytes across {} admissions",
        margins.len()
    );
}

#[then("a probe that refuses admission yields strict windowing")]
async fn then_margin_refused(w: &mut BddWorld) {
    let (materializations, stage_count, passes) = w
        .admission_refused_materializations
        .expect("the refused run");
    assert_eq!(
        materializations,
        (stage_count * passes) as u64,
        "refused admission must rematerialize every stage every pass — a projection, never a refusal"
    );
}

// ──────── S3 — derived-artifact closure (row `derived-artifact-kappa`) ───────

/// A growable staged session over the fixture with a derived store rooted at
/// `derived_root`; returns (completion, derived_hits).
fn generate_with_derived(
    store_path: &std::path::Path,
    derived_root: &std::path::Path,
) -> (String, u64) {
    use hologram_ai::commands::generate::{generate_stream, GenConfig};
    use hologram_ai::staged::DirDerivedStore;
    let windows = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let mut session = growable_staged_session(store_path, windows);
    session.set_derived_store(Box::new(DirDerivedStore::new(derived_root)));
    let cfg = GenConfig {
        max_tokens: Some(8),
        temperature: 0.0,
        ..Default::default()
    };
    let mut sink = Vec::new();
    let text = generate_stream(&mut session, &DecimalTok, "3 141 59 26 5", &cfg, &mut sink)
        .expect("derived-store generation completes");
    (text, session.derived_hits())
}

#[when("two sessions with identical inputs generate over a shared derived store")]
async fn when_derived_warm(w: &mut BddWorld) {
    let store = w.staged_store.as_ref().expect("the fixture κ-store");
    let derived = StoreDir::new("derived-warm");
    let (cold_text, cold_hits) = generate_with_derived(&store.path, &derived.path);
    assert_eq!(
        cold_hits, 0,
        "the first session derives — nothing to resolve yet"
    );
    let (warm_text, warm_hits) = generate_with_derived(&store.path, &derived.path);
    w.derived_completions = Some((cold_text, warm_text));
    w.derived_hits = Some(warm_hits);
    w.derived_root = Some(derived);
}

#[then("the second session resolves its window from the derived store")]
async fn then_derived_resolved(w: &mut BddWorld) {
    let hits = w.derived_hits.expect("the warm session ran");
    assert!(
        hits >= 1,
        "a warm session with identical inputs must resolve, not re-derive"
    );
    println!("[derived-artifact-kappa] warm session resolved {hits} window(s)");
    // Reuse the shared parity assertion.
    w.staged_completions = w.derived_completions.clone();
}

#[when("a session generates over a derived store with a corrupted entry")]
async fn when_derived_corrupt(w: &mut BddWorld) {
    let store = w.staged_store.as_ref().expect("the fixture κ-store");
    let derived = StoreDir::new("derived-corrupt");
    let (cold_text, _) = generate_with_derived(&store.path, &derived.path);
    // Corrupt the persisted stage-0 archive of the (single) derivation.
    let key_dir = std::fs::read_dir(&derived.path)
        .expect("the derived store lists")
        .filter_map(|e| e.ok())
        .find(|e| e.path().is_dir())
        .expect("one derivation persisted");
    let stage0 = key_dir.path().join("0.holo");
    let mut bytes = std::fs::read(&stage0).expect("the derived archive reads");
    bytes[0] ^= 0xFF;
    std::fs::write(&stage0, bytes).expect("the corruption writes");

    let (recovered_text, hits) = generate_with_derived(&store.path, &derived.path);
    w.derived_completions = Some((cold_text, recovered_text));
    w.derived_hits = Some(hits);
    w.derived_root = Some(derived);
}

#[then("the window is re-derived instead of resolved")]
async fn then_derived_rederived(w: &mut BddWorld) {
    assert_eq!(
        w.derived_hits.expect("the recovery session ran"),
        0,
        "a corrupted entry must not count as a resolution"
    );
}

#[then("the completion is unaffected")]
async fn then_derived_unaffected(w: &mut BddWorld) {
    let (cold, recovered) = w.derived_completions.as_ref().expect("both sessions ran");
    assert_eq!(
        cold, recovered,
        "derive-as-recovery must be output-identical to the original derivation"
    );
    assert!(!cold.is_empty(), "the completion must be non-empty");
    println!("[derived-artifact-kappa] recovery by derivation: {recovered:?}");
}

#[then("the derived store holds the fresh derivation")]
async fn then_derived_rewritten(w: &mut BddWorld) {
    use hologram_ai::staged::DerivedStore;
    let derived = w.derived_root.as_ref().expect("the derived store");
    let key = std::fs::read_dir(&derived.path)
        .expect("the derived store lists")
        .filter_map(|e| e.ok())
        .find(|e| e.path().is_dir())
        .expect("the fresh derivation persisted")
        .file_name()
        .to_string_lossy()
        .to_string();
    let entry = hologram_ai::staged::DirDerivedStore::new(&derived.path).load(&key);
    let (stages, kappas) = entry.expect("the fresh entry loads");
    assert!(
        stages.iter().zip(&kappas).all(|(s, k)| kappa_of(s) == *k),
        "the rewritten derivation must verify against its recorded κs"
    );
}

// ─────────── S3 — saturation residency (row `saturation-residency`) ──────────

/// A two-tier κ-store: a [`DirKappaStore`] cache over an in-memory
/// provenance tier — the native mirror of the browser's OPFS-over-recorded-
/// provenance resolver. `invalidate` evicts the cache entry only; resolution
/// then falls through to provenance.
struct TieredStore {
    cache: DirKappaStore,
    cache_root: std::path::PathBuf,
    provenance: std::collections::HashMap<String, Vec<u8>>,
}

impl hologram_ai::materialize::KappaStore for TieredStore {
    fn resolve(&mut self, kappa: &str) -> anyhow::Result<Vec<u8>> {
        if let Ok(bytes) = self.cache.resolve(kappa) {
            return Ok(bytes);
        }
        self.provenance
            .get(kappa)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("κ `{kappa}` not present in any tier"))
    }

    fn invalidate(&mut self, kappa: &str) {
        let _ = std::fs::remove_file(self.cache_root.join(format!("{kappa}.bin")));
    }
}

#[when("a stage materializes over a corrupted cache backed by a provenance tier")]
async fn when_unpin_recovers(w: &mut BddWorld) {
    use hologram_ai::materialize::materialize_archive_with;
    let store_dir = staged_store("unpin-recover");
    let kit = staged_kit();
    let corrupted = corrupt_one_kappa(&store_dir.path, &kit.stages[0]);
    let provenance: std::collections::HashMap<String, Vec<u8>> = staged_manifest(false)
        .into_iter()
        .map(|(name, dims)| {
            let bytes = staged_tensor_bytes(&name, &dims);
            (kappa_of(&bytes), bytes)
        })
        .collect();
    let others = std::fs::read_dir(&store_dir.path)
        .expect("the cache dir lists")
        .count()
        - 1; // the corrupted entry evaporates; the rest must not
    let mut store = TieredStore {
        cache: DirKappaStore::new(&store_dir.path),
        cache_root: store_dir.path.clone(),
        provenance,
    };
    let mut fresh = std::collections::HashSet::new();
    w.unpin_result = Some(materialize_archive_with(
        &kit.stages[0],
        &mut store,
        &mut fresh,
    ));
    w.unpin_corrupt_kappa = Some(corrupted);
    w.unpin_other_entries = Some(others);
    w.unpin_store = Some(store_dir);
}

#[then("materialization succeeds on content re-verified from the deeper tier")]
async fn then_unpin_recovered(w: &mut BddWorld) {
    let result = w.unpin_result.as_ref().expect("the recovery ran");
    assert!(
        result.is_ok(),
        "cache corruption must degrade to a stream, never a dead end: {:?}",
        result.as_ref().err()
    );
    println!("[saturation-residency] corrupted cache recovered through provenance");
}

#[then("the corrupted cache entry has evaporated")]
async fn then_unpin_evaporated(w: &mut BddWorld) {
    let store = w.unpin_store.as_ref().expect("the cache dir");
    let kappa = w.unpin_corrupt_kappa.as_ref().expect("the corrupted κ");
    assert!(
        !store.path.join(format!("{kappa}.bin")).exists(),
        "the failed entry must leave the cache by the law that admitted it"
    );
}

#[then("every other cached entry is untouched")]
async fn then_unpin_others_intact(w: &mut BddWorld) {
    let store = w.unpin_store.as_ref().expect("the cache dir");
    let expected = w.unpin_other_entries.expect("the pre-count");
    let remaining = std::fs::read_dir(&store.path)
        .expect("the cache dir lists")
        .count();
    assert_eq!(
        remaining, expected,
        "only the failing entry is unpinned — bound content is never evicted"
    );
    println!("[saturation-residency] {remaining} bound entries untouched");
}

#[when("the provenance tier itself serves corrupted content")]
async fn when_unpin_bad_provenance(w: &mut BddWorld) {
    use hologram_ai::materialize::materialize_archive_with;
    let store_dir = staged_store("unpin-bad-prov");
    let kit = staged_kit();
    let corrupted = corrupt_one_kappa(&store_dir.path, &kit.stages[0]);
    // The deeper tier serves the SAME corrupted bytes: recovery must reject.
    let bad = std::fs::read(store_dir.path.join(format!("{corrupted}.bin")))
        .expect("the corrupted content reads");
    let mut store = TieredStore {
        cache: DirKappaStore::new(&store_dir.path),
        cache_root: store_dir.path.clone(),
        provenance: std::collections::HashMap::from([(corrupted.clone(), bad)]),
    };
    let mut fresh = std::collections::HashSet::new();
    let err = materialize_archive_with(&kit.stages[0], &mut store, &mut fresh)
        .expect_err("unverifiable recovery must stay loud");
    w.verified_fresh_err = Some(format!("{err:#}"));
    w.verified_corrupt_kappa = Some(corrupted);
}

// ─────────── S3 — session verified-κ (row `session-verified-kappa`) ──────────

/// Corrupt the store entry of a κ the given archive actually consumes (flip
/// a byte, same length) and return the corrupted label. The file name IS the
/// κ, so the content no longer reproduces its label.
fn corrupt_one_kappa(store_path: &std::path::Path, archive: &[u8]) -> String {
    let kappa = kappa_set(archive)
        .into_iter()
        .next()
        .expect("the archive requires at least one κ");
    let path = store_path.join(format!("{kappa}.bin"));
    let mut bytes = std::fs::read(&path).expect("the κ content reads");
    bytes[0] ^= 0xFF;
    std::fs::write(&path, bytes).expect("the corruption writes");
    kappa
}

#[when("a stage materializes twice in one session over a store corrupted between the passes")]
async fn when_verified_skip(w: &mut BddWorld) {
    use hologram_ai::materialize::materialize_archive_with;
    // A private store copy: this scenario mutates it.
    let store = staged_store("verified-skip");
    let kit = staged_kit();
    let archive = &kit.stages[0];
    let mut verified = std::collections::HashSet::new();

    let mut dir = DirKappaStore::new(&store.path);
    materialize_archive_with(archive, &mut dir, &mut verified)
        .expect("the first materialization verifies at first touch");

    let corrupted = corrupt_one_kappa(&store.path, archive);
    let mut dir = DirKappaStore::new(&store.path);
    w.verified_second_pass = Some(materialize_archive_with(archive, &mut dir, &mut verified));
    w.verified_corrupt_kappa = Some(corrupted);
    w.staged_store = Some(store);
}

#[then("the second pass succeeds without re-hashing the session-verified content")]
async fn then_verified_skip(w: &mut BddWorld) {
    let second = w
        .verified_second_pass
        .as_ref()
        .expect("the second pass ran");
    assert!(
        second.is_ok(),
        "a session-verified κ must rematerialize as read-only I/O (no re-hash): {:?}",
        second.as_ref().err()
    );
    println!("[session-verified-kappa] second pass was read-only resolution");
}

#[when(
    "a fresh session materializes a stage over a store corrupted after another session verified it"
)]
async fn when_verified_fresh(w: &mut BddWorld) {
    use hologram_ai::materialize::materialize_archive_with;
    // Self-contained: a first session verifies the store, the store is
    // corrupted, then a FRESH session (empty verified set) must reject.
    let store = staged_store("verified-fresh");
    let kit = staged_kit();
    let mut first = std::collections::HashSet::new();
    let mut dir = DirKappaStore::new(&store.path);
    materialize_archive_with(&kit.stages[0], &mut dir, &mut first)
        .expect("the first session verifies at first touch");
    let corrupted = corrupt_one_kappa(&store.path, &kit.stages[0]);

    let mut fresh = std::collections::HashSet::new();
    let mut dir = DirKappaStore::new(&store.path);
    let err = materialize_archive_with(&kit.stages[0], &mut dir, &mut fresh)
        .expect_err("a fresh session must verify at first touch and reject");
    w.verified_fresh_err = Some(format!("{err:#}"));
    w.verified_corrupt_kappa = Some(corrupted);
}

#[then("materialization is rejected naming the corrupted label")]
async fn then_verified_fresh(w: &mut BddWorld) {
    let err = w.verified_fresh_err.as_ref().expect("the rejection");
    let kappa = w.verified_corrupt_kappa.as_ref().expect("the corrupted κ");
    assert!(
        err.contains("integrity") && err.contains(kappa.trim_start_matches("blake3:")),
        "the rejection must name the corrupted label `{kappa}`: {err}"
    );
    println!("[session-verified-kappa] fresh session rejected loud: {err}");
}

// ───────────────── S3 — bounded embedding (row `bounded-embedding`) ──────────

/// A large-vocabulary decoder fixture whose narrow-dtype embedding table would
/// be a `vocab · hidden · 4` byte F32 image if cast whole. The gather-then-cast
/// front leaves the table at its native dtype, so no such image is ever a
/// tensor — the embedding path is bounded by the token rows, not the vocab.
const BE_VOCAB: u64 = 200_000;
const BE_HIDDEN: u64 = 64;

struct BoundedEmbed {
    config_json: String,
    keys: Vec<String>,
    shapes: Vec<Vec<u64>>,
    dtypes: Vec<DType>,
    stages: Vec<AiGraph>,
}

fn bounded_embed_config() -> serde_json::Value {
    serde_json::json!({
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": BE_HIDDEN, "intermediate_size": 128,
        "num_hidden_layers": 1, "num_attention_heads": 4,
        "num_key_value_heads": 2, "vocab_size": BE_VOCAB,
        "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
        "max_position_embeddings": 64, "tie_word_embeddings": false,
    })
}

fn bounded_embed_manifest() -> Vec<(String, Vec<u64>)> {
    let (h, i, v, kv) = (BE_HIDDEN, 128u64, BE_VOCAB, 32u64);
    let p = "model.layers.0";
    vec![
        ("model.embed_tokens.weight".into(), vec![v, h]),
        (format!("{p}.input_layernorm.weight"), vec![h]),
        (format!("{p}.self_attn.q_proj.weight"), vec![h, h]),
        (format!("{p}.self_attn.k_proj.weight"), vec![kv, h]),
        (format!("{p}.self_attn.v_proj.weight"), vec![kv, h]),
        (format!("{p}.self_attn.o_proj.weight"), vec![h, h]),
        (format!("{p}.post_attention_layernorm.weight"), vec![h]),
        (format!("{p}.mlp.gate_proj.weight"), vec![i, h]),
        (format!("{p}.mlp.up_proj.weight"), vec![i, h]),
        (format!("{p}.mlp.down_proj.weight"), vec![h, i]),
        ("model.norm.weight".into(), vec![h]),
        ("lm_head.weight".into(), vec![v, h]),
    ]
}

#[given("a large-vocabulary decoder with a narrow-dtype embedding table")]
async fn given_bounded_embed(w: &mut BddWorld) {
    let (keys, shapes): (Vec<String>, Vec<Vec<u64>>) = bounded_embed_manifest().into_iter().unzip();
    let dtypes = vec![DType::BF16; keys.len()];
    let config = bounded_embed_config();
    let stages = build_parametric_stage_graphs(
        &config,
        &keys,
        &dtypes,
        None,
        std::num::NonZeroU64::new(1).expect("1 is non-zero"),
    )
    .expect("the large-vocabulary stage graphs build");
    w.bounded_embed = Some(BoundedEmbed {
        config_json: config.to_string(),
        keys,
        shapes,
        dtypes,
        stages,
    });
}

#[when("its embedding stage is compiled")]
async fn when_bounded_embed_compiled(w: &mut BddWorld) {
    let be = w
        .bounded_embed
        .as_ref()
        .expect("the large-vocabulary fixture");
    // Weightless (k-form) compile: synthetic κ labels bind the External
    // weights, so no [vocab, hidden] tensor bytes are ever allocated — the
    // compile proves the lowered STRUCTURE carries no whole-vocab F32 image.
    let kappas: Vec<String> = be.keys.iter().map(|k| kappa_of(k.as_bytes())).collect();
    let stages = hologram_ai::staged::compile_stages(
        &be.config_json,
        &be.keys,
        &kappas,
        &be.shapes,
        &be.dtypes,
        None,
        std::num::NonZeroU64::new(1).expect("1 is non-zero"),
    )
    .expect("the large-vocabulary staged compile succeeds");
    w.bounded_embed_compiled = Some(stages);
}

#[then("the embedding stage compiles with no whole [vocab, hidden] F32 tensor")]
async fn then_bounded_embed_no_f32_table(w: &mut BddWorld) {
    let be = w.bounded_embed.as_ref().expect("the fixture");
    let compiled = w
        .bounded_embed_compiled
        .as_ref()
        .expect("the compiled stages");
    assert!(!compiled.is_empty(), "the staged compile produced archives");
    let embed_stage = &be.stages[0];
    // No tensor in the embedding stage is the whole [vocab, hidden] table at
    // F32 (the head's own `lm_head.weight.f32` is a separate stage's floor).
    let table = vec![BE_VOCAB, BE_HIDDEN];
    for (id, info) in &embed_stage.tensor_info {
        let dims: Vec<u64> = info.shape.iter().filter_map(|d| d.as_concrete()).collect();
        let name = embed_stage
            .tensor_names
            .get(id)
            .map(String::as_str)
            .unwrap_or("");
        assert!(
            !(dims == table && info.storage_dtype == DType::F32),
            "no whole [vocab, hidden] F32 tensor may exist in the embedding stage \
             (found `{name}`) — the {BE_VOCAB}×{BE_HIDDEN} F32 image is never materialized"
        );
    }
    let embed_id = embed_stage
        .tensor_names
        .iter()
        .find(|(_, n)| n.as_str() == "model.embed_tokens.weight")
        .map(|(id, _)| *id)
        .expect("the embedding table");
    assert_eq!(
        embed_stage.tensor_info[&embed_id].storage_dtype,
        DType::BF16,
        "the embedding table stays at its native BF16 storage dtype"
    );
    println!(
        "[bounded-embedding] {BE_VOCAB}-vocab embedding stage compiled ({} stages total); \
         no [vocab, hidden] F32 image; table stays BF16",
        compiled.len()
    );
}

#[then(
    "the embedding table is gathered at its native dtype and only the gathered rows are widened"
)]
async fn then_bounded_embed_native_gather(w: &mut BddWorld) {
    let be = w.bounded_embed.as_ref().expect("the fixture");
    let embed_stage = &be.stages[0];
    let embed_id = embed_stage
        .tensor_names
        .iter()
        .find(|(_, n)| n.as_str() == "model.embed_tokens.weight")
        .map(|(id, _)| *id)
        .expect("the embedding table");
    let gather = embed_stage
        .nodes
        .iter()
        .find(|n| matches!(n.op, AiOp::Gather { .. }))
        .expect("the embedding gather");
    assert_eq!(
        gather.inputs.first().copied(),
        Some(embed_id),
        "the gather reads the NATIVE embedding table"
    );
    let gathered = gather.outputs[0];
    assert_eq!(
        embed_stage.tensor_info[&gathered].storage_dtype,
        DType::BF16,
        "the gathered rows are the native dtype"
    );
    assert!(
        embed_stage
            .nodes
            .iter()
            .any(|n| matches!(n.op, AiOp::Cast { to: DType::F32 }) && n.inputs.contains(&gathered)),
        "only the gathered [batch, seq, hidden] rows are widened to F32"
    );
    println!(
        "[bounded-embedding] gather reads native BF16 table; only the gathered rows widened to F32"
    );
}

// ───────────── S3 — fused Phi3 execution (row `bounded-embedding`) ───────────

/// A tiny fused Phi3-family decoder (qkv_proj / gate_up_proj carved by
/// compile-time Slice), compiled monolithically and as one-layer stages from
/// the same deterministic weights — the memory-bounding fixes must leave the
/// logits identical across both execution modes.
fn fused_phi3_config() -> serde_json::Value {
    serde_json::json!({
        "architectures": ["Phi3ForCausalLM"],
        "hidden_size": 64, "intermediate_size": 128,
        "num_hidden_layers": 2, "num_attention_heads": 4,
        "num_key_value_heads": 2, "vocab_size": 512,
        "rms_norm_eps": 1e-5, "rope_theta": 10000.0,
        "max_position_embeddings": 128, "tie_word_embeddings": false,
        "rope_scaling": null, "sliding_window": null,
    })
}

fn fused_phi3_manifest() -> Vec<(String, Vec<u64>)> {
    let (h, i, v) = (64u64, 128u64, 512u64);
    // qkv rows = q(heads·head_dim 64) + k(kv 32) + v(kv 32) = 128;
    // gate_up rows = 2 · intermediate = 256.
    let (qkv_rows, gate_up_rows) = (128u64, 256u64);
    let mut m: Vec<(String, Vec<u64>)> = vec![("model.embed_tokens.weight".into(), vec![v, h])];
    for l in 0..2 {
        let p = format!("model.layers.{l}");
        m.push((format!("{p}.input_layernorm.weight"), vec![h]));
        m.push((format!("{p}.self_attn.qkv_proj.weight"), vec![qkv_rows, h]));
        m.push((format!("{p}.self_attn.o_proj.weight"), vec![h, h]));
        m.push((format!("{p}.post_attention_layernorm.weight"), vec![h]));
        m.push((
            format!("{p}.mlp.gate_up_proj.weight"),
            vec![gate_up_rows, h],
        ));
        m.push((format!("{p}.mlp.down_proj.weight"), vec![h, i]));
    }
    m.push(("model.norm.weight".into(), vec![h]));
    m.push(("lm_head.weight".into(), vec![v, h]));
    m
}

struct FusedPhi3Kit {
    monolithic: Vec<u8>,
    stages: Vec<Vec<u8>>,
}

fn build_fused_phi3_kit() -> FusedPhi3Kit {
    let (keys, shapes): (Vec<String>, Vec<Vec<u64>>) = fused_phi3_manifest().into_iter().unzip();
    let kappas: Vec<String> = keys
        .iter()
        .zip(&shapes)
        .map(|(name, dims)| kappa_of(&staged_tensor_bytes(name, dims)))
        .collect();
    let dtypes = vec![DType::F32; keys.len()];
    let config = fused_phi3_config();

    let mut graph = build_parametric_graph_from_manifest(&config, &keys, &dtypes, None)
        .expect("the monolithic fused Phi3 graph builds");
    let name_to_id: HashMap<String, u32> = graph
        .tensor_names
        .iter()
        .map(|(id, name)| (name.clone(), *id))
        .collect();
    for (i, key) in keys.iter().enumerate() {
        let id = *name_to_id.get(key).expect("manifest tensor in the graph");
        let info = ti(DType::F32, &shapes[i]);
        graph.tensor_info.insert(id, info.clone());
        graph.params.insert(
            id,
            AiParam::External {
                kappa: kappas[i].clone(),
                info,
                range: None,
            },
        );
    }
    let monolithic = compile_graph(graph);

    let stages = hologram_ai::staged::compile_stages(
        &config.to_string(),
        &keys,
        &kappas,
        &shapes,
        &dtypes,
        None,
        std::num::NonZeroU64::new(1).expect("1 is non-zero"),
    )
    .expect("the staged fused Phi3 fixture compiles");

    FusedPhi3Kit { monolithic, stages }
}

fn fused_phi3_kit() -> &'static FusedPhi3Kit {
    static KIT: OnceLock<FusedPhi3Kit> = OnceLock::new();
    KIT.get_or_init(build_fused_phi3_kit)
}

fn fused_phi3_store() -> StoreDir {
    let store = StoreDir::new("fused-phi3");
    let dir = DirKappaStore::new(&store.path);
    for (name, dims) in fused_phi3_manifest() {
        dir.insert(&staged_tensor_bytes(&name, &dims))
            .expect("persisting a fused Phi3 weight");
    }
    store
}

#[given("the tiny fused Phi3-family decoder fixture with its weights in a κ-store")]
async fn given_fused_phi3_fixture(w: &mut BddWorld) {
    w.fused_phi3_store = Some(fused_phi3_store());
}

#[when("the fused fixture is executed monolithically and through the staged runner")]
async fn when_fused_phi3_executed(w: &mut BddWorld) {
    let kit = fused_phi3_kit();
    let store = w
        .fused_phi3_store
        .as_ref()
        .expect("the fused fixture κ-store");
    let ids = staged_window_ids();

    let mut dir = DirKappaStore::new(&store.path);
    let mono = materialize_archive(&kit.monolithic, &mut dir)
        .expect("the fused monolithic archive materializes");
    let mut runner = HoloRunner::from_bytes(mono).expect("the fused monolithic archive loads");
    let mono_out = runner
        .execute(&[&ids])
        .expect("the fused monolithic pass executes");
    let mono_logits = mono_out
        .into_iter()
        .next()
        .expect("the fused monolithic pass produces logits")
        .bytes;

    let mut staged = StagedRunner::from_archives(
        kit.stages.clone(),
        Box::new(DirKappaStore::new(&store.path)),
    )
    .expect("the fused staged runner builds");
    let staged_out =
        LmSession::execute(&mut staged, &[&ids]).expect("the fused staged pass executes");
    let staged_logits = staged_out
        .into_iter()
        .next()
        .expect("the fused staged pass produces logits")
        .bytes;

    w.fused_phi3_exec = Some((mono_logits, staged_logits));
}

#[then("the fused staged logits are byte-identical to the fused monolithic logits")]
async fn then_fused_phi3_logits_equal(w: &mut BddWorld) {
    let (mono, staged) = w
        .fused_phi3_exec
        .as_ref()
        .expect("both fused executions ran");
    assert_eq!(mono.len(), staged.len(), "fused logits sizes must agree");
    assert_eq!(
        mono, staged,
        "fused staged execution must reproduce the fused monolithic logits byte-for-byte"
    );
    let logits = le_f32(staged);
    assert!(
        logits.iter().all(|v| v.is_finite()) && logits.iter().any(|v| v.abs() > 1e-6),
        "the fused logits must be finite and non-trivial"
    );
    println!(
        "[bounded-embedding] fused Phi3 byte-identical logits: {} bytes ({} f32 elements)",
        staged.len(),
        logits.len()
    );
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
            max_tokens: Some(n),
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
