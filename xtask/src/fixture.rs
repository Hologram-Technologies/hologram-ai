//! `gen-fixture` — build the hermetic `handshake-tiny` model and its committed
//! deterministic references (oracle `journey-reference`).
//!
//! Everything is derived from a recorded seed through the REAL k-form
//! pipeline: parametric graph from config → weightless compile (`AiParam::
//! External` κs) → κ-store materialization → three-message handshake via the
//! real generation loop. The artifacts land under `oracles/fixture/` and their
//! sha256s are printed as a `model/oracles.toml` snippet.

use anyhow::{Context, Result};
use hologram_ai::commands::generate::{apply_template, generate_stream, GenConfig};
use hologram_ai::materialize::{kappa_of, materialize_archive, DirKappaStore};
use hologram_ai::runner::HoloRunner;
use hologram_ai::{FixedSession, ModelCompiler, ModelSource};
use hologram_ai_common::{shape_from_concrete, AiParam, DType, TensorInfo};
use hologram_ai_tokenizer::NativeTokenizer;
use std::collections::{BTreeMap, HashMap};
use std::path::Path;

/// The base of the recorded seed search — regeneration reproduces the
/// artifacts bit-for-bit. Weights are pure noise, so some seeds greedily emit
/// eos immediately (an empty handshake turn); `gen_fixture` deterministically
/// takes the first seed from this base whose three turns are all non-empty,
/// and records it in the transcript.
const SEED_BASE: u64 = 42;
const SEED_ATTEMPTS: u64 = 64;
/// The handshake user messages (docs/conceptual-model/02-user-journey.md S4).
const HANDSHAKE: [&str; 3] = ["Hello there!", "How are you today?", "Say goodbye."];
/// The plain `{prompt}`-substitution chat template (not Jinja; the same
/// `apply_template` semantics the browser applies). Plain ASCII on purpose:
/// the fixture vocabulary excludes `<`, `>`, `|`, so the browser's
/// assistant-text cleaning is a no-op and transcript equality is exact.
const TEMPLATE: &str = "User:\n{prompt}\nAssistant:\n";
/// The `{response}` turn separator the browser weaves prior turns with
/// (`buildMultiTurnPrompt` in apps/web/src/pages/Chat.tsx).
const SEPARATOR: &str = "\nAssistant: {response}\nUser: ";
const MAX_TOKENS: usize = 8;

/// The `handshake-tiny` configuration (must satisfy
/// `model/usecases.toml` `[usecase.expects]` for id `handshake-tiny`).
fn config_json() -> serde_json::Value {
    serde_json::json!({
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "vocab_size": 512,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "max_position_embeddings": 128,
        "tie_word_embeddings": false,
        "torch_dtype": "float32",
        "bos_token_id": 1,
        "eos_token_id": 2,
        "model_type": "llama"
    })
}

/// xorshift64* — deterministic weight synthesis, seeded per tensor.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    /// Uniform in [-scale, scale].
    fn f32(&mut self, scale: f32) -> f32 {
        let u = (self.next() >> 40) as f32 / (1u64 << 24) as f32;
        (u * 2.0 - 1.0) * scale
    }
}

fn tensor_seed(name: &str, seed: u64) -> u64 {
    // FNV-1a over the name, mixed with the recorded seed.
    let mut h = 0xcbf29ce484222325u64;
    for b in name.bytes() {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x100000001b3);
    }
    h ^ seed
}

/// The full LlamaForCausalLM tensor manifest for the tiny config (untied head).
fn manifest() -> Vec<(String, Vec<u64>)> {
    let (hidden, inter, vocab, kv_dim) = (64u64, 128u64, 512u64, 32u64);
    let mut m: Vec<(String, Vec<u64>)> = Vec::new();
    m.push(("model.embed_tokens.weight".into(), vec![vocab, hidden]));
    for l in 0..2 {
        let p = format!("model.layers.{l}");
        m.push((format!("{p}.input_layernorm.weight"), vec![hidden]));
        m.push((format!("{p}.self_attn.q_proj.weight"), vec![hidden, hidden]));
        m.push((format!("{p}.self_attn.k_proj.weight"), vec![kv_dim, hidden]));
        m.push((format!("{p}.self_attn.v_proj.weight"), vec![kv_dim, hidden]));
        m.push((format!("{p}.self_attn.o_proj.weight"), vec![hidden, hidden]));
        m.push((format!("{p}.post_attention_layernorm.weight"), vec![hidden]));
        m.push((format!("{p}.mlp.gate_proj.weight"), vec![inter, hidden]));
        m.push((format!("{p}.mlp.up_proj.weight"), vec![inter, hidden]));
        m.push((format!("{p}.mlp.down_proj.weight"), vec![hidden, inter]));
    }
    m.push(("model.norm.weight".into(), vec![hidden]));
    m.push(("lm_head.weight".into(), vec![vocab, hidden]));
    m
}

/// Synthesize a tensor's F32 bytes: norms near 1.0, projections small-uniform.
fn tensor_bytes(name: &str, dims: &[u64], seed: u64) -> Vec<u8> {
    let n: u64 = dims.iter().product();
    let mut rng = Rng(tensor_seed(name, seed));
    let is_norm = name.contains("layernorm") || name.ends_with(".norm.weight");
    let mut out = Vec::with_capacity((n * 4) as usize);
    for _ in 0..n {
        let v = if is_norm {
            1.0 + rng.f32(0.02)
        } else {
            rng.f32(0.08)
        };
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// A char-level BPE tokenizer.json loadable by `NativeTokenizer` (and thus by
/// the identical wasm tokenizer path).
fn tokenizer_json() -> serde_json::Value {
    let mut vocab = serde_json::Map::new();
    vocab.insert("<unk>".into(), 0.into());
    vocab.insert("<s>".into(), 1.into());
    vocab.insert("</s>".into(), 2.into());
    // No `<`, `>`, `|`: the only tokens containing them are the specials
    // above, so generated text can never trip the browser's tag-cleaning.
    let chars = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?'-:\n";
    let mut id = 3u32;
    for c in chars.chars() {
        vocab.insert(c.to_string(), id.into());
        id += 1;
    }
    serde_json::json!({
        "version": "1.0",
        "added_tokens": [
            {"id": 0, "content": "<unk>", "special": true},
            {"id": 1, "content": "<s>", "special": true},
            {"id": 2, "content": "</s>", "special": true}
        ],
        "model": {
            "type": "BPE",
            "vocab": serde_json::Value::Object(vocab),
            "merges": [],
            "byte_fallback": false,
            "unk_token": "<unk>"
        }
    })
}

fn generation_config_json() -> serde_json::Value {
    serde_json::json!({
        "bos_token_id": 1,
        "eos_token_id": 2,
        "chat_template": TEMPLATE
    })
}

/// Serialize the manifest to safetensors bytes (reference crate = the format
/// authority; BTreeMap gives a canonical, reproducible header order).
fn safetensors_bytes(tensors: &[(String, Vec<u64>, Vec<u8>)]) -> Result<Vec<u8>> {
    use safetensors::tensor::{Dtype, TensorView};
    let views: BTreeMap<String, TensorView> = tensors
        .iter()
        .map(|(name, dims, bytes)| {
            let shape: Vec<usize> = dims.iter().map(|&d| d as usize).collect();
            let view = TensorView::new(Dtype::F32, shape, bytes)
                .with_context(|| format!("tensor view for {name}"))?;
            Ok((name.clone(), view))
        })
        .collect::<Result<_>>()?;
    safetensors::serialize(&views, &None).context("serializing safetensors")
}

/// Compile the k-form archive: parametric graph + `AiParam::External` κs —
/// the exact injection the browser's streamed compile performs.
fn compile_kform(
    config: &serde_json::Value,
    tensors: &[(String, Vec<u64>, Vec<u8>)],
) -> Result<Vec<u8>> {
    let keys: Vec<String> = tensors.iter().map(|(n, _, _)| n.clone()).collect();
    let dtypes: Vec<DType> = vec![DType::F32; keys.len()];
    let mut graph = hologram_ai_safetensors::parametric::build_parametric_graph_from_manifest(
        config,
        &keys,
        &dtypes,
        Some(128),
    )?;

    let mut name_to_id: HashMap<String, u32> = HashMap::new();
    for (id, name) in &graph.tensor_names {
        name_to_id.insert(name.clone(), *id);
    }
    for (name, dims, bytes) in tensors {
        let id = *name_to_id
            .get(name)
            .with_context(|| format!("graph does not reference manifest tensor `{name}`"))?;
        let info = TensorInfo::new(DType::F32, shape_from_concrete(dims));
        graph.tensor_info.insert(id, info.clone());
        graph.params.insert(
            id,
            AiParam::External {
                kappa: kappa_of(bytes),
                info,
            },
        );
    }
    Ok(ModelCompiler::default()
        .compile(ModelSource::AiGraph(graph))?
        .bytes)
}

/// Run the three-message handshake through the real generation loop.
fn handshake(
    holo_kform: &[u8],
    tensors: &[(String, Vec<u64>, Vec<u8>)],
    tokenizer: &NativeTokenizer,
    seed: u64,
) -> Result<Vec<serde_json::Value>> {
    let dir = std::env::temp_dir().join(format!("hai-gen-fixture-{}", std::process::id()));
    let store = DirKappaStore::new(&dir);
    for (_, _, bytes) in tensors {
        store.insert(bytes)?;
    }
    let mut store = store;
    let material = materialize_archive(holo_kform, &mut store)?;
    std::fs::remove_dir_all(&dir).ok();

    let runner = HoloRunner::from_bytes(material)?;
    let mut provider = FixedSession::new(runner);
    // Mirror the browser's GenOpts exactly: greedy (temperature 0), the
    // catalogue stop string, eos derived from the tokenizer (both sides run
    // the same `generate_stream`, so derivation is shared code).
    let cfg = GenConfig {
        max_tokens: MAX_TOKENS,
        temperature: 0.0,
        top_k: None,
        stop: vec!["\nUser:".to_string()],
        eos: None,
        seed,
    };

    // Mirror the browser exactly (Chat.tsx): the outer template wraps a
    // multi-turn SLOT — prior turns woven with the `{response}` separator —
    // and each assistant reply is cleaned before it re-enters the history.
    let mut slot_parts: Vec<String> = Vec::new();
    let mut turns = Vec::new();
    for user in HANDSHAKE {
        slot_parts.push(user.to_string());
        let slot = slot_parts.concat();
        let prompt = apply_template(Some(TEMPLATE), &slot);
        let mut sink = Vec::new();
        let completion = generate_stream(&mut provider, tokenizer, &prompt, &cfg, &mut sink)
            .with_context(|| format!("handshake turn for {user:?}"))?;
        let cleaned = clean_assistant_text(&completion);
        slot_parts.push(SEPARATOR.replace("{response}", &cleaned));
        turns.push(serde_json::json!({
            "user": user,
            "prompt": prompt,
            "completion": cleaned,
        }));
    }
    Ok(turns)
}

/// The subset of the browser's `cleanAssistantText` that can fire on the
/// fixture alphabet: special-token removal + trailing-whitespace trim.
fn clean_assistant_text(text: &str) -> String {
    text.replace("</s>", "")
        .replace("<s>", "")
        .replace("<unk>", "")
        .trim_end()
        .to_string()
}

fn write(path: &Path, bytes: &[u8]) -> Result<()> {
    std::fs::create_dir_all(path.parent().context("artifact path has a parent")?)?;
    std::fs::write(path, bytes).with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

/// Generate all artifacts + print the oracles.toml snippet.
pub fn gen_fixture() -> Result<()> {
    let root = hologram_ai_model::workspace_root();
    let out = root.join("oracles/fixture");

    let config = config_json();
    let config_bytes = serde_json::to_vec_pretty(&config)?;
    let tok_json = tokenizer_json();
    let tok_bytes = serde_json::to_vec_pretty(&tok_json)?;
    let gen_bytes = serde_json::to_vec_pretty(&generation_config_json())?;

    let tokenizer = NativeTokenizer::from_tokenizer_json_bytes(&tok_bytes)
        .context("the fixture tokenizer must load")?;

    // Deterministic seed search: the first seed whose three greedy turns are
    // all non-empty becomes the recorded fixture.
    let mut chosen = None;
    for seed in SEED_BASE..SEED_BASE + SEED_ATTEMPTS {
        let tensors: Vec<(String, Vec<u64>, Vec<u8>)> = manifest()
            .into_iter()
            .map(|(name, dims)| {
                let bytes = tensor_bytes(&name, &dims, seed);
                (name, dims, bytes)
            })
            .collect();
        let holo = compile_kform(&config, &tensors)?;
        let turns = handshake(&holo, &tensors, &tokenizer, seed)?;
        let all_nonempty = turns
            .iter()
            .all(|t| !t["completion"].as_str().unwrap_or("").trim().is_empty());
        if all_nonempty {
            chosen = Some((seed, tensors, turns));
            break;
        }
        println!("seed {seed}: a turn was empty; continuing the search");
    }
    let (seed, tensors, turns) = chosen.context(format!(
        "no seed in {SEED_BASE}..{} yields three non-empty handshake turns",
        SEED_BASE + SEED_ATTEMPTS
    ))?;
    let st_bytes = safetensors_bytes(&tensors)?;
    let transcript = serde_json::json!({
        "seed": seed,
        "temperature": 0.0,
        "max_tokens": MAX_TOKENS,
        "template": TEMPLATE,
        "separator": SEPARATOR,
        "eos_token_id": 2,
        "turns": turns,
    });
    let transcript_bytes = serde_json::to_vec_pretty(&transcript)?;

    let artifacts: [(&str, &[u8]); 5] = [
        ("config.json", &config_bytes),
        ("model.safetensors", &st_bytes),
        ("tokenizer.json", &tok_bytes),
        ("generation_config.json", &gen_bytes),
        ("reference-transcript.json", &transcript_bytes),
    ];
    println!("# oracles.toml artifacts for `journey-reference`:");
    for (name, bytes) in artifacts {
        let path = out.join(name);
        write(&path, bytes)?;
        let sha = super::sha256_file(&path)?;
        println!("[[oracle.artifacts]]");
        println!("path = \"oracles/fixture/{name}\"");
        println!("sha256 = \"{sha}\"");
        println!();
    }
    println!("gen-fixture: artifacts written to {}", out.display());
    for t in transcript["turns"].as_array().expect("turns array") {
        println!(
            "turn: {:?} → {:?}",
            t["user"].as_str().expect("user"),
            t["completion"].as_str().expect("completion")
        );
    }
    Ok(())
}
