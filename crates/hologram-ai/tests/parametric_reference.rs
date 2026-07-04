//! Correctness witness for the parametric (safetensors) decoder path.
//!
//! A naive, self-contained f32 reference implementation of the tiny Llama
//! forward pass — embed lookup → per-layer (rms_norm → q/k/v proj → RoPE
//! (rotate-half, non-interleaved) → causal GQA attention → o_proj → residual →
//! rms_norm → SwiGLU MLP → residual) → final rms_norm → lm_head — is computed
//! directly from the same deterministic weights the parametric graph is built
//! from. The compiled pipeline's logits at every real position must match the
//! reference per-element within 1e-3 relative + 1e-4 absolute, and the
//! external-κ (materialized) compile must be byte-identical to the inline one.

use hologram_ai::materialize::{kappa_of, materialize_archive, DirKappaStore};
use hologram_ai::runner::HoloRunner;
use hologram_ai::{ModelCompiler, ModelSource};
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
const WINDOW: usize = 128;

// ── Deterministic fixture (config + manifest + weights) ─────────────────────

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

/// The fixture tensor decoded as f32 (all fixture weights are f32).
fn tensor_f32(name: &str) -> Vec<f32> {
    let dims = manifest()
        .into_iter()
        .find(|(n, _)| n == name)
        .unwrap_or_else(|| panic!("{name} not in manifest"))
        .1;
    bytes_for(name, &dims)
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

// ── Naive reference forward pass ─────────────────────────────────────────────

/// `x · Wᵀ` for one position: `w` is the row-major `[out, in]` weight.
fn project(x: &[f32], w: &[f32], out_dim: usize) -> Vec<f32> {
    let in_dim = x.len();
    (0..out_dim)
        .map(|o| {
            let mut acc = 0f32;
            for i in 0..in_dim {
                acc += x[i] * w[o * in_dim + i];
            }
            acc
        })
        .collect()
}

fn rms_norm(x: &[f32], gamma: &[f32]) -> Vec<f32> {
    let mut ms = 0f32;
    for &v in x {
        ms += v * v;
    }
    ms /= x.len() as f32;
    let denom = (ms + EPS).sqrt();
    x.iter().zip(gamma).map(|(&v, &g)| v / denom * g).collect()
}

/// Rotate-half (non-interleaved) RoPE on one head vector at position `pos`:
/// pair partner of element `j` is `j ± head_dim/2`, `inv_freq = θ^(-2j/d)`.
fn rope(head: &mut [f32], pos: usize) {
    let half = HEAD_DIM / 2;
    let orig = head.to_vec();
    for j in 0..half {
        let inv_freq = THETA.powf(-2.0 * j as f64 / HEAD_DIM as f64);
        let angle = pos as f64 * inv_freq;
        let (c, s) = (angle.cos() as f32, angle.sin() as f32);
        head[j] = orig[j] * c - orig[j + half] * s;
        head[j + half] = orig[j + half] * c + orig[j] * s;
    }
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Per-position logit rows for `tokens` (causal: row `t` depends only on
/// tokens `0..=t`, so trailing window padding in the pipeline is irrelevant).
fn reference_logits(tokens: &[u32]) -> Vec<Vec<f32>> {
    let n = tokens.len();
    let embed = tensor_f32("model.embed_tokens.weight");
    let mut xs: Vec<Vec<f32>> = tokens
        .iter()
        .map(|&t| embed[t as usize * HIDDEN..(t as usize + 1) * HIDDEN].to_vec())
        .collect();

    for l in 0..LAYERS {
        let p = format!("model.layers.{l}");
        let w_attn_norm = tensor_f32(&format!("{p}.input_layernorm.weight"));
        let wq = tensor_f32(&format!("{p}.self_attn.q_proj.weight"));
        let wk = tensor_f32(&format!("{p}.self_attn.k_proj.weight"));
        let wv = tensor_f32(&format!("{p}.self_attn.v_proj.weight"));
        let wo = tensor_f32(&format!("{p}.self_attn.o_proj.weight"));
        let w_mlp_norm = tensor_f32(&format!("{p}.post_attention_layernorm.weight"));
        let wg = tensor_f32(&format!("{p}.mlp.gate_proj.weight"));
        let wu = tensor_f32(&format!("{p}.mlp.up_proj.weight"));
        let wd = tensor_f32(&format!("{p}.mlp.down_proj.weight"));

        // Q/K/V projections + RoPE, per position.
        let mut q = Vec::with_capacity(n);
        let mut k = Vec::with_capacity(n);
        let mut v = Vec::with_capacity(n);
        for (t, x) in xs.iter().enumerate() {
            let normed = rms_norm(x, &w_attn_norm);
            let mut qt = project(&normed, &wq, HEADS * HEAD_DIM);
            let mut kt = project(&normed, &wk, KV_HEADS * HEAD_DIM);
            let vt = project(&normed, &wv, KV_HEADS * HEAD_DIM);
            for h in 0..HEADS {
                rope(&mut qt[h * HEAD_DIM..(h + 1) * HEAD_DIM], t);
            }
            for h in 0..KV_HEADS {
                rope(&mut kt[h * HEAD_DIM..(h + 1) * HEAD_DIM], t);
            }
            q.push(qt);
            k.push(kt);
            v.push(vt);
        }

        // Causal grouped-query attention: query head h reads kv head h/group.
        let group = HEADS / KV_HEADS;
        let scale = (HEAD_DIM as f32).sqrt();
        let mut ctx = vec![vec![0f32; HEADS * HEAD_DIM]; n];
        for t in 0..n {
            for h in 0..HEADS {
                let g = h / group;
                let mut scores = Vec::with_capacity(t + 1);
                for k_row in k.iter().take(t + 1) {
                    let mut dot = 0f32;
                    for j in 0..HEAD_DIM {
                        dot += q[t][h * HEAD_DIM + j] * k_row[g * HEAD_DIM + j];
                    }
                    scores.push(dot / scale);
                }
                let max = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let exps: Vec<f32> = scores.iter().map(|&s| (s - max).exp()).collect();
                let sum: f32 = exps.iter().sum();
                for (u, v_row) in v.iter().enumerate().take(t + 1) {
                    let w = exps[u] / sum;
                    for j in 0..HEAD_DIM {
                        ctx[t][h * HEAD_DIM + j] += w * v_row[g * HEAD_DIM + j];
                    }
                }
            }
        }

        // o_proj + residual, then SwiGLU MLP + residual.
        for t in 0..n {
            let o = project(&ctx[t], &wo, HIDDEN);
            for i in 0..HIDDEN {
                xs[t][i] += o[i];
            }
            let h2 = rms_norm(&xs[t], &w_mlp_norm);
            let gate = project(&h2, &wg, INTER);
            let up = project(&h2, &wu, INTER);
            let act: Vec<f32> = gate.iter().zip(&up).map(|(&g, &u)| silu(g) * u).collect();
            let down = project(&act, &wd, HIDDEN);
            for i in 0..HIDDEN {
                xs[t][i] += down[i];
            }
        }
    }

    let w_norm = tensor_f32("model.norm.weight");
    let head = tensor_f32("lm_head.weight");
    xs.iter()
        .map(|x| project(&rms_norm(x, &w_norm), &head, VOCAB))
        .collect()
}

// ── Pipeline side ────────────────────────────────────────────────────────────

fn fixture_tensors() -> Vec<(String, Vec<u64>, Vec<u8>)> {
    manifest()
        .into_iter()
        .map(|(n, d)| {
            let b = bytes_for(&n, &d);
            (n, d, b)
        })
        .collect()
}

/// Build the parametric graph with params injected inline or as external κs.
fn build_graph(inline: bool) -> AiGraph {
    let tensors = fixture_tensors();
    let keys: Vec<String> = tensors.iter().map(|(n, _, _)| n.clone()).collect();
    let dtypes = vec![DType::F32; keys.len()];
    let mut graph = hologram_ai_safetensors::parametric::build_parametric_graph_from_manifest(
        &config_json(),
        &keys,
        &dtypes,
        Some(WINDOW as u64),
    )
    .expect("parametric graph builds");
    let mut name_to_id: HashMap<String, u32> = HashMap::new();
    for (id, name) in &graph.tensor_names {
        name_to_id.insert(name.clone(), *id);
    }
    for (name, dims, bytes) in &tensors {
        let id = *name_to_id.get(name).expect("manifest tensor in graph");
        let info = TensorInfo::new(DType::F32, shape_from_concrete(dims));
        graph.tensor_info.insert(id, info.clone());
        let param = if inline {
            AiParam::inline(bytes.clone(), info)
        } else {
            AiParam::External {
                kappa: kappa_of(bytes),
                info,
            }
        };
        graph.params.insert(id, param);
    }
    graph
}

fn compile(graph: AiGraph) -> Vec<u8> {
    ModelCompiler::default()
        .compile(ModelSource::AiGraph(graph))
        .expect("compile")
        .bytes
}

/// The fixed-window `input_ids` buffer: real tokens left-aligned at positions
/// `0..n`, zero-padded to the compiled window (mirrors how `generate_stream`
/// feeds a fixed session — logits at real positions are unaffected by trailing
/// padding under causal attention).
fn window_ids(tokens: &[u32]) -> Vec<u8> {
    let mut ids = vec![0i64; WINDOW];
    for (i, &t) in tokens.iter().enumerate() {
        ids[i] = t as i64;
    }
    ids.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn run_logits(holo: Vec<u8>, tokens: &[u32]) -> Vec<u8> {
    let mut runner = HoloRunner::from_bytes(holo).expect("archive loads");
    let ids = window_ids(tokens);
    let outputs = runner.execute(&[&ids]).expect("forward pass");
    assert_eq!(outputs.len(), 1, "one logits output");
    outputs.into_iter().next().expect("logits").bytes
}

const TOKENS: [u32; 6] = [3, 141, 59, 26, 5, 35];

// ── Witnesses ────────────────────────────────────────────────────────────────

#[test]
fn pipeline_logits_match_naive_reference() {
    let logits_bytes = run_logits(compile(build_graph(true)), &TOKENS);
    let logits: &[f32] = bytemuck::cast_slice(&logits_bytes);
    assert_eq!(
        logits.len(),
        WINDOW * VOCAB,
        "logits are [1, window, vocab]"
    );

    let reference = reference_logits(&TOKENS);
    let mut max_abs = 0f32;
    let mut max_rel = 0f32;
    for (t, ref_row) in reference.iter().enumerate() {
        let row = &logits[t * VOCAB..(t + 1) * VOCAB];
        for (i, (&got, &want)) in row.iter().zip(ref_row).enumerate() {
            let diff = (got - want).abs();
            max_abs = max_abs.max(diff);
            max_rel = max_rel.max(diff / want.abs().max(1e-12));
            assert!(
                diff <= 1e-4 + 1e-3 * want.abs(),
                "logits[pos {t}][tok {i}]: pipeline {got} vs reference {want} (|Δ| = {diff})"
            );
        }
    }
    eprintln!("reference parity: max |Δ| = {max_abs:e}, max relative = {max_rel:e}");
}

#[test]
fn external_kappa_compile_is_byte_identical_to_inline() {
    let inline_logits = run_logits(compile(build_graph(true)), &TOKENS);

    let kform = compile(build_graph(false));
    let dir = std::env::temp_dir().join(format!("hai-parametric-reference-{}", std::process::id()));
    let store = DirKappaStore::new(&dir);
    for (_, _, bytes) in &fixture_tensors() {
        store.insert(bytes).expect("κ insert");
    }
    let mut store = store;
    let materialized = materialize_archive(&kform, &mut store).expect("materializes");
    std::fs::remove_dir_all(&dir).ok();

    let material_logits = run_logits(materialized, &TOKENS);
    assert_eq!(
        inline_logits, material_logits,
        "materialized execution must be byte-identical to the inline compile"
    );
}
