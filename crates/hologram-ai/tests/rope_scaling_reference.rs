//! Correctness witnesses for the SCALED rotary laws (`rope_scaling`,
//! `partial_rotary_factor`) through the real pipelines.
//!
//! A self-contained f32 reference forward pass restates the reference
//! (HuggingFace `modeling_rope_utils.py`) frequency laws INDEPENDENTLY of
//! `hologram_ai_common::rope` and freezes each position's rope at the law of
//! the forward that computed it (`seq_len = pos + 1` under a KV cache —
//! exactly the cached-generation semantics). The compiled plans must match it
//! per element:
//!
//! * length-INDEPENDENT laws (`llama3` + partial rotary, `yarn`) run through
//!   BOTH realizations — the whole-window plan (compile-time kernel tables via
//!   `rope_rotate`) and the decode plan (runtime rows via `RopeSpec::rows`) —
//!   two independent in-repo implementations against one reference;
//! * length-DEPENDENT laws (`longrope`, `dynamic`) run through the decode
//!   plan across their pretrained boundary, and the whole-window compile must
//!   REFUSE loud (padded compile-time tables would bake the wrong realized
//!   length — a silent wrong number).

use hologram_ai::runner::HoloRunner;
use hologram_ai::{DecodeSession, ModelCompiler, ModelSource};
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
const WINDOW: usize = 32;

const TOKENS: [u32; 6] = [3, 141, 59, 26, 5, 35];

// ── The independent law restatement (the test's own oracle) ─────────────────

/// One rotary law, restated directly from the reference formulas — never via
/// `RopeSpec` (the point is two independent derivations).
#[derive(Clone)]
enum Law {
    /// llama3 piecewise-by-wavelength; `rotary_dim` < HEAD_DIM exercises the
    /// partial pass-through alongside it.
    Llama3 {
        factor: f64,
        low: f64,
        high: f64,
        orig: f64,
        rotary_dim: usize,
    },
    /// yarn NTK-by-parts with the default attention temperature.
    Yarn { factor: f64, orig: f64 },
    /// longrope short/long per-dim divisors switching at `orig`.
    LongRope {
        short: Vec<f64>,
        long: Vec<f64>,
        orig: usize,
        max: usize,
    },
    /// dynamic NTK base growth beyond `orig`.
    Dynamic { factor: f64, orig: usize },
}

impl Law {
    fn rotary_dim(&self) -> usize {
        match self {
            Law::Llama3 { rotary_dim, .. } => *rotary_dim,
            _ => HEAD_DIM,
        }
    }

    /// inv_freq for pair `j` at realized length `seq_len`.
    fn inv_freq(&self, j: usize, seq_len: usize) -> f64 {
        let r = self.rotary_dim() as f64;
        let plain = |base: f64| 1.0 / base.powf(2.0 * j as f64 / r);
        match self {
            Law::Llama3 {
                factor,
                low,
                high,
                orig,
                ..
            } => {
                let f = plain(THETA);
                let wavelen = 2.0 * std::f64::consts::PI / f;
                let low_wl = orig / low;
                let high_wl = orig / high;
                if wavelen < high_wl {
                    f
                } else if wavelen > low_wl {
                    f / factor
                } else {
                    let smooth = (orig / wavelen - low) / (high - low);
                    (1.0 - smooth) * f / factor + smooth * f
                }
            }
            Law::Yarn { factor, orig } => {
                let dim = r;
                let corr = |n: f64| {
                    dim * (orig / (n * 2.0 * std::f64::consts::PI)).ln() / (2.0 * THETA.ln())
                };
                let lo = corr(32.0).floor().max(0.0);
                let mut hi = corr(1.0).ceil().min(dim - 1.0);
                if hi <= lo {
                    hi = lo + 0.001;
                }
                let ramp = ((j as f64 - lo) / (hi - lo)).clamp(0.0, 1.0);
                let extrapolation = 1.0 - ramp;
                let f = plain(THETA);
                (f / factor) * (1.0 - extrapolation) + f * extrapolation
            }
            Law::LongRope {
                short, long, orig, ..
            } => {
                let ext = if seq_len > *orig { long } else { short };
                plain(THETA) / ext[j]
            }
            Law::Dynamic { factor, orig } => {
                if seq_len <= *orig {
                    plain(THETA)
                } else {
                    let grown = THETA
                        * ((factor * seq_len as f64 / *orig as f64) - (factor - 1.0))
                            .powf(r / (r - 2.0));
                    plain(grown)
                }
            }
        }
    }

    /// The attention temperature multiplying the rotary cos/sin.
    fn attention_factor(&self) -> f32 {
        match self {
            Law::Yarn { factor, .. } => (0.1 * factor.ln() + 1.0) as f32,
            Law::LongRope { orig, max, .. } => {
                let factor = *max as f64 / *orig as f64;
                if factor <= 1.0 {
                    1.0
                } else {
                    (1.0 + factor.ln() / (*orig as f64).ln()).sqrt() as f32
                }
            }
            _ => 1.0,
        }
    }

    /// The `config.json` fragment publishing this law.
    fn config_extras(&self, config: &mut serde_json::Value) {
        match self {
            Law::Llama3 {
                factor,
                low,
                high,
                orig,
                rotary_dim,
            } => {
                config["rope_scaling"] = serde_json::json!({
                    "rope_type": "llama3", "factor": factor,
                    "low_freq_factor": low, "high_freq_factor": high,
                    "original_max_position_embeddings": orig,
                });
                config["partial_rotary_factor"] =
                    serde_json::json!(*rotary_dim as f64 / HEAD_DIM as f64);
            }
            Law::Yarn { factor, orig } => {
                config["rope_scaling"] = serde_json::json!({
                    "type": "yarn", "factor": factor,
                    "original_max_position_embeddings": orig,
                });
            }
            Law::LongRope {
                short, long, orig, ..
            } => {
                config["rope_scaling"] = serde_json::json!({
                    "type": "longrope", "short_factor": short, "long_factor": long,
                });
                config["original_max_position_embeddings"] = serde_json::json!(orig);
            }
            Law::Dynamic { factor, orig } => {
                config["rope_scaling"] = serde_json::json!({
                    "type": "dynamic", "factor": factor,
                    "original_max_position_embeddings": orig,
                });
            }
        }
    }
}

/// Rotate-half rope on one head at `pos`, frozen at the law of the forward
/// that realized it (`seq_len`) — pass-through dims `rotary_dim..` untouched.
fn ref_rope(head: &mut [f32], pos: usize, seq_len: usize, law: &Law) {
    let r = law.rotary_dim();
    let half = r / 2;
    let a = law.attention_factor();
    let orig = head.to_vec();
    for j in 0..half {
        let angle = pos as f64 * law.inv_freq(j, seq_len);
        let (c, s) = (angle.cos() as f32 * a, angle.sin() as f32 * a);
        head[j] = orig[j] * c - orig[j + half] * s;
        head[j + half] = orig[j + half] * c + orig[j] * s;
    }
}

// ── Deterministic fixture (config + manifest + weights) ─────────────────────

fn config_json(law: &Law) -> serde_json::Value {
    let mut config = serde_json::json!({
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": HIDDEN, "intermediate_size": INTER, "num_hidden_layers": LAYERS,
        "num_attention_heads": HEADS, "num_key_value_heads": KV_HEADS, "vocab_size": VOCAB,
        "rms_norm_eps": EPS, "rope_theta": THETA, "max_position_embeddings": WINDOW,
        "tie_word_embeddings": false, "torch_dtype": "float32",
        "bos_token_id": 1, "eos_token_id": 2, "model_type": "llama"
    });
    law.config_extras(&mut config);
    config
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

fn tensor_f32(name: &str) -> Vec<f32> {
    let dims = manifest()
        .into_iter()
        .find(|(n, _)| n == name)
        .map(|(_, d)| d)
        .expect("manifest tensor");
    bytemuck::cast_slice::<u8, f32>(&bytes_for(name, &dims)).to_vec()
}

// ── Reference forward (naive, causal GQA, frozen-law rope) ───────────────────

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

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Per-position logit rows under cached-generation semantics: position `t`'s
/// q/k are roped at the law realized when it was computed (`seq_len = t+1`)
/// and its K stays frozen in the cache thereafter.
fn reference_logits(tokens: &[u32], law: &Law) -> Vec<Vec<f32>> {
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

        let mut q = Vec::with_capacity(n);
        let mut k = Vec::with_capacity(n);
        let mut v = Vec::with_capacity(n);
        for (t, x) in xs.iter().enumerate() {
            let normed = rms_norm(x, &w_attn_norm);
            let mut qt = project(&normed, &wq, HEADS * HEAD_DIM);
            let mut kt = project(&normed, &wk, KV_HEADS * HEAD_DIM);
            let vt = project(&normed, &wv, KV_HEADS * HEAD_DIM);
            for h in 0..HEADS {
                ref_rope(&mut qt[h * HEAD_DIM..(h + 1) * HEAD_DIM], t, t + 1, law);
            }
            for h in 0..KV_HEADS {
                ref_rope(&mut kt[h * HEAD_DIM..(h + 1) * HEAD_DIM], t, t + 1, law);
            }
            q.push(qt);
            k.push(kt);
            v.push(vt);
        }

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

fn inject_params(graph: &mut AiGraph) {
    let mut name_to_id: HashMap<String, u32> = HashMap::new();
    for (id, name) in &graph.tensor_names {
        name_to_id.insert(name.clone(), *id);
    }
    for (name, dims, bytes) in &fixture_tensors() {
        let id = *name_to_id.get(name).expect("manifest tensor in graph");
        let info = TensorInfo::new(DType::F32, shape_from_concrete(dims));
        graph.tensor_info.insert(id, info.clone());
        graph
            .params
            .insert(id, AiParam::inline(bytes.clone(), info));
    }
}

fn build_window_graph(law: &Law) -> AiGraph {
    let tensors = fixture_tensors();
    let keys: Vec<String> = tensors.iter().map(|(n, _, _)| n.clone()).collect();
    let dtypes = vec![DType::F32; keys.len()];
    let mut graph = hologram_ai_safetensors::parametric::build_parametric_graph_from_manifest(
        &config_json(law),
        &keys,
        &dtypes,
        Some(WINDOW as u64),
    )
    .expect("parametric graph builds");
    inject_params(&mut graph);
    graph
}

fn build_decode_graph(law: &Law, bucket: u64) -> AiGraph {
    let tensors = fixture_tensors();
    let keys: Vec<String> = tensors.iter().map(|(n, _, _)| n.clone()).collect();
    let dtypes = vec![DType::F32; keys.len()];
    let mut graph =
        hologram_ai_safetensors::parametric::build_parametric_decode_graph_from_manifest(
            &config_json(law),
            &keys,
            &dtypes,
            bucket,
        )
        .expect("decode graph builds");
    inject_params(&mut graph);
    graph
}

fn compile(graph: AiGraph) -> anyhow::Result<Vec<u8>> {
    Ok(ModelCompiler::default()
        .compile(ModelSource::AiGraph(graph))?
        .bytes)
}

fn decode_session(law: &Law) -> DecodeSession<HoloRunner> {
    let bytes = compile(build_decode_graph(law, 8)).expect("decode compiles");
    let runner = HoloRunner::from_bytes(bytes).expect("decode loads");
    // The session's law comes through the REAL parse — config.json → spec.
    let spec = hologram_ai_safetensors::parametric::rope_spec_from_config(&config_json(law))
        .expect("the rotary law parses");
    DecodeSession::new(runner, spec, WINDOW as u64).expect("decode session opens")
}

fn window_logits_at(runner: &mut HoloRunner, tokens: &[u32], pos: usize) -> Vec<f32> {
    let mut ids = vec![0i64; WINDOW];
    for (i, &t) in tokens.iter().enumerate() {
        ids[i] = t as i64;
    }
    let ids: Vec<u8> = ids.iter().flat_map(|v| v.to_le_bytes()).collect();
    let lp = (pos as i64).to_le_bytes();
    let outputs = runner.execute(&[&ids, &lp]).expect("forward pass");
    assert_eq!(outputs.len(), 1, "one logits output");
    bytemuck::cast_slice::<u8, f32>(&outputs.into_iter().next().expect("logits").bytes).to_vec()
}

fn assert_rows_match(got: &[f32], want: &[f32], label: &str, t: usize) {
    assert_eq!(got.len(), want.len(), "{label}: row width at pos {t}");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= 1e-4 + 1e-3 * w.abs(),
            "{label}: logits[pos {t}][tok {i}]: got {g} vs reference {w}"
        );
    }
}

/// Decode the fixture step by step and compare every position against the
/// frozen-law reference.
fn assert_decode_matches_reference(law: &Law, label: &str) {
    let reference = reference_logits(&TOKENS, law);
    let mut session = decode_session(law);
    for (t, &tok) in TOKENS.iter().enumerate() {
        let step = session.step(tok as i64).expect("decode step");
        assert_rows_match(&step, &reference[t], label, t);
    }
}

/// Run the whole-window plan and compare every real position against the
/// same reference (length-independent laws only: frozen == global).
fn assert_window_matches_reference(law: &Law, label: &str) {
    let reference = reference_logits(&TOKENS, law);
    let bytes = compile(build_window_graph(law)).expect("window compiles");
    let mut runner = HoloRunner::from_bytes(bytes).expect("window loads");
    for (t, want) in reference.iter().enumerate() {
        let row = window_logits_at(&mut runner, &TOKENS, t);
        assert_rows_match(&row, want, label, t);
    }
}

// ── Witnesses ────────────────────────────────────────────────────────────────

#[test]
fn llama3_with_partial_rotary_matches_reference_on_both_plans() {
    // Bounds chosen so the fixture's frequencies land in all three llama3
    // pieces (kept / ramped / interpolated) at HEAD_DIM 16 — and half the
    // head passes through unrotated.
    let law = Law::Llama3 {
        factor: 8.0,
        low: 1.0,
        high: 4.0,
        orig: 64.0,
        rotary_dim: HEAD_DIM / 2,
    };
    assert_decode_matches_reference(&law, "llama3+partial decode");
    assert_window_matches_reference(&law, "llama3+partial window");
}

#[test]
fn yarn_attention_temperature_flows_through_both_plans() {
    let law = Law::Yarn {
        factor: 4.0,
        orig: 16.0,
    };
    assert_decode_matches_reference(&law, "yarn decode");
    assert_window_matches_reference(&law, "yarn window");
}

#[test]
fn longrope_decode_crosses_the_boundary_and_the_padded_window_refuses() {
    let half = HEAD_DIM / 2;
    let law = Law::LongRope {
        short: vec![1.0; half],
        long: (0..half).map(|j| 2.0 + j as f64).collect(),
        orig: 3,
        max: WINDOW,
    };
    // The decode path realizes the boundary exactly: steps 0..3 use the short
    // set, steps 3.. the long set (TOKENS crosses mid-generation).
    assert_decode_matches_reference(&law, "longrope decode");
    // The padded whole-window plan would bake the law at seq = WINDOW —
    // wrong for every realized length ≤ the boundary — so it refuses loud.
    let err = compile(build_window_graph(&law)).expect_err("padded tables must refuse");
    assert!(
        format!("{err:#}").contains("length-dependent"),
        "names the refusal: {err:#}"
    );
}

#[test]
fn dynamic_ntk_decode_grows_the_base_and_the_padded_window_refuses() {
    let law = Law::Dynamic {
        factor: 2.0,
        orig: 3,
    };
    assert_decode_matches_reference(&law, "dynamic decode");
    let err = compile(build_window_graph(&law)).expect_err("padded tables must refuse");
    assert!(
        format!("{err:#}").contains("length-dependent"),
        "names the refusal: {err:#}"
    );
}
