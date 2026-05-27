//! Browser (WebAssembly) entry point for hologram-ai — ADR-0017.
//!
//! GitHub Pages is static hosting with no server, so the platform runs
//! **client-side**. This crate is a thin `wasm-bindgen` shell over the **real**
//! pipeline — it reuses `ModelCompiler`, `HoloRunner`, and the generation loop
//! from the `hologram-ai` facade (built `default-features = false`: no native
//! downloader, no rayon — neither compiles on wasm32). No logic is
//! reimplemented; the browser drives the same code paths as the CLI.
//!
//! Verbs (over byte buffers): `compile` (ONNX → `.holo`), `describe` (ports),
//! `run` (arbitrary forward pass, `--fill`-style), `generate` (autoregressive).

use hologram_ai::commands::generate::{apply_template, generate_stream, GenConfig};
use hologram_ai::{HoloRunner, ModelCompiler, ModelSource};
use hologram_ai_tokenizer::NativeTokenizer;
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

/// Surface Rust panics in the browser console. Runs on module init.
#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
}

fn err(e: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&e.to_string())
}

// ── compile ─────────────────────────────────────────────────────────────────

/// Compile an ONNX model (bytes) to a `.holo` archive (bytes). The real
/// `ModelCompiler` pipeline — import → optimize → lower → compile — runs in the
/// browser. Returns the archive bytes.
#[wasm_bindgen]
pub fn compile(onnx: &[u8]) -> Result<Vec<u8>, JsValue> {
    let archive = ModelCompiler::default()
        .compile(ModelSource::OnnxBytes(onnx.to_vec()))
        .map_err(|e| err(format!("compile: {e:#}")))?;
    Ok(archive.bytes)
}

// ── describe ────────────────────────────────────────────────────────────────

#[derive(Serialize, Deserialize)]
pub struct Port {
    pub name: String,
    pub dtype: u8,
    pub dtype_name: String,
    pub element_count: usize,
    pub shape: Vec<usize>,
    pub bytes: usize,
}

#[derive(Serialize, Deserialize)]
pub struct ModelInfo {
    pub inputs: Vec<Port>,
    pub outputs: Vec<Port>,
}

fn dtype_name(tag: u8) -> &'static str {
    match tag {
        0 => "bool", 1 => "u8", 2 => "i8", 3 => "u64", 4 => "i32", 5 => "i64",
        6 => "f16", 7 => "bf16", 8 => "f32", 9 => "f64", 10 => "i4", _ => "?",
    }
}

fn ports(info: &[hologram_ai::runner::PortInfo], sizes: &[usize]) -> Vec<Port> {
    info.iter()
        .zip(sizes.iter())
        .map(|(p, &bytes)| Port {
            name: p.name.clone(),
            dtype: p.dtype,
            dtype_name: dtype_name(p.dtype).to_string(),
            element_count: p.element_count,
            shape: p.shape.clone(),
            bytes,
        })
        .collect()
}

/// Inspect a compiled `.holo`: its named input/output ports.
#[wasm_bindgen]
pub fn describe(holo: &[u8]) -> Result<JsValue, JsValue> {
    let runner = HoloRunner::from_bytes(holo.to_vec()).map_err(err)?;
    let info = ModelInfo {
        inputs: ports(&runner.input_port_info(), &runner.input_byte_sizes()),
        outputs: ports(&runner.output_port_info(), &runner.output_byte_sizes()),
    };
    serde_wasm_bindgen::to_value(&info).map_err(err)
}

// ── run (arbitrary forward pass) ──────────────────────────────────────────────

#[derive(Serialize, Deserialize)]
pub struct Output {
    pub dtype: u8,
    pub dtype_name: String,
    pub element_count: usize,
    pub values: Vec<f64>,
}

/// Synthesize an input buffer from a fill value (`None` ⇒ zeros). Total over
/// every dtype, so any port is fillable.
fn synth(byte_size: usize, element_count: usize, dtype: u8, fill: Option<f64>) -> Vec<u8> {
    let Some(v) = fill else { return vec![0u8; byte_size] };
    if dtype == 10 {
        let nib = (v as i64 as u8) & 0x0F;
        return vec![nib | (nib << 4); byte_size];
    }
    let mut out = Vec::with_capacity(byte_size);
    for _ in 0..element_count {
        match dtype {
            0 | 1 => out.push(v as u8),
            2 => out.push(v as i8 as u8),
            3 => out.extend_from_slice(&(v as u64).to_le_bytes()),
            4 => out.extend_from_slice(&(v as i32).to_le_bytes()),
            5 => out.extend_from_slice(&(v as i64).to_le_bytes()),
            6 => out.extend_from_slice(&half::f16::from_f64(v).to_le_bytes()),
            7 => out.extend_from_slice(&half::bf16::from_f64(v).to_le_bytes()),
            9 => out.extend_from_slice(&v.to_le_bytes()),
            _ => out.extend_from_slice(&(v as f32).to_le_bytes()),
        }
    }
    out
}

/// Decode an output buffer to `f64` values for every dtype (total).
fn decode(bytes: &[u8], dtype: u8) -> Vec<f64> {
    let conv = |w: usize, f: &dyn Fn(&[u8]) -> f64| bytes.chunks_exact(w).map(f).collect::<Vec<_>>();
    match dtype {
        0 | 1 => bytes.iter().map(|&b| b as f64).collect(),
        2 => bytes.iter().map(|&b| b as i8 as f64).collect(),
        3 => conv(8, &|c| u64::from_le_bytes(c.try_into().unwrap()) as f64),
        4 => conv(4, &|c| i32::from_le_bytes(c.try_into().unwrap()) as f64),
        5 => conv(8, &|c| i64::from_le_bytes(c.try_into().unwrap()) as f64),
        6 => conv(2, &|c| f64::from(half::f16::from_le_bytes(c.try_into().unwrap()))),
        7 => conv(2, &|c| f64::from(half::bf16::from_le_bytes(c.try_into().unwrap()))),
        8 => conv(4, &|c| f32::from_le_bytes(c.try_into().unwrap()) as f64),
        9 => conv(8, &|c| f64::from_le_bytes(c.try_into().unwrap())),
        10 => bytes
            .iter()
            .flat_map(|&b| {
                let s = |n: i8| if n >= 8 { (n - 16) as f64 } else { n as f64 };
                [s((b & 0x0F) as i8), s((b >> 4) as i8)]
            })
            .collect(),
        _ => bytes.iter().map(|&b| b as f64).collect(),
    }
}

/// Run one forward pass over an arbitrary compiled model (mirrors `run --fill`).
/// `inputs` is a JS array of byte arrays by graph-input index; empty/omitted
/// entries are synthesized from `fill` (a number, or undefined ⇒ zeros).
#[wasm_bindgen]
pub fn run(holo: &[u8], inputs: JsValue, fill: Option<f64>) -> Result<JsValue, JsValue> {
    let provided: Vec<Vec<u8>> = if inputs.is_undefined() || inputs.is_null() {
        Vec::new()
    } else {
        serde_wasm_bindgen::from_value(inputs).map_err(err)?
    };
    let mut runner = HoloRunner::from_bytes(holo.to_vec()).map_err(err)?;
    let in_info = runner.input_port_info();
    let in_sizes = runner.input_byte_sizes();
    if !provided.is_empty() && provided.len() != in_info.len() {
        return Err(err(format!("expected {} input(s), got {}", in_info.len(), provided.len())));
    }

    let mut owned: Vec<Vec<u8>> = Vec::with_capacity(in_info.len());
    for (i, p) in in_info.iter().enumerate() {
        let want = in_sizes[i];
        match provided.get(i).filter(|b| !b.is_empty()) {
            Some(b) if b.len() == want => owned.push(b.clone()),
            Some(b) => return Err(err(format!("input[{i}] is {} bytes but the model expects {want}", b.len()))),
            None => owned.push(synth(want, p.element_count, p.dtype, fill)),
        }
    }

    let refs: Vec<&[u8]> = owned.iter().map(|v| v.as_slice()).collect();
    let outputs = runner.execute(&refs).map_err(|e| err(format!("execute: {e:#}")))?;
    let out_info = runner.output_port_info();
    let results: Vec<Output> = outputs
        .iter()
        .enumerate()
        .map(|(i, o)| {
            let dtype = out_info.get(i).map(|p| p.dtype).unwrap_or(8);
            Output {
                dtype,
                dtype_name: dtype_name(dtype).to_string(),
                element_count: out_info.get(i).map(|p| p.element_count).unwrap_or(0),
                values: decode(&o.bytes, dtype),
            }
        })
        .collect();
    serde_wasm_bindgen::to_value(&results).map_err(err)
}

// ── generate (autoregressive) ─────────────────────────────────────────────────

/// Generation options (all optional; sensible defaults applied).
#[derive(Deserialize, Default)]
pub struct GenOpts {
    pub prompt_template: Option<String>,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_k: Option<usize>,
    #[serde(default)]
    pub stop: Vec<String>,
    pub eos: Option<u32>,
    pub seed: Option<u64>,
}

/// Autoregressive text generation over a compiled causal LM — the real loop
/// (`generate_stream`). The tokenizer comes from `tokenizer_json` (bytes) when
/// given, else from the archive's baked-in extension. Returns the generated text.
#[wasm_bindgen]
pub fn generate(
    holo: &[u8],
    tokenizer_json: Option<Vec<u8>>,
    prompt: &str,
    opts: JsValue,
) -> Result<String, JsValue> {
    let opts: GenOpts = if opts.is_undefined() || opts.is_null() {
        GenOpts::default()
    } else {
        serde_wasm_bindgen::from_value(opts).map_err(err)?
    };
    let mut runner = HoloRunner::from_bytes(holo.to_vec()).map_err(err)?;

    let tokenizer = match tokenizer_json {
        Some(bytes) => NativeTokenizer::from_tokenizer_json_bytes(&bytes).map_err(err)?,
        None => {
            let embedded = runner
                .extension("tokenizer.json")
                .ok_or_else(|| err("no tokenizer: none embedded in the archive and none supplied"))?;
            NativeTokenizer::from_tokenizer_json_bytes(embedded).map_err(err)?
        }
    };

    let cfg = GenConfig {
        max_tokens: opts.max_tokens.unwrap_or(64),
        temperature: opts.temperature.unwrap_or(0.0),
        top_k: opts.top_k,
        stop: opts.stop,
        eos: opts.eos,
        seed: opts.seed.unwrap_or(0x9E3779B97F4A7C15),
    };
    let templated = apply_template(opts.prompt_template.as_deref(), prompt);

    let mut sink: Vec<u8> = Vec::new();
    generate_stream(&mut runner, &tokenizer, &templated, &cfg, &mut sink)
        .map_err(|e| err(format!("generate: {e:#}")))?;
    String::from_utf8(sink).map_err(err)
}

#[cfg(test)]
mod tests {
    use super::*;
    use hologram_ai::compiler::ArchiveSections;
    use hologram_ai_common::{shape_from_concrete, AiGraph, AiNode, AiOp, AiParam, DType, TensorInfo};
    use std::collections::HashMap;
    use wasm_bindgen_test::*;

    fn ti(dt: DType, dims: &[u64]) -> TensorInfo {
        TensorInfo::new(dt, shape_from_concrete(dims))
    }

    // [1,4]·[4,4 identity] matmul — for describe/run.
    fn matmul_onnxless() -> Vec<u8> {
        let (x, w, y) = (0u32, 1u32, 2u32);
        let mut t = HashMap::new();
        t.insert(x, ti(DType::F32, &[1, 4]));
        t.insert(w, ti(DType::F32, &[4, 4]));
        t.insert(y, ti(DType::F32, &[1, 4]));
        let mut wb = vec![0u8; 64];
        for k in 0..4 {
            wb[(k * 4 + k) * 4..(k * 4 + k) * 4 + 4].copy_from_slice(&1.0f32.to_le_bytes());
        }
        let mut params = HashMap::new();
        params.insert(w, AiParam::inline(wb, t[&w].clone()));
        let g = AiGraph {
            name: "mm".into(),
            nodes: vec![AiNode::new(0, AiOp::MatMul, vec![x, w], vec![y])],
            inputs: vec![x],
            outputs: vec![y],
            input_names: vec!["x".into()],
            output_names: vec!["y".into()],
            params,
            tensor_info: t,
            metadata: HashMap::new(),
            warnings: vec![],
            dim_vars: Default::default(),
            shape_constraints: Default::default(),
            subgraphs: HashMap::new(),
            tensor_names: HashMap::new(),
            topo_cache: Default::default(),
        };
        ModelCompiler::default()
            .compile(ModelSource::AiGraph(g))
            .unwrap()
            .bytes
    }

    // Causal LM (Gather over a table whose every row argmaxes to token 1) with a
    // tiny tokenizer baked in — generation always emits token 1 ("a").
    fn lm_with_tokenizer() -> Vec<u8> {
        let (seq, v) = (4u64, 3u64);
        let (ids, w, logits) = (0u32, 1u32, 2u32);
        let mut t = HashMap::new();
        t.insert(ids, ti(DType::INT64, &[1, seq]));
        t.insert(w, ti(DType::F32, &[v, v]));
        t.insert(logits, ti(DType::F32, &[1, seq, v]));
        let mut wb = vec![0u8; (v * v) as usize * 4]; // every row → column 1
        for r in 0..v as usize {
            wb[(r * v as usize + 1) * 4..(r * v as usize + 1) * 4 + 4]
                .copy_from_slice(&1.0f32.to_le_bytes());
        }
        let mut params = HashMap::new();
        params.insert(w, AiParam::inline(wb, t[&w].clone()));
        let g = AiGraph {
            name: "lm".into(),
            nodes: vec![AiNode::new(0, AiOp::Gather { axis: 0 }, vec![w, ids], vec![logits])],
            inputs: vec![ids],
            outputs: vec![logits],
            input_names: vec!["input_ids".into()],
            output_names: vec!["logits".into()],
            params,
            tensor_info: t,
            metadata: HashMap::new(),
            warnings: vec![],
            dim_vars: Default::default(),
            shape_constraints: Default::default(),
            subgraphs: HashMap::new(),
            tensor_names: HashMap::new(),
            topo_cache: Default::default(),
        };
        let tok = br#"{"added_tokens":[{"id":0,"content":"</s>","special":true}],"model":{"type":"BPE","vocab":{"</s>":0,"a":1,"b":2},"merges":[]}}"#;
        let mut sections = ArchiveSections::new();
        sections.add_extension("tokenizer.json", tok.to_vec());
        ModelCompiler::default()
            .compile_with_sections(ModelSource::AiGraph(g), sections)
            .unwrap()
            .bytes
    }

    #[wasm_bindgen_test]
    fn describe_in_wasm() {
        let info: ModelInfo = serde_wasm_bindgen::from_value(describe(&matmul_onnxless()).unwrap()).unwrap();
        assert_eq!(info.inputs.len(), 1);
        assert_eq!(info.inputs[0].dtype_name, "f32");
        assert_eq!(info.inputs[0].element_count, 4);
        assert_eq!(info.outputs[0].element_count, 4);
    }

    #[wasm_bindgen_test]
    fn run_in_wasm() {
        let holo = matmul_onnxless();
        let outs: Vec<Output> =
            serde_wasm_bindgen::from_value(run(&holo, JsValue::NULL, Some(1.0)).unwrap()).unwrap();
        assert_eq!(outs[0].values, vec![1.0, 1.0, 1.0, 1.0]); // identity·ones
    }

    #[wasm_bindgen_test]
    fn compile_and_generate_in_wasm() {
        // The LM + tokenizer were compiled in-wasm (above). Generate reads the
        // embedded tokenizer and runs the real loop entirely in the browser.
        let holo = lm_with_tokenizer();
        let opts = serde_wasm_bindgen::to_value(&serde_json::json!({"max_tokens": 3})).unwrap();
        let out = generate(&holo, None, "a", opts).unwrap();
        // Every step argmaxes to token 1 ("a") ⇒ output is all 'a', non-empty.
        assert!(!out.is_empty() && out.chars().all(|c| c == 'a'), "got {out:?}");
        // Deterministic (greedy).
        let opts2 = serde_wasm_bindgen::to_value(&serde_json::json!({"max_tokens": 3})).unwrap();
        assert_eq!(generate(&holo, None, "a", opts2).unwrap(), out);
    }
}
