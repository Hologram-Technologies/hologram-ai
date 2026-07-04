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
use hologram_ai::{FixedSession, HoloRunner, ModelCompiler, ModelSource};
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
        .compile(ModelSource::OnnxBytes {
            model_bytes: onnx.to_vec(),
            external_data: None,
        })
        .map_err(|e| err(format!("compile: {e:#}")))?;
    Ok(archive.bytes)
}

#[wasm_bindgen]
pub fn compile_onnx_with_data(onnx: &[u8], external_data_bytes: &[u8]) -> Result<Vec<u8>, JsValue> {
    let archive = ModelCompiler::default()
        .compile(ModelSource::OnnxBytes {
            model_bytes: onnx.to_vec(),
            external_data: Some(external_data_bytes.to_vec()),
        })
        .map_err(|e| err(format!("compile: {e:#}")))?;
    Ok(archive.bytes)
}

#[wasm_bindgen]
pub fn compile_safetensors(
    config_json: &str,
    safetensors_shards_js: &js_sys::Array,
) -> Result<Vec<u8>, JsValue> {
    let mut safetensors_shards = Vec::new();
    for i in 0..safetensors_shards_js.length() {
        let val = safetensors_shards_js.get(i);
        let u8_array = js_sys::Uint8Array::new(&val);
        safetensors_shards.push(u8_array.to_vec());
    }

    let archive = ModelCompiler::default()
        .compile(ModelSource::Safetensors {
            config_json: config_json.to_string(),
            safetensors_shards,
        })
        .map_err(|e| err(format!("compile_safetensors: {e:#}")))?;
    Ok(archive.bytes)
}

/// One streamed-manifest row parsed out of the JS arrays.
struct ManifestRows {
    keys: Vec<String>,
    shapes: Vec<Vec<u64>>,
    dtypes: Vec<hologram_ai_common::DType>,
}

/// Parse the manifest arrays (names, shapes, dtypes) — fail loud on anything
/// unmapped: a mislabeled dtype corrupts every weight downstream.
fn parse_manifest(
    keys_js: &js_sys::Array,
    tensor_shapes_js: &js_sys::Array,
    tensor_dtypes_js: &js_sys::Array,
) -> Result<ManifestRows, JsValue> {
    let mut rows = ManifestRows {
        keys: Vec::new(),
        shapes: Vec::new(),
        dtypes: Vec::new(),
    };
    for i in 0..keys_js.length() {
        let key = keys_js
            .get(i)
            .as_string()
            .ok_or_else(|| err(format!("manifest key {i} is not a string")))?;
        let shape_str = tensor_shapes_js
            .get(i)
            .as_string()
            .ok_or_else(|| err(format!("shape for `{key}` is not a string")))?;
        let dtype_str = tensor_dtypes_js
            .get(i)
            .as_string()
            .ok_or_else(|| err(format!("dtype for `{key}` is not a string")))?;

        let shape: Vec<u64> = serde_json::from_str(&shape_str)
            .map_err(|e| err(format!("shape for `{key}` does not parse: {e}")))?;

        let dtype = match dtype_str.as_str() {
            "F32" => hologram_ai_common::DType::F32,
            "F16" => hologram_ai_common::DType::F16,
            "BF16" => hologram_ai_common::DType::BF16,
            "F64" => hologram_ai_common::DType::F64,
            "I64" | "INT64" => hologram_ai_common::DType::INT64,
            "I32" | "INT32" => hologram_ai_common::DType::INT32,
            "I8" | "INT8" => hologram_ai_common::DType::INT8,
            "U8" => hologram_ai_common::DType::U8,
            "BOOL" => hologram_ai_common::DType::BOOL,
            other => {
                return Err(err(format!(
                    "tensor `{key}` has unsupported safetensors dtype `{other}`"
                )))
            }
        };

        rows.keys.push(key);
        rows.shapes.push(shape);
        rows.dtypes.push(dtype);
    }
    Ok(rows)
}

/// The architecture-family names the registry supports (drives the browser's
/// supported-only model search — dictionary row `supported-search`).
#[wasm_bindgen]
pub fn supported_families() -> js_sys::Array {
    let out = js_sys::Array::new();
    for name in hologram_ai_safetensors::parametric::supported_families() {
        out.push(&JsValue::from_str(name));
    }
    out
}

/// Config-only preflight (journey S1, step a): the architecture family must
/// be registered and the family's required keys present — checked BEFORE even
/// the shard headers are fetched. Fails naming the family or the missing key.
#[wasm_bindgen]
pub fn validate_model_config(config_json: &str) -> Result<(), JsValue> {
    let config: serde_json::Value =
        serde_json::from_str(config_json).map_err(|e| err(format!("config.json: {e}")))?;
    hologram_ai_safetensors::parametric::validate_config(&config).map_err(|e| err(format!("{e:#}")))
}

/// Preflight (journey S1): validate that the parametric graph builds from
/// config.json plus the header-only tensor manifest — BEFORE any shard byte
/// moves. An unsupported family, a missing config key, or a manifest the
/// family cannot realize fails here, naming the reason. Weight-free: only
/// names/shapes/dtypes are consulted.
#[wasm_bindgen]
pub fn validate_streamed_manifest(
    config_json: &str,
    keys_js: &js_sys::Array,
    tensor_shapes_js: &js_sys::Array,
    tensor_dtypes_js: &js_sys::Array,
    context_length: Option<u32>,
) -> Result<(), JsValue> {
    let rows = parse_manifest(keys_js, tensor_shapes_js, tensor_dtypes_js)?;
    let config: serde_json::Value =
        serde_json::from_str(config_json).map_err(|e| err(format!("config.json: {e}")))?;
    hologram_ai_safetensors::parametric::build_parametric_graph_from_manifest(
        &config,
        &rows.keys,
        &rows.dtypes,
        context_length.map(u64::from),
    )
    .map_err(|e| err(format!("{e:#}")))?;
    Ok(())
}

#[wasm_bindgen]
pub fn compile_safetensors_streamed(
    config_json: &str,
    keys_js: &js_sys::Array,
    kappas_js: &js_sys::Array,
    tensor_shapes_js: &js_sys::Array,
    tensor_dtypes_js: &js_sys::Array,
    context_length: Option<u32>,
) -> Result<Vec<u8>, JsValue> {
    let rows = parse_manifest(keys_js, tensor_shapes_js, tensor_dtypes_js)?;
    let mut kappas = Vec::new();
    for (i, key) in rows.keys.iter().enumerate() {
        let kappa = kappas_js
            .get(i as u32)
            .as_string()
            .ok_or_else(|| err(format!("κ for `{key}` is not a string")))?;
        kappas.push(kappa);
    }
    let (keys, shapes, dtypes) = (rows.keys, rows.shapes, rows.dtypes);

    let config: serde_json::Value =
        serde_json::from_str(config_json).map_err(|e| err(e.to_string()))?;
    let mut graph = hologram_ai_safetensors::parametric::build_parametric_graph_from_manifest(
        &config,
        &keys,
        &dtypes,
        context_length.map(u64::from),
    )
    .map_err(|e| err(e.to_string()))?;

    // Inject AiParam::External for each key so the compiled `.holo` has the `holospaces.kappa_map`.
    // The parametric compiler doesn't add parameters, so we add them here.
    let mut next_id = graph.tensor_names.keys().max().copied().unwrap_or(0) + 1;
    let mut name_to_id = std::collections::HashMap::new();
    for (id, name) in &graph.tensor_names {
        name_to_id.insert(name.clone(), *id);
    }

    for (i, key) in keys.iter().enumerate() {
        let id = if let Some(existing_id) = name_to_id.get(key) {
            *existing_id
        } else {
            let new_id = next_id;
            next_id += 1;
            graph.tensor_names.insert(new_id, key.clone());
            new_id
        };

        let info = hologram_ai_common::TensorInfo::new(
            dtypes[i],
            hologram_ai_common::shape_from_concrete(&shapes[i]),
        );
        graph.tensor_info.insert(id, info.clone());
        graph.params.insert(
            id,
            hologram_ai_common::AiParam::External {
                kappa: kappas[i].clone(),
                info,
            },
        );
    }

    let archive = ModelCompiler::default()
        .compile(ModelSource::AiGraph(graph))
        .map_err(|e| err(format!("compile_safetensors_streamed: {e:#}")))?;

    Ok(archive.bytes)
}

// ── κ-materialization (journey stage S3) ────────────────────────────────────

/// The κ-labels a k-form archive requires (its `holospaces.kappa_map`
/// entries), as a JS array of strings. Empty for a material archive.
#[wasm_bindgen]
pub fn kappa_requirements(holo: &[u8]) -> Result<js_sys::Array, JsValue> {
    let reqs = hologram_ai::materialize::kappa_requirements(holo)
        .map_err(|e| err(format!("kappa_requirements: {e:#}")))?;
    let out = js_sys::Array::new();
    for r in reqs {
        out.push(&JsValue::from_str(&r.kappa));
    }
    Ok(out)
}

/// Materialize a k-form archive against a κ-store resolver.
///
/// `resolve` is a synchronous JS function `(kappa: string) => Uint8Array` —
/// in the browser this reads `tensors/{κ}.bin` from OPFS via a sync access
/// handle (worker context). Every resolved buffer is re-hashed and must
/// reproduce its κ (content addressing is the integrity check); a missing or
/// corrupt κ aborts naming the label. Returns the executable archive bytes.
#[wasm_bindgen]
pub fn materialize(holo: &[u8], resolve: &js_sys::Function) -> Result<Vec<u8>, JsValue> {
    let mut store = |kappa: &str| -> anyhow::Result<Vec<u8>> {
        let value = resolve
            .call1(&JsValue::NULL, &JsValue::from_str(kappa))
            .map_err(|e| anyhow::anyhow!("κ resolver threw for `{kappa}`: {e:?}"))?;
        if value.is_null() || value.is_undefined() {
            anyhow::bail!("κ `{kappa}` not present in store");
        }
        Ok(js_sys::Uint8Array::new(&value).to_vec())
    };
    hologram_ai::materialize::materialize_archive(holo, &mut store)
        .map_err(|e| err(format!("materialize: {e:#}")))
}

#[wasm_bindgen]
pub fn compute_kappa(bytes: &[u8]) -> String {
    holospaces::address(bytes).as_str().to_string()
}

#[wasm_bindgen]
pub struct KappaHasher {
    hasher: blake3::Hasher,
}

impl Default for KappaHasher {
    fn default() -> Self {
        Self::new()
    }
}

#[wasm_bindgen]
impl KappaHasher {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            hasher: blake3::Hasher::new(),
        }
    }

    pub fn update(&mut self, bytes: &[u8]) {
        self.hasher.update(bytes);
    }

    pub fn finalize(self) -> String {
        let hash = self.hasher.finalize();
        format!("blake3:{}", hash.to_hex())
    }
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
        0 => "bool",
        1 => "u8",
        2 => "i8",
        3 => "u64",
        4 => "i32",
        5 => "i64",
        6 => "f16",
        7 => "bf16",
        8 => "f32",
        9 => "f64",
        10 => "i4",
        _ => "?",
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
    let Some(v) = fill else {
        return vec![0u8; byte_size];
    };
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
    let conv =
        |w: usize, f: &dyn Fn(&[u8]) -> f64| bytes.chunks_exact(w).map(f).collect::<Vec<_>>();
    match dtype {
        0 | 1 => bytes.iter().map(|&b| b as f64).collect(),
        2 => bytes.iter().map(|&b| b as i8 as f64).collect(),
        3 => conv(8, &|c| u64::from_le_bytes(c.try_into().unwrap()) as f64),
        4 => conv(4, &|c| i32::from_le_bytes(c.try_into().unwrap()) as f64),
        5 => conv(8, &|c| i64::from_le_bytes(c.try_into().unwrap()) as f64),
        6 => conv(2, &|c| {
            f64::from(half::f16::from_le_bytes(c.try_into().unwrap()))
        }),
        7 => conv(2, &|c| {
            f64::from(half::bf16::from_le_bytes(c.try_into().unwrap()))
        }),
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
        return Err(err(format!(
            "expected {} input(s), got {}",
            in_info.len(),
            provided.len()
        )));
    }

    let mut owned: Vec<Vec<u8>> = Vec::with_capacity(in_info.len());
    for (i, p) in in_info.iter().enumerate() {
        let want = in_sizes[i];
        match provided.get(i).filter(|b| !b.is_empty()) {
            Some(b) if b.len() == want => owned.push(b.clone()),
            Some(b) => {
                return Err(err(format!(
                    "input[{i}] is {} bytes but the model expects {want}",
                    b.len()
                )))
            }
            None => owned.push(synth(want, p.element_count, p.dtype, fill)),
        }
    }

    let refs: Vec<&[u8]> = owned.iter().map(|v| v.as_slice()).collect();
    let outputs = runner
        .execute(&refs)
        .map_err(|e| err(format!("execute: {e:#}")))?;
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
    callback: Option<js_sys::Function>,
) -> Result<String, JsValue> {
    let opts: GenOpts = if opts.is_undefined() || opts.is_null() {
        GenOpts::default()
    } else {
        serde_wasm_bindgen::from_value(opts).map_err(err)?
    };
    let runner = HoloRunner::from_bytes(holo.to_vec()).map_err(err)?;

    let tokenizer = match tokenizer_json {
        Some(bytes) => NativeTokenizer::from_tokenizer_json_bytes(&bytes).map_err(err)?,
        None => {
            let embedded = runner.extension("tokenizer.json").ok_or_else(|| {
                err("no tokenizer: none embedded in the archive and none supplied")
            })?;
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

    struct CallbackSink<'a> {
        buffer: Vec<u8>,
        callback: Option<&'a js_sys::Function>,
    }

    impl<'a> std::io::Write for CallbackSink<'a> {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.buffer.extend_from_slice(buf);
            if let Some(cb) = self.callback {
                if let Ok(s) = String::from_utf8(self.buffer.clone()) {
                    let _ = cb.call1(&JsValue::NULL, &JsValue::from_str(&s));
                }
            }
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    // A precompiled `.holo` is a fixed-window session.
    let mut session = FixedSession::new(runner);
    let mut sink = CallbackSink {
        buffer: Vec::new(),
        callback: callback.as_ref(),
    };
    generate_stream(&mut session, &tokenizer, &templated, &cfg, &mut sink)
        .map_err(|e| err(format!("generate: {e:#}")))?;
    String::from_utf8(sink.buffer).map_err(err)
}

#[cfg(test)]
mod tests {
    use super::*;
    use hologram_ai::compiler::ArchiveSections;
    use hologram_ai_common::{
        shape_from_concrete, AiGraph, AiNode, AiOp, AiParam, DType, TensorInfo,
    };
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
        let info: ModelInfo =
            serde_wasm_bindgen::from_value(describe(&matmul_onnxless()).unwrap()).unwrap();
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
        let out = generate(&holo, None, "a", opts, None).unwrap();
        // Every step argmaxes to token 1 ("a") ⇒ output is all 'a', non-empty.
        assert!(
            !out.is_empty() && out.chars().all(|c| c == 'a'),
            "got {out:?}"
        );
        // Deterministic (greedy).
        let opts2 = serde_wasm_bindgen::to_value(&serde_json::json!({"max_tokens": 3})).unwrap();
        assert_eq!(generate(&holo, None, "a", opts2, None).unwrap(), out);
    }

    #[wasm_bindgen_test]
    fn compute_kappa_works() {
        let bytes = b"hello world";
        let expected = holospaces::address(bytes).as_str().to_string();
        let result = compute_kappa(bytes);
        assert_eq!(result, expected);
    }
}
