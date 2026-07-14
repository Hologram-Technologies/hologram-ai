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
//!
//! Multi-threaded decode (ADR-0018): the `wasm-threads` feature turns on the
//! substrate's embedder worker pool. See the `wasm_futex` module (compiled only
//! on the `+atomics` build) and `apps/web`'s worker pool for the two halves of
//! the embedder contract.

// The nightly shared-memory build (`--features wasm-threads`, `+atomics`) uses
// the native wasm atomic wait/notify intrinsics for the pool futex (see
// `wasm_futex`). They are unstable on stable Rust (rust-lang/rust#77839) — the
// very reason the substrate imports them from the embedder — but this build is
// already nightly (for `-Z build-std`), so gate the feature on `atomics` only.
#![cfg_attr(target_feature = "atomics", feature(stdarch_wasm_atomic_wait))]

use hologram_ai::commands::generate::{
    apply_template, generate_stream, generate_stream_decode, GenConfig,
};
use hologram_ai::{FixedSession, HoloRunner, ModelCompiler, ModelSource};
use hologram_ai_tokenizer::{NativeTokenizer, Tokenizer};
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

/// Embedder futex for the substrate's wasm worker pool (ADR-0018).
///
/// The no_std backend (`hologram-backend/wasm-threads`) declares
/// `hologram_host_wait32` / `hologram_host_notify` as `extern "C"` imports —
/// the embedder is expected to supply `Atomics.wait` / `Atomics.notify` over
/// the shared linear memory. On this nightly `+atomics` build we satisfy the
/// imports *by definition* with the native wasm `memory.atomic.wait32` /
/// `memory.atomic.notify` instructions, avoiding a JS round-trip per idle wait
/// and keeping the wasm-bindgen import object free of `env` entries.
///
/// Soundness: the substrate only ever waits from a pool worker or the executing
/// worker (never the browser main thread), where a blocking `atomic.wait` is
/// permitted; `hologram_worker_run` and `execute` both run off-main-thread by
/// the embedder contract. `notify` is legal from any agent.
#[cfg(all(target_arch = "wasm32", target_feature = "atomics"))]
mod wasm_futex {
    /// Block while `*ptr == expect`, up to `timeout_ns` (negative = infinite).
    /// Returns 0 (woken), 1 (`*ptr != expect`), or 2 (timed out).
    #[no_mangle]
    pub extern "C" fn hologram_host_wait32(ptr: *const i32, expect: i32, timeout_ns: i64) -> i32 {
        // SAFETY: the substrate passes the address of an `AtomicU32` living in
        // the shared linear memory; a wait on a valid, aligned i32 is defined.
        unsafe { core::arch::wasm32::memory_atomic_wait32(ptr as *mut i32, expect, timeout_ns) }
    }

    /// Wake up to `count` agents waiting on `ptr`; returns the number woken.
    #[no_mangle]
    pub extern "C" fn hologram_host_notify(ptr: *const i32, count: u32) -> u32 {
        // SAFETY: as above — `ptr` addresses a shared-memory `AtomicU32`.
        unsafe { core::arch::wasm32::memory_atomic_notify(ptr as *mut i32, count) }
    }
}

/// The last Rust panic message. A trapped wasm call surfaces to JS as a bare
/// `RuntimeError: unreachable` — the worker reads this to attach the actual
/// panic text to the error it reports (a user must never see an undiagnosable
/// crash).
static LAST_PANIC: std::sync::Mutex<Option<String>> = std::sync::Mutex::new(None);

/// Surface Rust panics: the browser console gets the full hook output, and
/// the message is RECORDED for [`last_panic`]. Runs on module init.
#[wasm_bindgen(start)]
pub fn start() {
    std::panic::set_hook(Box::new(|info| {
        if let Ok(mut slot) = LAST_PANIC.lock() {
            *slot = Some(info.to_string());
        }
        console_error_panic_hook::hook(info);
    }));
}

/// The most recent Rust panic message, if any — cleared on read so a stale
/// panic is never attributed to a later, unrelated failure.
#[wasm_bindgen]
pub fn last_panic() -> Option<String> {
    LAST_PANIC.lock().ok().and_then(|mut s| s.take())
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
    layers_per_stage: Option<u32>,
) -> Result<(), JsValue> {
    let rows = parse_manifest(keys_js, tensor_shapes_js, tensor_dtypes_js)?;
    let config: serde_json::Value =
        serde_json::from_str(config_json).map_err(|e| err(format!("config.json: {e}")))?;
    // Validate the graphs the PLAN will build: a staged plan builds stage
    // graphs (whose head chunks at the pipeline's own granularity — no head
    // is too large to execute); only a monolithic plan builds the monolithic
    // graph, whose whole-head working set the floor guard checks.
    match layers_per_stage.and_then(|n| std::num::NonZeroU64::new(u64::from(n))) {
        Some(block) => {
            hologram_ai_safetensors::parametric::build_parametric_stage_graphs(
                &config,
                &rows.keys,
                &rows.dtypes,
                context_length.map(u64::from),
                block,
            )
            .map_err(|e| err(format!("{e:#}")))?;
        }
        None => {
            hologram_ai_safetensors::parametric::build_parametric_graph_from_manifest(
                &config,
                &rows.keys,
                &rows.dtypes,
                context_length.map(u64::from),
            )
            .map_err(|e| err(format!("{e:#}")))?;
        }
    }
    Ok(())
}

/// Parse the parallel κ array of a streamed manifest — fail loud on a
/// missing or non-string κ, naming the tensor.
fn parse_kappas(keys: &[String], kappas_js: &js_sys::Array) -> Result<Vec<String>, JsValue> {
    let mut kappas = Vec::with_capacity(keys.len());
    for (i, key) in keys.iter().enumerate() {
        let kappa = kappas_js
            .get(i as u32)
            .as_string()
            .ok_or_else(|| err(format!("κ for `{key}` is not a string")))?;
        kappas.push(kappa);
    }
    Ok(kappas)
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
    let kappas = parse_kappas(&rows.keys, kappas_js)?;
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
                range: None,
            },
        );
    }

    let archive = ModelCompiler::default()
        .compile(ModelSource::AiGraph(graph))
        .map_err(|e| err(format!("compile_safetensors_streamed: {e:#}")))?;

    // Canonicalize so the same model yields a byte-identical k-form archive
    // (a stable κ) across processes/platforms — content-addressing requires it
    // (the substrate emits the Weights section in per-process hashbrown order).
    hologram_ai::materialize::canonicalize_archive(&archive.bytes)
        .map_err(|e| err(format!("canonicalize: {e:#}")))
}

/// Staged (windowed) compilation — dictionary row `staged-execution`.
///
/// Partitions the parametric decoder into stage graphs (embedding,
/// `layers_per_stage`-layer blocks, head) and compiles each into its own
/// k-form archive with `AiParam::External` κ-bindings, exactly like the
/// monolithic streamed compile. Returns the stage archives in execution
/// order as a JS `Array` of `Uint8Array`. The model's weights stay in the
/// κ-store; execution materializes one stage at a time
/// ([`StagedChatSession`]), so peak weight residency is the largest stage —
/// the window — never the model.
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn compile_safetensors_staged(
    config_json: &str,
    keys_js: &js_sys::Array,
    kappas_js: &js_sys::Array,
    tensor_shapes_js: &js_sys::Array,
    tensor_dtypes_js: &js_sys::Array,
    context_length: Option<u32>,
    layers_per_stage: u32,
) -> Result<js_sys::Array, JsValue> {
    let rows = parse_manifest(keys_js, tensor_shapes_js, tensor_dtypes_js)?;
    let kappas = parse_kappas(&rows.keys, kappas_js)?;
    let layers_per_stage = std::num::NonZeroU64::new(u64::from(layers_per_stage))
        .ok_or_else(|| err("layers_per_stage must be at least 1"))?;

    let stages = hologram_ai::staged::compile_stages(
        config_json,
        &rows.keys,
        &kappas,
        &rows.shapes,
        &rows.dtypes,
        context_length.map(u64::from),
        layers_per_stage,
    )
    .map_err(|e| err(format!("compile_safetensors_staged: {e:#}")))?;

    let out = js_sys::Array::new();
    for stage in stages {
        // Canonical per-stage κ (see compile_safetensors_streamed).
        let canonical = hologram_ai::materialize::canonicalize_archive(&stage)
            .map_err(|e| err(format!("canonicalize stage: {e:#}")))?;
        out.push(&js_sys::Uint8Array::from(canonical.as_slice()).into());
    }
    Ok(out)
}

/// Parse the JSON quant map the web tier records in `stages.json`:
/// `[{"wide": κ, "artifact": κ, "out": n, "in": n}, …]`. A whole projection
/// carries just its wide κ; a **head chunk** additionally carries `offset`/`len`
/// (its byte range within the wide LM-head/embedding tensor), and its map key is
/// the composite [`quant_key`](hologram_ai_common::lower::quant_key)`(κ,
/// Some((offset, len)))` — so the graph matcher and this loader mint the
/// identical key from the one shared function, never a re-implemented format.
fn parse_quant_json(
    quant_json: Option<String>,
) -> Result<Option<hologram_ai_common::lower::QuantMap>, JsValue> {
    let Some(json) = quant_json.filter(|j| !j.is_empty()) else {
        return Ok(None);
    };
    #[derive(serde::Deserialize)]
    struct Entry {
        wide: String,
        artifact: String,
        out: u64,
        #[serde(rename = "in")]
        inf: u64,
        #[serde(default)]
        offset: Option<u64>,
        #[serde(default)]
        len: Option<u64>,
        /// Tier tag (`"int8"` / `"int4"`); absent ⇒ the int8 default. The web tier
        /// records the tier its artifact was derived to, so the binder declares
        /// the weight slot with the matching dtype and byte ranges.
        #[serde(default)]
        tier: Option<String>,
    }
    let entries: Vec<Entry> =
        serde_json::from_str(&json).map_err(|e| err(format!("quant map JSON: {e}")))?;
    Ok(Some(
        entries
            .into_iter()
            .map(|e| {
                let key = hologram_ai_common::lower::quant_key(&e.wide, e.offset.zip(e.len));
                let tier = hologram_ai_common::lower::QuantTier::from_tag(e.tier.as_deref());
                (key, (e.artifact, e.out, e.inf, tier))
            })
            .collect(),
    ))
}

/// [`compile_safetensors_staged`] on the quantized tier (row
/// `quantized-transit`): stage graphs bind projection weights to their
/// quantized derived artifacts per the recorded map.
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn compile_safetensors_staged_quantized(
    config_json: &str,
    keys_js: &js_sys::Array,
    kappas_js: &js_sys::Array,
    tensor_shapes_js: &js_sys::Array,
    tensor_dtypes_js: &js_sys::Array,
    context_length: Option<u32>,
    layers_per_stage: u32,
    quant_json: Option<String>,
) -> Result<js_sys::Array, JsValue> {
    let rows = parse_manifest(keys_js, tensor_shapes_js, tensor_dtypes_js)?;
    let kappas = parse_kappas(&rows.keys, kappas_js)?;
    let layers_per_stage = std::num::NonZeroU64::new(u64::from(layers_per_stage))
        .ok_or_else(|| err("layers_per_stage must be at least 1"))?;
    let quant = parse_quant_json(quant_json)?;

    let stages = hologram_ai::staged::compile_stages_with(
        config_json,
        &rows.keys,
        &kappas,
        &rows.shapes,
        &rows.dtypes,
        context_length.map(u64::from),
        layers_per_stage,
        quant.as_ref(),
    )
    .map_err(|e| err(format!("compile_safetensors_staged_quantized: {e:#}")))?;

    let out = js_sys::Array::new();
    for stage in stages {
        let canonical = hologram_ai::materialize::canonicalize_archive(&stage)
            .map_err(|e| err(format!("canonicalize stage: {e:#}")))?;
        out.push(&js_sys::Uint8Array::from(canonical.as_slice()).into());
    }
    Ok(out)
}

/// The wide κs the staged plan can rewrite onto quantized artifacts and
/// fully retire (browser tier of row `quantized-transit`): the download
/// derives artifacts for exactly these and their wide blobs go gas-phase.
#[wasm_bindgen]
pub fn quantizable_weights(
    config_json: &str,
    keys_js: &js_sys::Array,
    kappas_js: &js_sys::Array,
    tensor_shapes_js: &js_sys::Array,
    tensor_dtypes_js: &js_sys::Array,
    context_length: Option<u32>,
    layers_per_stage: u32,
) -> Result<js_sys::Array, JsValue> {
    let rows = parse_manifest(keys_js, tensor_shapes_js, tensor_dtypes_js)?;
    let kappas = parse_kappas(&rows.keys, kappas_js)?;
    let layers_per_stage = std::num::NonZeroU64::new(u64::from(layers_per_stage))
        .ok_or_else(|| err("layers_per_stage must be at least 1"))?;
    let eligible = hologram_ai::staged::quantizable_weights(
        config_json,
        &rows.keys,
        &kappas,
        &rows.shapes,
        &rows.dtypes,
        context_length.map(u64::from),
        layers_per_stage,
    )
    .map_err(|e| err(format!("quantizable_weights: {e:#}")))?;
    let out = js_sys::Array::new();
    for kappa in eligible {
        out.push(&JsValue::from_str(&kappa));
    }
    Ok(out)
}

/// The head-chunk quantization targets of the staged plan (row
/// `quantized-transit`, chunked head): the vocab-row ranges of a large LM head
/// the int8 tier derives into per-chunk artifacts, so a chunked head is a
/// dequant-fused int8 matmul instead of a bf16 matmul whose whole-panel F32
/// image thrashes residency. Returns a JSON array
/// `[{"kappa": κ, "offset": n, "len": n, "out": n, "in": n}, …]` — the download
/// derives each artifact from `[offset, offset+len)` of the wide κ (the tied
/// head's is the embedding table's, kept wide for the Gather; only its slice is
/// crystallized) and records a quant entry keyed by that κ AND range. Empty
/// where the head is a single chunk (small vocabulary).
#[wasm_bindgen]
pub fn head_quant_chunks(
    config_json: &str,
    keys_js: &js_sys::Array,
    kappas_js: &js_sys::Array,
    tensor_shapes_js: &js_sys::Array,
    tensor_dtypes_js: &js_sys::Array,
    context_length: Option<u32>,
    layers_per_stage: u32,
) -> Result<String, JsValue> {
    let rows = parse_manifest(keys_js, tensor_shapes_js, tensor_dtypes_js)?;
    let kappas = parse_kappas(&rows.keys, kappas_js)?;
    let layers_per_stage = std::num::NonZeroU64::new(u64::from(layers_per_stage))
        .ok_or_else(|| err("layers_per_stage must be at least 1"))?;
    let targets = hologram_ai::staged::head_quant_chunks(
        config_json,
        &rows.keys,
        &kappas,
        &rows.shapes,
        &rows.dtypes,
        context_length.map(u64::from),
        layers_per_stage,
    )
    .map_err(|e| err(format!("head_quant_chunks: {e:#}")))?;
    let json: Vec<serde_json::Value> = targets
        .into_iter()
        .map(|t| {
            serde_json::json!({
                "kappa": t.kappa,
                "offset": t.offset,
                "len": t.len,
                "out": t.out_features,
                "in": t.in_features,
            })
        })
        .collect();
    serde_json::to_string(&json).map_err(|e| err(format!("head_quant_chunks JSON: {e}")))
}

/// Derive the quantized artifact of a wide `[out, in]` weight (row
/// `quantized-transit`): matmul-ready per-channel symmetric codes,
/// `q ‖ scales_f32(4·out)`. `tier` selects the width — "int8" (default, one
/// byte/code) or "int4" (packed nibbles, half the bytes). Deterministic — the
/// caller mints the artifact's κ from the returned bytes and records the same
/// tier in stages.json.
#[wasm_bindgen]
pub fn derive_quantized_artifact(
    wide: &[u8],
    dtype: &str,
    out_features: u32,
    in_features: u32,
    tier: Option<String>,
) -> Result<Vec<u8>, JsValue> {
    let dtype = match dtype {
        "F32" => hologram_ai_common::DType::F32,
        "F16" => hologram_ai_common::DType::F16,
        "BF16" => hologram_ai_common::DType::BF16,
        other => {
            return Err(err(format!(
                "quantized derivation from dtype `{other}` is not defined"
            )))
        }
    };
    // Tier tag (`"int8"` / `"int4"`); absent ⇒ int8. int4 halves the artifact's
    // weight bytes (packed nibbles) — the caller records the same tier in
    // stages.json so the binder declares the matching weight dtype.
    let target = hologram_ai_common::lower::QuantTier::from_tag(tier.as_deref());
    hologram_ai::quantized::derive_quantized_artifact_tier(
        wide,
        dtype,
        target,
        u64::from(out_features),
        u64::from(in_features),
    )
    .map_err(|e| err(format!("derive_quantized_artifact: {e:#}")))
}

/// The PARAMETRIC optimal quant tier (`"int8"` / `"int4"`) for a model of
/// `params` weights on a host address space of `address_space` bytes — the ONE
/// law the browser consults so a selected model is AUTOMATICALLY, optimally
/// compiled (no user knob). int8 for quality whenever it fits resident; int4
/// only to keep a larger model resident/interactive when int8 cannot; int8
/// (paged) for anything larger. `f64` at the boundary — param/byte counts are
/// well under 2^53. See `QuantTier::optimal_for`.
#[wasm_bindgen]
pub fn optimal_quant_tier(params: f64, address_space: f64) -> String {
    let tier = hologram_ai_common::lower::QuantTier::optimal_for(
        params.max(0.0) as u64,
        address_space.max(0.0) as u64,
    );
    match tier {
        hologram_ai_common::lower::QuantTier::Int8 => "int8",
        hologram_ai_common::lower::QuantTier::Int4 => "int4",
    }
    .to_string()
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
/// A κ-store backed by JS callbacks. `resolve` returns the bytes for a κ (or
/// null/undefined for "not present"); the optional `invalidate` is the
/// UNPIN hook (row `saturation-residency`): called when resolved content
/// fails verification, it must evict the cache tier's entry so the next
/// `resolve` falls through to recorded provenance.
struct JsKappaStore {
    resolve: js_sys::Function,
    invalidate: Option<js_sys::Function>,
    /// Optional ranged read `(kappa, offset, len) => Uint8Array | null` —
    /// the seekable-tier hook of sub-tensor κ-resolution (row `chunked-head`):
    /// a session-verified ranged binding moves only its slice (an OPFS
    /// `read({at})` or a ranged provenance GET), never the whole tensor.
    /// Absent, or returning null, falls back to whole-resolve + slice.
    resolve_range: Option<js_sys::Function>,
    /// Optional size stat `(kappa) => number | null` — the weight-tier pager
    /// (row `lazy-constant-residency`) sizes a paged constant's slot from an
    /// OPFS `getFile().size`, never reading the body. Absent falls back to
    /// resolve-and-measure.
    size: Option<js_sys::Function>,
}

impl hologram_ai::materialize::KappaStore for JsKappaStore {
    fn resolve(&mut self, kappa: &str) -> anyhow::Result<Vec<u8>> {
        let value = self
            .resolve
            .call1(&JsValue::NULL, &JsValue::from_str(kappa))
            .map_err(|e| anyhow::anyhow!("κ resolver threw for `{kappa}`: {e:?}"))?;
        if value.is_null() || value.is_undefined() {
            anyhow::bail!("κ `{kappa}` not present in store");
        }
        Ok(js_sys::Uint8Array::new(&value).to_vec())
    }

    fn invalidate(&mut self, kappa: &str) {
        if let Some(f) = &self.invalidate {
            let _ = f.call1(&JsValue::NULL, &JsValue::from_str(kappa));
        }
    }

    fn resolve_range(&mut self, kappa: &str, offset: u64, len: u64) -> anyhow::Result<Vec<u8>> {
        if let Some(f) = &self.resolve_range {
            let value = f
                .call3(
                    &JsValue::NULL,
                    &JsValue::from_str(kappa),
                    &JsValue::from_f64(offset as f64),
                    &JsValue::from_f64(len as f64),
                )
                .map_err(|e| anyhow::anyhow!("κ range resolver threw for `{kappa}`: {e:?}"))?;
            if !value.is_null() && !value.is_undefined() {
                let bytes = js_sys::Uint8Array::new(&value).to_vec();
                anyhow::ensure!(
                    bytes.len() as u64 == len,
                    "κ range resolver returned {} bytes for `{kappa}` range {offset}+{len}",
                    bytes.len()
                );
                return Ok(bytes);
            }
        }
        // No seekable tier (or a miss): whole-resolve + slice, the default law.
        let bytes = self.resolve(kappa)?;
        let (start, end) = (offset as usize, (offset + len) as usize);
        anyhow::ensure!(
            end <= bytes.len() && start <= end,
            "range {offset}+{len} exceeds the {}-byte content of `{kappa}`",
            bytes.len()
        );
        Ok(bytes[start..end].to_vec())
    }

    fn content_size(&mut self, kappa: &str) -> anyhow::Result<u64> {
        if let Some(f) = &self.size {
            let value = f
                .call1(&JsValue::NULL, &JsValue::from_str(kappa))
                .map_err(|e| anyhow::anyhow!("κ size stat threw for `{kappa}`: {e:?}"))?;
            if let Some(n) = value.as_f64() {
                return Ok(n as u64);
            }
        }
        Ok(self.resolve(kappa)?.len() as u64)
    }
}

/// A `Send` κ-store over the OPFS callbacks for the weight-tier pager's
/// provider (row `lazy-constant-residency`). hologram's `load_paged` requires
/// `WeightProvider: Send + Sync`; the browser runs single-threaded, so the JS
/// callbacks are never actually shared across threads and the `unsafe impl` is
/// sound — a wasm-only escape hatch, exactly as the store callbacks are. It
/// delegates to a full [`JsKappaStore`], so the provider inherits resolve,
/// invalidate (the unpin/recover hook), the ranged seek, and the size stat.
struct SendStore(JsKappaStore);
// SAFETY: under ADR-0018 the wasm module runs on N+1 threads over one shared
// memory, but the pool workers only ever execute `pool_exec_gemv` over raw
// pointers — they never touch this JS-backed κ-store or its callbacks. The store
// is created and invoked solely on the single EXECUTE thread, so it is never
// shared across threads. (Narrowed from "wasm32 is single-threaded", which
// ADR-0018 made false; the impl stays sound on the execute-thread-only argument.)
unsafe impl Send for SendStore {}

impl hologram_ai::materialize::KappaStore for SendStore {
    fn resolve(&mut self, kappa: &str) -> anyhow::Result<Vec<u8>> {
        self.0.resolve(kappa)
    }
    fn invalidate(&mut self, kappa: &str) {
        self.0.invalidate(kappa)
    }
    fn resolve_range(&mut self, kappa: &str, offset: u64, len: u64) -> anyhow::Result<Vec<u8>> {
        self.0.resolve_range(kappa, offset, len)
    }
    fn content_size(&mut self, kappa: &str) -> anyhow::Result<u64> {
        self.0.content_size(kappa)
    }
}

/// A derived-artifact store backed by JS callbacks (row
/// `derived-artifact-kappa`): `load(key)` returns
/// `{ stages: Uint8Array[], kappas: string[] }` or undefined; `store(key,
/// stages, kappas)` persists a fresh derivation (async on the JS side —
/// persistence is an optimization, a lost write only costs re-derivation);
/// `evaporate(key)` unpins a corrupted entry.
struct JsDerivedStore {
    load: js_sys::Function,
    store: js_sys::Function,
    evaporate: js_sys::Function,
}

impl hologram_ai::staged::DerivedStore for JsDerivedStore {
    fn load(&mut self, key: &str) -> Option<(Vec<Vec<u8>>, Vec<String>)> {
        let value = self
            .load
            .call1(&JsValue::NULL, &JsValue::from_str(key))
            .ok()?;
        if value.is_null() || value.is_undefined() {
            return None;
        }
        let stages_js = js_sys::Reflect::get(&value, &JsValue::from_str("stages")).ok()?;
        let kappas_js = js_sys::Reflect::get(&value, &JsValue::from_str("kappas")).ok()?;
        let stages: Vec<Vec<u8>> = js_sys::Array::from(&stages_js)
            .iter()
            .map(|v| js_sys::Uint8Array::new(&v).to_vec())
            .collect();
        let kappas: Vec<String> = js_sys::Array::from(&kappas_js)
            .iter()
            .filter_map(|v| v.as_string())
            .collect();
        Some((stages, kappas))
    }

    fn store(&mut self, key: &str, stages: &[Vec<u8>], kappas: &[String]) {
        let stages_js = js_sys::Array::new();
        for stage in stages {
            stages_js.push(&js_sys::Uint8Array::from(stage.as_slice()).into());
        }
        let kappas_js = js_sys::Array::new();
        for kappa in kappas {
            kappas_js.push(&JsValue::from_str(kappa));
        }
        let _ = self.store.call3(
            &JsValue::NULL,
            &JsValue::from_str(key),
            &stages_js,
            &kappas_js,
        );
    }

    fn evaporate(&mut self, key: &str) {
        let _ = self
            .evaporate
            .call1(&JsValue::NULL, &JsValue::from_str(key));
    }
}

#[wasm_bindgen]
pub fn materialize(
    holo: &[u8],
    resolve: &js_sys::Function,
    invalidate: Option<js_sys::Function>,
) -> Result<Vec<u8>, JsValue> {
    let mut store = JsKappaStore {
        resolve: resolve.clone(),
        invalidate,
        resolve_range: None,
        size: None,
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

// ── tokenize ──────────────────────────────────────────────────────────────────

/// Token count of `text` under a HuggingFace `tokenizer.json` (bytes) — the
/// same `NativeTokenizer::encode` the generation loop runs, so the count is
/// exact, specials included. The browser uses it for template-aware session
/// trimming: the templated prompt must fit the model's context (the same
/// `prompt_tokens ≤ context` bound [`generate`] enforces).
#[wasm_bindgen]
pub fn count_tokens(tokenizer_json: &[u8], text: &str) -> Result<u32, JsValue> {
    let tokenizer = NativeTokenizer::from_tokenizer_json_bytes(tokenizer_json).map_err(err)?;
    let count = tokenizer.encode(text).len();
    u32::try_from(count).map_err(|_| err(format!("token count {count} exceeds u32::MAX")))
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
    /// Speculative decode (row `speculative-decode`): the draft width `K` (also
    /// the verify pass's chunk). `None`/`0`/`1` decode plainly; `≥ 2` drafts the
    /// next tokens from the realized sequence's recurrence and verifies them in
    /// one `M = K` pass. Works at ANY temperature — the accept rule samples per
    /// absolute position, so the output is byte-identical to plain decode at
    /// that temperature (greedy when temperature ≤ 0).
    pub speculative_draft: Option<usize>,
}

impl GenOpts {
    /// Parse a JS options object (`undefined`/`null` ⇒ defaults).
    fn from_js(opts: JsValue) -> Result<Self, JsValue> {
        if opts.is_undefined() || opts.is_null() {
            return Ok(Self::default());
        }
        serde_wasm_bindgen::from_value(opts).map_err(err)
    }

    /// The [`GenConfig`] these options select (defaults applied). An absent
    /// `max_tokens` stays `None`: generation is bounded by the model's stop
    /// conditions and the remaining context window, never a fixed token cap
    /// (journey S4).
    fn config(&self) -> GenConfig {
        GenConfig {
            max_tokens: self.max_tokens,
            temperature: self.temperature.unwrap_or(0.0),
            top_k: self.top_k,
            stop: self.stop.clone(),
            eos: self.eos,
            seed: self.seed.unwrap_or(0x9E3779B97F4A7C15),
        }
    }
}

/// Advance the streaming UTF-8 boundary: append `buf` to `pending`, split off
/// everything that now forms complete UTF-8 and return it, leaving the 0–3
/// trailing bytes of a character whose remaining bytes have not arrived yet in
/// `pending` for the next call. Definitely invalid UTF-8 — a sequence no
/// continuation byte could ever complete — is an error, never silently dropped
/// or reinterpreted. Pure (no JS types), so the boundary law is host-testable:
/// the concatenation of every returned chunk equals the concatenation of every
/// `buf`, and no chunk ever ends inside a character.
fn drain_complete_utf8(pending: &mut Vec<u8>, buf: &[u8]) -> Result<String, std::str::Utf8Error> {
    pending.extend_from_slice(buf);
    match std::str::from_utf8(pending) {
        // Common case — the shared loops write whole `StreamingDecoder` deltas,
        // so the pending bytes are usually already complete.
        Ok(_) => Ok(String::from_utf8(std::mem::take(pending)).expect("just validated")),
        Err(e) if e.error_len().is_none() => {
            // The tail is an incomplete character: hold it back, emit the rest.
            let tail = pending.split_off(e.valid_up_to());
            let complete = std::mem::replace(pending, tail);
            Ok(String::from_utf8(complete).expect("prefix validated by from_utf8"))
        }
        Err(e) => Err(e),
    }
}

/// A `Write` sink that accumulates the generated text (the final return value)
/// and streams each newly complete UTF-8 DELTA to an optional JS callback —
/// shared by [`generate`], [`StagedChatSession::generate`] and
/// [`DecodeChatSession::generate`]. Writes arrive as incremental chunks (the
/// shared loops stream `StreamingDecoder` deltas), so the callback receives
/// DELTAS, never the running string; bytes of a character split across writes
/// are held back until it completes ([`drain_complete_utf8`]). The law: the
/// concatenation of every callback delta is byte-identical to the returned
/// text.
struct CallbackSink<'a> {
    /// Every byte written — the final returned text.
    buffer: Vec<u8>,
    /// Trailing bytes of a not-yet-complete UTF-8 character, held back from
    /// the callback and prepended to the next write.
    pending: Vec<u8>,
    callback: Option<&'a js_sys::Function>,
}

impl<'a> CallbackSink<'a> {
    fn new(callback: Option<&'a js_sys::Function>) -> Self {
        Self {
            buffer: Vec::new(),
            pending: Vec::new(),
            callback,
        }
    }
}

impl std::io::Write for CallbackSink<'_> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.buffer.extend_from_slice(buf);
        if let Some(cb) = self.callback {
            let delta = drain_complete_utf8(&mut self.pending, buf).map_err(|e| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("generated stream is not UTF-8: {e}"),
                )
            })?;
            if !delta.is_empty() {
                let _ = cb.call1(&JsValue::NULL, &JsValue::from_str(&delta));
            }
        }
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
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
    let opts = GenOpts::from_js(opts)?;
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

    let cfg = opts.config();
    let templated = apply_template(opts.prompt_template.as_deref(), prompt);

    // A precompiled `.holo` is a fixed-window session.
    let mut session = FixedSession::new(runner);
    let mut sink = CallbackSink::new(callback.as_ref());
    generate_stream(&mut session, &tokenizer, &templated, &cfg, &mut sink)
        .map_err(|e| err(format!("generate: {e:#}")))?;
    String::from_utf8(sink.buffer).map_err(err)
}

/// A persistent staged chat session (rows `staged-execution`,
/// `staged-window-growth`, `stage-residency-cache`, `warm-turn`) — the
/// browser realization of the growable staged session that SURVIVES across
/// chat turns. The compiled window, the resident stage sessions (measured
/// admission), the session verified-κ set, and the derived-artifact hits all
/// carry from one `generate` call to the next, so a warm turn pays decode —
/// never recompile, never rematerialization, never re-verification. The
/// window still follows the SEQUENCE (geometric buckets up to the model's
/// own context); κs resolve through `resolve_kappa` (content-verified at
/// first touch, invalidate-and-recover on failure); stage archives resolve
/// from the derived store when present.
/// One wasm32 linear memory's hard address ceiling. A property of the HOST (32-bit
/// addressing), not of any model — nothing to derive.
// Only a host with a HARD address ceiling (wasm32) budgets residency this way;
// elsewhere there is no ceiling to divide, so the policy is unreferenced.
#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
const STRUCTURAL_CEILING: u64 = 4 << 30;

/// Headroom the HOST needs beside the resident weight set: the wasm runtime, the
/// allocator's slack, the JS heap. A property of the host and its allocator — no
/// model, input, or use-case quantity hides inside it. The two MODEL quantities
/// that used to be lumped into this reserve are now accounted where they belong:
/// the session's carried K/V is DERIVED from the model's own attention shape and
/// its current bucket ([`hologram_ai::decode::DecodeGeometry::carried_kv_bytes`]),
/// and walk transients are reserved inside the residency ledger (`max_walk`).
// Only a host with a HARD address ceiling (wasm32) budgets residency this way;
// elsewhere there is no ceiling to divide, so the policy is unreferenced.
#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
const HOST_HEADROOM: u64 = 1 << 30;

/// The resident-weight budget under the host ceiling: everything that is NOT
/// resident stage weights, subtracted. Because `carried_kv_bytes` is the model's
/// own K/V at its current geometry, the budget SHRINKS as the bucket grows — at a
/// long context the carried K/V alone runs to gigabytes, and a budget that
/// ignored it would over-admit straight into an allocation abort.
// Only a host with a HARD address ceiling (wasm32) budgets residency this way;
// elsewhere there is no ceiling to divide, so the policy is unreferenced.
#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
fn decode_residency_budget(carried_kv_bytes: u64) -> u64 {
    STRUCTURAL_CEILING
        .saturating_sub(HOST_HEADROOM)
        .saturating_sub(carried_kv_bytes)
}

/// A growable staged session shared by the wasm chat sessions.
type GrowableRc = std::rc::Rc<std::cell::RefCell<hologram_ai::staged::GrowableStagedSession>>;

/// The carried K/V of the decode PAIR — `(target, draft)` — which share one
/// address space, so admission must charge their sum.
type KvCharge = std::rc::Rc<std::cell::Cell<(u64, u64)>>;

fn kv_total(charge: &KvCharge) -> u64 {
    let (target, draft) = charge.get();
    target.saturating_add(draft)
}

/// Re-budget residency for every growable that shares this address space, having
/// charged the pair's carried K/V against the host ceiling. Under a hard address
/// ceiling (wasm32) this is what keeps a long-context K/V from over-committing;
/// elsewhere the budget is a κ-store bandwidth cache limit and the K/V lives
/// outside it, so this is a no-op.
#[cfg(target_arch = "wasm32")]
fn charge_carried_kv(growables: &[&GrowableRc], carried_kv_bytes: u64) {
    let budget = decode_residency_budget(carried_kv_bytes);
    for g in growables {
        g.borrow_mut().set_residency_budget(budget);
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn charge_carried_kv(_growables: &[&GrowableRc], _carried_kv_bytes: u64) {}

/// The shared growable-session builder behind [`StagedChatSession`] and
/// [`DecodeChatSession`]: manifest parsing, the JS κ-store, the quant tier,
/// the derived-artifact store, wasm heap admission, and progress narration.
#[allow(clippy::too_many_arguments)]
fn build_growable_session(
    config_json: &str,
    keys_js: &js_sys::Array,
    kappas_js: &js_sys::Array,
    tensor_shapes_js: &js_sys::Array,
    tensor_dtypes_js: &js_sys::Array,
    context_length: Option<u32>,
    layers_per_stage: u32,
    resolve_kappa: &js_sys::Function,
    invalidate_kappa: Option<js_sys::Function>,
    resolve_kappa_range: Option<js_sys::Function>,
    quant_json: Option<String>,
    derived_load: Option<js_sys::Function>,
    derived_store: Option<js_sys::Function>,
    derived_evaporate: Option<js_sys::Function>,
    weight_budget: Option<u32>,
    size_kappa: Option<js_sys::Function>,
    on_progress: Option<js_sys::Function>,
) -> Result<hologram_ai::staged::GrowableStagedSession, JsValue> {
    let rows = parse_manifest(keys_js, tensor_shapes_js, tensor_dtypes_js)?;
    let kappas = parse_kappas(&rows.keys, kappas_js)?;
    let layers_per_stage = std::num::NonZeroU64::new(u64::from(layers_per_stage))
        .ok_or_else(|| err("layers_per_stage must be at least 1"))?;

    // Clone the OPFS callbacks the paged provider needs BEFORE they move into
    // the main store (the paged store is independent per hologram's Send
    // requirement, but drives the same OPFS tiers).
    let paging_cbs = weight_budget.map(|budget| {
        (
            budget,
            resolve_kappa.clone(),
            invalidate_kappa.clone(),
            resolve_kappa_range.clone(),
            size_kappa.clone(),
        )
    });

    let store = Box::new(JsKappaStore {
        resolve: resolve_kappa.clone(),
        invalidate: invalidate_kappa,
        resolve_range: resolve_kappa_range,
        size: size_kappa,
    });

    let mut session = hologram_ai::staged::GrowableStagedSession::new(
        config_json.to_string(),
        rows.keys,
        kappas,
        rows.shapes,
        rows.dtypes,
        context_length.map(u64::from),
        layers_per_stage,
        store,
    )
    .map_err(|e| err(format!("staged session: {e:#}")))?;

    if let Some(quant) = parse_quant_json(quant_json)? {
        session.set_quant_map(quant);
    }

    if let (Some(load), Some(store), Some(evaporate)) =
        (derived_load, derived_store, derived_evaporate)
    {
        session.set_derived_store(Box::new(JsDerivedStore {
            load,
            store,
            evaporate,
        }));
    }

    // Residency admission against the wasm32 4 GiB address-space ceiling. The
    // budget is a BYTE budget on the resident weight set, not a heap
    // measurement: `memory_size` only ever GROWS (wasm memory never shrinks), so
    // after the compile/derivation phase peaks it stays pinned near the ceiling
    // and refuses residency for the rest of the session — every stage then
    // re-materializes from the κ-store on every token. The byte budget tracks
    // what the session actually holds, and the admission check adds the model's
    // own largest-stage transient (see `StagedRunner`), so a model whose weights
    // fit under the ceiling minus the runtime reserve stays resident across
    // tokens AND turns; a larger one falls back to windowing, never refused. The
    // reserve is the fixed non-weight headroom (activations, K/V, the runtime) —
    // the margin, which adapts to the model, is subtracted inside the check.
    #[cfg(target_arch = "wasm32")]
    {
        // No decode geometry yet, so no carried K/V to charge; a decode session
        // re-budgets with its own K/V as soon as its bucket is known, and again
        // at every regrow (`charge_carried_kv`).
        session.set_residency_budget(decode_residency_budget(0));
        // The budget is a HARD 32-bit address ceiling here: gate residency on
        // each stage's TRUE footprint (weights + the transients the buffer pool
        // retains — a float LM-head chunk's F32 image is several times its
        // packed weight) plus a largest-walk reserve. Without this, resident
        // head chunks' retained F32 scratch accumulates past 4 GiB → an opaque
        // `RuntimeError: unreachable` allocation abort on a large-vocabulary
        // model whose float head does not fit alongside the resident layers.
        session.set_bound_by_footprint(true);
    }

    // Weight-tier paging (row `lazy-constant-residency`): each stage loads
    // PAGED against `weight_budget` resident bytes, so a stage whose weights
    // exceed the wasm heap window still runs — the arena is a bounded window
    // over the OPFS κ-store. The provider's store is a fresh clone of the OPFS
    // callbacks per stage, wrapped `Send` (wasm is single-threaded), so it
    // inherits verify/invalidate/seek/size from `JsKappaStore`.
    if let Some((budget, resolve, invalidate, resolve_range, size)) = paging_cbs {
        session.set_weight_paging(
            budget as usize,
            std::rc::Rc::new(move || {
                Box::new(SendStore(JsKappaStore {
                    resolve: resolve.clone(),
                    invalidate: invalidate.clone(),
                    resolve_range: resolve_range.clone(),
                    size: size.clone(),
                })) as hologram_ai::runner::PagedStore
            }),
        );
    }

    if let Some(progress) = on_progress {
        let for_window = progress.clone();
        session.set_window_observer(Box::new(move |window, resolved| {
            let verb = if resolved {
                "resolving (derived κ)"
            } else {
                "compiling"
            };
            let _ = for_window.call1(
                &JsValue::NULL,
                &JsValue::from_str(&format!("{verb} a {window}-token window")),
            );
        }));
        session.set_stage_observer(Box::new(move |stage, count, bytes| {
            let mb = bytes as f64 / (1024.0 * 1024.0);
            let _ = progress.call1(
                &JsValue::NULL,
                &JsValue::from_str(&format!(
                    "stage {}/{count} materialized ({mb:.0} MB)",
                    stage + 1
                )),
            );
        }));
    }

    Ok(session)
}

#[wasm_bindgen]
pub struct StagedChatSession {
    session: hologram_ai::staged::GrowableStagedSession,
    tokenizer: NativeTokenizer,
}

#[wasm_bindgen]
impl StagedChatSession {
    /// Build the session from the streamed-download manifest — the same
    /// inputs the download compiled with. Stage archives embed no tokenizer,
    /// so `tokenizer_json` is required. `on_progress` (optional) narrates
    /// window compiles and per-stage materialization for the session's whole
    /// lifetime.
    #[wasm_bindgen(constructor)]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        config_json: &str,
        keys_js: &js_sys::Array,
        kappas_js: &js_sys::Array,
        tensor_shapes_js: &js_sys::Array,
        tensor_dtypes_js: &js_sys::Array,
        context_length: Option<u32>,
        layers_per_stage: u32,
        resolve_kappa: &js_sys::Function,
        invalidate_kappa: Option<js_sys::Function>,
        resolve_kappa_range: Option<js_sys::Function>,
        quant_json: Option<String>,
        derived_load: Option<js_sys::Function>,
        derived_store: Option<js_sys::Function>,
        derived_evaporate: Option<js_sys::Function>,
        weight_budget: Option<u32>,
        size_kappa: Option<js_sys::Function>,
        tokenizer_json: Vec<u8>,
        on_progress: Option<js_sys::Function>,
    ) -> Result<StagedChatSession, JsValue> {
        let tokenizer = NativeTokenizer::from_tokenizer_json_bytes(&tokenizer_json).map_err(err)?;
        let session = build_growable_session(
            config_json,
            keys_js,
            kappas_js,
            tensor_shapes_js,
            tensor_dtypes_js,
            context_length,
            layers_per_stage,
            resolve_kappa,
            invalidate_kappa,
            resolve_kappa_range,
            quant_json,
            derived_load,
            derived_store,
            derived_evaporate,
            weight_budget,
            size_kappa,
            on_progress,
        )?;
        Ok(StagedChatSession { session, tokenizer })
    }

    /// One chat turn over the warm session: the same `generate_stream` loop,
    /// streaming each newly decoded text DELTA to `callback` (the consumer
    /// accumulates). Returns the generated text.
    pub fn generate(
        &mut self,
        prompt: &str,
        opts: JsValue,
        callback: Option<js_sys::Function>,
    ) -> Result<String, JsValue> {
        let opts = GenOpts::from_js(opts)?;
        let cfg = opts.config();
        let templated = apply_template(opts.prompt_template.as_deref(), prompt);
        let mut sink = CallbackSink::new(callback.as_ref());
        generate_stream(
            &mut self.session,
            &self.tokenizer,
            &templated,
            &cfg,
            &mut sink,
        )
        .map_err(|e| err(format!("staged generate: {e:#}")))?;
        String::from_utf8(sink.buffer).map_err(err)
    }

    /// Stage materializations performed by the resident window's runner so
    /// far — the cross-turn bandwidth instrument a warm turn leaves
    /// unchanged.
    pub fn materialization_count(&self) -> u64 {
        self.session.materialization_count()
    }

    /// Window regrows resolved from the derived store instead of compiled.
    pub fn derived_hits(&self) -> u64 {
        self.session.derived_hits()
    }

    /// Idle pre-derivation (row `idle-derivation`): derive the next window
    /// bucket's stage archives into the derived store, off the per-token
    /// path — no weights move, the resident window is untouched. Returns
    /// the pre-derived bucket, or undefined at the ceiling.
    pub fn prederive_next_window(&mut self) -> Result<Option<u32>, JsValue> {
        self.session
            .prederive_next_window()
            .map(|w| w.map(|w| w as u32))
            .map_err(|e| err(format!("prederive: {e:#}")))
    }
}

/// A persistent **decode-plan** chat session (row `decode-plan`, browser
/// realization): every token — prompt prefill included — is one
/// single-position pass over the staged decode pipeline, never a
/// window-sized forward. Carried K/V lives in the engine's buffers and moves
/// through the pipeline's named ports; positions are runtime data the engine
/// synthesizes per step. The session survives across turns: the materialized
/// pipeline, the verified-κ set, and the derived-store hits all carry; each
/// turn rewinds the position and replays the templated transcript through
/// decode steps (elision recognizes the unchanged prefix). Bucket exhaustion
/// regrows geometrically through the same derived-artifact store the
/// whole-window plan uses — under a decode-specific derivation key.
#[wasm_bindgen]
pub struct DecodeChatSession {
    growable: std::rc::Rc<std::cell::RefCell<hologram_ai::staged::GrowableStagedSession>>,
    session: Option<hologram_ai::decode::DecodeSession<hologram_ai::staged::StagedRunner<'static>>>,
    tokenizer: NativeTokenizer,
    /// The model's complete rotary law, parsed from its own config.json
    /// (`rope_theta`, `rope_scaling`, `partial_rotary_factor`).
    rope: hologram_ai::RopeSpec,
    context_length: u64,
    /// The model's own vocabulary size (from config.json) — the pairing guard:
    /// a paired draft must cover the target's vocabulary, because the draft
    /// consumes the TARGET's token ids (row `speculative-draft-pairing`).
    vocab_size: u64,
    /// A catalogue-paired speculative DRAFT model, once `attach_draft`ed: its own
    /// growable (the archive factory sharing this target's residency ledger), and
    /// the runtime constants its lazily-built decode session needs. `None` until
    /// a draft is paired — then speculative decode drafts from this model instead
    /// of by prompt-lookup.
    draft_growable:
        Option<std::rc::Rc<std::cell::RefCell<hologram_ai::staged::GrowableStagedSession>>>,
    draft_session:
        Option<hologram_ai::decode::DecodeSession<hologram_ai::staged::StagedRunner<'static>>>,
    draft_rope: Option<hologram_ai::RopeSpec>,
    draft_context_length: u64,
    /// The carried K/V of this decode PAIR, charged against the host ceiling so
    /// residency admission shrinks as the bucket grows (row `decode-plan`).
    kv_charge: KvCharge,
}

#[wasm_bindgen]
impl DecodeChatSession {
    /// Build from the streamed-download manifest — the same inputs (and the
    /// same κ-store/derived-store/progress wiring) as [`StagedChatSession`].
    #[wasm_bindgen(constructor)]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        config_json: &str,
        keys_js: &js_sys::Array,
        kappas_js: &js_sys::Array,
        tensor_shapes_js: &js_sys::Array,
        tensor_dtypes_js: &js_sys::Array,
        context_length: Option<u32>,
        layers_per_stage: u32,
        resolve_kappa: &js_sys::Function,
        invalidate_kappa: Option<js_sys::Function>,
        resolve_kappa_range: Option<js_sys::Function>,
        quant_json: Option<String>,
        derived_load: Option<js_sys::Function>,
        derived_store: Option<js_sys::Function>,
        derived_evaporate: Option<js_sys::Function>,
        weight_budget: Option<u32>,
        size_kappa: Option<js_sys::Function>,
        tokenizer_json: Vec<u8>,
        on_progress: Option<js_sys::Function>,
    ) -> Result<DecodeChatSession, JsValue> {
        let tokenizer = NativeTokenizer::from_tokenizer_json_bytes(&tokenizer_json).map_err(err)?;
        let growable = build_growable_session(
            config_json,
            keys_js,
            kappas_js,
            tensor_shapes_js,
            tensor_dtypes_js,
            context_length,
            layers_per_stage,
            resolve_kappa,
            invalidate_kappa,
            resolve_kappa_range,
            quant_json,
            derived_load,
            derived_store,
            derived_evaporate,
            weight_budget,
            size_kappa,
            on_progress,
        )?;
        let context_length = hologram_ai::SessionProvider::max_window(&growable) as u64;
        // Rope tables are runtime data the ENGINE synthesizes; the law comes
        // from the model's own config — the same parse the parametric recipe
        // ran at build (`rope_theta`, `rope_scaling`, `partial_rotary_factor`).
        let config: serde_json::Value = serde_json::from_str(config_json).map_err(err)?;
        let rope = hologram_ai_safetensors::parametric::rope_spec_from_config(&config)
            .map_err(|e| err(format!("rotary law: {e:#}")))?;
        // The vocabulary the pairing guard compares (a draft must cover it). A
        // config without `vocab_size` cannot be pairing-checked, so a later
        // `attach_draft` refuses rather than risk an out-of-range draft Gather.
        let vocab_size = config
            .get("vocab_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        Ok(DecodeChatSession {
            growable: std::rc::Rc::new(std::cell::RefCell::new(growable)),
            session: None,
            tokenizer,
            rope,
            context_length,
            vocab_size,
            draft_growable: None,
            draft_session: None,
            draft_rope: None,
            draft_context_length: 0,
            kv_charge: std::rc::Rc::new(std::cell::Cell::new((0, 0))),
        })
    }

    /// Pair a speculative DRAFT model with this target (row
    /// `speculative-draft-pairing`): `draft` is a second `DecodeChatSession`
    /// built from the paired model's own dir, whose growable this session absorbs
    /// so speculative decode drafts from the paired model (`ModelDrafter`) rather
    /// than by prompt-lookup. Consumes `draft` — its growable lives on inside
    /// this target; its own tokenizer is discarded (the draft consumes the
    /// TARGET's token ids, never tokenizing text itself).
    ///
    /// Two invariants make the pairing SAFE, not merely fast:
    ///
    ///  * VOCABULARY. The draft embeds the target's token ids, so its vocabulary
    ///    must COVER the target's; an incompatible (or unknown) vocabulary is
    ///    refused (`Err`) and the caller falls back to prompt-lookup. Because the
    ///    output is the target's byte for byte regardless of the drafter, this
    ///    guard is about avoiding an out-of-range draft Gather, never correctness.
    ///  * RESIDENCY. The draft's growable adopts THIS target's residency ledger
    ///    (`share_residency_with`) BEFORE either wires a runner (both build
    ///    lazily on first `generate`), so the pair charges ONE combined footprint
    ///    and never over-commits the wasm 4 GiB ceiling.
    pub fn attach_draft(&mut self, draft: DecodeChatSession) -> Result<(), JsValue> {
        // The pairing compatibility policy (vocabulary + context) lives in the
        // native crate as a pure, tested function — the ONE source of the rule.
        // A refusal degrades to prompt-lookup; it never affects correctness.
        if let Some(reason) = hologram_ai::speculative::draft_pairing_refusal(
            self.vocab_size,
            self.context_length,
            draft.vocab_size,
            draft.context_length,
        ) {
            return Err(err(format!(
                "refusing the draft pairing: {reason}; drafting by prompt-lookup instead"
            )));
        }
        // Share ONE residency ledger across the pair before either builds a
        // runner — the draft's admission is then charged against the combined
        // footprint. Two distinct `Rc<RefCell<…>>`, so no double borrow.
        {
            let mut target = self.growable.borrow_mut();
            let mut paired = draft.growable.borrow_mut();
            target.share_residency_with(&mut paired);
        }
        self.draft_growable = Some(std::rc::Rc::clone(&draft.growable));
        self.draft_rope = Some(draft.rope.clone());
        self.draft_context_length = draft.context_length;
        Ok(())
    }

    /// One chat turn over the decode loop — prompt prefill as decode steps,
    /// one step per generated token — streaming each newly decoded text DELTA
    /// to `callback`. Cross-turn K/V retention lives in the loop itself: a
    /// transcript extending its own history rewinds to the shared prefix
    /// and pays only its novel suffix. Returns the generated text.
    pub fn generate(
        &mut self,
        prompt: &str,
        opts: JsValue,
        callback: Option<js_sys::Function>,
    ) -> Result<String, JsValue> {
        let opts = GenOpts::from_js(opts)?;
        let cfg = opts.config();
        let templated = apply_template(opts.prompt_template.as_deref(), prompt);

        if self.session.is_none() {
            // First turn: size the bucket to the prompt AND the generation the
            // caller DECLARED (`max_tokens`), so a turn of known length never
            // regrows mid-way — a regrow re-materializes every stage, which on a
            // weight-heavy model dominates the turn. An UNDECLARED budget could
            // run to the context, and pinning a context-sized K/V is impossible
            // at scale, so it starts at the prompt's window and climbs the
            // geometric ladder. Later turns KEEP the session (its carried K/V is
            // the retained prefix) and grow through the rebuild closure, which
            // copies the rows.
            let prompt_len = self
                .tokenizer
                .encode(&templated)
                .len()
                .max(1)
                .min(self.context_length as usize);
            let want = hologram_ai::engine::decode_bucket_for_turn(
                prompt_len,
                cfg.max_tokens.unwrap_or(0),
                self.context_length as usize,
            );
            let runner = self
                .growable
                .borrow_mut()
                .decode_runner_for(want)
                .map_err(|e| err(format!("decode pipeline: {e:#}")))?;
            let session = hologram_ai::decode::DecodeSession::new(
                runner,
                self.rope.clone(),
                self.context_length,
            )
            .map_err(|e| err(format!("decode session: {e:#}")))?;

            // Charge this session's carried K/V — a MODEL quantity, derived from
            // its own attention shape — against the host ceiling, and re-charge at
            // every regrow, because the K/V grows with the bucket.
            let geom = session.geometry();
            let kv_row = geom
                .carried_kv_bytes_per_row()
                .map_err(|e| err(format!("carried K/V: {e:#}")))?;
            let charge = std::rc::Rc::clone(&self.kv_charge);
            charge.set((kv_row.saturating_mul(geom.bucket as u64), charge.get().1));
            let mut sharers: Vec<&GrowableRc> = vec![&self.growable];
            if let Some(d) = &self.draft_growable {
                sharers.push(d);
            }
            charge_carried_kv(&sharers, kv_total(&charge));

            let g = std::rc::Rc::clone(&self.growable);
            let dg = self.draft_growable.clone();
            let session = session.with_rebuild(Box::new(move |bucket| {
                charge.set((kv_row.saturating_mul(bucket), charge.get().1));
                let mut sharers: Vec<&GrowableRc> = vec![&g];
                if let Some(d) = &dg {
                    sharers.push(d);
                }
                charge_carried_kv(&sharers, kv_total(&charge));
                g.borrow_mut().decode_runner_for(bucket as usize)
            }));
            self.session = Some(session);
        }

        // Build the paired draft model's decode session (row
        // `speculative-draft-pairing`), once, on the first turn a draft is
        // present: sized to the prompt (capped by the DRAFT's own context) and
        // grown through its own rebuild closure. It shares the target's
        // residency ledger (adopted at `attach_draft`, before this first runner).
        // A draft that cannot build degrades to prompt-lookup — a projection of
        // speed, never a refusal of the turn.
        if self.draft_session.is_none() {
            if let Some(dg) = self.draft_growable.clone() {
                let dctx = self.draft_context_length;
                let drope = self.draft_rope.clone();
                let dprompt = self
                    .tokenizer
                    .encode(&templated)
                    .len()
                    .max(1)
                    .min(dctx.max(1) as usize);
                let dwant = hologram_ai::engine::decode_bucket_for_turn(
                    dprompt,
                    cfg.max_tokens.unwrap_or(0),
                    dctx.max(1) as usize,
                );
                let tg = std::rc::Rc::clone(&self.growable);
                let charge = std::rc::Rc::clone(&self.kv_charge);
                let built = (|| -> anyhow::Result<
                    hologram_ai::decode::DecodeSession<hologram_ai::staged::StagedRunner<'static>>,
                > {
                    let drunner = dg.borrow_mut().decode_runner_for(dwant)?;
                    let drope = drope
                        .ok_or_else(|| anyhow::anyhow!("the paired draft carries no rotary law"))?;
                    let dsession = hologram_ai::decode::DecodeSession::new(drunner, drope, dctx)?;

                    // The pair shares ONE address space: charge the DRAFT's own
                    // carried K/V alongside the target's, at build and at regrow.
                    let dgeom = dsession.geometry();
                    let dkv_row = dgeom.carried_kv_bytes_per_row()?;
                    charge.set((charge.get().0, dkv_row.saturating_mul(dgeom.bucket as u64)));
                    charge_carried_kv(&[&tg, &dg], kv_total(&charge));

                    let dg2 = std::rc::Rc::clone(&dg);
                    let tg2 = std::rc::Rc::clone(&tg);
                    Ok(dsession.with_rebuild(Box::new(move |bucket| {
                        charge.set((charge.get().0, dkv_row.saturating_mul(bucket)));
                        charge_carried_kv(&[&tg2, &dg2], kv_total(&charge));
                        dg2.borrow_mut().decode_runner_for(bucket as usize)
                    })))
                })();
                match built {
                    Ok(s) => self.draft_session = Some(s),
                    Err(e) => web_sys::console::warn_1(&JsValue::from_str(&format!(
                        "paired draft model unavailable (drafting by prompt-lookup instead): {e:#}"
                    ))),
                }
            }
        }

        let session = self.session.as_mut().expect("session just ensured");

        // Chunked-prefill seeder (row `chunked-prefill`): the prompt suffix
        // seeds in ceil(n/chunk) BATCHED passes instead of n single-position
        // steps. The dominant benefit is compute, not the weight stream — an
        // M=chunk forward processes many positions in ~one pass, so prefill is
        // ~10x faster than stepping even when the weights are fully resident
        // (measured: 24-token prefill 10.8s stepping vs 1.1s seeded, INCLUDING
        // the seeder's own materialization — `tests/decode_perf.rs`). So install
        // it whenever the model is chunkable, NOT gated on residency; the
        // seeder's contention-reclaim (see `feed`) frees its residency after
        // prefill when it cannot coexist with the step runner. Installed lazily
        // (growth drops it), cached per bucket in the derived store; a failed
        // build degrades prefill to steps — a projection, never a refusal.
        if session.seeder_chunk().is_none() {
            // Parametric prefill chunk: the system's own geometric window base
            // (`geometric_window(1, context)` = min(MIN_WINDOW, context)),
            // capped by the bucket (the seeder's past span) — cache-friendly
            // and derived from the window policy, never a magic width. A
            // context below 2 has no seeder (prefill is one step per token).
            let bucket = session.geometry().bucket;
            let base =
                hologram_ai::engine::geometric_window(1, self.context_length as usize) as u64;
            let chunk = base.min(bucket as u64);
            if chunk >= 2 {
                // The seeder shares the session's ONE residency ledger with the
                // step runner (see `set_bound_by_footprint`): where both fit they
                // both stay resident (a warm turn re-materializes nothing); where
                // they don't, the shared gate windows — never an over-commit.
                match self
                    .growable
                    .borrow_mut()
                    .chunk_runner_for(bucket, chunk)
                    .and_then(|seeder| session.set_seeder(seeder))
                {
                    Ok(()) => {}
                    Err(e) => web_sys::console::warn_1(&JsValue::from_str(&format!(
                        "prefill seeder unavailable (stepping instead): {e:#}"
                    ))),
                }
            }
        }

        let mut sink = CallbackSink::new(callback.as_ref());

        // Speculative decode (row `speculative-decode`): a `K ≥ 2` draft at ANY
        // temperature. Build a verify runner at the session's OWN bucket (they
        // share the carried past) and draft the next K tokens from the realized
        // sequence's recurrence, verifying them in one M=K pass. The accept rule
        // is per-position sampling (greedy when temperature ≤ 0), so the output
        // is byte-identical to plain decode at that temperature. A failed/absent
        // verify runner falls back to plain decode — a projection of speed,
        // never meaning.
        let draft = opts.speculative_draft.unwrap_or(0);
        if draft >= 2 {
            let bucket = session.geometry().bucket;
            match self
                .growable
                .borrow_mut()
                .verify_runner_for(bucket, draft as u64)
            {
                Ok(mut verify) => {
                    // The drafter is parametric (row `speculative-draft-pairing`):
                    // a catalogue-paired DRAFT MODEL when one is attached, else
                    // the zero-weight prompt-lookup default. Either way the output
                    // is the target's byte for byte — the drafter only changes the
                    // acceptance rate. The warm draft session is taken into the
                    // per-turn drafter and reclaimed by `into_session`, so its
                    // resident pipeline survives across turns like the target's.
                    let result = if let Some(draft_session) = self.draft_session.take() {
                        let mut drafter =
                            hologram_ai::speculative::ModelDrafter::new(draft_session);
                        let r = hologram_ai::commands::generate::generate_stream_speculative(
                            session,
                            &mut verify,
                            &self.tokenizer,
                            &templated,
                            &cfg,
                            &mut drafter,
                            draft,
                            &mut sink,
                        );
                        self.draft_session = Some(drafter.into_session());
                        r
                    } else {
                        let mut drafter = hologram_ai::speculative::PromptLookupDrafter;
                        hologram_ai::commands::generate::generate_stream_speculative(
                            session,
                            &mut verify,
                            &self.tokenizer,
                            &templated,
                            &cfg,
                            &mut drafter,
                            draft,
                            &mut sink,
                        )
                    };
                    result.map_err(|e| err(format!("speculative decode: {e:#}")))?;
                    return String::from_utf8(sink.buffer).map_err(err);
                }
                Err(e) => web_sys::console::warn_1(&JsValue::from_str(&format!(
                    "verify pipeline unavailable (decoding plainly): {e:#}"
                ))),
            }
        }

        generate_stream_decode(session, &self.tokenizer, &templated, &cfg, &mut sink)
            .map_err(|e| err(format!("decode generate: {e:#}")))?;
        String::from_utf8(sink.buffer).map_err(err)
    }

    /// Stage materializations performed by the resident decode pipeline so
    /// far — the cross-turn bandwidth instrument a warm turn leaves
    /// unchanged.
    pub fn materialization_count(&self) -> u64 {
        self.session
            .as_ref()
            .map_or(0, |s| s.runner().materialization_count())
    }

    /// Bucket builds resolved from the derived store instead of compiled.
    pub fn derived_hits(&self) -> u64 {
        self.growable.borrow().derived_hits()
    }

    /// Idle pre-derivation (row `idle-derivation`, decode plan): derive the
    /// next geometric bucket's decode archives into the derived store, off
    /// the per-token path. Returns the pre-derived bucket, or undefined at
    /// the ceiling (or before the first turn sizes the pipeline).
    pub fn prederive_next_window(&mut self) -> Result<Option<u32>, JsValue> {
        let Some(session) = &self.session else {
            return Ok(None);
        };
        let current = session.geometry().bucket;
        self.growable
            .borrow_mut()
            .prederive_next_decode_bucket(current)
            .map(|w| w.map(|w| w as u32))
            .map_err(|e| err(format!("prederive: {e:#}")))
    }
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

    /// The `stages.json` quant map round-trips through `parse_quant_json`: a
    /// whole projection keys by its bare κ, and a head chunk — carrying
    /// `offset`/`len` — keys by the composite `κ@offset+len` (the one
    /// `quant_key` law), so the several chunks sharing one wide κ each resolve
    /// their own artifact. A plain host `#[test]` (not `wasm_bindgen_test`): the
    /// function is pure Rust, and this is the ranged-persistence seam the browser
    /// download→session round-trip depends on.
    #[test]
    fn parse_quant_json_builds_composite_keys_for_ranged_head_chunks() {
        let json = r#"[
            {"wide":"proj-k","artifact":"proj-art","out":4,"in":8},
            {"wide":"embed-k","artifact":"chunk-art-a","out":3,"in":8,"offset":0,"len":48},
            {"wide":"embed-k","artifact":"chunk-art-b","out":3,"in":8,"offset":48,"len":48}
        ]"#;
        let map = parse_quant_json(Some(json.to_string()))
            .expect("valid quant JSON parses")
            .expect("a non-empty map");
        // Whole projection: bare-κ key.
        assert_eq!(
            map.get("proj-k").map(|v| (v.0.as_str(), v.1, v.2)),
            Some(("proj-art", 4, 8))
        );
        // Head chunks: κ@offset+len keys, one artifact each, sharing the wide κ.
        assert_eq!(
            map.get("embed-k@0+48").map(|v| v.0.as_str()),
            Some("chunk-art-a")
        );
        assert_eq!(
            map.get("embed-k@48+48").map(|v| v.0.as_str()),
            Some("chunk-art-b")
        );
        assert_eq!(
            map.len(),
            3,
            "three distinct keys — the two chunks do not collide"
        );
    }

    /// [`drain_complete_utf8`] under the WORST chunking — one byte per write:
    /// a multibyte character split across writes never emits partial bytes
    /// (every emitted chunk ends on a character boundary, so every prefix of
    /// the accumulation is a prefix of the text), and the accumulation equals
    /// the concatenation with nothing held back at the end. Plain host
    /// `#[test]`: the helper is pure Rust — this is the delta-boundary law
    /// [`CallbackSink`]'s JS callback stream rests on.
    #[test]
    fn utf8_delta_boundary_never_splits_a_character_and_accumulates_exactly() {
        // 1-, 2-, 3- and 4-byte characters, adjacent in every order.
        let text = "a\u{00E9}\u{2192}\u{1F980}x\u{1F980}\u{2192}\u{00E9}b";
        let bytes = text.as_bytes();
        let mut pending = Vec::new();
        let mut emitted = String::new();
        for b in bytes {
            let delta =
                drain_complete_utf8(&mut pending, std::slice::from_ref(b)).expect("valid stream");
            emitted.push_str(&delta);
            assert!(
                text.starts_with(&emitted),
                "emitted a byte sequence the text does not begin with: {emitted:?}"
            );
        }
        assert_eq!(emitted, text, "accumulation equals concatenation");
        assert!(pending.is_empty(), "a finished stream holds nothing back");
    }

    /// An incomplete character emits NOTHING until its last byte arrives, then
    /// emits exactly the whole character — the held-back tail is prepended to
    /// the next write, not dropped or flushed early.
    #[test]
    fn utf8_delta_boundary_holds_an_incomplete_tail_across_writes() {
        let crab = "\u{1F980}".as_bytes(); // 4 bytes
        let mut pending = Vec::new();
        let first = drain_complete_utf8(&mut pending, &crab[..2]).expect("incomplete, not invalid");
        assert_eq!(first, "", "a half-written character must not surface");
        assert_eq!(pending, &crab[..2], "the partial bytes are held back");
        let rest = drain_complete_utf8(&mut pending, &crab[2..]).expect("now complete");
        assert_eq!(rest, "\u{1F980}", "completion emits the whole character");
        assert!(pending.is_empty());
    }

    /// Definitely invalid UTF-8 — bytes no continuation could complete — is an
    /// error, never silently skipped (fail loud, not stream corrupt text).
    #[test]
    fn utf8_delta_boundary_rejects_invalid_bytes_loudly() {
        let mut pending = Vec::new();
        drain_complete_utf8(&mut pending, &[b'o', b'k', 0xFF])
            .expect_err("0xFF can never begin a UTF-8 sequence");
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

    /// Build the bare v0.9.0 fused decode step in OUR IR and drive it once over a
    /// REALIZED past: `attn = DecodeAttention(q, k_past, v_past, k_new, v_new,
    /// mask)` (κ119) plus two `KvCacheWrite`s (κ120) at `pos = 3` with a mask
    /// that reveals columns 0..=3 — so κ119 reads real keys, exactly as the
    /// deployed model does on the step after prefill. The native witness
    /// `v090_fused_decode_lowering.rs` does this at head_dim 16 and PASSES; this
    /// helper compiles AND executes it entirely in-wasm at an arbitrary head_dim.
    fn fused_decode_over_realized_past(d: u64) {
        use hologram_ai::HoloRunner;
        const B: u64 = 1;
        const H: u64 = 4;
        const HKV: u64 = 2;
        const BUCKET: u64 = 8;

        let (q, kp, vp, kn, vn, mask, pos, attn, kc, vc) = (0u32, 1, 2, 3, 4, 5, 6, 7, 8, 9);
        let mut tinfo = HashMap::new();
        tinfo.insert(q, ti(DType::F32, &[B, H, 1, d]));
        tinfo.insert(kp, ti(DType::F32, &[B, HKV, BUCKET, d]));
        tinfo.insert(vp, ti(DType::F32, &[B, HKV, BUCKET, d]));
        tinfo.insert(kn, ti(DType::F32, &[B, HKV, 1, d]));
        tinfo.insert(vn, ti(DType::F32, &[B, HKV, 1, d]));
        tinfo.insert(mask, ti(DType::F32, &[1, BUCKET + 1]));
        tinfo.insert(pos, ti(DType::INT32, &[1]));
        tinfo.insert(attn, ti(DType::F32, &[B, H, 1, d]));
        tinfo.insert(kc, ti(DType::F32, &[B, HKV, BUCKET, d]));
        tinfo.insert(vc, ti(DType::F32, &[B, HKV, BUCKET, d]));
        let g = AiGraph {
            name: "fused_decode_step".into(),
            nodes: vec![
                AiNode::new(
                    0,
                    AiOp::DecodeAttention,
                    vec![q, kp, vp, kn, vn, mask],
                    vec![attn],
                ),
                AiNode::new(1, AiOp::KvCacheWrite, vec![kp, kn, pos], vec![kc]),
                AiNode::new(2, AiOp::KvCacheWrite, vec![vp, vn, pos], vec![vc]),
            ],
            inputs: vec![q, kp, vp, kn, vn, mask, pos],
            outputs: vec![attn, kc, vc],
            input_names: Vec::new(),
            output_names: Vec::new(),
            params: HashMap::new(),
            tensor_info: tinfo,
            metadata: HashMap::new(),
            warnings: Vec::new(),
            dim_vars: Default::default(),
            shape_constraints: Default::default(),
            subgraphs: HashMap::new(),
            tensor_names: HashMap::new(),
            topo_cache: Default::default(),
        };
        let archive = ModelCompiler::default()
            .compile(ModelSource::AiGraph(g))
            .expect("the fused decode step compiles in-wasm");

        let planes = (B * HKV) as usize;
        let (bucket, dd) = (BUCKET as usize, d as usize);
        let f32s = |n: usize, seed: usize| -> Vec<f32> {
            (0..n)
                .map(|i| (((i * 13 + seed * 7) % 41) as f32 - 20.0) * 0.043)
                .collect()
        };
        let to_le = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let pos_v = 3u32;
        let mask_v: Vec<f32> = (0..(bucket + 1))
            .map(|j| {
                if j <= pos_v as usize {
                    0.0
                } else {
                    f32::NEG_INFINITY
                }
            })
            .collect();
        let inputs: Vec<Vec<u8>> = vec![
            to_le(&f32s((B * H * d) as usize, 1)),
            to_le(&f32s(planes * bucket * dd, 2)),
            to_le(&f32s(planes * bucket * dd, 3)),
            to_le(&f32s(planes * dd, 4)),
            to_le(&f32s(planes * dd, 5)),
            to_le(&mask_v),
            pos_v.to_le_bytes().to_vec(),
        ];
        let refs: Vec<&[u8]> = inputs.iter().map(|v| v.as_slice()).collect();
        let mut runner = HoloRunner::from_bytes(archive.bytes)
            .expect("archive loads in-wasm through HoloRunner");
        let out = runner
            .execute(&refs)
            .expect("the fused decode step executes in-wasm");
        let attn_out: Vec<f32> = out[0]
            .bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(
            attn_out.len(),
            (B * H * d) as usize,
            "head_dim {d}: attention output is q-shaped"
        );
        assert!(
            attn_out.iter().all(|x| x.is_finite()),
            "head_dim {d}: fused decode over a realized past is finite in wasm — no trap"
        );
    }

    /// REPRO A — the bare κ119/κ120 kernel over a realized past, in-wasm. The
    /// deployed browser streams token 1, then traps `RuntimeError: unreachable`
    /// on the first decode step that reads a realized past (Qwen2.5-1.5B,
    /// head_dim 128). This drives the SAME bare fused step the native witness
    /// passes — compiled and executed entirely in-wasm — at head_dim 16 (the
    /// native size) then the production head_dim 128. RESULT: both pass, so the
    /// bare kernel over a realized past is sound in wasm; the trap is elsewhere.
    #[wasm_bindgen_test]
    fn fused_decode_over_realized_past_in_wasm() {
        fused_decode_over_realized_past(16);
        web_sys::console::log_1(&JsValue::from_str(
            "REPRO: head_dim 16 bare fused decode over a realized past — OK in wasm",
        ));
        fused_decode_over_realized_past(128);
        web_sys::console::log_1(&JsValue::from_str(
            "REPRO: head_dim 128 bare fused decode over a realized past — OK in wasm",
        ));
    }

    /// Drive the fused resident-KV decode over TWO carried walks in-wasm — the
    /// path NO existing test exercised (`v090_fused_equals_legacy` drives bare
    /// `execute`, never `execute_kv_resident`). Build a 1-layer GQA decoder,
    /// `rewrite_decode_attention(.., resident_kv = true)` (bypassing the wasm
    /// gate that forces `false`), then two `execute_kv_resident` walks: walk 1
    /// `carry = false` (host K/V is the truth — the first decode step), then
    /// walk 2 `carry = true` — the FIRST walk that `release_label`s the retained
    /// cache so κ120 does the in-place MOVE and κ119 reads the carried past by
    /// label. That is exactly the second decode step the deployed Qwen traps on.
    fn fused_resident_two_walks(h: u64, kv: u64, dh: u64, bucket: u64) {
        use hologram_ai::HoloRunner;
        use hologram_ai_common::opt::decode_plan::{
            past_key_port, past_value_port, rewrite_decode_attention, DECODE_MASK_PORT,
            DECODE_POS_PORT, DECODE_ROPE_COS_K_PORT, DECODE_ROPE_COS_Q_PORT,
            DECODE_ROPE_SIN_K_PORT, DECODE_ROPE_SIN_Q_PORT,
        };

        let (q, k, v, attn) = (0u32, 1, 2, 3);
        let mut tinfo = HashMap::new();
        tinfo.insert(q, ti(DType::F32, &[1, h, 1, dh]));
        tinfo.insert(k, ti(DType::F32, &[1, kv, 1, dh]));
        tinfo.insert(v, ti(DType::F32, &[1, kv, 1, dh]));
        tinfo.insert(attn, ti(DType::F32, &[1, h, 1, dh]));
        let mut graph = AiGraph {
            name: "gqa1".into(),
            nodes: vec![AiNode::new(
                0,
                AiOp::GroupedQueryAttention {
                    num_heads: h as u32,
                    num_kv_heads: kv as u32,
                    head_dim: dh as u32,
                    scale: None,
                    causal: true,
                    heads_first: true,
                    qk_norm: false,
                    rope: Some(hologram_ai_common::RopeSpec::plain(10000.0)),
                },
                vec![q, k, v],
                vec![attn],
            )],
            inputs: vec![q, k, v],
            outputs: vec![attn],
            input_names: vec!["q".into(), "k".into(), "v".into()],
            output_names: vec!["attn".into()],
            params: HashMap::new(),
            tensor_info: tinfo,
            metadata: HashMap::new(),
            warnings: Vec::new(),
            dim_vars: Default::default(),
            shape_constraints: Default::default(),
            subgraphs: HashMap::new(),
            tensor_names: HashMap::new(),
            topo_cache: Default::default(),
        };
        rewrite_decode_attention(&mut graph, bucket, 1, 0, true)
            .expect("rewrite the single GQA layer to fused resident-KV");
        let archive = ModelCompiler::default()
            .compile(ModelSource::AiGraph(graph))
            .expect("fused resident-KV decoder compiles in-wasm");
        let mut runner = HoloRunner::from_bytes(archive.bytes)
            .expect("archive loads in-wasm through HoloRunner");

        let f32s = |n: usize, seed: usize| -> Vec<f32> {
            (0..n)
                .map(|i| (((i * 31 + seed * 17) % 53) as f32 - 26.0) * 0.037)
                .collect()
        };
        let to_le = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let realized = (bucket / 2).max(1);
        let mut named: HashMap<String, Vec<u8>> = HashMap::new();
        named.insert("q".into(), to_le(&f32s((h * dh) as usize, 1)));
        named.insert("k".into(), to_le(&f32s((kv * dh) as usize, 2)));
        named.insert("v".into(), to_le(&f32s((kv * dh) as usize, 3)));
        named.insert(
            DECODE_ROPE_COS_Q_PORT.into(),
            to_le(&vec![1.0; (h * dh) as usize]),
        );
        named.insert(
            DECODE_ROPE_SIN_Q_PORT.into(),
            to_le(&vec![0.0; (h * dh) as usize]),
        );
        named.insert(
            DECODE_ROPE_COS_K_PORT.into(),
            to_le(&vec![1.0; (kv * dh) as usize]),
        );
        named.insert(
            DECODE_ROPE_SIN_K_PORT.into(),
            to_le(&vec![0.0; (kv * dh) as usize]),
        );
        let kv_bytes = (kv * bucket * dh) as usize;
        named.insert(past_key_port(0), to_le(&f32s(kv_bytes, 4)));
        named.insert(past_value_port(0), to_le(&f32s(kv_bytes, 5)));
        // One query row's visibility over [bucket past ∥ 1 new key] (resident
        // form → exactly one mask row).
        let row: Vec<f32> = (0..bucket + 1)
            .map(|j| {
                if j < realized || j == bucket {
                    0.0
                } else {
                    f32::NEG_INFINITY
                }
            })
            .collect();
        named.insert(DECODE_MASK_PORT.into(), to_le(&row));
        named.insert(
            DECODE_POS_PORT.into(),
            (realized as u32).to_le_bytes().to_vec(),
        );

        let order = |runner: &HoloRunner, named: &HashMap<String, Vec<u8>>| -> Vec<Vec<u8>> {
            runner
                .input_port_info()
                .iter()
                .map(|p| {
                    named
                        .get(&p.name)
                        .unwrap_or_else(|| panic!("no test data for input port `{}`", p.name))
                        .clone()
                })
                .collect()
        };

        // Walk 1: carry = false — the first decode step (host K/V is the truth).
        let bufs1 = order(&runner, &named);
        let refs1: Vec<&[u8]> = bufs1.iter().map(|b| b.as_slice()).collect();
        runner
            .execute_kv_resident(&refs1, false)
            .expect("walk 1 (carry=false) executes in-wasm");
        assert!(
            runner.has_kv_carry(),
            "walk 1 must leave a resident K/V carry"
        );

        // Walk 2: carry = true — releases the retained cache label so κ120 MOVES
        // in place and κ119 reads the carried past by label. THE deployed step.
        let bufs2 = order(&runner, &named);
        let refs2: Vec<&[u8]> = bufs2.iter().map(|b| b.as_slice()).collect();
        let out2 = runner
            .execute_kv_resident(&refs2, true)
            .expect("walk 2 (carry=true) executes in-wasm — the deployed trap step");
        let attn2 = out2
            .iter()
            .flatten()
            .next()
            .expect("walk 2 yields the attention output");
        let vals: Vec<f32> = attn2
            .bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert!(
            !vals.is_empty() && vals.iter().all(|x| x.is_finite()),
            "head_dim {dh}: fused resident-KV carry (walk 2, the in-place move) is finite in wasm — no trap"
        );
    }

    /// REPRO B — the resident-KV carry/steal path in-wasm, at head_dim 16 then
    /// the production head_dim 128. The console marker pins which head_dim a
    /// trap lands on. Native `decode_family_coverage` drives this exact carry
    /// (int8, staged, evicted) and PASSES — so a trap here is wasm-specific.
    #[wasm_bindgen_test]
    fn fused_resident_carry_two_walks_in_wasm() {
        fused_resident_two_walks(4, 2, 16, 6);
        web_sys::console::log_1(&JsValue::from_str(
            "REPRO: resident-KV carry (2 walks) head_dim 16 — OK in wasm",
        ));
        fused_resident_two_walks(4, 2, 128, 8);
        web_sys::console::log_1(&JsValue::from_str(
            "REPRO: resident-KV carry (2 walks) head_dim 128 — OK in wasm",
        ));
    }

    /// Guard: the browser (wasm) MUST ship the legacy decode decomposition, not
    /// the fused resident-KV path that traps `unreachable` on the deployed
    /// model's staged carry-across-eviction step. This locks the mitigation in
    /// place until the substrate's wasm staged-carry path is fixed and verified
    /// — flipping the gate back on for wasm turns this red before it can deploy.
    #[wasm_bindgen_test]
    fn browser_ships_legacy_decode_not_the_trapping_fused_path() {
        assert!(
            !hologram_ai_safetensors::parametric::fused_resident_decode_enabled(),
            "the wasm build must NOT compile the fused resident-KV decode — it \
             traps `unreachable` on the deployed model's staged carry-across-\
             eviction step (docs/notes/upstream-issue-v090-wasm-decode-unreachable.md)"
        );
    }
}

#[cfg(test)]
mod residency_budget_tests {
    use super::{decode_residency_budget, HOST_HEADROOM, STRUCTURAL_CEILING};
    use hologram_ai::decode::DecodeGeometry;

    fn geom(bucket: usize) -> DecodeGeometry {
        DecodeGeometry {
            layers: 28,
            kv_heads: 2,
            heads: 12,
            head_dim: 128,
            bucket,
            chunk: 1,
            vocab: 151936,
            resident_kv: false,
        }
    }

    #[test]
    fn the_budget_shrinks_as_the_carried_kv_grows() {
        // Nothing carried: the whole ceiling less the HOST's own headroom.
        assert_eq!(
            decode_residency_budget(0),
            STRUCTURAL_CEILING - HOST_HEADROOM
        );
        // The model's own K/V is charged, so a wider bucket leaves less room for
        // resident weights — the term a fixed reserve silently omitted.
        let small = geom(128).carried_kv_bytes().unwrap();
        let large = geom(32768).carried_kv_bytes().unwrap();
        assert!(large > small);
        assert!(decode_residency_budget(large) < decode_residency_budget(small));
        assert_eq!(
            decode_residency_budget(small),
            STRUCTURAL_CEILING - HOST_HEADROOM - small
        );
    }

    #[test]
    fn a_long_context_kv_dwarfs_any_fixed_reserve() {
        // The whole point: at a 32 k bucket this model carries ~1.9 GiB of K/V —
        // nearly twice the host headroom. A budget that lumped K/V into a fixed
        // reserve would admit weights into memory the K/V already owns.
        let kv = geom(32768).carried_kv_bytes().unwrap();
        assert!(
            kv > HOST_HEADROOM,
            "carried K/V {kv} must exceed the host headroom"
        );
    }

    #[test]
    fn the_budget_saturates_rather_than_wrapping() {
        // A carried K/V larger than the whole ceiling yields zero residency, not
        // an underflowed enormous budget.
        assert_eq!(decode_residency_budget(u64::MAX), 0);
        assert_eq!(decode_residency_budget(STRUCTURAL_CEILING), 0);
    }
}
