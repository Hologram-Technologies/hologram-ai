//! Parametric decoder-graph builder.
//!
//! Builds a decoder-only transformer graph from a model's own `config.json`
//! plus its tensor manifest (names and storage dtypes). Every quantity is a
//! function of the configuration — hidden size, layers, heads, KV heads, head
//! dim, vocabulary, `rope_theta`, `rms_norm_eps`, `tie_word_embeddings`,
//! context length, tensor dtypes. Architecture families are selected from
//! `config.architectures[0]` via the family registry; anything else fails
//! loud. No canonical model constant appears in this code path.

use crate::builder::GraphBuilder;
use anyhow::{anyhow, ensure, Result};
use hologram_ai_common::ir::{
    dtype::DType, graph::AiGraph, node::TensorId, op::AiOp, shape::DimExpr,
};
use hologram_ai_common::MetaValue;
use safetensors::{Dtype as SafeDtype, SafeTensors};
use serde_json::Value;
use std::collections::HashMap;
use std::num::NonZeroU64;

fn map_dtype(d: SafeDtype) -> Result<DType> {
    match d {
        SafeDtype::F32 => Ok(DType::F32),
        SafeDtype::F16 => Ok(DType::F16),
        SafeDtype::BF16 => Ok(DType::BF16),
        SafeDtype::I64 => Ok(DType::INT64),
        SafeDtype::I32 => Ok(DType::INT32),
        _ => Err(anyhow!("Unsupported safetensors dtype: {:?}", d)),
    }
}

/// One architecture family the parametric builder knows how to assemble.
///
/// The registry is the single place where family-specific structure lives;
/// every other quantity comes from the model's own `config.json`.
struct FamilySpec {
    /// The `config.architectures[0]` value this entry matches.
    name: &'static str,
    /// The attention Q/K/V projections carry bias tensors
    /// (`model.layers.N.self_attn.{q,k,v}_proj.bias`) as a structural property
    /// of the family (Qwen2), independent of any `attention_bias` config flag.
    attention_qkv_bias: bool,
    /// The checkpoint ships the attention Q/K/V weights as one fused
    /// `model.layers.N.self_attn.qkv_proj.weight` tensor whose rows are
    /// `[q (heads·head_dim); k (kv_heads·head_dim); v (kv_heads·head_dim)]`
    /// (Phi3). The builder declares the *fused* tensor — the name the κ-map
    /// binds — and carves the three operands out with compile-time `Slice`
    /// nodes.
    attention_fused_qkv: bool,
    /// The checkpoint ships the MLP gate/up weights as one fused
    /// `model.layers.N.mlp.gate_up_proj.weight` tensor whose rows are
    /// `[gate (intermediate); up (intermediate)]` (Phi3), realized by the
    /// same compile-time `Slice` of the fused operand.
    mlp_fused_gate_up: bool,
    /// The family's published checkpoints attend with a sliding window: a
    /// non-null `sliding_window` in the config clamps the effective context
    /// length to `min(max_position_embeddings, sliding_window)` — see
    /// [`effective_context_ceiling`] for why anything longer is forbidden.
    sliding_window_clamp: bool,
    /// Semantic config keys this builder does not implement for the family.
    /// A key that is present and non-null fails the build loud, naming the
    /// knob — a semantic config key is never silently ignored.
    unsupported_knobs: &'static [&'static str],
}

/// The architecture-family registry: `config.architectures[0]` → structure.
const SUPPORTED_FAMILIES: &[FamilySpec] = &[
    FamilySpec {
        name: "LlamaForCausalLM",
        attention_qkv_bias: false,
        attention_fused_qkv: false,
        mlp_fused_gate_up: false,
        sliding_window_clamp: false,
        unsupported_knobs: &[],
    },
    FamilySpec {
        name: "Qwen2ForCausalLM",
        attention_qkv_bias: true,
        attention_fused_qkv: false,
        mlp_fused_gate_up: false,
        sliding_window_clamp: false,
        unsupported_knobs: &[],
    },
    // Tensor-identical to Llama (separate q/k/v and gate/up projections,
    // untied lm_head, no biases); sliding-window checkpoints clamp the
    // effective context length.
    FamilySpec {
        name: "MistralForCausalLM",
        attention_qkv_bias: false,
        attention_fused_qkv: false,
        mlp_fused_gate_up: false,
        sliding_window_clamp: true,
        unsupported_knobs: &[],
    },
    // Llama-family compute with fused qkv_proj / gate_up_proj checkpoint
    // tensors, realized by compile-time Slice. Partial-rotary and
    // rope-scaling variants (Phi-3-mini) are not implemented: those knobs
    // change the RoPE semantics, so they fail loud instead of being ignored.
    FamilySpec {
        name: "Phi3ForCausalLM",
        attention_qkv_bias: false,
        attention_fused_qkv: true,
        mlp_fused_gate_up: true,
        sliding_window_clamp: true,
        unsupported_knobs: &["partial_rotary_factor", "rope_scaling"],
    },
];

/// The architecture-family names the registry supports — the single source
/// the browser's search filter and the coverage probe read (dictionary rows
/// `supported-search`, `family-registry-support`).
pub fn supported_families() -> Vec<&'static str> {
    SUPPORTED_FAMILIES.iter().map(|f| f.name).collect()
}

/// The registry family selected for a config — the same selection the
/// builder performs. The external-authority witness (dictionary row
/// `family-registry-support`) asserts this against the published model's own
/// `config.json`.
pub fn selected_family(config: &Value) -> Result<&'static str> {
    Ok(select_family(config)?.name)
}

fn select_family(config: &Value) -> Result<&'static FamilySpec> {
    let name = config
        .get("architectures")
        .and_then(Value::as_array)
        .and_then(|a| a.first())
        .and_then(Value::as_str)
        .ok_or_else(|| {
            anyhow!(
                "config.json is missing required key `architectures` \
                 (expected `architectures[0]` to name the model family)"
            )
        })?;
    SUPPORTED_FAMILIES
        .iter()
        .find(|f| f.name == name)
        .ok_or_else(|| {
            let supported: Vec<&str> = SUPPORTED_FAMILIES.iter().map(|f| f.name).collect();
            anyhow!(
                "unsupported architecture family `{name}` — supported families: {}",
                supported.join(", ")
            )
        })
}

/// Every parametric quantity of the decoder graph, extracted from the model's
/// own `config.json`. Required keys fail loud when absent; the only permitted
/// derivations are the transformers conventions documented on the fields.
struct ModelConfig {
    hidden_size: u64,
    num_hidden_layers: u64,
    num_attention_heads: u64,
    /// `num_key_value_heads`; when absent, the transformers convention is
    /// plain multi-head attention: one KV head per query head.
    num_key_value_heads: u64,
    /// `head_dim`; when absent, the transformers convention splits the hidden
    /// dimension evenly across the attention heads.
    head_dim: u64,
    vocab_size: u64,
    intermediate_size: u64,
    rms_norm_eps: f32,
    rope_theta: f32,
    max_position_embeddings: u64,
    /// `tie_word_embeddings`; absent means untied unless the manifest carries
    /// no separate `lm_head.weight` (see the LM-head wiring below).
    tie_word_embeddings: bool,
    /// Llama-style `attention_bias`: Q/K/V/O projections carry bias tensors.
    attention_bias: bool,
    /// Llama-style `mlp_bias`: gate/up/down projections carry bias tensors.
    mlp_bias: bool,
    /// `sliding_window`: a non-null value is the trained attention span of a
    /// sliding-window checkpoint. Explicit `null` (the published
    /// full-attention convention) reads as absent.
    sliding_window: Option<u64>,
}

fn require_u64(config: &Value, key: &str) -> Result<u64> {
    config.get(key).and_then(Value::as_u64).ok_or_else(|| {
        anyhow!("config.json is missing required key `{key}` (or it is not a positive integer)")
    })
}

fn require_f64(config: &Value, key: &str) -> Result<f64> {
    config.get(key).and_then(Value::as_f64).ok_or_else(|| {
        anyhow!("config.json is missing required key `{key}` (or it is not a number)")
    })
}

/// A boolean config flag; absent means `false` (the transformers default for
/// `tie_word_embeddings`, `attention_bias`, and `mlp_bias`).
fn config_flag(config: &Value, key: &str) -> bool {
    config.get(key).and_then(Value::as_bool).unwrap_or(false)
}

impl ModelConfig {
    fn from_json(config: &Value) -> Result<Self> {
        let hidden_size = require_u64(config, "hidden_size")?;
        let num_hidden_layers = require_u64(config, "num_hidden_layers")?;
        let num_attention_heads = require_u64(config, "num_attention_heads")?;
        let vocab_size = require_u64(config, "vocab_size")?;
        let intermediate_size = require_u64(config, "intermediate_size")?;
        let rms_norm_eps = require_f64(config, "rms_norm_eps")? as f32;
        let rope_theta = require_f64(config, "rope_theta")? as f32;
        let max_position_embeddings = require_u64(config, "max_position_embeddings")?;
        ensure!(
            num_attention_heads > 0,
            "config.json key `num_attention_heads` must be positive"
        );

        // Transformers convention: a config without `num_key_value_heads` is
        // plain multi-head attention (one KV head per query head).
        let num_key_value_heads = match config.get("num_key_value_heads") {
            Some(v) => v.as_u64().ok_or_else(|| {
                anyhow!("config.json key `num_key_value_heads` is not a positive integer")
            })?,
            None => num_attention_heads,
        };
        ensure!(
            num_key_value_heads > 0 && num_attention_heads % num_key_value_heads == 0,
            "config.json: num_attention_heads ({num_attention_heads}) must be a positive \
             multiple of num_key_value_heads ({num_key_value_heads})"
        );

        // Transformers convention: a config without `head_dim` splits the
        // hidden dimension evenly across the attention heads.
        let head_dim = match config.get("head_dim") {
            Some(v) => v
                .as_u64()
                .ok_or_else(|| anyhow!("config.json key `head_dim` is not a positive integer"))?,
            None => {
                ensure!(
                    hidden_size % num_attention_heads == 0,
                    "config.json: hidden_size ({hidden_size}) is not divisible by \
                     num_attention_heads ({num_attention_heads}) and no `head_dim` is given"
                );
                hidden_size / num_attention_heads
            }
        };

        // `sliding_window`: non-null must be a positive integer; explicit
        // null is the full-attention convention and reads as absent.
        let sliding_window = match config.get("sliding_window") {
            None | Some(Value::Null) => None,
            Some(v) => Some(v.as_u64().ok_or_else(|| {
                anyhow!("config.json key `sliding_window` is not a positive integer or null")
            })?),
        };

        Ok(Self {
            hidden_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            vocab_size,
            intermediate_size,
            rms_norm_eps,
            rope_theta,
            max_position_embeddings,
            tie_word_embeddings: config_flag(config, "tie_word_embeddings"),
            attention_bias: config_flag(config, "attention_bias"),
            mlp_bias: config_flag(config, "mlp_bias"),
            sliding_window,
        })
    }
}

/// A semantic config key the registry entry does not implement is never
/// silently ignored: present and non-null, it fails the build loud naming
/// the knob (e.g. Phi-3-mini's `partial_rotary_factor` / `rope_scaling`,
/// which change the RoPE semantics the builder would otherwise misstate).
fn reject_unsupported_knobs(family: &FamilySpec, config: &Value) -> Result<()> {
    for knob in family.unsupported_knobs {
        let carried = config.get(*knob).map(|v| !v.is_null()).unwrap_or(false);
        ensure!(
            !carried,
            "config.json carries `{knob}` — the `{}` family builder does not implement \
             this knob and refuses to silently ignore a semantic config key",
            family.name
        );
    }
    Ok(())
}

/// The tensor manifest: the names and storage dtypes of the model's weights.
struct TensorManifest<'a> {
    dtypes: HashMap<&'a str, DType>,
}

impl<'a> TensorManifest<'a> {
    fn new(keys: &'a [String], dtypes: &[DType]) -> Result<Self> {
        ensure!(
            keys.len() == dtypes.len(),
            "tensor manifest has {} keys but {} dtypes",
            keys.len(),
            dtypes.len()
        );
        Ok(Self {
            dtypes: keys
                .iter()
                .map(String::as_str)
                .zip(dtypes.iter().copied())
                .collect(),
        })
    }

    fn contains(&self, name: &str) -> bool {
        self.dtypes.contains_key(name)
    }

    /// Storage dtype of a manifest tensor. A name outside the manifest (the
    /// keys-only compatibility path, [`build_parametric_graph_from_keys`])
    /// carries no dtype information and is declared at the F32 compute type.
    fn dtype_of(&self, name: &str) -> DType {
        self.dtypes.get(name).copied().unwrap_or(DType::F32)
    }
}

/// Config-only preflight: the architecture family must be registered and the
/// family's required keys present and well-formed. Weight- and manifest-free —
/// the earliest, cheapest rejection point of the journey (S1 preflight, step
/// a). Fails loud naming the family or the missing key.
pub fn validate_config(config: &Value) -> Result<()> {
    let family = select_family(config)?;
    reject_unsupported_knobs(family, config)?;
    let _cfg = ModelConfig::from_json(config)?;
    Ok(())
}

/// Build the graph directly from safetensors shard bytes: the manifest (keys,
/// dtypes) is read from the shard headers and the weights are injected inline.
pub fn build_parametric_graph(config: &Value, safetensors_shards: &[&[u8]]) -> Result<AiGraph> {
    let mut st_instances = Vec::new();
    for shard in safetensors_shards {
        let st = SafeTensors::deserialize(shard)?;
        st_instances.push(st);
    }

    // Collect the manifest in a deterministic (name-sorted) order. `safetensors`
    // stores its tensor index in a `HashMap`, so `SafeTensors::tensors()`
    // iteration order varies per call; sorting by name makes tensor-id
    // allocation — and thus the emitted archive — a pure function of the shard
    // bytes, never of a map seed (content addressing requires a stable κ).
    let mut tensors: Vec<(String, safetensors::tensor::TensorView<'_>)> = Vec::new();
    for st in &st_instances {
        for (k, view) in st.tensors() {
            tensors.push((k, view));
        }
    }
    tensors.sort_by(|(a, _), (b, _)| a.cmp(b));

    let mut keys = Vec::with_capacity(tensors.len());
    let mut dtypes = Vec::with_capacity(tensors.len());
    for (k, view) in &tensors {
        dtypes.push(map_dtype(view.dtype())?);
        keys.push(k.clone());
    }

    let mut graph = build_parametric_graph_from_manifest(config, &keys, &dtypes, None)?;

    // Inject the actual safetensors weights into the graph's params, in the same
    // name-sorted order so any tensor the builder did not declare (an unused
    // checkpoint buffer) is allocated a deterministic tensor id.
    let mut name_to_id = HashMap::new();
    for (id, name) in &graph.tensor_names {
        name_to_id.insert(name.clone(), *id);
    }

    let mut next_id = graph.tensor_names.keys().max().copied().unwrap_or(0) + 1;
    for (k, tensor_view) in &tensors {
        let id = if let Some(existing_id) = name_to_id.get(k) {
            *existing_id
        } else {
            let new_id = next_id;
            next_id += 1;
            graph.tensor_names.insert(new_id, k.clone());
            new_id
        };

        let dtype = map_dtype(tensor_view.dtype())?;
        let shape = hologram_ai_common::shape_from_concrete(
            &tensor_view
                .shape()
                .iter()
                .map(|&x| x as u64)
                .collect::<Vec<_>>(),
        );
        let info = hologram_ai_common::TensorInfo::new(dtype, shape);
        graph.tensor_info.insert(id, info.clone());

        let data = tensor_view.data().to_vec();
        graph.params.insert(
            id,
            hologram_ai_common::ir::param::AiParam::inline(data, info),
        );
    }

    Ok(graph)
}

/// Keys-only compatibility entry: the manifest carries no dtype information,
/// so every weight is declared at the F32 compute type and the context length
/// is the model's own `max_position_embeddings`. Callers holding per-tensor
/// dtypes should use [`build_parametric_graph_from_manifest`].
pub fn build_parametric_graph_from_keys(config: &Value, keys: &[String]) -> Result<AiGraph> {
    let dtypes = vec![DType::F32; keys.len()];
    build_parametric_graph_from_manifest(config, keys, &dtypes, None)
}

/// Shape and naming parameters for one `x · Wᵀ` linear layer.
struct LinearLayerParams<'a> {
    weight_name: &'a str,
    in_features: DimExpr,
    out_features: DimExpr,
    output_name: &'a str,
    output_shape: Vec<DimExpr>,
}

/// A linear layer over an already-declared (and F32-typed) `[out, in]` weight:
/// transpose to `[in, out]`, then matmul.
fn add_linear_layer_from_tensor(
    builder: &mut GraphBuilder,
    input: TensorId,
    weight: TensorId,
    params: LinearLayerParams<'_>,
) -> TensorId {
    let transposed_weight = builder.add_tensor(
        &format!("{}_transposed", params.weight_name),
        DType::F32,
        vec![params.in_features, params.out_features],
    );
    builder.add_node(
        AiOp::Transpose { perm: vec![1, 0] },
        vec![weight],
        vec![transposed_weight],
    );
    let output = builder.add_tensor(params.output_name, DType::F32, params.output_shape);
    builder.add_node(AiOp::MatMul, vec![input, transposed_weight], vec![output]);
    output
}

/// Declare a weight tensor at its manifest storage dtype and return an
/// F32-typed view of it. When the stored dtype is narrower than the F32
/// compute type, the IR's canonical `Cast` node is inserted — the same
/// conversion ONNX-imported graphs carry for F16/BF16 weights.
fn add_weight_f32(
    builder: &mut GraphBuilder,
    manifest: &TensorManifest<'_>,
    name: &str,
    shape: Vec<DimExpr>,
) -> TensorId {
    let dtype = manifest.dtype_of(name);
    let weight = builder.add_tensor(name, dtype, shape.clone());
    if dtype == DType::F32 {
        return weight;
    }
    let cast = builder.add_tensor(&format!("{name}.f32"), DType::F32, shape);
    builder.add_node(AiOp::Cast { to: DType::F32 }, vec![weight], vec![cast]);
    cast
}

/// Declare a weight tensor at its manifest storage dtype and return it **as
/// stored** — no widening `Cast`. Used where a downstream op consumes the
/// weight in its native dtype and only a small, bounded result needs widening:
/// the embedding gathers the token rows from the native-dtype table, then
/// widens the `[batch, seq, hidden]` result — never the whole `[vocab, hidden]`
/// matrix (whose F32 image is ~`vocab · hidden · 4` bytes, the 32-bit-heap
/// allocation trap for a large-vocabulary model).
fn add_weight_native(
    builder: &mut GraphBuilder,
    manifest: &TensorManifest<'_>,
    name: &str,
    shape: Vec<DimExpr>,
) -> TensorId {
    let dtype = manifest.dtype_of(name);
    builder.add_tensor(name, dtype, shape)
}

/// Widen an already-declared native-dtype weight to the F32 compute type,
/// inserting the IR's canonical `Cast` only when the stored dtype is narrower.
/// Returns the weight itself when it is already F32 (the F32 checkpoint path,
/// where the tied head reuses the embedding weight's single κ-bound tensor).
fn widen_weight_to_f32(
    builder: &mut GraphBuilder,
    weight: TensorId,
    weight_dtype: DType,
    name: &str,
    shape: Vec<DimExpr>,
) -> TensorId {
    if weight_dtype == DType::F32 {
        return weight;
    }
    let cast = builder.add_tensor(&format!("{name}.f32"), DType::F32, shape);
    builder.add_node(AiOp::Cast { to: DType::F32 }, vec![weight], vec![cast]);
    cast
}

/// Bytes a dense F32 `[rows, cols]` weight occupies.
fn f32_weight_bytes(rows: u64, cols: u64) -> u64 {
    rows.saturating_mul(cols).saturating_mul(4)
}

/// The largest dense F32 weight the build target's heap can materialize before
/// the allocator aborts. A 32-bit (wasm) tab traps well short of its 4 GiB
/// address space; a large-vocabulary LM head at F32 is past this ceiling. A
/// 64-bit host has no such ceiling — the same weight is unremarkable there.
const fn f32_materialization_ceiling() -> u64 {
    if usize::BITS <= 32 {
        1 << 31 // 2 GiB
    } else {
        u64::MAX
    }
}

/// The loud-fail message when widening `name` to a dense `[rows, cols]` F32
/// weight would exceed `ceiling`; `None` when it fits. Split from the guard so
/// the floor logic is witnessed on a 64-bit host (where the live ceiling is
/// unbounded) with an explicit ceiling.
fn f32_head_floor_error(name: &str, rows: u64, cols: u64, ceiling: u64) -> Option<String> {
    let bytes = f32_weight_bytes(rows, cols);
    (bytes > ceiling).then(|| {
        format!(
            "the language-model head weight `{name}` is [{rows}, {cols}] = {bytes} bytes at \
             F32, exceeding the {ceiling}-byte heap ceiling of a 32-bit (wasm) target. The \
             whole-vocabulary F32 head is a genuine floor there: the matmul kernel consults \
             the weight in full and widens even a narrow-dtype operand to a full F32 panel, \
             so no storage dtype avoids the allocation. Run this model on a 64-bit host."
        )
    })
}

/// Fail loud — naming the tensor and its byte size — when widening the LM-head
/// weight to a dense `[rows, cols]` F32 matrix would exceed the build target's
/// heap ceiling, rather than proceeding to an opaque `RuntimeError: unreachable`
/// allocation trap. No-op on a 64-bit host.
fn guard_f32_head_materialization(name: &str, rows: u64, cols: u64) -> Result<()> {
    match f32_head_floor_error(name, rows, cols, f32_materialization_ceiling()) {
        Some(msg) => Err(anyhow!(msg)),
        None => Ok(()),
    }
}

/// Shape, naming, and bias parameters for one projection.
struct ProjectionParams<'a> {
    /// Manifest name of the `[out, in]` weight tensor.
    weight_name: &'a str,
    /// Manifest name of the optional `[out]` bias tensor.
    bias_name: &'a str,
    /// The family registry or the config's bias flags say this bias must be
    /// present in the manifest.
    bias_expected: bool,
    in_features: DimExpr,
    out_features: DimExpr,
    output_name: &'a str,
    output_shape: Vec<DimExpr>,
}

/// A projection `x · Wᵀ (+ b)`. The weight is declared from the manifest; a
/// bias the manifest carries is consumed as an explicit broadcast `Add` node
/// (the IR's canonical biased-linear form). A bias that is expected — by the
/// family registry or by `attention_bias`/`mlp_bias` — but missing from the
/// manifest fails loud.
fn add_projection(
    builder: &mut GraphBuilder,
    manifest: &TensorManifest<'_>,
    input: TensorId,
    params: ProjectionParams<'_>,
) -> Result<TensorId> {
    let has_bias = manifest.contains(params.bias_name);
    if params.bias_expected && !has_bias {
        return Err(anyhow!(
            "tensor manifest is missing expected bias tensor `{}` \
             (required by the architecture family / bias config)",
            params.bias_name
        ));
    }

    let weight = add_weight_f32(
        builder,
        manifest,
        params.weight_name,
        vec![params.out_features.clone(), params.in_features.clone()],
    );
    let matmul_name = if has_bias {
        format!("{}_matmul", params.output_name)
    } else {
        params.output_name.to_string()
    };
    let projected = add_linear_layer_from_tensor(
        builder,
        input,
        weight,
        LinearLayerParams {
            weight_name: params.weight_name,
            in_features: params.in_features,
            out_features: params.out_features.clone(),
            output_name: &matmul_name,
            output_shape: params.output_shape.clone(),
        },
    );
    if !has_bias {
        return Ok(projected);
    }

    let bias = add_weight_f32(
        builder,
        manifest,
        params.bias_name,
        vec![params.out_features],
    );
    let output = builder.add_tensor(params.output_name, DType::F32, params.output_shape);
    builder.add_node(AiOp::Add, vec![projected, bias], vec![output]);
    Ok(output)
}

/// Declare a fused checkpoint weight from the manifest and return its
/// F32-typed view. Fused families slice this single operand at compile time;
/// the manifest (hence the κ-map) binds the *fused* name — exactly what the
/// downloader persists under κ. A fused family whose manifest lacks the
/// fused tensor cannot be realized and fails loud naming it.
fn add_fused_weight_f32(
    builder: &mut GraphBuilder,
    manifest: &TensorManifest<'_>,
    family: &FamilySpec,
    name: &str,
    shape: Vec<DimExpr>,
) -> Result<TensorId> {
    ensure!(
        manifest.contains(name),
        "tensor manifest is missing fused tensor `{name}` \
         (the `{}` family ships this projection fused)",
        family.name
    );
    Ok(add_weight_f32(builder, manifest, name, shape))
}

/// One compile-time row-block `Slice` of a fused weight's F32 view: rows
/// `[start, start + rows)` become the `[rows, in_features]` operand named
/// `slice_name`. This realizes fused checkpoint tensors (Phi3 `qkv_proj`,
/// `gate_up_proj`) without any runtime split: the fused tensor stays the one
/// External κ-bound param and the slices are ordinary graph nodes.
fn add_row_slice(
    builder: &mut GraphBuilder,
    fused_f32: TensorId,
    slice_name: &str,
    start: u64,
    rows: u64,
    in_features: DimExpr,
) -> TensorId {
    let slice = builder.add_tensor(
        slice_name,
        DType::F32,
        vec![DimExpr::Concrete(rows), in_features],
    );
    builder.add_node(
        AiOp::Slice {
            axes: vec![0],
            starts: vec![start as i64],
            ends: vec![(start + rows) as i64],
            steps: vec![1],
        },
        vec![fused_f32],
        vec![slice],
    );
    slice
}

/// Cross-check `config.num_hidden_layers` against the layer indices named by
/// the tensor manifest — a mismatch means config and weights disagree.
fn validate_layer_count(cfg: &ModelConfig, keys: &[String]) -> Result<()> {
    let manifest_layers = keys
        .iter()
        .filter_map(|k| extract_layer_idx(k))
        .max()
        .map(|max| max as u64 + 1);
    let Some(manifest_layers) = manifest_layers else {
        return Err(anyhow!(
            "tensor manifest names no `model.layers.N` tensors — cannot build a decoder graph"
        ));
    };
    ensure!(
        manifest_layers == cfg.num_hidden_layers,
        "config.json declares num_hidden_layers = {} but the tensor manifest names {} layers",
        cfg.num_hidden_layers,
        manifest_layers
    );
    Ok(())
}

/// The model's effective context ceiling. For families whose published
/// checkpoints attend with a sliding window (the registry's
/// `sliding_window_clamp`), a non-null `sliding_window` clamps the ceiling to
/// `min(max_position_embeddings, sliding_window)`: this builder compiles
/// full-causal attention, which is exactly sliding-window attention only up
/// to the window length — a longer window would let positions attend beyond
/// the trained span and the semantics would silently diverge, and silent
/// divergence is forbidden.
fn effective_context_ceiling(family: &FamilySpec, cfg: &ModelConfig) -> u64 {
    match cfg.sliding_window {
        Some(window) if family.sliding_window_clamp => cfg.max_position_embeddings.min(window),
        _ => cfg.max_position_embeddings,
    }
}

/// The compile-time context length. An explicit request is validated against
/// the model's effective ceiling — trained `max_position_embeddings`, clamped
/// by any family sliding window ([`effective_context_ceiling`]); no request
/// means the ceiling itself.
fn resolve_context_length(
    family: &FamilySpec,
    cfg: &ModelConfig,
    requested: Option<u64>,
) -> Result<u64> {
    let ceiling = effective_context_ceiling(family, cfg);
    let Some(n) = requested else {
        return Ok(ceiling);
    };
    ensure!(n >= 1, "requested context_length {n} must be at least 1");
    ensure!(
        n <= ceiling,
        "requested context_length {n} exceeds the model's effective context ceiling {ceiling} \
         (max_position_embeddings {}, sliding_window {:?})",
        cfg.max_position_embeddings,
        cfg.sliding_window
    );
    Ok(n)
}

/// Per-builder dimension expressions — all functions of the config. `batch`
/// and `seq` are registered on the builder's own dim-var table, so every
/// graph (monolithic or stage slice) concretizes identically at compile.
struct DecoderDims {
    batch: DimExpr,
    seq: DimExpr,
    vocab: DimExpr,
    hidden: DimExpr,
    ffn_hidden: DimExpr,
    n_heads: DimExpr,
    n_kv_heads: DimExpr,
    head_dim: DimExpr,
    q_out: DimExpr,
    kv_out: DimExpr,
}

impl DecoderDims {
    fn register(builder: &mut GraphBuilder, cfg: &ModelConfig) -> Self {
        Self {
            batch: builder.register_var("batch"),
            seq: builder.register_var("seq"),
            vocab: DimExpr::Concrete(cfg.vocab_size),
            hidden: DimExpr::Concrete(cfg.hidden_size),
            ffn_hidden: DimExpr::Concrete(cfg.intermediate_size),
            n_heads: DimExpr::Concrete(cfg.num_attention_heads),
            n_kv_heads: DimExpr::Concrete(cfg.num_key_value_heads),
            head_dim: DimExpr::Concrete(cfg.head_dim),
            q_out: DimExpr::Concrete(cfg.num_attention_heads * cfg.head_dim),
            kv_out: DimExpr::Concrete(cfg.num_key_value_heads * cfg.head_dim),
        }
    }

    /// The `[batch, seq, hidden]` activation shape flowing between layers —
    /// and between stage archives in the staged partition.
    fn hidden_state(&self) -> Vec<DimExpr> {
        vec![self.batch.clone(), self.seq.clone(), self.hidden.clone()]
    }
}

/// The two dangling operands of a decoder layer whose closing residual add
/// has not been emitted yet: `attn_residual` is the post-attention residual
/// stream (`res1_l`) and `mlp_down` is the MLP down-projection output. The
/// layer is completed by [`DecoderRecipe::seal_layer`], which emits
/// `res2_l = attn_residual + mlp_down`.
struct LayerTail {
    attn_residual: TensorId,
    mlp_down: TensorId,
}

/// The validated decoder recipe: family + config + manifest + resolved
/// context. **The single owner of layer emission** — the monolithic builder
/// ([`build_parametric_graph_from_manifest`]) and the staged builder
/// ([`build_parametric_stage_graphs`]) both assemble their graphs from these
/// emitters, so a stage slice contains exactly the nodes the monolithic
/// graph contains for the same layers.
struct DecoderRecipe<'a> {
    family: &'static FamilySpec,
    cfg: ModelConfig,
    manifest: TensorManifest<'a>,
    context_length: u64,
}

impl<'a> DecoderRecipe<'a> {
    /// Shared validation front door: family selection, knob rejection,
    /// config extraction, manifest/layer cross-checks, context resolution.
    fn prepare(
        config: &Value,
        keys: &'a [String],
        dtypes: &[DType],
        context_length: Option<u64>,
    ) -> Result<Self> {
        let family = select_family(config)?;
        reject_unsupported_knobs(family, config)?;
        let cfg = ModelConfig::from_json(config)?;
        let manifest = TensorManifest::new(keys, dtypes)?;
        validate_layer_count(&cfg, keys)?;
        let context_length = resolve_context_length(family, &cfg, context_length)?;
        Ok(Self {
            family,
            cfg,
            manifest,
            context_length,
        })
    }

    /// The embedding front: the `input_ids` graph input, the embedding weight
    /// declared at its **native** manifest dtype, and the token gather producing
    /// `hidden_states`. Returns `(embedding_native_weight, hidden_states)` — the
    /// monolithic tied head reuses the native weight (widening it once, for the
    /// head matmul); a stage graph outputs the hidden states.
    ///
    /// **Gather first, then cast.** Widening the whole `[vocab, hidden]` table
    /// to F32 before selecting rows would materialize a `vocab · hidden · 4`
    /// byte tensor (past the 32-bit wasm heap for a large-vocabulary model — the
    /// confirmed allocation trap). Row selection is dtype-agnostic (the Gather
    /// desugars to `Slice`/`Concat`), so the table stays at its native storage
    /// dtype, the gather yields native-dtype `[batch, seq, hidden]` rows, and a
    /// single `Cast` widens only that bounded result before the first RmsNorm.
    /// This is mathematically identical to cast-then-gather for a row-selection
    /// gather, so numeric parity is unchanged.
    fn emit_embedding(
        &self,
        builder: &mut GraphBuilder,
        dims: &DecoderDims,
    ) -> (TensorId, TensorId) {
        let input_ids = builder.add_input(
            "input_ids",
            DType::INT64,
            vec![dims.batch.clone(), dims.seq.clone()],
        );
        let embed_dtype = self.manifest.dtype_of("model.embed_tokens.weight");
        let embed_weight = add_weight_native(
            builder,
            &self.manifest,
            "model.embed_tokens.weight",
            vec![dims.vocab.clone(), dims.hidden.clone()],
        );

        if embed_dtype == DType::F32 {
            // Already the compute type: gather straight into the F32 hidden
            // states — identical to the previous graph for an F32 checkpoint.
            let hidden = builder.add_tensor("hidden_states", DType::F32, dims.hidden_state());
            builder.add_node(
                AiOp::Gather { axis: 0 },
                vec![embed_weight, input_ids],
                vec![hidden],
            );
            return (embed_weight, hidden);
        }

        // Narrow storage: gather the native-dtype rows, then widen ONLY the
        // `[batch, seq, hidden]` result — never the `[vocab, hidden]` table.
        let gathered = builder.add_tensor("embedded_tokens", embed_dtype, dims.hidden_state());
        builder.add_node(
            AiOp::Gather { axis: 0 },
            vec![embed_weight, input_ids],
            vec![gathered],
        );
        let hidden = builder.add_tensor("hidden_states", DType::F32, dims.hidden_state());
        builder.add_node(AiOp::Cast { to: DType::F32 }, vec![gathered], vec![hidden]);
        (embed_weight, hidden)
    }

    /// One decoder layer up to (but excluding) its closing residual add:
    /// attention norm → Q/K/V (per the family registry) → RoPE-fused causal
    /// GQA → O projection → residual 1 → MLP norm → gate/up (per the family
    /// registry) → SwiGLU → down projection. Returns the [`LayerTail`] whose
    /// residual add [`Self::seal_layer`] emits — split out so the staged
    /// builder can defer the *final* layer's add into the head stage, where
    /// the compiler fuses it into the final norm exactly as the monolithic
    /// compile does.
    fn emit_layer_core(
        &self,
        builder: &mut GraphBuilder,
        dims: &DecoderDims,
        l: u64,
        current: TensorId,
    ) -> Result<LayerTail> {
        let family = self.family;
        let cfg = &self.cfg;
        let manifest = &self.manifest;
        let DecoderDims {
            batch,
            seq,
            hidden,
            ffn_hidden,
            n_heads: n_heads_expr,
            n_kv_heads: n_kv_heads_expr,
            head_dim: head_dim_expr,
            q_out: q_out_dim,
            kv_out: kv_out_dim,
            ..
        } = dims;

        // Bias expectations: family structure (Qwen2 Q/K/V) or Llama-style flags.
        let qkv_bias_expected = family.attention_qkv_bias || cfg.attention_bias;

        // Attention Norm — ε from the model's own `rms_norm_eps`.
        let attn_norm_weight = add_weight_f32(
            builder,
            manifest,
            &format!("model.layers.{l}.input_layernorm.weight"),
            vec![hidden.clone()],
        );
        let attn_norm_out = builder.add_tensor(
            &format!("attn_norm_{l}"),
            DType::F32,
            vec![batch.clone(), seq.clone(), hidden.clone()],
        );
        builder.add_node(
            AiOp::RmsNorm {
                epsilon: cfg.rms_norm_eps,
            },
            vec![current, attn_norm_weight],
            vec![attn_norm_out],
        );

        // Q/K/V projections — per the family registry: a fused `qkv_proj`
        // checkpoint tensor sliced at compile time (Phi3), or three separate
        // manifest projections with biases per the registry/`attention_bias`.
        let (q_flat, k_flat, v_flat) = if family.attention_fused_qkv {
            ensure!(
                !qkv_bias_expected,
                "the `{}` family ships Q/K/V fused without biases — a Q/K/V bias \
                 expectation cannot be realized against a fused `qkv_proj` checkpoint",
                family.name
            );
            let q_rows = cfg.num_attention_heads * cfg.head_dim;
            let kv_rows = cfg.num_key_value_heads * cfg.head_dim;
            // Fused row layout: [q (heads·head_dim); k (kv_dim); v (kv_dim)].
            let fused = add_fused_weight_f32(
                builder,
                manifest,
                family,
                &format!("model.layers.{l}.self_attn.qkv_proj.weight"),
                vec![DimExpr::Concrete(q_rows + 2 * kv_rows), hidden.clone()],
            )?;
            let q_weight = add_row_slice(
                builder,
                fused,
                &format!("q_weight_{l}"),
                0,
                q_rows,
                hidden.clone(),
            );
            let k_weight = add_row_slice(
                builder,
                fused,
                &format!("k_weight_{l}"),
                q_rows,
                kv_rows,
                hidden.clone(),
            );
            let v_weight = add_row_slice(
                builder,
                fused,
                &format!("v_weight_{l}"),
                q_rows + kv_rows,
                kv_rows,
                hidden.clone(),
            );
            let q_flat = add_linear_layer_from_tensor(
                builder,
                attn_norm_out,
                q_weight,
                LinearLayerParams {
                    weight_name: &format!("q_weight_{l}"),
                    in_features: hidden.clone(),
                    out_features: q_out_dim.clone(),
                    output_name: &format!("q_flat_{l}"),
                    output_shape: vec![batch.clone(), seq.clone(), q_out_dim.clone()],
                },
            );
            let k_flat = add_linear_layer_from_tensor(
                builder,
                attn_norm_out,
                k_weight,
                LinearLayerParams {
                    weight_name: &format!("k_weight_{l}"),
                    in_features: hidden.clone(),
                    out_features: kv_out_dim.clone(),
                    output_name: &format!("k_flat_{l}"),
                    output_shape: vec![batch.clone(), seq.clone(), kv_out_dim.clone()],
                },
            );
            let v_flat = add_linear_layer_from_tensor(
                builder,
                attn_norm_out,
                v_weight,
                LinearLayerParams {
                    weight_name: &format!("v_weight_{l}"),
                    in_features: hidden.clone(),
                    out_features: kv_out_dim.clone(),
                    output_name: &format!("v_flat_{l}"),
                    output_shape: vec![batch.clone(), seq.clone(), kv_out_dim.clone()],
                },
            );
            (q_flat, k_flat, v_flat)
        } else {
            let q_flat = add_projection(
                builder,
                manifest,
                attn_norm_out,
                ProjectionParams {
                    weight_name: &format!("model.layers.{l}.self_attn.q_proj.weight"),
                    bias_name: &format!("model.layers.{l}.self_attn.q_proj.bias"),
                    bias_expected: qkv_bias_expected,
                    in_features: hidden.clone(),
                    out_features: q_out_dim.clone(),
                    output_name: &format!("q_flat_{l}"),
                    output_shape: vec![batch.clone(), seq.clone(), q_out_dim.clone()],
                },
            )?;
            let k_flat = add_projection(
                builder,
                manifest,
                attn_norm_out,
                ProjectionParams {
                    weight_name: &format!("model.layers.{l}.self_attn.k_proj.weight"),
                    bias_name: &format!("model.layers.{l}.self_attn.k_proj.bias"),
                    bias_expected: qkv_bias_expected,
                    in_features: hidden.clone(),
                    out_features: kv_out_dim.clone(),
                    output_name: &format!("k_flat_{l}"),
                    output_shape: vec![batch.clone(), seq.clone(), kv_out_dim.clone()],
                },
            )?;
            let v_flat = add_projection(
                builder,
                manifest,
                attn_norm_out,
                ProjectionParams {
                    weight_name: &format!("model.layers.{l}.self_attn.v_proj.weight"),
                    bias_name: &format!("model.layers.{l}.self_attn.v_proj.bias"),
                    bias_expected: qkv_bias_expected,
                    in_features: hidden.clone(),
                    out_features: kv_out_dim.clone(),
                    output_name: &format!("v_flat_{l}"),
                    output_shape: vec![batch.clone(), seq.clone(), kv_out_dim.clone()],
                },
            )?;
            (q_flat, k_flat, v_flat)
        };

        let q_out = builder.add_tensor(
            &format!("q_{l}"),
            DType::F32,
            vec![
                batch.clone(),
                seq.clone(),
                n_heads_expr.clone(),
                head_dim_expr.clone(),
            ],
        );
        let k_out = builder.add_tensor(
            &format!("k_{l}"),
            DType::F32,
            vec![
                batch.clone(),
                seq.clone(),
                n_kv_heads_expr.clone(),
                head_dim_expr.clone(),
            ],
        );
        let v_out = builder.add_tensor(
            &format!("v_{l}"),
            DType::F32,
            vec![
                batch.clone(),
                seq.clone(),
                n_kv_heads_expr.clone(),
                head_dim_expr.clone(),
            ],
        );

        // Reshape flat QKV to 4D for GQA
        builder.add_node(
            AiOp::Reshape { allow_zero: false },
            vec![q_flat],
            vec![q_out],
        );
        builder.add_node(
            AiOp::Reshape { allow_zero: false },
            vec![k_flat],
            vec![k_out],
        );
        builder.add_node(
            AiOp::Reshape { allow_zero: false },
            vec![v_flat],
            vec![v_out],
        );

        // GQA — RoPE base from the model's own `rope_theta`.
        let attn_out = builder.add_tensor(
            &format!("attn_out_{l}"),
            DType::F32,
            vec![
                batch.clone(),
                seq.clone(),
                n_heads_expr.clone(),
                head_dim_expr.clone(),
            ],
        );
        builder.add_node(
            AiOp::GroupedQueryAttention {
                num_heads: cfg.num_attention_heads as u32,
                num_kv_heads: cfg.num_key_value_heads as u32,
                head_dim: cfg.head_dim as u32,
                scale: None,
                causal: true,
                heads_first: false,
                qk_norm: false,
                rope: true,
                rope_base: cfg.rope_theta,
            },
            vec![q_out, k_out, v_out],
            vec![attn_out],
        );

        let attn_out_flat = builder.add_tensor(
            &format!("attn_out_flat_{l}"),
            DType::F32,
            vec![batch.clone(), seq.clone(), hidden.clone()],
        );
        builder.add_node(
            AiOp::Reshape { allow_zero: false },
            vec![attn_out],
            vec![attn_out_flat],
        );

        // O Projection — bias only per Llama-style `attention_bias`.
        let o_out = add_projection(
            builder,
            manifest,
            attn_out_flat,
            ProjectionParams {
                weight_name: &format!("model.layers.{l}.self_attn.o_proj.weight"),
                bias_name: &format!("model.layers.{l}.self_attn.o_proj.bias"),
                bias_expected: cfg.attention_bias,
                in_features: q_out_dim.clone(),
                out_features: hidden.clone(),
                output_name: &format!("o_out_{l}"),
                output_shape: vec![batch.clone(), seq.clone(), hidden.clone()],
            },
        )?;

        // Add (residual 1)
        let res1_out = builder.add_tensor(
            &format!("res1_{l}"),
            DType::F32,
            vec![batch.clone(), seq.clone(), hidden.clone()],
        );
        builder.add_node(AiOp::Add, vec![current, o_out], vec![res1_out]);

        // MLP Norm — ε from the model's own `rms_norm_eps`.
        let mlp_norm_weight = add_weight_f32(
            builder,
            manifest,
            &format!("model.layers.{l}.post_attention_layernorm.weight"),
            vec![hidden.clone()],
        );
        let mlp_norm_out = builder.add_tensor(
            &format!("mlp_norm_{l}"),
            DType::F32,
            vec![batch.clone(), seq.clone(), hidden.clone()],
        );
        builder.add_node(
            AiOp::RmsNorm {
                epsilon: cfg.rms_norm_eps,
            },
            vec![res1_out, mlp_norm_weight],
            vec![mlp_norm_out],
        );

        // MLP Gate + Up — per the family registry: a fused `gate_up_proj`
        // checkpoint tensor sliced at compile time (Phi3), or two separate
        // manifest projections with biases per Llama-style `mlp_bias`.
        let (gate_out, up_out) = if family.mlp_fused_gate_up {
            ensure!(
                !cfg.mlp_bias,
                "the `{}` family ships gate/up fused without biases — `mlp_bias` \
                 cannot be realized against a fused `gate_up_proj` checkpoint",
                family.name
            );
            // Fused row layout: [gate (intermediate); up (intermediate)].
            let fused = add_fused_weight_f32(
                builder,
                manifest,
                family,
                &format!("model.layers.{l}.mlp.gate_up_proj.weight"),
                vec![DimExpr::Concrete(2 * cfg.intermediate_size), hidden.clone()],
            )?;
            let gate_weight = add_row_slice(
                builder,
                fused,
                &format!("gate_weight_{l}"),
                0,
                cfg.intermediate_size,
                hidden.clone(),
            );
            let up_weight = add_row_slice(
                builder,
                fused,
                &format!("up_weight_{l}"),
                cfg.intermediate_size,
                cfg.intermediate_size,
                hidden.clone(),
            );
            let gate_out = add_linear_layer_from_tensor(
                builder,
                mlp_norm_out,
                gate_weight,
                LinearLayerParams {
                    weight_name: &format!("gate_weight_{l}"),
                    in_features: hidden.clone(),
                    out_features: ffn_hidden.clone(),
                    output_name: &format!("gate_out_{l}"),
                    output_shape: vec![batch.clone(), seq.clone(), ffn_hidden.clone()],
                },
            );
            let up_out = add_linear_layer_from_tensor(
                builder,
                mlp_norm_out,
                up_weight,
                LinearLayerParams {
                    weight_name: &format!("up_weight_{l}"),
                    in_features: hidden.clone(),
                    out_features: ffn_hidden.clone(),
                    output_name: &format!("up_out_{l}"),
                    output_shape: vec![batch.clone(), seq.clone(), ffn_hidden.clone()],
                },
            );
            (gate_out, up_out)
        } else {
            let gate_out = add_projection(
                builder,
                manifest,
                mlp_norm_out,
                ProjectionParams {
                    weight_name: &format!("model.layers.{l}.mlp.gate_proj.weight"),
                    bias_name: &format!("model.layers.{l}.mlp.gate_proj.bias"),
                    bias_expected: cfg.mlp_bias,
                    in_features: hidden.clone(),
                    out_features: ffn_hidden.clone(),
                    output_name: &format!("gate_out_{l}"),
                    output_shape: vec![batch.clone(), seq.clone(), ffn_hidden.clone()],
                },
            )?;
            let up_out = add_projection(
                builder,
                manifest,
                mlp_norm_out,
                ProjectionParams {
                    weight_name: &format!("model.layers.{l}.mlp.up_proj.weight"),
                    bias_name: &format!("model.layers.{l}.mlp.up_proj.bias"),
                    bias_expected: cfg.mlp_bias,
                    in_features: hidden.clone(),
                    out_features: ffn_hidden.clone(),
                    output_name: &format!("up_out_{l}"),
                    output_shape: vec![batch.clone(), seq.clone(), ffn_hidden.clone()],
                },
            )?;
            (gate_out, up_out)
        };

        let silu_out = builder.add_tensor(
            &format!("silu_out_{l}"),
            DType::F32,
            vec![batch.clone(), seq.clone(), ffn_hidden.clone()],
        );
        builder.add_node(AiOp::Silu, vec![gate_out], vec![silu_out]);

        let mul_out = builder.add_tensor(
            &format!("mul_out_{l}"),
            DType::F32,
            vec![batch.clone(), seq.clone(), ffn_hidden.clone()],
        );
        builder.add_node(AiOp::Mul, vec![silu_out, up_out], vec![mul_out]);

        // MLP Down — bias per Llama-style `mlp_bias`.
        let down_out = add_projection(
            builder,
            manifest,
            mul_out,
            ProjectionParams {
                weight_name: &format!("model.layers.{l}.mlp.down_proj.weight"),
                bias_name: &format!("model.layers.{l}.mlp.down_proj.bias"),
                bias_expected: cfg.mlp_bias,
                in_features: ffn_hidden.clone(),
                out_features: hidden.clone(),
                output_name: &format!("down_out_{l}"),
                output_shape: vec![batch.clone(), seq.clone(), hidden.clone()],
            },
        )?;

        Ok(LayerTail {
            attn_residual: res1_out,
            mlp_down: down_out,
        })
    }

    /// The closing residual add of layer `l`:
    /// `res2_l = attn_residual + mlp_down`. Kept separate from
    /// [`Self::emit_layer_core`] so the staged builder can emit the model's
    /// final add inside the head stage (see [`build_parametric_stage_graphs`]).
    fn seal_layer(
        &self,
        builder: &mut GraphBuilder,
        dims: &DecoderDims,
        l: u64,
        tail: LayerTail,
    ) -> TensorId {
        let res2_out = builder.add_tensor(&format!("res2_{l}"), DType::F32, dims.hidden_state());
        builder.add_node(
            AiOp::Add,
            vec![tail.attn_residual, tail.mlp_down],
            vec![res2_out],
        );
        res2_out
    }

    /// Final norm + LM head over `current`, returning the `logits` tensor.
    ///
    /// The head is tied when the config says so or when the manifest carries
    /// no separate `lm_head.weight` (the transformers convention for tied
    /// models). A tied monolithic graph reuses the embedding weight
    /// (`embed_native = Some(..)` — the same κ-bound tensor id, widened once
    /// here for the matmul); a tied *head stage* declares
    /// `model.embed_tokens.weight` itself (`embed_native = None`), so its κ-map
    /// binds the same κ the embedding stage binds — one κ-store blob shared by
    /// two stage archives, which is exactly the k-form sharing the staged
    /// partition witnesses.
    ///
    /// **The whole-vocabulary head is a genuine F32 floor.** Unlike the
    /// embedding — which selects a few rows and so stays at native dtype — the
    /// head is a dense matmul over the entire vocabulary: the weight is
    /// consulted in full. The hologram matmul kernel takes a single operand
    /// dtype and widens a narrow (BF16/F16) operand to a whole-matrix F32 panel
    /// internally, so no storage dtype avoids the F32 image; the dequant-fused
    /// (tiled-panel) kernel accepts only i8/u8/i4, not BF16. A large-vocabulary
    /// head's F32 image therefore exceeds a 32-bit heap. Rather than trap
    /// opaquely, [`guard_f32_head_materialization`] fails loud naming the tensor
    /// and its byte size.
    fn emit_head(
        &self,
        builder: &mut GraphBuilder,
        dims: &DecoderDims,
        current: TensorId,
        embed_native: Option<TensorId>,
    ) -> Result<TensorId> {
        let manifest = &self.manifest;

        // Final Norm — ε from the model's own `rms_norm_eps`.
        let norm_weight = add_weight_f32(
            builder,
            manifest,
            "model.norm.weight",
            vec![dims.hidden.clone()],
        );
        let norm_out = builder.add_tensor("norm_out", DType::F32, dims.hidden_state());
        builder.add_node(
            AiOp::RmsNorm {
                epsilon: self.cfg.rms_norm_eps,
            },
            vec![current, norm_weight],
            vec![norm_out],
        );

        // The tied head is transposed by the shared linear-layer wiring
        // ([vocab, hidden] → [hidden, vocab]) for the matmul orientation.
        let tied = self.cfg.tie_word_embeddings || !manifest.contains("lm_head.weight");
        let head_shape = vec![dims.vocab.clone(), dims.hidden.clone()];
        let (head_weight, head_weight_name) = if tied {
            let head_bytes_source = "model.embed_tokens.weight";
            guard_f32_head_materialization(
                head_bytes_source,
                self.cfg.vocab_size,
                self.cfg.hidden_size,
            )?;
            let embed = match embed_native {
                // Monolithic: widen the native embedding weight to F32 for the
                // matmul (the head's whole-vocabulary F32 image).
                Some(id) => widen_weight_to_f32(
                    builder,
                    id,
                    manifest.dtype_of(head_bytes_source),
                    head_bytes_source,
                    head_shape,
                ),
                // Head stage: declare + widen the embedding weight itself, so
                // its κ-map binds the same κ the embedding stage binds.
                None => add_weight_f32(builder, manifest, head_bytes_source, head_shape),
            };
            (embed, "lm_head.tied")
        } else {
            guard_f32_head_materialization(
                "lm_head.weight",
                self.cfg.vocab_size,
                self.cfg.hidden_size,
            )?;
            let weight = add_weight_f32(builder, manifest, "lm_head.weight", head_shape);
            (weight, "lm_head.weight")
        };
        Ok(add_linear_layer_from_tensor(
            builder,
            norm_out,
            head_weight,
            LinearLayerParams {
                weight_name: head_weight_name,
                in_features: dims.hidden.clone(),
                out_features: dims.vocab.clone(),
                output_name: "logits",
                output_shape: vec![dims.batch.clone(), dims.seq.clone(), dims.vocab.clone()],
            },
        ))
    }

    /// Stamp the standard parametric metadata (all functions of the config)
    /// onto a built graph.
    fn apply_metadata(&self, graph: &mut AiGraph) {
        let cfg = &self.cfg;
        let metadata = [
            ("arch", MetaValue::Str("parametric_transformer".to_string())),
            ("family", MetaValue::Str(self.family.name.to_string())),
            ("vocab_size", MetaValue::Int(cfg.vocab_size as i64)),
            ("n_layers", MetaValue::Int(cfg.num_hidden_layers as i64)),
            ("n_embd", MetaValue::Int(cfg.hidden_size as i64)),
            ("n_kv_heads", MetaValue::Int(cfg.num_key_value_heads as i64)),
            ("head_dim", MetaValue::Int(cfg.head_dim as i64)),
            ("context_length", MetaValue::Int(self.context_length as i64)),
        ];
        for (key, value) in metadata {
            graph.metadata.insert(key.to_string(), value);
        }
        // Surface the sliding-window clamp next to the context it clamped: the
        // `context_length` above already carries the clamped ceiling, and this
        // names its cause so no consumer mistakes the clamp for the trained
        // `max_position_embeddings`.
        if let (true, Some(window)) = (self.family.sliding_window_clamp, cfg.sliding_window) {
            graph
                .metadata
                .insert("sliding_window".to_string(), MetaValue::Int(window as i64));
        }
    }
}

/// Build the decoder graph from `config.json` plus the tensor manifest
/// (parallel `keys`/`dtypes` slices).
///
/// `context_length`: `Some(n)` bakes `n` (validated against the model's
/// `max_position_embeddings`); `None` uses the model's own trained context.
pub fn build_parametric_graph_from_manifest(
    config: &Value,
    keys: &[String],
    dtypes: &[DType],
    context_length: Option<u64>,
) -> Result<AiGraph> {
    let recipe = DecoderRecipe::prepare(config, keys, dtypes, context_length)?;
    let mut builder = GraphBuilder::new("parametric_model".to_string());
    let dims = DecoderDims::register(&mut builder, &recipe.cfg);

    let (embed_f32, mut current) = recipe.emit_embedding(&mut builder, &dims);
    for l in 0..recipe.cfg.num_hidden_layers {
        let tail = recipe.emit_layer_core(&mut builder, &dims, l, current)?;
        current = recipe.seal_layer(&mut builder, &dims, l, tail);
    }
    let logits = recipe.emit_head(&mut builder, &dims, current, Some(embed_f32))?;
    builder.add_output(logits, "logits");

    let mut graph = builder.build();
    recipe.apply_metadata(&mut graph);
    Ok(graph)
}

/// `stage_role` metadata value of the embedding stage (`input_ids → hidden_states`).
pub const STAGE_ROLE_EMBEDDING: &str = "embedding";
/// `stage_role` metadata value of a decoder-layer stage.
pub const STAGE_ROLE_LAYERS: &str = "layers";
/// `stage_role` metadata value of the head stage (final residual + norm + lm_head).
pub const STAGE_ROLE_HEAD: &str = "head";

/// Partition the parametric decoder into **stage graphs** for windowed
/// execution over k (dictionary row `staged-execution`):
///
/// * stage `0` — embedding: `input_ids → hidden_states`;
/// * stages `1..=ceil(L / layers_per_stage)` — consecutive decoder layers:
///   `hidden_states → hidden_states`;
/// * final stage — the model's closing residual add + final norm + LM head:
///   `(hidden_states, hidden_residual) → logits`.
///
/// Every stage is an ordinary [`AiGraph`] over the same family registry,
/// emitted by the same layer-emission recipe the monolithic builder uses — a
/// stage contains exactly the monolithic graph's nodes for its layers, with
/// absolute layer indices, the same compile-time context length (positions
/// are absolute, so RoPE angles and the causal mask are identical in every
/// stage), and the same per-tensor manifest declarations for the κ-map.
///
/// **The head-stage boundary carries two activations.** In the monolithic
/// compile, the final layer's residual add is the final norm's only consumer,
/// so the optimizer fuses `Add + RmsNorm` into one `FusedLayerNormResidual`
/// kernel (ε-preconditioning is applied to each operand before the in-kernel
/// sum). Cutting *after* that add would execute a different kernel sequence
/// (`Add`, then a norm preconditioned on the sum) whose f32 rounding differs
/// in the last bits — silently breaking the staged-equals-monolithic logits
/// equality this partition must witness. The cut therefore lands on the fused
/// kernel's own operands: the last layer stage outputs the post-attention
/// residual stream (`hidden_states`) and the MLP down-projection
/// (`hidden_residual`), and the head stage emits the closing add itself,
/// where the optimizer fuses it exactly as the monolithic compile does. Every
/// other layer boundary is a genuine two-consumer residual in the monolithic
/// graph (never fused there), so those stages exchange one `hidden_states`
/// activation and execute the identical `Add`/`RmsNorm` kernels.
///
/// Tied embeddings: the head stage declares `model.embed_tokens.weight`
/// itself, so the embedding tensor's κ appears in **both** stage 0's and the
/// final stage's κ-map — correct k-form sharing (one κ-store blob), which the
/// partition witness documents rather than double-counts.
pub fn build_parametric_stage_graphs(
    config: &Value,
    keys: &[String],
    dtypes: &[DType],
    context_length: Option<u64>,
    layers_per_stage: NonZeroU64,
) -> Result<Vec<AiGraph>> {
    let recipe = DecoderRecipe::prepare(config, keys, dtypes, context_length)?;
    let layers = recipe.cfg.num_hidden_layers;
    let block = layers_per_stage.get();
    let layer_stages = layers.div_ceil(block);
    let stage_count = layer_stages + 2;
    let mut graphs: Vec<AiGraph> = Vec::with_capacity(stage_count as usize);

    let stage_metadata =
        |recipe: &DecoderRecipe<'_>, graph: &mut AiGraph, index: u64, role: &str| {
            recipe.apply_metadata(graph);
            graph
                .metadata
                .insert("stage_index".to_string(), MetaValue::Int(index as i64));
            graph.metadata.insert(
                "stage_count".to_string(),
                MetaValue::Int(stage_count as i64),
            );
            graph
                .metadata
                .insert("stage_role".to_string(), MetaValue::Str(role.to_string()));
        };

    // Stage 0 — embedding.
    let mut builder = GraphBuilder::new("parametric_stage_embedding".to_string());
    let dims = DecoderDims::register(&mut builder, &recipe.cfg);
    let (_embed_f32, hidden) = recipe.emit_embedding(&mut builder, &dims);
    builder.add_output(hidden, "hidden_states");
    let mut graph = builder.build();
    stage_metadata(&recipe, &mut graph, 0, STAGE_ROLE_EMBEDDING);
    graphs.push(graph);

    // Decoder-layer stages.
    for s in 0..layer_stages {
        let start = s * block;
        let end = (start + block).min(layers);
        let mut builder = GraphBuilder::new(format!("parametric_stage_layers_{start}_{end}"));
        let dims = DecoderDims::register(&mut builder, &recipe.cfg);
        let mut current = builder.add_input("hidden_states", DType::F32, dims.hidden_state());
        let mut deferred: Option<LayerTail> = None;
        for l in start..end {
            let tail = recipe.emit_layer_core(&mut builder, &dims, l, current)?;
            if l + 1 == layers {
                // The model's last layer: its closing residual add belongs to
                // the head stage (see the partition contract above).
                deferred = Some(tail);
            } else {
                current = recipe.seal_layer(&mut builder, &dims, l, tail);
            }
        }
        match deferred {
            Some(tail) => {
                builder.add_output(tail.attn_residual, "hidden_states");
                builder.add_output(tail.mlp_down, "hidden_residual");
            }
            None => builder.add_output(current, "hidden_states"),
        }
        let mut graph = builder.build();
        stage_metadata(&recipe, &mut graph, 1 + s, STAGE_ROLE_LAYERS);
        graph.metadata.insert(
            "stage_layer_start".to_string(),
            MetaValue::Int(start as i64),
        );
        graph
            .metadata
            .insert("stage_layer_end".to_string(), MetaValue::Int(end as i64));
        graphs.push(graph);
    }

    // Head stage — the final layer's residual add + final norm + LM head.
    let mut builder = GraphBuilder::new("parametric_stage_head".to_string());
    let dims = DecoderDims::register(&mut builder, &recipe.cfg);
    let attn_residual = builder.add_input("hidden_states", DType::F32, dims.hidden_state());
    let mlp_down = builder.add_input("hidden_residual", DType::F32, dims.hidden_state());
    let current = recipe.seal_layer(
        &mut builder,
        &dims,
        layers - 1,
        LayerTail {
            attn_residual,
            mlp_down,
        },
    );
    let logits = recipe.emit_head(&mut builder, &dims, current, None)?;
    builder.add_output(logits, "logits");
    let mut graph = builder.build();
    stage_metadata(&recipe, &mut graph, stage_count - 1, STAGE_ROLE_HEAD);
    graphs.push(graph);

    Ok(graphs)
}

fn extract_layer_idx(key: &str) -> Option<usize> {
    let parts: Vec<&str> = key.split('.').collect();
    for (i, part) in parts.iter().enumerate() {
        if (*part == "layers" || *part == "h" || *part == "blocks") && i + 1 < parts.len() {
            if let Ok(idx) = parts[i + 1].parse::<usize>() {
                return Some(idx);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tensor keys of a Llama-family checkpoint (`model.*` naming).
    fn decoder_keys(layers: usize, tied: bool, qkv_bias: bool) -> Vec<String> {
        let mut keys = vec![
            "model.embed_tokens.weight".to_string(),
            "model.norm.weight".to_string(),
        ];
        if !tied {
            keys.push("lm_head.weight".to_string());
        }
        for l in 0..layers {
            for suffix in [
                "input_layernorm.weight",
                "post_attention_layernorm.weight",
                "self_attn.q_proj.weight",
                "self_attn.k_proj.weight",
                "self_attn.v_proj.weight",
                "self_attn.o_proj.weight",
                "mlp.gate_proj.weight",
                "mlp.up_proj.weight",
                "mlp.down_proj.weight",
            ] {
                keys.push(format!("model.layers.{l}.{suffix}"));
            }
            if qkv_bias {
                for suffix in [
                    "self_attn.q_proj.bias",
                    "self_attn.k_proj.bias",
                    "self_attn.v_proj.bias",
                ] {
                    keys.push(format!("model.layers.{l}.{suffix}"));
                }
            }
        }
        keys
    }

    fn tiny_llama_config() -> Value {
        serde_json::json!({
            "architectures": ["LlamaForCausalLM"],
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "vocab_size": 512,
            "intermediate_size": 128,
            "rope_theta": 10000.0,
            "rms_norm_eps": 1e-6,
            "max_position_embeddings": 2048,
            "tie_word_embeddings": false,
        })
    }

    fn tiny_qwen2_config() -> Value {
        serde_json::json!({
            "architectures": ["Qwen2ForCausalLM"],
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "vocab_size": 512,
            "intermediate_size": 128,
            "rope_theta": 1000000.0,
            "rms_norm_eps": 1e-6,
            "max_position_embeddings": 1024,
            "tie_word_embeddings": true,
        })
    }

    /// A Mistral-family config: Llama-identical keys plus `sliding_window`
    /// (null in the published v0.3 convention, an integer when the
    /// checkpoint attends with a sliding window).
    fn tiny_mistral_config(sliding_window: Value) -> Value {
        serde_json::json!({
            "architectures": ["MistralForCausalLM"],
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "vocab_size": 512,
            "intermediate_size": 128,
            "rope_theta": 1000000.0,
            "rms_norm_eps": 1e-5,
            "max_position_embeddings": 4096,
            "tie_word_embeddings": false,
            "sliding_window": sliding_window,
        })
    }

    fn tiny_phi3_config() -> Value {
        serde_json::json!({
            "architectures": ["Phi3ForCausalLM"],
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "vocab_size": 512,
            "intermediate_size": 128,
            "rope_theta": 250000.0,
            "rms_norm_eps": 1e-5,
            "max_position_embeddings": 2048,
            "tie_word_embeddings": false,
            "rope_scaling": null,
            "sliding_window": null,
        })
    }

    /// Tensor keys of a Phi3-style checkpoint: fused qkv_proj/gate_up_proj,
    /// untied lm_head, no biases (the microsoft/phi-4 layout).
    fn fused_decoder_keys(layers: usize) -> Vec<String> {
        let mut keys = vec![
            "model.embed_tokens.weight".to_string(),
            "model.norm.weight".to_string(),
            "lm_head.weight".to_string(),
        ];
        for l in 0..layers {
            for suffix in [
                "input_layernorm.weight",
                "post_attention_layernorm.weight",
                "self_attn.qkv_proj.weight",
                "self_attn.o_proj.weight",
                "mlp.gate_up_proj.weight",
                "mlp.down_proj.weight",
            ] {
                keys.push(format!("model.layers.{l}.{suffix}"));
            }
        }
        keys
    }

    fn tensor_id(graph: &AiGraph, name: &str) -> TensorId {
        graph
            .tensor_names
            .iter()
            .find(|(_, n)| n.as_str() == name)
            .map(|(id, _)| *id)
            .unwrap_or_else(|| panic!("tensor `{name}` not found in graph"))
    }

    fn meta_int(graph: &AiGraph, key: &str) -> Option<i64> {
        match graph.metadata.get(key) {
            Some(MetaValue::Int(i)) => Some(*i),
            _ => None,
        }
    }

    #[test]
    fn llama_family_builds_with_config_eps_theta_and_untied_head() {
        let config = tiny_llama_config();
        let keys = decoder_keys(2, false, false);
        let dtypes = vec![DType::F32; keys.len()];
        let graph =
            build_parametric_graph_from_manifest(&config, &keys, &dtypes, None).expect("build");

        // ε comes from the config's `rms_norm_eps`: 2 layers × 2 norms + final.
        let eps: Vec<f32> = graph
            .nodes
            .iter()
            .filter_map(|n| match n.op {
                AiOp::RmsNorm { epsilon } => Some(epsilon),
                _ => None,
            })
            .collect();
        assert_eq!(eps.len(), 5);
        assert!(eps.iter().all(|&e| (e - 1e-6).abs() < 1e-12));

        // θ, heads, KV heads, head dim come from the config.
        let gqa: Vec<(u32, u32, u32, f32)> = graph
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
        assert_eq!(gqa.len(), 2);
        assert!(gqa.iter().all(|&(h, kv, d, base)| h == 4
            && kv == 2
            && d == 16
            && (base - 10000.0).abs() < 1e-3));

        // Untied: a separate `lm_head.weight` is declared.
        assert!(graph.tensor_names.values().any(|n| n == "lm_head.weight"));

        // No context request → the model's own `max_position_embeddings`.
        assert_eq!(meta_int(&graph, "context_length"), Some(2048));
        assert_eq!(meta_int(&graph, "n_layers"), Some(2));
        assert_eq!(meta_int(&graph, "n_kv_heads"), Some(2));
        assert_eq!(meta_int(&graph, "head_dim"), Some(16));
    }

    #[test]
    fn qwen2_family_builds_with_biases_tied_head_and_manifest_dtypes() {
        let config = tiny_qwen2_config();
        let keys = decoder_keys(2, true, true);
        let dtypes = vec![DType::BF16; keys.len()];
        let graph = build_parametric_graph_from_manifest(&config, &keys, &dtypes, Some(256))
            .expect("build");

        // The embedding table is declared at its BF16 storage dtype and gathered
        // NATIVELY — no whole-table F32 widening. Only the gathered
        // [batch, seq, hidden] rows are cast to the F32 compute type.
        let embed_id = tensor_id(&graph, "model.embed_tokens.weight");
        assert_eq!(graph.tensor_info[&embed_id].storage_dtype, DType::BF16);
        let gather = graph
            .nodes
            .iter()
            .find(|n| matches!(n.op, AiOp::Gather { .. }))
            .expect("the embedding gather");
        assert_eq!(
            gather.inputs.first().copied(),
            Some(embed_id),
            "the gather reads the NATIVE embedding table, not an F32 view of it"
        );
        let gathered = gather.outputs[0];
        assert!(
            graph
                .nodes
                .iter()
                .any(|n| matches!(n.op, AiOp::Cast { to: DType::F32 })
                    && n.inputs.contains(&gathered)),
            "only the gathered rows are widened to F32"
        );

        // Every non-embedding weight is widened to F32 via the canonical Cast
        // (one per such tensor); the embedding is widened once more, for the
        // tied head's whole-vocabulary matmul.
        let cast_count = graph
            .nodes
            .iter()
            .filter(|n| matches!(n.op, AiOp::Cast { to: DType::F32 }))
            .count();
        assert_eq!(cast_count, keys.len() + 1);

        // Q/K/V biases are consumed as explicit broadcast Add operands.
        let q_bias_f32 = tensor_id(&graph, "model.layers.0.self_attn.q_proj.bias.f32");
        assert!(graph
            .nodes
            .iter()
            .any(|n| matches!(n.op, AiOp::Add) && n.inputs.contains(&q_bias_f32)));

        // Tied head: no separate `lm_head.weight`; the head widens the
        // embedding weight to F32 (its whole-vocabulary matmul operand) and
        // transposes THAT view — the token Gather still reads the native table,
        // so the F32 image is never fed back into the gather.
        assert!(!graph.tensor_names.values().any(|n| n == "lm_head.weight"));
        let embed_f32 = tensor_id(&graph, "model.embed_tokens.weight.f32");
        let consumers: Vec<&AiOp> = graph
            .nodes
            .iter()
            .filter(|n| n.inputs.contains(&embed_f32))
            .map(|n| &n.op)
            .collect();
        assert!(
            consumers
                .iter()
                .any(|op| matches!(op, AiOp::Transpose { .. })),
            "the head transposes the embedding's F32 view"
        );
        assert!(
            !consumers.iter().any(|op| matches!(op, AiOp::Gather { .. })),
            "the F32 view must NOT feed the token gather"
        );

        // θ comes from the config's `rope_theta` (Qwen2.5 convention: 1e6).
        assert!(graph.nodes.iter().any(|n| matches!(
            n.op,
            AiOp::GroupedQueryAttention { rope_base, .. } if (rope_base - 1_000_000.0).abs() < 1.0
        )));

        // An explicit context request flows into the metadata.
        assert_eq!(meta_int(&graph, "context_length"), Some(256));
    }

    #[test]
    fn kv_heads_and_head_dim_default_by_transformers_convention() {
        let mut config = tiny_llama_config();
        let obj = config.as_object_mut().expect("config is an object");
        obj.remove("num_key_value_heads");
        let keys = decoder_keys(2, false, false);
        let dtypes = vec![DType::F32; keys.len()];
        let graph =
            build_parametric_graph_from_manifest(&config, &keys, &dtypes, None).expect("build");

        // Absent `num_key_value_heads` → one KV head per query head (MHA);
        // absent `head_dim` → hidden_size / num_attention_heads.
        assert!(graph.nodes.iter().any(|n| matches!(
            n.op,
            AiOp::GroupedQueryAttention {
                num_heads: 4,
                num_kv_heads: 4,
                head_dim: 16,
                ..
            }
        )));
    }

    #[test]
    fn missing_required_key_fails_naming_the_key() {
        let mut config = tiny_llama_config();
        config
            .as_object_mut()
            .expect("config is an object")
            .remove("hidden_size");
        let keys = decoder_keys(2, false, false);
        let dtypes = vec![DType::F32; keys.len()];
        let err = build_parametric_graph_from_manifest(&config, &keys, &dtypes, None)
            .err()
            .expect("must fail loud");
        assert!(
            err.to_string().contains("hidden_size"),
            "error should name the missing key: {err}"
        );
    }

    #[test]
    fn unknown_family_fails_naming_family_and_supported_set() {
        let mut config = tiny_llama_config();
        config["architectures"] = serde_json::json!(["MambaForCausalLM"]);
        let keys = decoder_keys(2, false, false);
        let dtypes = vec![DType::F32; keys.len()];
        let err = build_parametric_graph_from_manifest(&config, &keys, &dtypes, None)
            .err()
            .expect("must fail loud");
        let msg = err.to_string();
        assert!(msg.contains("MambaForCausalLM"), "names the family: {msg}");
        assert!(
            msg.contains("LlamaForCausalLM") && msg.contains("Qwen2ForCausalLM"),
            "names the supported set: {msg}"
        );
    }

    #[test]
    fn supported_families_lists_all_registered() {
        assert_eq!(
            supported_families(),
            vec![
                "LlamaForCausalLM",
                "Qwen2ForCausalLM",
                "MistralForCausalLM",
                "Phi3ForCausalLM",
            ]
        );
    }

    #[test]
    fn mistral_family_builds_llama_identical_with_null_sliding_window() {
        // The published Mistral-7B-Instruct-v0.3 convention: `sliding_window`
        // is explicit null → full attention at the trained context.
        let config = tiny_mistral_config(Value::Null);
        let keys = decoder_keys(2, false, false);
        let dtypes = vec![DType::F32; keys.len()];
        let graph =
            build_parametric_graph_from_manifest(&config, &keys, &dtypes, None).expect("build");
        assert_eq!(meta_int(&graph, "context_length"), Some(4096));
        assert!(meta_int(&graph, "sliding_window").is_none());
        assert!(graph.tensor_names.values().any(|n| n == "lm_head.weight"));
        match graph.metadata.get("family") {
            Some(MetaValue::Str(s)) => assert_eq!(s, "MistralForCausalLM"),
            other => panic!("family metadata missing or mistyped: {other:?}"),
        }
    }

    #[test]
    fn mistral_sliding_window_clamps_the_effective_context() {
        // Full-causal attention equals sliding-window attention only up to
        // the window length — the effective context clamps there.
        let config = tiny_mistral_config(serde_json::json!(1024));
        let keys = decoder_keys(2, false, false);
        let dtypes = vec![DType::F32; keys.len()];
        let graph =
            build_parametric_graph_from_manifest(&config, &keys, &dtypes, None).expect("build");
        assert_eq!(meta_int(&graph, "context_length"), Some(1024));
        assert_eq!(meta_int(&graph, "sliding_window"), Some(1024));

        // A request within the window still resolves.
        let graph = build_parametric_graph_from_manifest(&config, &keys, &dtypes, Some(256))
            .expect("build within the window");
        assert_eq!(meta_int(&graph, "context_length"), Some(256));

        // A request beyond the window fails naming the clamp.
        let err = build_parametric_graph_from_manifest(&config, &keys, &dtypes, Some(2048))
            .err()
            .expect("a context beyond the sliding window must fail loud");
        let msg = err.to_string();
        assert!(
            msg.contains("sliding_window") && msg.contains("1024"),
            "error should name the sliding-window ceiling: {msg}"
        );
    }

    #[test]
    fn embedding_gathers_native_then_widens_only_the_result() {
        // A BF16 embedding table must be gathered at its native dtype (row
        // selection is dtype-agnostic) and only the [batch, seq, hidden] result
        // widened to F32 — the whole [vocab, hidden] table is never an F32
        // tensor (that image is the 32-bit-heap allocation trap).
        let config = tiny_llama_config();
        let keys = decoder_keys(2, false, false);
        let dtypes = vec![DType::BF16; keys.len()];
        let graph =
            build_parametric_graph_from_manifest(&config, &keys, &dtypes, None).expect("build");

        let embed_id = tensor_id(&graph, "model.embed_tokens.weight");
        assert_eq!(graph.tensor_info[&embed_id].storage_dtype, DType::BF16);

        // The embedding table itself is never widened wholesale: no F32 image
        // of `model.embed_tokens.weight` exists (an untied model gathers it and
        // nothing else). The head's own `lm_head.weight.f32` is a separate,
        // acknowledged whole-vocabulary floor.
        assert!(
            !graph
                .tensor_names
                .values()
                .any(|n| n == "model.embed_tokens.weight.f32"),
            "the embedding table must never be widened to a whole [vocab, hidden] F32 tensor"
        );

        // The gather reads the native table; its output is widened by a Cast.
        let gather = graph
            .nodes
            .iter()
            .find(|n| matches!(n.op, AiOp::Gather { .. }))
            .expect("the embedding gather");
        assert_eq!(gather.inputs.first().copied(), Some(embed_id));
        let gathered = gather.outputs[0];
        assert_eq!(graph.tensor_info[&gathered].storage_dtype, DType::BF16);
        assert!(
            graph
                .nodes
                .iter()
                .any(|n| matches!(n.op, AiOp::Cast { to: DType::F32 })
                    && n.inputs.contains(&gathered))
        );
    }

    #[test]
    fn head_f32_floor_fails_loud_naming_the_tensor_and_bytes() {
        // A whole-vocabulary F32 head past the heap ceiling fails loud (naming
        // the tensor and its byte size) rather than trapping; a head that fits
        // returns no error. Uses an explicit ceiling so the floor logic is
        // witnessed on a 64-bit host, where the live ceiling is unbounded.
        let (rows, cols) = (100_000u64, 5_000u64);
        let bytes = f32_weight_bytes(rows, cols);
        assert_eq!(bytes, rows * cols * 4);

        let err = f32_head_floor_error("lm_head.weight", rows, cols, 1_000_000_000)
            .expect("a 2 GB F32 head must exceed a 1 GB ceiling");
        assert!(err.contains("lm_head.weight"), "names the tensor: {err}");
        assert!(
            err.contains(&bytes.to_string()),
            "names the byte size: {err}"
        );

        assert!(
            f32_head_floor_error("lm_head.weight", rows, cols, u64::MAX).is_none(),
            "an unbounded ceiling (64-bit host) admits the head"
        );
        // The live guard is a no-op on a 64-bit host.
        assert!(guard_f32_head_materialization("lm_head.weight", rows, cols).is_ok());
    }

    #[test]
    fn phi3_family_slices_fused_qkv_and_gate_up_at_compile_time() {
        let config = tiny_phi3_config();
        let keys = fused_decoder_keys(2);
        let dtypes = vec![DType::BF16; keys.len()];
        let graph =
            build_parametric_graph_from_manifest(&config, &keys, &dtypes, None).expect("build");

        // The manifest (κ-bound) names are the FUSED tensors; no per-part
        // projection weights are declared.
        assert!(graph
            .tensor_names
            .values()
            .any(|n| n == "model.layers.0.self_attn.qkv_proj.weight"));
        assert!(!graph
            .tensor_names
            .values()
            .any(|n| n.contains("q_proj") || n.contains("gate_proj")));

        // 3 QKV slices + 2 gate/up slices per layer, all compile-time Slice.
        let slices: Vec<(&Vec<i64>, &Vec<i64>)> = graph
            .nodes
            .iter()
            .filter_map(|n| match &n.op {
                AiOp::Slice { starts, ends, .. } => Some((starts, ends)),
                _ => None,
            })
            .collect();
        assert_eq!(slices.len(), 2 * 5, "five row slices per layer");

        // The fused BF16 tensor is widened once; its F32 view feeds exactly
        // the three row-block slices [q; k; v] = [0,64) [64,96) [96,128)
        // (hidden 64, heads 4, kv 2 → head_dim 16, q 64 rows, kv 32 rows).
        let qkv_f32 = tensor_id(&graph, "model.layers.0.self_attn.qkv_proj.weight.f32");
        let qkv_slices: Vec<(i64, i64)> = graph
            .nodes
            .iter()
            .filter(|n| n.inputs.contains(&qkv_f32))
            .map(|n| match &n.op {
                AiOp::Slice { starts, ends, .. } => (starts[0], ends[0]),
                other => panic!("the fused qkv view must feed Slice nodes only, got {other:?}"),
            })
            .collect();
        assert_eq!(qkv_slices, vec![(0, 64), (64, 96), (96, 128)]);

        // Same for the fused gate/up: [gate; up] = [0,128) [128,256).
        let gu_f32 = tensor_id(&graph, "model.layers.0.mlp.gate_up_proj.weight.f32");
        let gu_slices: Vec<(i64, i64)> = graph
            .nodes
            .iter()
            .filter(|n| n.inputs.contains(&gu_f32))
            .map(|n| match &n.op {
                AiOp::Slice { starts, ends, .. } => (starts[0], ends[0]),
                other => panic!("the fused gate/up view must feed Slice nodes only, got {other:?}"),
            })
            .collect();
        assert_eq!(gu_slices, vec![(0, 128), (128, 256)]);

        // Each sliced operand wires into the shared linear-layer form
        // (Transpose → MatMul), e.g. the layer-0 Q slice.
        let q_weight = tensor_id(&graph, "q_weight_0");
        assert!(graph
            .nodes
            .iter()
            .any(|n| matches!(n.op, AiOp::Transpose { .. }) && n.inputs.contains(&q_weight)));

        // Untied head, family surfaced, no biases anywhere.
        assert!(graph.tensor_names.values().any(|n| n == "lm_head.weight"));
        match graph.metadata.get("family") {
            Some(MetaValue::Str(s)) => assert_eq!(s, "Phi3ForCausalLM"),
            other => panic!("family metadata missing or mistyped: {other:?}"),
        }
    }

    #[test]
    fn phi3_manifest_without_the_fused_tensor_fails_naming_it() {
        let config = tiny_phi3_config();
        // A Llama-style (unfused) manifest cannot realize a fused family.
        let keys = decoder_keys(2, false, false);
        let dtypes = vec![DType::F32; keys.len()];
        let err = build_parametric_graph_from_manifest(&config, &keys, &dtypes, None)
            .err()
            .expect("a fused family without its fused tensor must fail loud");
        let msg = err.to_string();
        assert!(
            msg.contains("qkv_proj.weight") && msg.contains("Phi3ForCausalLM"),
            "error should name the missing fused tensor and the family: {msg}"
        );
    }

    #[test]
    fn phi3_unsupported_rope_knobs_fail_naming_the_knob() {
        let keys = fused_decoder_keys(2);
        let dtypes = vec![DType::F32; keys.len()];

        let mut config = tiny_phi3_config();
        config["rope_scaling"] = serde_json::json!({"type": "longrope"});
        let err = build_parametric_graph_from_manifest(&config, &keys, &dtypes, None)
            .err()
            .expect("non-null rope_scaling must fail loud");
        assert!(
            err.to_string().contains("rope_scaling"),
            "error should name the knob: {err}"
        );

        let mut config = tiny_phi3_config();
        config["partial_rotary_factor"] = serde_json::json!(0.4);
        let err = build_parametric_graph_from_manifest(&config, &keys, &dtypes, None)
            .err()
            .expect("partial_rotary_factor must fail loud");
        assert!(
            err.to_string().contains("partial_rotary_factor"),
            "error should name the knob: {err}"
        );

        // The config-only preflight rejects the same knob with no manifest.
        let mut config = tiny_phi3_config();
        config["rope_scaling"] = serde_json::json!({"type": "longrope"});
        let err = validate_config(&config).expect_err("preflight must reject the knob too");
        assert!(
            err.to_string().contains("rope_scaling"),
            "preflight error should name the knob: {err}"
        );
    }

    #[test]
    fn context_length_above_max_position_embeddings_fails() {
        let config = tiny_llama_config();
        let keys = decoder_keys(2, false, false);
        let dtypes = vec![DType::F32; keys.len()];
        let err = build_parametric_graph_from_manifest(&config, &keys, &dtypes, Some(3000))
            .err()
            .expect("must fail loud");
        assert!(
            err.to_string().contains("max_position_embeddings"),
            "error should name the ceiling: {err}"
        );
    }

    // ───────────────── staged partition (row `staged-execution`) ─────────────

    fn meta_str<'g>(graph: &'g AiGraph, key: &str) -> Option<&'g str> {
        match graph.metadata.get(key) {
            Some(MetaValue::Str(s)) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Manifest names declared by a graph (the names a κ binding can attach to).
    fn declared_manifest_names(graph: &AiGraph, keys: &[String]) -> Vec<String> {
        let declared: std::collections::HashSet<&str> =
            graph.tensor_names.values().map(String::as_str).collect();
        keys.iter()
            .filter(|k| declared.contains(k.as_str()))
            .cloned()
            .collect()
    }

    #[test]
    fn staged_partition_has_embedding_layer_blocks_and_head() {
        let config = tiny_llama_config();
        let keys = decoder_keys(2, false, false);
        let dtypes = vec![DType::F32; keys.len()];
        let stages = build_parametric_stage_graphs(
            &config,
            &keys,
            &dtypes,
            Some(128),
            NonZeroU64::new(1).expect("non-zero"),
        )
        .expect("the staged build succeeds");

        // 2 layers at 1 layer/stage → embedding + 2 layer stages + head.
        assert_eq!(stages.len(), 4);
        for (i, g) in stages.iter().enumerate() {
            assert_eq!(meta_int(g, "stage_index"), Some(i as i64));
            assert_eq!(meta_int(g, "stage_count"), Some(4));
            assert_eq!(meta_int(g, "context_length"), Some(128));
            assert_eq!(meta_str(g, "arch"), Some("parametric_transformer"));
        }

        // Stage 0: input_ids → hidden_states.
        assert_eq!(
            meta_str(&stages[0], "stage_role"),
            Some(STAGE_ROLE_EMBEDDING)
        );
        assert_eq!(stages[0].input_names, vec!["input_ids"]);
        assert_eq!(stages[0].output_names, vec!["hidden_states"]);

        // Middle layer stage: hidden_states → hidden_states.
        assert_eq!(meta_str(&stages[1], "stage_role"), Some(STAGE_ROLE_LAYERS));
        assert_eq!(meta_int(&stages[1], "stage_layer_start"), Some(0));
        assert_eq!(meta_int(&stages[1], "stage_layer_end"), Some(1));
        assert_eq!(stages[1].input_names, vec!["hidden_states"]);
        assert_eq!(stages[1].output_names, vec!["hidden_states"]);

        // The last layer stage defers its closing residual add to the head
        // stage (the monolithic compile fuses that add into the final norm),
        // so it carries the fused kernel's two operands.
        assert_eq!(meta_str(&stages[2], "stage_role"), Some(STAGE_ROLE_LAYERS));
        assert_eq!(stages[2].input_names, vec!["hidden_states"]);
        assert_eq!(
            stages[2].output_names,
            vec!["hidden_states", "hidden_residual"]
        );

        // Head stage: (hidden_states, hidden_residual) → logits.
        assert_eq!(meta_str(&stages[3], "stage_role"), Some(STAGE_ROLE_HEAD));
        assert_eq!(
            stages[3].input_names,
            vec!["hidden_states", "hidden_residual"]
        );
        assert_eq!(stages[3].output_names, vec!["logits"]);
    }

    #[test]
    fn staged_partition_declares_every_manifest_tensor_exactly_once_per_consumer() {
        let config = tiny_llama_config();
        let keys = decoder_keys(2, false, false);
        let dtypes = vec![DType::F32; keys.len()];
        let stages = build_parametric_stage_graphs(
            &config,
            &keys,
            &dtypes,
            None,
            NonZeroU64::new(1).expect("non-zero"),
        )
        .expect("the staged build succeeds");

        // Union over stages = the whole manifest; each name in exactly the
        // stage that consumes it (untied → all singletons).
        let mut seen: HashMap<String, Vec<usize>> = HashMap::new();
        for (i, g) in stages.iter().enumerate() {
            for name in declared_manifest_names(g, &keys) {
                seen.entry(name).or_default().push(i);
            }
        }
        assert_eq!(seen.len(), keys.len(), "every manifest tensor is declared");
        for (name, in_stages) in &seen {
            let expected = if name == "model.embed_tokens.weight" {
                vec![0]
            } else if name == "model.norm.weight" || name == "lm_head.weight" {
                vec![3]
            } else {
                let l = extract_layer_idx(name).expect("layer tensors name their layer");
                vec![1 + l]
            };
            assert_eq!(
                in_stages, &expected,
                "`{name}` must be declared by exactly its consuming stage"
            );
        }
    }

    #[test]
    fn staged_tied_head_declares_the_embedding_tensor_in_both_stages() {
        // Tied embeddings: the head stage declares `model.embed_tokens.weight`
        // itself — the κ-map of stage 0 and the head stage bind the SAME κ,
        // one κ-store blob shared by two stage archives.
        let config = tiny_qwen2_config();
        let keys = decoder_keys(2, true, true);
        let dtypes = vec![DType::F32; keys.len()];
        let stages = build_parametric_stage_graphs(
            &config,
            &keys,
            &dtypes,
            Some(256),
            NonZeroU64::new(1).expect("non-zero"),
        )
        .expect("the staged build succeeds");
        let head = stages.last().expect("a head stage");
        assert!(head
            .tensor_names
            .values()
            .any(|n| n == "model.embed_tokens.weight"));
        assert!(!head.tensor_names.values().any(|n| n == "lm_head.weight"));
        assert!(stages[0]
            .tensor_names
            .values()
            .any(|n| n == "model.embed_tokens.weight"));
    }

    #[test]
    fn staged_blocks_group_consecutive_layers() {
        // 2 layers at 2 layers/stage → embedding + ONE layer stage + head,
        // and the single layer stage ends unsealed (its last layer is the
        // model's last layer).
        let config = tiny_llama_config();
        let keys = decoder_keys(2, false, false);
        let dtypes = vec![DType::F32; keys.len()];
        let stages = build_parametric_stage_graphs(
            &config,
            &keys,
            &dtypes,
            None,
            NonZeroU64::new(2).expect("non-zero"),
        )
        .expect("the staged build succeeds");
        assert_eq!(stages.len(), 3);
        assert_eq!(meta_int(&stages[1], "stage_layer_start"), Some(0));
        assert_eq!(meta_int(&stages[1], "stage_layer_end"), Some(2));
        assert_eq!(
            stages[1].output_names,
            vec!["hidden_states", "hidden_residual"]
        );
        // Both layers' weights live in the one layer stage.
        let declared = declared_manifest_names(&stages[1], &keys);
        assert!(declared.iter().any(|n| n.contains("layers.0")));
        assert!(declared.iter().any(|n| n.contains("layers.1")));
    }
}
