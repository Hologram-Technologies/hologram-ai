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

    let mut keys = Vec::new();
    let mut dtypes = Vec::new();
    for st in &st_instances {
        for (k, view) in st.tensors() {
            dtypes.push(map_dtype(view.dtype())?);
            keys.push(k.clone());
        }
    }

    let mut graph = build_parametric_graph_from_manifest(config, &keys, &dtypes, None)?;

    // Inject the actual safetensors weights into the graph's params.
    let mut name_to_id = HashMap::new();
    for (id, name) in &graph.tensor_names {
        name_to_id.insert(name.clone(), *id);
    }

    let mut next_id = graph.tensor_names.keys().max().copied().unwrap_or(0) + 1;
    for st in &st_instances {
        for (k, tensor_view) in st.tensors() {
            let id = if let Some(existing_id) = name_to_id.get(&k) {
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
    let family = select_family(config)?;
    reject_unsupported_knobs(family, config)?;
    let cfg = ModelConfig::from_json(config)?;
    let manifest = TensorManifest::new(keys, dtypes)?;
    validate_layer_count(&cfg, keys)?;
    let context_length = resolve_context_length(family, &cfg, context_length)?;

    let mut builder = GraphBuilder::new("parametric_model".to_string());

    // Dimension expressions — all functions of the config.
    let batch = builder.register_var("batch");
    let seq = builder.register_var("seq");
    let vocab = DimExpr::Concrete(cfg.vocab_size);
    let hidden = DimExpr::Concrete(cfg.hidden_size);
    let ffn_hidden = DimExpr::Concrete(cfg.intermediate_size);
    let n_heads_expr = DimExpr::Concrete(cfg.num_attention_heads);
    let n_kv_heads_expr = DimExpr::Concrete(cfg.num_key_value_heads);
    let head_dim_expr = DimExpr::Concrete(cfg.head_dim);
    let q_out_dim = DimExpr::Concrete(cfg.num_attention_heads * cfg.head_dim);
    let kv_out_dim = DimExpr::Concrete(cfg.num_key_value_heads * cfg.head_dim);

    // Bias expectations: family structure (Qwen2 Q/K/V) or Llama-style flags.
    let qkv_bias_expected = family.attention_qkv_bias || cfg.attention_bias;

    // Inputs
    let input_ids = builder.add_input("input_ids", DType::INT64, vec![batch.clone(), seq.clone()]);

    // 1. Embedding — declared at its manifest dtype, widened to F32 for compute.
    let embed_weight = add_weight_f32(
        &mut builder,
        &manifest,
        "model.embed_tokens.weight",
        vec![vocab.clone(), hidden.clone()],
    );
    let mut current = builder.add_tensor(
        "hidden_states",
        DType::F32,
        vec![batch.clone(), seq.clone(), hidden.clone()],
    );
    builder.add_node(
        AiOp::Gather { axis: 0 },
        vec![embed_weight, input_ids],
        vec![current],
    );

    // 2. Transformer blocks
    for l in 0..cfg.num_hidden_layers {
        // Attention Norm — ε from the model's own `rms_norm_eps`.
        let attn_norm_weight = add_weight_f32(
            &mut builder,
            &manifest,
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
                &mut builder,
                &manifest,
                family,
                &format!("model.layers.{l}.self_attn.qkv_proj.weight"),
                vec![DimExpr::Concrete(q_rows + 2 * kv_rows), hidden.clone()],
            )?;
            let q_weight = add_row_slice(
                &mut builder,
                fused,
                &format!("q_weight_{l}"),
                0,
                q_rows,
                hidden.clone(),
            );
            let k_weight = add_row_slice(
                &mut builder,
                fused,
                &format!("k_weight_{l}"),
                q_rows,
                kv_rows,
                hidden.clone(),
            );
            let v_weight = add_row_slice(
                &mut builder,
                fused,
                &format!("v_weight_{l}"),
                q_rows + kv_rows,
                kv_rows,
                hidden.clone(),
            );
            let q_flat = add_linear_layer_from_tensor(
                &mut builder,
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
                &mut builder,
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
                &mut builder,
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
                &mut builder,
                &manifest,
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
                &mut builder,
                &manifest,
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
                &mut builder,
                &manifest,
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
            &mut builder,
            &manifest,
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
            &mut builder,
            &manifest,
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
                &mut builder,
                &manifest,
                family,
                &format!("model.layers.{l}.mlp.gate_up_proj.weight"),
                vec![DimExpr::Concrete(2 * cfg.intermediate_size), hidden.clone()],
            )?;
            let gate_weight = add_row_slice(
                &mut builder,
                fused,
                &format!("gate_weight_{l}"),
                0,
                cfg.intermediate_size,
                hidden.clone(),
            );
            let up_weight = add_row_slice(
                &mut builder,
                fused,
                &format!("up_weight_{l}"),
                cfg.intermediate_size,
                cfg.intermediate_size,
                hidden.clone(),
            );
            let gate_out = add_linear_layer_from_tensor(
                &mut builder,
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
                &mut builder,
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
                &mut builder,
                &manifest,
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
                &mut builder,
                &manifest,
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
            &mut builder,
            &manifest,
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

        // Add (residual 2)
        let res2_out = builder.add_tensor(
            &format!("res2_{l}"),
            DType::F32,
            vec![batch.clone(), seq.clone(), hidden.clone()],
        );
        builder.add_node(AiOp::Add, vec![res1_out, down_out], vec![res2_out]);

        current = res2_out;
    }

    // Final Norm — ε from the model's own `rms_norm_eps`.
    let norm_weight = add_weight_f32(
        &mut builder,
        &manifest,
        "model.norm.weight",
        vec![hidden.clone()],
    );
    let norm_out = builder.add_tensor(
        "norm_out",
        DType::F32,
        vec![batch.clone(), seq.clone(), hidden.clone()],
    );
    builder.add_node(
        AiOp::RmsNorm {
            epsilon: cfg.rms_norm_eps,
        },
        vec![current, norm_weight],
        vec![norm_out],
    );

    // LM Head — tied when the config says so or when the manifest carries no
    // separate `lm_head.weight` (the transformers convention for tied models).
    // The tied head reuses the embedding weight tensor — the same tensor id,
    // hence the same κ param — transposed by the shared linear-layer wiring
    // ([vocab, hidden] → [hidden, vocab]) for the matmul orientation.
    let tied = cfg.tie_word_embeddings || !manifest.contains("lm_head.weight");
    let (head_weight, head_weight_name) = if tied {
        (embed_weight, "lm_head.tied")
    } else {
        let weight = add_weight_f32(
            &mut builder,
            &manifest,
            "lm_head.weight",
            vec![vocab.clone(), hidden.clone()],
        );
        (weight, "lm_head.weight")
    };
    let logits = add_linear_layer_from_tensor(
        &mut builder,
        norm_out,
        head_weight,
        LinearLayerParams {
            weight_name: head_weight_name,
            in_features: hidden.clone(),
            out_features: vocab.clone(),
            output_name: "logits",
            output_shape: vec![batch.clone(), seq.clone(), vocab.clone()],
        },
    );

    // Output
    builder.add_output(logits, "logits");

    let mut graph = builder.build();

    let metadata = [
        ("arch", MetaValue::Str("parametric_transformer".to_string())),
        ("family", MetaValue::Str(family.name.to_string())),
        ("vocab_size", MetaValue::Int(cfg.vocab_size as i64)),
        ("n_layers", MetaValue::Int(cfg.num_hidden_layers as i64)),
        ("n_embd", MetaValue::Int(cfg.hidden_size as i64)),
        ("n_kv_heads", MetaValue::Int(cfg.num_key_value_heads as i64)),
        ("head_dim", MetaValue::Int(cfg.head_dim as i64)),
        ("context_length", MetaValue::Int(context_length as i64)),
    ];
    for (key, value) in metadata {
        graph.metadata.insert(key.to_string(), value);
    }
    // Surface the sliding-window clamp next to the context it clamped: the
    // `context_length` above already carries the clamped ceiling, and this
    // names its cause so no consumer mistakes the clamp for the trained
    // `max_position_embeddings`.
    if let (true, Some(window)) = (family.sliding_window_clamp, cfg.sliding_window) {
        graph
            .metadata
            .insert("sliding_window".to_string(), MetaValue::Int(window as i64));
    }

    Ok(graph)
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

        // Every weight is declared at its manifest storage dtype and widened
        // to F32 via the IR's canonical Cast — one per manifest tensor.
        let embed_id = tensor_id(&graph, "model.embed_tokens.weight");
        assert_eq!(graph.tensor_info[&embed_id].storage_dtype, DType::BF16);
        let cast_count = graph
            .nodes
            .iter()
            .filter(|n| matches!(n.op, AiOp::Cast { to: DType::F32 }))
            .count();
        assert_eq!(cast_count, keys.len());

        // Q/K/V biases are consumed as explicit broadcast Add operands.
        let q_bias_f32 = tensor_id(&graph, "model.layers.0.self_attn.q_proj.bias.f32");
        assert!(graph
            .nodes
            .iter()
            .any(|n| matches!(n.op, AiOp::Add) && n.inputs.contains(&q_bias_f32)));

        // Tied head: no separate `lm_head.weight`; the embedding weight's F32
        // view feeds both the token Gather and the head Transpose.
        assert!(!graph.tensor_names.values().any(|n| n == "lm_head.weight"));
        let embed_f32 = tensor_id(&graph, "model.embed_tokens.weight.f32");
        let consumers: Vec<&AiOp> = graph
            .nodes
            .iter()
            .filter(|n| n.inputs.contains(&embed_f32))
            .map(|n| &n.op)
            .collect();
        assert!(consumers.iter().any(|op| matches!(op, AiOp::Gather { .. })));
        assert!(consumers
            .iter()
            .any(|op| matches!(op, AiOp::Transpose { .. })));

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
}
