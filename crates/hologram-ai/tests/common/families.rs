//! Parametric decoder-family definitions shared by the family-coverage decode
//! test and the parametric family memory sweep.
//!
//! There are NO per-model magic numbers here: a family's structural layout
//! (fused vs separate q/k/v, fused vs separate gate/up, q/k/v bias, tied vs
//! untied head) comes from the parametric registry's *faithful set*
//! (`hologram_ai_safetensors::parametric::supported_families` — Llama, Qwen2,
//! Mistral, Phi3), and the *scale* is an independent knob. The same scale can
//! be applied to any family, and the same family can be built at any scale —
//! the tests exercise architectural coverage, not any single model's shape.

#![allow(dead_code)]

use serde_json::Value;

/// The structural layout of a decoder family — exactly the knobs the parametric
/// registry's `FamilySpec` carries (`attention_qkv_bias`,
/// `attention_fused_qkv`, `mlp_fused_gate_up`) plus the tied-head convention.
/// One entry per family in the registry's faithful set.
#[derive(Clone, Copy, Debug)]
pub struct FamilyLayout {
    /// `config.architectures[0]` — the family name the registry selects on.
    pub arch: &'static str,
    /// Q/K/V projections carry bias tensors (Qwen2), a structural property of
    /// the family independent of any `attention_bias` config flag.
    pub qkv_bias: bool,
    /// Attention Q/K/V ship as one fused `self_attn.qkv_proj.weight` (Phi3),
    /// carved into the three operands by compile-time `Slice`.
    pub fused_qkv: bool,
    /// MLP gate/up ship as one fused `mlp.gate_up_proj.weight` (Phi3).
    pub fused_gate_up: bool,
    /// The LM head reuses the embedding table — no separate `lm_head.weight`
    /// (Qwen2's published convention); the others ship an untied head.
    pub tied_head: bool,
}

/// Llama: separate q/k/v/o projections, no attention bias, gate/up/down SwiGLU
/// MLP, an untied `lm_head.weight`.
pub const LLAMA: FamilyLayout = FamilyLayout {
    arch: "LlamaForCausalLM",
    qkv_bias: false,
    fused_qkv: false,
    fused_gate_up: false,
    tied_head: false,
};

/// Qwen2: the Llama tensor shape plus attention q/k/v biases, with a tied head.
pub const QWEN2: FamilyLayout = FamilyLayout {
    arch: "Qwen2ForCausalLM",
    qkv_bias: true,
    fused_qkv: false,
    fused_gate_up: false,
    tied_head: true,
};

/// Mistral: tensor-identical to Llama (separate q/k/v and gate/up, no bias,
/// untied head).
pub const MISTRAL: FamilyLayout = FamilyLayout {
    arch: "MistralForCausalLM",
    qkv_bias: false,
    fused_qkv: false,
    fused_gate_up: false,
    tied_head: false,
};

/// Phi3: fused `self_attn.qkv_proj.weight` + fused `mlp.gate_up_proj.weight`,
/// no bias, untied head.
pub const PHI3: FamilyLayout = FamilyLayout {
    arch: "Phi3ForCausalLM",
    qkv_bias: false,
    fused_qkv: true,
    fused_gate_up: true,
    tied_head: false,
};

/// The registry's faithful decoder families — the coverage frontier the
/// parametric compiler builds end to end (mirrors
/// `hologram_ai_safetensors::parametric::supported_families`).
pub const FAITHFUL_FAMILIES: &[FamilyLayout] = &[LLAMA, QWEN2, MISTRAL, PHI3];

/// Model dimensions — the scale knob, held separately from the family layout so
/// the SAME scale sweeps across families of different layout. No per-model
/// magic numbers: the coverage scale is deliberately small (architecture
/// coverage, not size); the memory sweep supplies a larger, representative
/// multi-billion-parameter-class template.
#[derive(Clone, Copy, Debug)]
pub struct Dims {
    pub hidden_size: u64,
    pub layers: u64,
    pub num_attention_heads: u64,
    pub num_key_value_heads: u64,
    pub head_dim: u64,
    pub intermediate_size: u64,
    pub vocab_size: u64,
    pub max_position_embeddings: u64,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
}

impl Dims {
    /// A modest, fast scale for the default per-family coverage test: small
    /// enough to compile and decode all four families in seconds, yet still
    /// exercising grouped-query attention (`kv_heads < heads`), several layers,
    /// a non-trivial vocabulary, and a gated MLP. NOT any real model's numbers —
    /// `head_dim * heads == hidden_size` is the only structural constraint.
    pub const MODEST: Dims = Dims {
        hidden_size: 512,
        layers: 4,
        num_attention_heads: 8,
        num_key_value_heads: 2,
        head_dim: 64,
        intermediate_size: 1536,
        vocab_size: 2048,
        max_position_embeddings: 4096,
        rope_theta: 10_000.0,
        rms_norm_eps: 1e-6,
    };

    /// LARGE — a ~20-billion-parameter-class template (round, tied to no single
    /// published model): at the full 44 layers this is ≈ 20 B parameters. The
    /// embed/head (128 k × 6144) and each layer stage are already at 20B-class
    /// WIDTH, so the memory-critical terms — the resident window and the float
    /// head chunk's F32 image — are exercised at true scale even when the memory
    /// sweep turns `with_layers` DOWN for a feasible native run (the per-stage
    /// footprint that drives the 4 GiB ceiling does not depend on layer count).
    pub const LARGE: Dims = Dims {
        hidden_size: 6144,
        layers: 44,
        num_attention_heads: 48,
        num_key_value_heads: 8,
        head_dim: 128,
        intermediate_size: 16384,
        vocab_size: 128_000,
        max_position_embeddings: 8192,
        rope_theta: 1_000_000.0,
        rms_norm_eps: 1e-6,
    };

    /// EXTRA_LARGE — a ~500-billion-parameter-class template: at the full 140
    /// layers this is ≈ 500 B+ parameters. Its embed (256 k × 16384 ≈ 8 GB as a
    /// single tensor) exceeds the whole residency budget, so this tier exercises
    /// the weight-tier paging / windowing frontier: the peak must still stay
    /// bounded by the WINDOW, never the model. As with LARGE the width is the
    /// point; `with_layers` scales the depth for a runnable native measurement.
    pub const EXTRA_LARGE: Dims = Dims {
        hidden_size: 16384,
        layers: 140,
        num_attention_heads: 128,
        num_key_value_heads: 8,
        head_dim: 128,
        intermediate_size: 65536,
        vocab_size: 256_000,
        max_position_embeddings: 8192,
        rope_theta: 1_000_000.0,
        rms_norm_eps: 1e-6,
    };

    /// The same dimensions at a different transformer-layer count — the scale
    /// knob the memory sweep turns. At full depth `LARGE` ≈ 20 B params and
    /// `EXTRA_LARGE` ≈ 500 B+; turned down, the width (embed/head/per-layer)
    /// still exercises the memory-critical, ceiling-driving stages.
    #[must_use]
    pub fn with_layers(mut self, layers: u64) -> Self {
        self.layers = layers;
        self
    }
}

/// A decoder model at a chosen scale for a chosen family: produces the
/// `config.json` and the family's characteristic tensor manifest (fused vs
/// separate q/k/v and gate/up, q/k/v bias, tied vs untied head) — the exact
/// layouts the parametric registry recognizes. Parametric in both the family
/// and the scale; no term is specialized to a particular model.
#[derive(Clone, Copy, Debug)]
pub struct FamilyScale {
    pub layout: FamilyLayout,
    pub dims: Dims,
}

impl FamilyScale {
    /// The generic constructor — a family layout at a scale.
    pub fn new(layout: FamilyLayout, dims: Dims) -> Self {
        Self { layout, dims }
    }

    /// Per-family constructors — thin selectors over the shared scale, so the
    /// numbers stay parametric while each supported family is nameable.
    pub fn llama(dims: Dims) -> Self {
        Self::new(LLAMA, dims)
    }
    pub fn qwen2(dims: Dims) -> Self {
        Self::new(QWEN2, dims)
    }
    pub fn mistral(dims: Dims) -> Self {
        Self::new(MISTRAL, dims)
    }
    pub fn phi3(dims: Dims) -> Self {
        Self::new(PHI3, dims)
    }

    pub fn arch(&self) -> &'static str {
        self.layout.arch
    }

    /// Fused-Q output rows: `heads · head_dim`.
    pub fn q_out(&self) -> u64 {
        self.dims.num_attention_heads * self.dims.head_dim
    }

    /// Per-KV-projection output rows: `kv_heads · head_dim`.
    pub fn kv_out(&self) -> u64 {
        self.dims.num_key_value_heads * self.dims.head_dim
    }

    /// The model's `config.json` as a `serde_json::Value` (for the parametric
    /// graph builders, which take `&Value`).
    pub fn config_value(&self) -> Value {
        let d = &self.dims;
        serde_json::json!({
            "architectures": [self.layout.arch],
            "hidden_size": d.hidden_size,
            "intermediate_size": d.intermediate_size,
            "num_hidden_layers": d.layers,
            "num_attention_heads": d.num_attention_heads,
            "num_key_value_heads": d.num_key_value_heads,
            "head_dim": d.head_dim,
            "vocab_size": d.vocab_size,
            "rms_norm_eps": d.rms_norm_eps,
            "rope_theta": d.rope_theta,
            "max_position_embeddings": d.max_position_embeddings,
            "tie_word_embeddings": self.layout.tied_head,
            "torch_dtype": "bfloat16",
        })
    }

    /// The model's `config.json` as a string (for the staged pipeline, which
    /// takes `config_json: &str` / `String`).
    pub fn config_json(&self) -> String {
        self.config_value().to_string()
    }

    /// The family's characteristic tensor manifest `(name, shape)`: the exact
    /// fused/separate layout, bias set, and head convention the parametric
    /// registry recognizes for this family.
    pub fn manifest(&self) -> Vec<(String, Vec<u64>)> {
        let d = &self.dims;
        let h = d.hidden_size;
        let q_out = self.q_out();
        let kv_out = self.kv_out();
        let i = d.intermediate_size;
        let v = d.vocab_size;

        let mut m: Vec<(String, Vec<u64>)> = vec![
            ("model.embed_tokens.weight".into(), vec![v, h]),
            ("model.norm.weight".into(), vec![h]),
        ];
        if !self.layout.tied_head {
            m.push(("lm_head.weight".into(), vec![v, h]));
        }
        for l in 0..d.layers {
            let p = format!("model.layers.{l}");
            m.push((format!("{p}.input_layernorm.weight"), vec![h]));
            m.push((format!("{p}.post_attention_layernorm.weight"), vec![h]));

            // Attention: one fused qkv_proj (Phi3) or separate q/k/v (others).
            if self.layout.fused_qkv {
                m.push((
                    format!("{p}.self_attn.qkv_proj.weight"),
                    vec![q_out + 2 * kv_out, h],
                ));
            } else {
                m.push((format!("{p}.self_attn.q_proj.weight"), vec![q_out, h]));
                m.push((format!("{p}.self_attn.k_proj.weight"), vec![kv_out, h]));
                m.push((format!("{p}.self_attn.v_proj.weight"), vec![kv_out, h]));
            }
            m.push((format!("{p}.self_attn.o_proj.weight"), vec![h, q_out]));

            // MLP: one fused gate_up_proj (Phi3) or separate gate/up (others).
            if self.layout.fused_gate_up {
                m.push((format!("{p}.mlp.gate_up_proj.weight"), vec![2 * i, h]));
            } else {
                m.push((format!("{p}.mlp.gate_proj.weight"), vec![i, h]));
                m.push((format!("{p}.mlp.up_proj.weight"), vec![i, h]));
            }
            m.push((format!("{p}.mlp.down_proj.weight"), vec![h, i]));

            // Attention q/k/v biases (Qwen2 only).
            if self.layout.qkv_bias {
                m.push((format!("{p}.self_attn.q_proj.bias"), vec![q_out]));
                m.push((format!("{p}.self_attn.k_proj.bias"), vec![kv_out]));
                m.push((format!("{p}.self_attn.v_proj.bias"), vec![kv_out]));
            }
        }
        m
    }
}

/// A norm weight (RMSNorm gain) — kept at exactly 1.0 so the forward stays
/// well-conditioned; also excluded from the quantizable set (it is 1-D).
pub fn is_norm(name: &str) -> bool {
    name.contains("layernorm") || name.ends_with(".norm.weight")
}

/// FNV-1a 64-bit of the tensor name — a per-tensor seed so every tensor's dummy
/// bytes are a DISTINCT pseudo-random stream. Content addressing deduplicates
/// identical bytes, so identically-shaped dummy weights would otherwise collapse
/// onto one κ and under-count resident memory; a distinct seed keeps every
/// load-bearing weight its own κ (the real model's every-weight-distinct set).
pub fn name_seed(name: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in name.as_bytes() {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h | 1 // nonzero xorshift seed
}

/// Deterministic dummy weight VALUES for a tensor. VALUES do not carry meaning
/// (the SHAPES drive the tests); they only need to be finite, bounded, and
/// distinct per tensor so the forward pass does not NaN and κs do not dedup.
/// Norm weights are exactly 1.0; everything else is a name-seeded xorshift64
/// stream in `[-0.05, 0.05]` — small enough that either a bf16 or an f32
/// encoding stays finite through the whole stack.
fn dummy_values(name: &str, dims: &[u64]) -> Vec<f32> {
    let n: u64 = dims.iter().product();
    if is_norm(name) {
        return vec![1.0f32; n as usize];
    }
    let mut state = name_seed(name);
    (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let unit = (state >> 11) as f32 / (1u64 << 53) as f32; // [0, 1)
            (unit * 2.0 - 1.0) * 0.05 // [-0.05, 0.05]
        })
        .collect()
}

/// The tensor's dummy weight bytes as little-endian bf16 (2 bytes/element) — the
/// download byte set the staged / int8-quantized κ path consumes.
pub fn dummy_bf16_bytes(name: &str, dims: &[u64]) -> Vec<u8> {
    let vals = dummy_values(name, dims);
    let mut out = Vec::with_capacity(vals.len() * 2);
    for v in vals {
        let bf16 = (v.to_bits() >> 16) as u16;
        out.extend_from_slice(&bf16.to_le_bytes());
    }
    out
}

/// The tensor's dummy weight bytes as little-endian f32 (4 bytes/element) — the
/// correctness oracle's reference path, which decodes and compiles in f32.
pub fn dummy_f32_bytes(name: &str, dims: &[u64]) -> Vec<u8> {
    dummy_values(name, dims)
        .into_iter()
        .flat_map(|v| v.to_le_bytes())
        .collect()
}
