# Unified hologram-ai: ONNX + GGUF + SafeTensors Support

## Implementation Status: COMPLETE

| Phase | Description | Status |
|-------|-------------|--------|
| 1-2 | Workspace restructure | ✅ Complete |
| 4 | Move ONNX code | ✅ Complete |
| 3 | GenericTransformerBuilder | ✅ Complete |
| 5 | GGUF support | ✅ Complete |
| 6 | SafeTensors support | ✅ Complete |
| 7 | Unified CLI | ✅ Complete |
| 8 | Testing & validation | ✅ Complete |

### Cleanup & Features

| Item | Status |
|------|--------|
| Remove old `src/` directory | ✅ Complete |
| Remove old `proto/` and `build.rs` | ✅ Complete |
| Implement SiLU activation | ✅ Complete (composed from sigmoid + mul) |
| Implement RoPE position encoding | ✅ Complete (in AttentionBuilder) |
| Fix doctest imports | ✅ Complete |
| Mark external fixture tests as #[ignore] | ✅ Complete |

### Test Results

* **hologram-ai-common**: 41 tests passing
* **hologram-ai-gguf**: 22 tests passing
* **hologram-ai-safetensors**: 13 tests passing
* **hologram-ai-onnx**: 973 tests passing
* **Workspace**: Compiles successfully with all features
* **All doctests**: Passing (with ignored for external fixtures)

***

## Decisions Made

Based on user requirements:

* **Architectures**: LLaMA/LLaMA2/LLaMA3, Mistral/Mixtral, Qwen2, DeepSeek
* **Quantization**: Dequantize to F32 on load (for GGUF quantized models)
* **Structure**: Rename `hologram-onnx` → `hologram-ai` as a Cargo workspace
* **Multi-format**: Support ONNX, GGUF, and SafeTensors
* **Crate design**: Hybrid with feature flags (`onnx`, `gguf`, `safetensors`)

***

## Background: Format Comparison

| Aspect | ONNX | GGUF | SafeTensors |
|--------|------|------|-------------|
| **Format Type** | Computational graph | Weight storage + metadata | Weight storage + config.json |
| **Operations** | Explicit nodes | Implicit (from architecture) | Implicit (from config.json) |
| **Architectures** | Any DAG of ops | Fixed (LLaMA, etc.) | Fixed (from config.json) |
| **Weights** | F32/F16 | Quantized (Q4\_K, Q8\_0) | F32/F16/BF16 |
| **Source** | PyTorch export | llama.cpp ecosystem | HuggingFace models |
| **Translation** | 1:1 op mapping | Arch → graph rebuild | Arch → graph rebuild |

**Key insight**: All formats compile to the same hologram IR, then to `.holo` files with embedded weights.

### Format Support Scope

**ONNX: Any model (in principle)**

* ONNX contains explicit computational graph
* Any model with implemented operation translators works
* Missing ops → add translator, model works

**GGUF + SafeTensors: Architecture-specific**

* Both are weight-only formats, NO computational graph
* Must reconstruct graph from architecture metadata
* Only works for architectures with implemented builders
* Unknown architecture → clear error message

**Generic TransformerBuilder (no architecture-specific code):**

```
GGUF file                      SafeTensors + config.json
    │                                   │
    ▼                                   ▼
 GGUF Parser                    SafeTensors Parser
    │                                   │
    └──────────┬────────────────────────┘
               ▼
      TransformerConfig
      ├── num_layers: 32
      ├── hidden_size: 4096
      ├── num_attention_heads: 32
      ├── norm_type: RMSNorm
      └── activation: SiLU
               │
               ▼
      GenericTransformerBuilder
      (builds IR from config params)
               │
               ▼
          hologram IR → .holo
```

**Key insight:** Most LLMs are the same transformer architecture with different parameters. No need for LlamaBuilder, MistralBuilder, etc. - just ONE generic builder that reads config.

### Weight Handling

All weights are embedded in `.holo` files:

* No external weight files
* No SafeTensors runtime dependency (we only PARSE the format)
* Everything runs through hologram runtime
* Aligned with project philosophy: "Everything is a .holo file"

***

## Implementation Plan

### Phase 1: Restructure to Workspace

Rename and restructure from single crate to workspace:

```
hologram-ai/                      # Renamed from hologram-onnx
├── Cargo.toml                    # Workspace manifest
├── README.md
├── CLAUDE.md                     # Updated instructions
├── crates/
│   ├── hologram-ai/              # Main crate (CLI + unified API)
│   │   ├── Cargo.toml            # Features: onnx, gguf, safetensors
│   │   └── src/
│   │       ├── lib.rs            # Conditional re-exports
│   │       └── main.rs           # Unified CLI
│   │
│   ├── hologram-ai-onnx/         # ONNX format support
│   │   ├── Cargo.toml
│   │   ├── build.rs              # prost-build for proto
│   │   ├── proto/onnx.proto3
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── parser.rs
│   │       ├── translator.rs
│   │       └── ops/              # ONNX op translators
│   │
│   ├── hologram-ai-gguf/         # GGUF format support
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── parser.rs         # GGUF parsing (uses gguf-rs-lib)
│   │       └── dequant/          # Q4_K, Q8_0 → F32
│   │
│   ├── hologram-ai-safetensors/  # SafeTensors format support
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── parser.rs         # SafeTensors parsing (no runtime)
│   │       └── config.rs         # config.json parsing
│   │
│   └── hologram-ai-common/       # Shared utilities + generic builder
│       ├── Cargo.toml
│       └── src/
│           ├── lib.rs
│           ├── error.rs          # Unified error types
│           ├── weights.rs        # Weight handling
│           ├── serialization.rs  # .holo format
│           ├── transformer/      # Generic transformer builder
│           │   ├── mod.rs
│           │   ├── config.rs     # TransformerConfig struct
│           │   ├── builder.rs    # GenericTransformerBuilder
│           │   ├── attention.rs  # Attention block builder
│           │   ├── ffn.rs        # Feed-forward block builder
│           │   └── norm.rs       # Normalization builders
│           └── weight_map.rs     # Tensor name → weight mapping
│
├── tests/                        # Integration tests
├── examples/                     # Example usage
└── configs/                      # Pipeline configs
```

**Files to modify:**

* `/workspace/Cargo.toml` → workspace manifest
* Create new crate directories and Cargo.toml files
* Move existing code into `hologram-ai-onnx`

### Phase 2: Create Workspace Cargo.toml

```toml
# /hologram-ai/Cargo.toml (root workspace)
[workspace]
resolver = "2"
members = [
    "crates/hologram-ai",
    "crates/hologram-ai-onnx",
    "crates/hologram-ai-gguf",
    "crates/hologram-ai-safetensors",
    "crates/hologram-ai-common",
]

[workspace.package]
version = "0.1.0"
edition = "2024"
license = "MIT OR Apache-2.0"
repository = "https://github.com/uor-framework/hologram-ai"

[workspace.dependencies]
# Hologram
hologram = { path = "/hologram" }

# Internal crates
hologram-ai-common = { path = "crates/hologram-ai-common" }
hologram-ai-onnx = { path = "crates/hologram-ai-onnx" }
hologram-ai-gguf = { path = "crates/hologram-ai-gguf" }
hologram-ai-safetensors = { path = "crates/hologram-ai-safetensors" }

# Shared
thiserror = "2.0"
anyhow = "1.0"
bytemuck = { version = "1.14", features = ["derive"] }
half = "2.3"
tracing = "0.1"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"

# CLI
clap = { version = "4.5", features = ["derive"] }

# Format-specific
gguf-rs-lib = "0.1"
prost = "0.13"
```

```toml
# crates/hologram-ai/Cargo.toml (main crate with features)
[package]
name = "hologram-ai"
version.workspace = true
edition.workspace = true

[[bin]]
name = "hologram-ai"
path = "src/main.rs"

[dependencies]
hologram-ai-common.workspace = true
hologram-ai-onnx = { workspace = true, optional = true }
hologram-ai-gguf = { workspace = true, optional = true }
hologram-ai-safetensors = { workspace = true, optional = true }

clap.workspace = true
anyhow.workspace = true

[features]
default = ["onnx", "gguf", "safetensors"]
onnx = ["dep:hologram-ai-onnx"]
gguf = ["dep:hologram-ai-gguf"]
safetensors = ["dep:hologram-ai-safetensors"]
```

**Usage examples:**

```toml
# Full support (default)
hologram-ai = "0.1"

# ONNX only (smaller binary, faster compile)
hologram-ai = { version = "0.1", default-features = false, features = ["onnx"] }

# GGUF only (for quantized LLMs)
hologram-ai = { version = "0.1", default-features = false, features = ["gguf"] }

# SafeTensors only (for HuggingFace models)
hologram-ai = { version = "0.1", default-features = false, features = ["safetensors"] }

# LLM formats only (no ONNX)
hologram-ai = { version = "0.1", default-features = false, features = ["gguf", "safetensors"] }
```

### Phase 3: Unified CLI

```bash
# Auto-detect format from extension
hologram-ai compile model.onnx -o model.holo
hologram-ai compile model.gguf -o model.holo
hologram-ai compile model_dir/ -o model.holo  # SafeTensors directory

# Explicit format
hologram-ai compile --format onnx model.bin -o model.holo
hologram-ai compile --format gguf model.bin -o model.holo
hologram-ai compile --format safetensors ./model/ -o model.holo

# Info commands
hologram-ai info model.onnx
hologram-ai info model.gguf
hologram-ai info ./model/  # SafeTensors directory

# Run compiled models
hologram-ai run model.holo --input data.json
```

**Format detection logic:**

* `.onnx` extension → ONNX format
* `.gguf` extension → GGUF format
* Directory with `config.json` + `*.safetensors` → SafeTensors format

### Phase 4: Move Existing ONNX Code

Move existing code from `src/` to `crates/hologram-ai-onnx/src/`:

| Current Location | New Location |
|------------------|--------------|
| `src/core/parser.rs` | `crates/hologram-ai-onnx/src/parser.rs` |
| `src/core/translator.rs` | `crates/hologram-ai-onnx/src/translator.rs` |
| `src/ops/` | `crates/hologram-ai-onnx/src/ops/` |
| `src/translators/` | `crates/hologram-ai-onnx/src/translators/` |
| `src/core/weights.rs` | `crates/hologram-ai-common/src/weights.rs` |
| `src/core/serialization.rs` | `crates/hologram-ai-common/src/serialization.rs` |
| `src/cli/` | `crates/hologram-ai/src/cli/` |
| `src/runtime/` | `crates/hologram-ai/src/runtime/` |

### Phase 5: Implement GGUF Support

Add GGUF support in `crates/hologram-ai-gguf/`:

#### 5.1: GGUF Parsing & Metadata

Implement typed metadata extraction wrapping `gguf-rs-lib`:

```rust
// crates/hologram-ai-gguf/src/metadata.rs
pub struct GgufMetadata {
    pub architecture: Architecture,
    pub block_count: u32,
    pub embedding_length: u32,
    pub attention_head_count: u32,
    pub attention_head_count_kv: u32,  // For GQA
    pub feed_forward_length: u32,
    pub rope_freq_base: f32,
    pub context_length: u32,
    pub vocab_size: u32,
}

pub enum Architecture {
    Llama,
    Mistral,
    Unknown(String),
}
```

#### 5.2: Dequantization Module

**Quantization types to support (priority order):**

1. F32, F16, BF16 (trivial)
2. Q8\_0 (simple 8-bit)
3. Q4\_0 (legacy 4-bit)
4. Q4\_K (K-quant 4-bit, most common)
5. Q6\_K, Q5\_K (stretch goals)

#### 5.3: Architecture Builders

**LLaMA transformer block structure:**

```
Input → RMSNorm → QKV Projection → RoPE → Attention → Add (residual)
                                                            ↓
                                               RMSNorm → FFN (gate*up→down) → Add (residual)
```

***

## Required hologram Operations

Operations needed for LLaMA/Mistral - verified against hologram IR:

| Operation | Status | Location/Notes |
|-----------|--------|----------------|
| MatMul | ✅ | `ops/complex/matmul.rs` |
| Add | ✅ | `ops/binary/add.rs` |
| Mul | ✅ | `ops/binary/mul.rs` |
| Div | ✅ | `ops/binary/div.rs` |
| Softmax | ✅ | `ops/activation/softmax.rs` |
| Reshape | ✅ | `ops/shape/reshape.rs` |
| Transpose | ✅ | `ops/shape/transpose.rs` |
| Gather | ✅ | `ops/advanced/gather.rs` |
| **RMSNorm** | ✅ | `ops/normalization/rms_norm.rs` - includes `llama_style()` |
| Sigmoid | ✅ | `ops/activation/sigmoid.rs` |
| Sin/Cos | ✅ | `ops/unary/sin.rs`, `ops/unary/cos.rs` |
| Concat | ✅ | `ops/advanced/concat.rs` |
| Split | ✅ | `ops/shape/split.rs` |
| **SiLU** | 🔧 | Compose: `x * sigmoid(x)` using Mul + Sigmoid |
| **RoPE** | 🔧 | Compose using Sin/Cos/Mul/Reshape or add dedicated op |

**SiLU Implementation**: Can be composed from existing ops:

```rust
// SiLU(x) = x * sigmoid(x)
let sig_x = builder.sigmoid(x)?;
let silu_output = builder.mul(x, sig_x)?;
```

**RoPE Implementation**: Can be composed from primitives, or we could add a dedicated `NodeOp::RoPE` for efficiency. RoPE formula:

```
q_rot = q * cos(θ) + rotate_half(q) * sin(θ)
```

Where `rotate_half` swaps adjacent pairs and negates. This can be built from:

* Reshape (split pairs)
* Neg, Mul, Add
* Precomputed sin/cos tables as constants

***

## New Dependencies for GGUF

Add to workspace `Cargo.toml`:

```toml
[workspace.dependencies]
# GGUF parsing (for hologram-ai-gguf)
gguf-rs-lib = "0.1"
```

The ONNX crate keeps its existing dependencies (prost, prost-build, etc.).

***

## Verification Plan

1. **Unit tests**: Dequantization correctness vs reference implementations
2. **Integration test**: Load small GGUF model, compile to IR, verify graph structure
3. **E2E test**: Compile GGUF → .holo → run with hologram CLI → verify output
4. **Reference comparison**: Compare outputs against llama.cpp for same model/input

***

## Sources

* [GGUF Specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
* [Hugging Face GGUF Docs](https://huggingface.co/docs/hub/en/gguf)
* [gguf-rs-lib crate](https://lib.rs/crates/gguf-rs-lib)
* [GGUF Quantization Types](https://huggingface.co/docs/hub/en/gguf)
* [llama.cpp Model Architecture](https://deepwiki.com/ggml-org/llama.cpp/3.2-model-loading-and-management)
* [K-Quants Overview](https://gist.github.com/Artefact2/b5f810600771265fc1e39442288e8ec9)

***

## Model Support Summary

### Diffusion Models (Stable Diffusion, SDXL, Flux)

**Supported via ONNX path.** Diffusion models are commonly exported to ONNX:

* UNet, VAE, CLIP text encoder are standard NN operations
* Existing ONNX translator supports Conv2D, attention, GroupNorm
* No additional work needed for ONNX-exported diffusion models

### LLM Models (via GGUF or SafeTensors)

**Generic TransformerBuilder handles all transformer-based LLMs:**

| Model | Config Source | Notes |
|-------|--------------|-------|
| LLaMA 1/2/3 | GGUF metadata, config.json | Standard transformer |
| Mistral | GGUF metadata, config.json | Sliding window attention |
| Mixtral | GGUF metadata, config.json | MoE (requires MoE support) |
| Qwen/Qwen2 | GGUF metadata, config.json | Standard transformer |
| DeepSeek | GGUF metadata, config.json | MoE variants |
| Phi-2/3 | GGUF metadata, config.json | Standard transformer |
| Gemma | GGUF metadata, config.json | Standard transformer |

**All handled by ONE GenericTransformerBuilder** - config parameters determine the exact structure.

### Generic TransformerBuilder Details

The builder reads config parameters and constructs the graph - no model-specific code needed:

```rust
/// TransformerConfig - extracted from GGUF metadata or config.json
pub struct TransformerConfig {
    pub num_layers: u32,
    pub hidden_size: u32,
    pub num_attention_heads: u32,
    pub num_kv_heads: Option<u32>,      // For GQA (None = same as attention heads)
    pub intermediate_size: u32,
    pub vocab_size: u32,
    pub max_position_embeddings: u32,

    // Normalization
    pub norm_type: NormType,            // RMSNorm | LayerNorm
    pub norm_eps: f32,

    // Activation
    pub hidden_act: Activation,         // SiLU | GELU | ReLU

    // Position encoding
    pub rope_theta: Option<f32>,        // RoPE base frequency
    pub rope_scaling: Option<RoPEScaling>,

    // FFN style
    pub ffn_type: FFNType,              // Gated (LLaMA-style) | Standard
}

impl GenericTransformerBuilder {
    pub fn build(&self, config: &TransformerConfig, weights: &WeightMap) -> Result<IRFunction> {
        let mut builder = GraphBuilder::new();

        // 1. Input embedding
        let input_ids = builder.input("input_ids", ...);
        let hidden = self.build_embedding(&mut builder, input_ids, weights)?;

        // 2. Transformer layers (loop based on config.num_layers)
        for i in 0..config.num_layers {
            hidden = self.build_layer(&mut builder, hidden, i, config, weights)?;
        }

        // 3. Output head
        let logits = self.build_output(&mut builder, hidden, config, weights)?;

        builder.output("logits", logits);
        Ok(builder.build())
    }
}
```

**Supported model families (same generic builder):**

* LLaMA, LLaMA 2, LLaMA 3
* Mistral, Mixtral
* Qwen, Qwen2
* DeepSeek
* Phi-2, Phi-3
* Gemma

All use the same `GenericTransformerBuilder` with different `TransformerConfig` parameters.

***

## Implementation Order

1. **Workspace restructure** (Phase 1-2)
   * Convert to Cargo workspace
   * Create all crate directories
   * Update root Cargo.toml

2. **Move ONNX code** (Phase 4)
   * Move existing files to `hologram-ai-onnx`
   * Extract common code to `hologram-ai-common`
   * Update imports and module paths
   * Verify ONNX tests still pass

3. **Generic TransformerBuilder** (`hologram-ai-common/transformer/`)
   * `TransformerConfig` struct (num\_layers, hidden\_size, etc.)
   * Attention block builder (multi-head, GQA support)
   * FFN block builder (gated, standard)
   * Normalization builders (RMSNorm, LayerNorm)
   * Full transformer graph builder
   * Weight name mapping (GGUF style ↔ SafeTensors style)

4. **GGUF support** (`hologram-ai-gguf`)
   * Parser (using gguf-rs-lib)
   * Metadata → TransformerConfig conversion
   * Dequantization (F16, Q8\_0, Q4\_K, Q4\_0)
   * Integration with GenericTransformerBuilder

5. **SafeTensors support** (`hologram-ai-safetensors`)
   * Parser (simple binary format, no runtime)
   * config.json → TransformerConfig conversion
   * Integration with GenericTransformerBuilder

6. **Unified CLI** (Phase 3)
   * Create main crate with format detection
   * Integrate all format compilers
   * Add info/validate commands

7. **Testing & validation**
   * Unit tests for dequant
   * Unit tests for transformer builder
   * Integration tests with small models
   * E2E comparison with llama.cpp (GGUF)
   * E2E comparison with transformers (SafeTensors)
