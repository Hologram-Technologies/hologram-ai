# Falcon-Perception: Lessons for hologram-ai (Rust Compiler)

## Context

We reviewed [tiiuae/Falcon-Perception](https://github.com/tiiuae/Falcon-Perception) — a 1B-param natively multimodal vision-language transformer that does object detection, segmentation, and OCR. The goal is to extract architectural patterns and techniques that inform hologram-ai's compiler IR and execution backend, with an eye toward **Rust implementation** of the key subsystems.

---

## 1. Paged KV Cache → Rust Implementation Notes

### What Falcon Does (Python/CUDA)
- **Single unified KV tensor** `[layers, 2, 1, heads, total_pages * page_size, head_dim]`
- **Virtual page tables** `[max_batch, max_pages]` mapping logical → physical page indices
- CPU-side metadata (`page_table_cpu`, `capacity` arrays) to avoid GPU→CPU sync for allocation checks
- LIFO free-page stack for O(1) allocation
- Address calc: `physical_addr = page_table[batch, token_pos / page_size] * page_size + token_pos % page_size`
- Prefill skips paging entirely (sequential writes to page 0)
- Preemption: evict newest sequence, free all its pages, re-prefill later

### Rust Translation for hologram

**Data structures:**
```rust
struct PagedKvCache {
    /// Physical storage: [layers][kv=2][heads][total_slots][head_dim]
    storage: Vec<f32>,  // or backend-specific allocation

    /// Logical→physical page table per sequence slot
    page_table: Vec<Vec<u32>>,  // page_table[batch_slot][logical_page] = physical_page

    /// LIFO free page stack
    free_pages: Vec<u32>,

    /// Tokens capacity per batch slot
    capacity: Vec<usize>,

    page_size: usize,
    n_heads: usize,
    head_dim: usize,
    n_layers: usize,
}
```

**Key operations to expose in FloatOp / hologram IR:**
- `KvPagedWrite { layer, batch_idx, token_pos }` — lookup page table, compute physical addr, scatter write
- `KvPagedRead { layer, batch_idx, seq_range }` — gather read across pages for attention
- `PageAlloc` / `PageFree` — runtime memory management (not compiled into graph)

**Design decision:** Page table management is a **runtime** concern (scheduler), not a **compiled graph** concern. The compiled graph should emit `KvPagedWrite`/`KvPagedRead` ops that the runtime resolves via page tables. This matches Falcon's separation where `assign()` is called by the engine, not by the model.

---

## 2. FlexAttention / Composable Masks → Rust IR Design

### What Falcon Does
- `mask_mod(b, h, q_idx, kv_idx) → bool` — composable callable functions
- `and_masks(causal, document, non_left_pad)` — intersection
- `or_masks(image_prefix, block_causal)` — union
- **Block-level sparsity** (`BlockMask.kv_indices`) + **element-level** (`mask_mod` evaluated per pair)
- Two-tier: coarse blocks skip entire KV chunks, fine mask_mod handles boundaries

### Rust IR: Replace `causal: bool` with `AttentionMask` enum

Current hologram-ai attention: `AiOp::Attention { causal: bool, ... }`

Proposed extension:
```rust
enum AttentionMaskKind {
    /// Standard: all queries attend to all keys
    Full,
    /// Causal: q_idx >= kv_idx
    Causal,
    /// Hybrid vision-language: bidirectional within image regions, causal for text
    /// Encoded as segment boundaries in a metadata tensor
    HybridCausalPrefix { prefix_segments: Vec<(usize, usize)> },
    /// Windowed: each query attends to keys within a spatial window
    /// Used for upsampler cross-attention (AnyUp pattern)
    SpatialWindow { window_ratio: f32, q_grid: (usize, usize), kv_grid: (usize, usize) },
    /// Arbitrary block-sparse mask (fallback)
    BlockSparse { block_size: usize, block_mask: Vec<u8> },
}
```

**Why not full callable mask_mod?** We're a compiler — we need static analysis for fusion and scheduling. The enum captures the patterns we've seen (causal, prefix-bidirectional, spatial window) without requiring a Turing-complete mask language. `BlockSparse` is the escape hatch.

**Compiler fusion opportunity:** `HybridCausalPrefix` can be lowered to two separate attention calls (one bidirectional for image prefix, one causal for text suffix) and fused back at the block level.

---

## 3. 3D RoPE (Temporal + Spatial) → Rust Implementation

### What Falcon Does
- Head dim split in half: lower = 1D temporal RoPE (standard), upper = 2D spatial RoPE (golden-ratio learned frequencies)
- `apply_3d_rotary_emb`: chunk → apply_rotary_emb on lower half → apply_golden_rotary_emb on upper half → concat
- Text tokens: spatial position = (0,0) → identity rotation (no branch needed)
- Image patches: spatial position = normalized (h, w) grid scaled by aspect ratio
- Golden frequencies are **learned parameters** (loaded from checkpoint), not computed from theta

### Rust Implementation

```rust
enum RoPEKind {
    /// Standard 1D: freqs from theta, applied to full head_dim
    Standard { theta: f32 },
    /// 3D split: lower half temporal (theta-based), upper half spatial (learned freqs)
    SplitTemporalSpatial {
        theta: f32,
        /// Learned per-head spatial frequencies: [n_heads, rope_dim/4, 2]
        spatial_freqs: TensorRef,
    },
}
```

**Lowering to FloatOp:**
- `RoPE1D { theta, pos }` — existing path, applied to lower half
- `RoPE2D { freqs_tensor, pos_hw }` — new op for golden spatial RoPE on upper half
- At compile time, the split is a `Slice` + two RoPE applications + `Concat`

**Key insight:** Text tokens get zero spatial positions → the golden RoPE degenerates to identity (cos(0)=1, sin(0)=0). No branching needed in the kernel — just pass zeros for text positions. This is clean for compiled code.

---

## 4. Vision Pipeline Components → What hologram-ai Would Need

### Image Patch Embedding
```
Image (T, H, W, C) → Rearrange to (T*H_p*W_p, patch_size²*C) → Linear(768, 1024)
```
- The rearrange is a zero-copy reshape/stride manipulation
- The linear projection is a standard MatMul
- **Already expressible** in our IR as `Reshape` + `MatMul`

### Fourier Coordinate Encoding
```
(x, y) → Linear(2, 256) → 2π*result → [cos, sin] → Linear(512, 1024)
```
- Sequence: `MatMul → Scale(2π) → Cos/Sin → Concat → MatMul`
- **Already expressible** — all standard float ops

### AnyUp Upsampler (Segmentation)
- Windowed cross-attention between high-res queries and low-res keys
- ResBlocks with SiLU activation
- The attention uses `SpatialWindow` mask pattern (see §2)
- **New pattern** for us but structurally just cross-attention + conv blocks

### Perception Heads (Bbox Decoding)
- Hidden state → Linear → argmax → normalize to [0,1]
- Standard dense layer + argmax — trivially expressible

---

## 5. Techniques Worth Adopting

### 5a. Sink Tokens (Attention Stabilization)
```python
# Per-head learned scalar
output = output * sigmoid(log_sum_exp - sink_param)
```
- Prevents attention score outliers from dominating
- One learned scalar per head — negligible parameter cost
- In Rust: `FloatOp::SinkGate { lse, sink_params }` or fused into attention output

### 5b. Squared-ReLU Gated FFN
```
output = W_down(relu(W_gate(x))² * W_up(x))
```
- Fuses gate activation into single kernel (no temporary for relu² result)
- In Rust: `FloatOp::FusedSquaredReluGate` or decomposed as `ReLU → Square → Mul`
- Our compiler could recognize this pattern via a fusion pass

### 5c. QK-Norm (RMSNorm on Q and K after projection)
```python
xq = rms_norm(xq)  # Per-head normalization
xk = rms_norm(xk)
```
- Stabilizes attention scores, especially with large head_dim
- Already have RMSNorm; just need to recognize per-head application pattern

### 5d. GQA repeat_kv
- 16 query heads, 8 KV heads → `repeat_interleave(k, 2, dim=heads)`
- Pure reshape/expand — we already handle this

---

## 6. What's NOT Worth Porting

- **FlexAttention Triton kernels** — we write our own compute kernels in Rust
- **CUDA graph capture** — runtime optimization, not compiler concern
- **Continuous batching scheduler** — runtime concern (hologram-exec territory)
- **MLX backend** — we target our own backend
- **FastAPI server** — application layer, not compiler

---

## Summary: Prioritized Rust Work Items

| Priority | Item | Complexity | Blocks |
|----------|------|-----------|--------|
| **P0** | Paged KV cache runtime in hologram-exec | Medium | Plan 016 |
| **P1** | `AttentionMaskKind` enum (replace `causal: bool`) | Small | Vision-language support |
| **P1** | `RoPE2D` / split temporal-spatial RoPE | Small | Vision-language support |
| **P2** | `FusedSquaredReluGate` fusion pass | Small | Performance |
| **P2** | Sink token gating | Trivial | Attention quality |
| **P3** | AnyUp-style windowed cross-attention | Medium | Segmentation models |
| **P3** | Fourier coordinate encoding | Trivial | Detection models |

The P0/P1 items are prerequisites for compiling any vision-language model. P2 items are performance wins applicable to current LLM support. P3 items are needed only for perception/segmentation tasks.

---

## 7. Import Path Gap: SafeTensors vs ONNX

### Current State
- hologram-ai's **only** import path is ONNX (`hologram-ai-onnx` crate, 60+ ops supported)
- Falcon-Perception ships as **SafeTensors** (HuggingFace format) — no ONNX export available
- Many newer vision-language models (LLaVA, Qwen-VL, Falcon-Perception) only publish SafeTensors + PyTorch code

### Options

**Option A: ONNX conversion pipeline (recommended short-term)**
- Use `torch.onnx.export()` or `optimum` to convert Falcon-Perception → ONNX
- Pros: leverages our existing 60+ op ONNX importer, no new parsing code
- Cons: export can lose custom ops (FlexAttention → decomposed attention), requires Python tooling as a pre-step
- This is what we do today for all models

**Option B: SafeTensors importer (medium-term)**
- Add `hologram-ai-safetensors` crate using the `safetensors` Rust crate (mature, HuggingFace-maintained)
- SafeTensors only contains **weights**, not the **graph** — we'd also need to parse `config.json` to reconstruct the model architecture
- This means building architecture templates in Rust (e.g., `FalconPerceptionArchitecture` that knows the layer structure)
- Pros: direct loading, no Python dependency, supports the growing SafeTensors ecosystem
- Cons: each new model family needs a Rust architecture template — more maintenance

**Option C: GGUF (already removed)**
- Was removed in Plan 061 Stage 0 — ONNX is the only format going forward
- Not relevant for Falcon-Perception anyway (no GGUF published)

### Recommendation
**Option A now, Option B later.** For immediate experimentation, export Falcon-Perception to ONNX via `torch.onnx.export()`. For production support of the HuggingFace ecosystem, build a SafeTensors weight loader paired with config-driven architecture reconstruction.

---

## 8. Tokenizer Gap: Vision-Language Special Tokens

### Current State (`hologram-ai-tokenizer`)
- Supports BPE, Unigram, WordPiece — all via HuggingFace `tokenizer.json` format
- Stores special tokens: `<s>`, `</s>`, `<unk>`, `<pad>` + arbitrary `additional` HashMap
- **Assumes pure text input** — `fn encode(&self, text: &str) -> Vec<u32>`

### What Falcon-Perception Needs
- `<|image|>` placeholder token — replaced by image patch embeddings at runtime
- `<coord>`, `<size>` — trigger perception head decoders during generation
- `<seg>` — triggers segmentation upsampler
- `<|start_of_query|>`, `<|REF_SEG|>` — prompt structure tokens
- Image token insertion: the tokenizer must know where to place N image tokens in the sequence

### Required Rust Changes

```rust
/// Extend TokenizerSectionData (in archive.rs)
struct VisionTokenConfig {
    /// Token ID that marks "insert image patches here"
    image_placeholder_id: u32,
    /// Token IDs for perception-specific decoding
    coord_token_id: Option<u32>,
    size_token_id: Option<u32>,
    seg_token_id: Option<u32>,
    /// Start/end of image markers
    soi_token_id: Option<u32>,
    eoi_token_id: Option<u32>,
}

/// Extend encode API to handle multimodal input
trait MultimodalTokenizer: Tokenizer {
    /// Encode text with image placeholders, returning token IDs
    /// and a list of (position, n_patches) for each image
    fn encode_with_images(
        &self,
        text: &str,
        images: &[ImageInfo],  // contains patch counts
    ) -> (Vec<u32>, Vec<ImageInsertionPoint>);
}
```

**Key insight from Falcon:** The tokenizer uses HuggingFace's Rust `tokenizers` crate internally — the same ecosystem our `native.rs` already loads from. The special tokens are just entries in `added_tokens` with specific string patterns. We can load them today; we just lack the multimodal encoding logic.

---

## 9. Variable-Resolution Shapes: DimExpr Implications

### Current State (`hologram-ai-common/src/ir/shape/`)
- `DimExpr` supports symbolic dims: `Concrete`, `Var`, `Add/Mul/Div`, `Dynamic`
- `DimVarTable` manages named variables with bounds
- Canonical vars: `BATCH`, `SEQ_LEN`, `VOCAB_SIZE`, `HIDDEN_DIM`, etc.
- **All batch elements assumed to have the same shape** (rectangular tensors only)

### What Falcon-Perception Needs
- Each image can be a different resolution → different patch count per image
- Image patches are flattened into the sequence dimension alongside text tokens
- Total sequence length = text_tokens + sum(image_patches[i]) — **per-batch-element variable**
- Spatial RoPE positions are per-image grids of different sizes

### Options

**Option A: Pad to max + mask (pragmatic)**
- Pad all images to the largest resolution in the batch
- Use attention mask to ignore padding
- This is what Falcon's `BatchInferenceEngine` does (left-padding to max seq len)
- **Works with our existing DimExpr** — just use `max_patches` as a concrete dim
- Wastes compute on padding but avoids ragged tensor support
- **Recommended for initial support**

**Option B: Ragged dimension support (future)**
- Add `DimExpr::RaggedVar { per_element_sizes: Vec<u64> }` or similar
- Requires ragged tensor support throughout the compiler pipeline
- Significant complexity — only needed for high-throughput production serving
- Falcon's `PagedInferenceEngine` uses this (continuous batching, no padding waste)

**Option C: Single-image compilation (simplest)**
- Compile for batch_size=1, single image at a time
- No padding, no ragged tensors — just concrete dimensions per invocation
- Sequence length is known at runtime: `text_tokens + H_patches * W_patches`
- **This is how we'd start** — matches our LLM pipeline approach (compile prefill + decode graphs)

### Recommendation
**Start with Option C** (single-image, concrete dims at compile time). Graduate to **Option A** (padded batching) when we need throughput. **Option B** is a long-term goal that aligns with Plan 016 paged attention.

### New canonical DimVars needed:
```rust
const IMAGE_HEIGHT: &str = "image_height";     // in patches, not pixels
const IMAGE_WIDTH: &str = "image_width";        // in patches
const NUM_IMAGES: &str = "num_images";           // images per sequence
const PATCH_DIM: &str = "patch_dim";             // patch_size² * channels (e.g., 768)
```

---

## 10. Updated Priority Table

| Priority | Item | Complexity | Blocks |
|----------|------|-----------|--------|
| **P0** | Paged KV cache runtime in hologram-exec | Medium | Plan 016 |
| **P0** | ONNX export script for Falcon-Perception (Python) | Small | Any testing |
| **P1** | `AttentionMaskKind` enum (replace `causal: bool`) | Small | Vision-language support |
| **P1** | `RoPE2D` / split temporal-spatial RoPE | Small | Vision-language support |
| **P1** | Vision special tokens in tokenizer (`VisionTokenConfig`) | Small | Multimodal encoding |
| **P1** | New DimVars for image dimensions | Trivial | Shape inference |
| **P2** | `FusedSquaredReluGate` fusion pass | Small | Performance |
| **P2** | Sink token gating | Trivial | Attention quality |
| **P2** | SafeTensors weight loader crate | Medium | HuggingFace ecosystem |
| **P3** | `MultimodalTokenizer` trait + encode_with_images | Medium | End-to-end VLM pipeline |
| **P3** | AnyUp-style windowed cross-attention | Medium | Segmentation models |
| **P3** | Ragged dimension support in DimExpr | Large | Production batched serving |

## Verification

To validate this analysis end-to-end:
1. Export Falcon-Perception-300M to ONNX via `torch.onnx.export()` with a dummy image+text input
2. Attempt `import_onnx()` — identify which ops fail to import
3. Compare the ONNX graph structure against our AiOp coverage
4. Run the imported graph through our compiler pipeline to find lowering gaps
