# Tokenizers as Hologram Operations

## Philosophy

**Everything is a computational graph that compiles to hologram IR.**

Tokenizers are NOT special - they're just:
- Vocabulary lookups → `Gather` operations
- Padding → `Pad` operations
- Masking → Comparison operations
- String processing → Byte manipulations

## Architecture

```
tokenizer.json ──→ TokenizerIR ──→ OperationGraph ──→ tokenizer.holo
                   (parse vocab)    (hologram-ir)      (compiled)

At runtime:
text_input ──→ tokenizer.holo ──→ [input_ids, attention_mask]
               (hologram backend)
```

## Tokenizer as IR Operations

### Example: SentencePiece Tokenization

```rust
// Input: text as bytes [batch, text_len]
// Output: token_ids [batch, seq_len], attention_mask [batch, seq_len]

OperationGraph {
    nodes: [
        // 1. Vocabulary lookup table (32K entries)
        Constant { data: vocab_table, shape: [32128] },

        // 2. Input bytes
        Input { name: "text_bytes", shape: [1, Dynamic] },

        // 3. Lookup tokens via piece matching
        // For SentencePiece, this is greedy longest-match
        // Can be implemented as a series of Gather operations
        Gather {
            data: vocab_table,
            indices: matched_indices,
            axis: 0
        },

        // 4. Pad to max_length
        Pad {
            input: token_ids,
            pads: [0, 0, 0, max_length - dynamic_len],
            constant_value: pad_token_id
        },

        // 5. Generate attention mask (1 for tokens, 0 for padding)
        NotEqual {
            a: padded_tokens,
            b: pad_token_id
        },
        Cast { input: mask, to: Float32 },

        // Outputs
        Output { name: "input_ids", value: padded_tokens },
        Output { name: "attention_mask", value: float_mask },
    ]
}
```

### Example: BPE Tokenization (GPT-style)

```rust
OperationGraph {
    nodes: [
        // 1. Vocabulary + merges table
        Constant { data: vocab, shape: [50257] },
        Constant { data: merges, shape: [50000, 2] },

        // 2. Byte-level encoding
        Input { name: "text", shape: [1, Dynamic] },

        // 3. Apply byte-pair merges iteratively
        // Each merge is: Compare pairs → Replace matches
        Loop {
            iterations: num_merges,
            body: [
                // Find all pair matches
                Compare { pairs, current_merge_rule },
                // Replace matches with merged token
                Scatter { ... },
            ]
        },

        // 4. Vocabulary lookup
        Gather { data: vocab, indices: byte_pairs },

        // 5. Add special tokens (BOS, EOS)
        Concat { tensors: [bos_token, tokens, eos_token], axis: 1 },

        // 6. Pad + mask
        Pad { ... },
        NotEqual { ... },
    ]
}
```

## Compilation Pipeline

### Step 1: Parse Tokenizer Config

```rust
// tokenizer.json → TokenizerSpec
struct TokenizerSpec {
    type: TokenizerType,           // SentencePiece, BPE, WordPiece
    vocab: HashMap<String, u32>,    // Token → ID
    merges: Vec<(String, String)>,  // BPE merges
    special_tokens: SpecialTokens,
}
```

### Step 2: Build IR Graph

```rust
fn compile_tokenizer_to_ir(spec: &TokenizerSpec) -> OperationGraph {
    let mut builder = GraphBuilder::new();

    match spec.type {
        TokenizerType::SentencePiece => {
            compile_sentencepiece(&mut builder, spec)
        }
        TokenizerType::BPE => {
            compile_bpe(&mut builder, spec)
        }
        TokenizerType::WordPiece => {
            compile_wordpiece(&mut builder, spec)
        }
    }

    builder.build()
}
```

### Step 3: Compile to .holo

```rust
// Same as ONNX compilation
let ir_graph = compile_tokenizer_to_ir(&spec)?;
let backend_plan = hologram_compiler::compile_ir(&ir_graph, BackendType::Cpu)?;
let holo_bytes = serialize_backend_plan(&backend_plan)?;
fs::write("tokenizer.holo", holo_bytes)?;
```

### Step 4: Execute via Hologram Runtime

```rust
// Load compiled tokenizer
let tokenizer = ModelExecutor::from_holo_file("tokenizer.holo")?;

// Execute tokenization
let text_bytes = text.as_bytes();
let inputs = hashmap! {
    "text_bytes" => Tensor::new(text_bytes_as_f32, [1, text.len()])
};

let outputs = tokenizer.execute(inputs)?;
let input_ids = outputs["input_ids"];
let attention_mask = outputs["attention_mask"];
```

## Why This Approach?

### ✅ Unified Framework
- Tokenization, model inference, post-processing all use hologram
- Single execution path
- Single optimization pipeline

### ✅ SIMD Acceleration
- Vocabulary lookups parallelized via hologram's SIMD kernels
- Batch tokenization automatically optimized
- Hardware-agnostic (CPU, GPU via hologram backend)

### ✅ Config-Driven Everything
```toml
# tokenizer.toml
type = "sentencepiece"
vocab_path = "vocab.json"
output_path = "tokenizer.holo"

# Compile once
$ hologram-onnx compile-tokenizer tokenizer.toml

# Use everywhere
$ hologram-onnx run --tokenizer tokenizer.holo --model model.holo
```

### ✅ Cacheable & Portable
- Compile once, run forever
- Distribute `.holo` files, not JSON
- Version control compiled tokenizers

### ✅ Composable
```toml
[[stages]]
type = "model"
model = "tokenizer"  # tokenizer.holo
inputs = { text = "prompt" }
outputs = ["input_ids", "attention_mask"]

[[stages]]
type = "model"
model = "encoder"    # encoder.holo
inputs = { input_ids = "input_ids", attention_mask = "attention_mask" }
outputs = ["hidden_states"]
```

Tokenizer is just another model!

## Implementation Strategy

### Phase 1: Simple Vocabulary Lookup
- Parse tokenizer.json → vocabulary HashMap
- Create Gather-based IR for lookups
- Compile to .holo
- **This gets 80% of cases working**

### Phase 2: Full SentencePiece
- Implement greedy longest-match algorithm as IR ops
- Handle unicode properly
- Add special token insertion

### Phase 3: BPE & WordPiece
- Implement merge rules as IR operations
- Add subword splitting logic

### Phase 4: Optimization
- Fuse operations in compiler
- Use hologram's optimization passes
- Benchmark vs native implementations

## File Structure

```
src/tokenizers/
├── mod.rs              # Trait + loader
├── compiler.rs         # Main: tokenizer → IR → .holo
├── sentencepiece.rs    # SentencePiece IR compiler
├── bpe.rs              # BPE IR compiler
├── wordpiece.rs        # WordPiece IR compiler
└── ops.rs              # Common tokenizer operations as IR
```

## CLI Integration

```bash
# Compile tokenizer to .holo
hologram-onnx compile-tokenizer \
    --type sentencepiece \
    --vocab models/t5-small/tokenizer.json \
    --output models/t5-small/tokenizer.holo

# Use in pipeline
hologram-onnx run --config configs/t5-generate.toml
```

Config references compiled tokenizer:
```toml
[tokenizer]
precompiled = "models/t5-small/tokenizer.holo"

# OR compile on-the-fly
[tokenizer]
type = "sentencepiece"
vocab_path = "models/t5-small/tokenizer.json"
```

## Benefits Over Native Tokenizers

1. **Performance**: SIMD-accelerated lookups via hologram kernels
2. **Portability**: `.holo` files run anywhere hologram runs
3. **Composability**: Tokenizer = model = post-processor (all `.holo`)
4. **Optimization**: hologram compiler optimizes tokenization graph
5. **Future-proof**: New tokenizer types = new IR patterns

## The Vision

```
Everything is a .holo file:
├── tokenizer.holo       (text → tokens)
├── encoder.holo         (tokens → hidden)
├── decoder.holo         (hidden → logits)
└── post_process.holo    (logits → text)

All execute on hologram backend.
All benefit from hologram optimizations.
All are config-driven and cacheable.
```

**This is the hologram way**: Universal computational framework where tokenizers, models, and post-processing are all first-class IR operations.
