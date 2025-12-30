# Project: `onnx-pipe`

Build a config-driven ONNX compilation and execution framework in Rust. The system should support image generation (Stable Diffusion), audio models (Whisper), LLMs (Phi/Llama), and generic ONNX models through declarative TOML/YAML configuration files.

## Core Principles

1. **Convention over configuration** - Sensible defaults, users only specify what differs
2. **No model-specific code** - Everything driven by config, works with any ONNX model
3. **Minimal configs** - A simple classifier should need only 4 lines of config
4. **Auto-inference** - Infer model names, types, and output handlers where possible
5. **Zero-copy where possible** - Memory-mapped weights, Arc<Vec<f32>> for tensors

---

## 1. Project Structure

```
onnx-pipe/
├── Cargo.toml
├── config/examples/           # Example configs (SD, Whisper, Phi-2)
├── crates/
│   ├── onnx-pipe-core/        # Core types (Value, TensorBuffer, Error)
│   ├── onnx-pipe-config/      # TOML/YAML parsing & validation
│   ├── onnx-pipe-expr/        # Expression parser & evaluator
│   ├── onnx-pipe-compiler/    # ONNX → compiled format
│   ├── onnx-pipe-runtime/     # Pipeline executor
│   ├── onnx-pipe-builtins/    # Built-in operations
│   ├── onnx-pipe-handlers/    # Output handlers (image, audio, text)
│   └── onnx-pipe-cli/         # CLI application
└── tests/
```

---

## 2. Config Schema (Simplified with Defaults)

### Design Principle: Convention Over Configuration

- Most settings have sensible defaults
- Only specify what differs from defaults
- Auto-infer types, shapes, and handlers where possible

### Minimal Config Example (Classifier)

```toml
# Absolute minimum - just models and stages
[[models]]
path = "model.onnx"

[[stages]]
model = "model"
```

Everything else (inputs, outputs, handlers, compiler settings) is inferred.

### Simple SD Config (Real World)

```toml
name = "sd-turbo"

# Inputs with defaults - only specify non-string types
[inputs]
seed = { type = "int", default = -1 }
width = { type = "int", default = 512 }
height = { type = "int", default = 512 }
num_steps = { type = "int", default = 4 }

# Models - auto-named from filename if not specified
[[models]]
path = "text_encoder/model.onnx"

[[models]]
path = "unet/model.onnx"

[[models]]
path = "vae_decoder/model.onnx"

# Stages - minimal syntax
[[stages]]
type = "tokenize"
text = "prompt"               # Direct assignment instead of inputs = {...}

[[stages]]
model = "text_encoder"

[[stages]]
type = "randn"
shape = [1, 4, "height/8", "width/8"]
seed = "seed"

[[stages]]
loop = "num_steps"            # Simple loop syntax
var = "step"

  [[stages.stages]]
  model = "unet"
  sample = "latents"
  timestep = "timesteps[step]"

  [[stages.stages]]
  type = "euler_step"

[[stages]]
model = "vae_decoder"
latent_sample = "latents / 0.18215"

# Output auto-detected as image from shape/values
```

### Full Defaults Reference

```toml
# These are ALL the defaults - you never need to write them

[compiler]
optimization_level = 2        # Default: balanced optimization
target = "cpu"                # Default: CPU
memory_budget_mb = 0          # Default: 0 = unlimited
enable_fp16 = false           # Default: full precision
parallel_compile = true       # Default: use all cores
quantization.enabled = false  # Default: no quantization
fusion.enabled = true         # Default: enable op fusion

[defaults]
# Stage defaults
builtin_prefix = "builtin."   # Can omit "builtin." from type

# Input defaults
input_type = "string"         # Untyped inputs are strings
input_required = true         # Inputs required unless default given

# Output handler auto-detection
auto_detect_handlers = true   # Infer handler from output shape/type
image_value_range = "auto"    # Detect [-1,1], [0,1], or [0,255]
image_layout = "auto"         # Detect NCHW or NHWC from shape

# Model naming
auto_name_models = true       # Use filename as model name
```

### Shorthand Syntax

| Verbose                           | Shorthand                        |
| --------------------------------- | -------------------------------- |
| `inputs = { text = "prompt" }`    | `text = "prompt"`                |
| `type = "builtin.tokenize"`       | `type = "tokenize"`              |
| `loop = { count = 4, var = "i" }` | `loop = 4` (var defaults to "i") |
| `outputs = ["result"]`            | (auto-inferred from stage)       |

### Auto-Inference Rules

1. **Model names**: `path = "models/unet/model.onnx"` → name = `"unet"`
2. **Input types**: String unless specified otherwise
3. **Output handlers**:
   - 4D tensor with C=3/4 → Image handler
   - 1D float array → Audio handler
   - Integer array → Text handler (tokens)
4. **Value ranges**: Detect from actual output values at runtime
5. **Loop outputs**: Last stage's output becomes loop output

### Explicit Override When Needed

```toml
# Override auto-detected handler
[output_handlers.my_output]
type = "image"
format = "rgba"           # Override default rgb
layout = "NHWC"           # Override auto-detected
output_format = "webp"    # Override default png
quality = 90

# Override compiler for specific needs
[compiler]
target = { cuda = { device = 1 } }
quantization = { enabled = true, method = "dynamic" }
```

---

## 3. Expression System

### Grammar

```ebnf
expression  = conditional
conditional = logical_or [ "?" expr ":" expr ]
logical_or  = logical_and { "||" logical_and }
logical_and = equality { "&&" equality }
equality    = comparison { ("==" | "!=") comparison }
comparison  = additive { ("<" | "<=" | ">" | ">=") additive }
additive    = multiplicative { ("+" | "-") multiplicative }
multiplicative = unary { ("*" | "/" | "%") unary }
unary       = ("!" | "-") unary | postfix
postfix     = primary { "[" expr "]" | "." ident | "??" expr }
primary     = literal | variable | func_call | array | "(" expr ")"
```

### Supported Features

| Feature       | Example                             |
| ------------- | ----------------------------------- |
| Arithmetic    | `height / 8`, `latents * 0.18215`   |
| Comparison    | `guidance_scale > 1.0`              |
| Conditional   | `seed >= 0 ? seed : random()`       |
| Indexing      | `timesteps[step]`, `arr[-1]`        |
| Null coalesce | `guided_noise ?? noise_pred`        |
| Functions     | `len(arr)`, `min(a, b)`, `random()` |

### Built-in Functions

- **Math**: `abs`, `min`, `max`, `floor`, `ceil`, `sqrt`, `pow`, `log`, `exp`
- **Array**: `len`, `sum`, `mean`, `range`, `linspace`
- **String**: `lower`, `upper`, `trim`, `split`, `join`, `format`
- **Type**: `int`, `float`, `str`, `bool`
- **Random**: `random`, `random_int`

---

## 4. Compilation Options

```toml
[compiler]
output_path = "./model.pipe"
optimization_level = 2        # 0=none, 1=basic, 2=standard, 3=aggressive
target = "cpu"                # cpu, cuda, metal, rocm
memory_budget_mb = 4096
enable_fp16 = false
parallel_compile = true
cache_dir = "./.cache"

[compiler.quantization]
enabled = false
method = "dynamic"            # dynamic, static, qat
calibration_samples = 100
exclude_layers = ["lm_head"]

[compiler.fusion]
enabled = true
patterns = ["conv_bn", "matmul_add", "gelu", "attention"]
```

### Target Platforms

- **CPU**: x86_64/aarch64, AVX/NEON features, thread count
- **CUDA**: Compute capability, device selection
- **Metal**: Apple Silicon optimization
- **ROCm**: AMD GPU support

---

## 5. Built-in Operations

### Categories

| Category      | Operations                                                             |
| ------------- | ---------------------------------------------------------------------- |
| **Common**    | `add`, `sub`, `mul`, `div`, `concat`, `reshape`, `slice`, `gather`     |
| **Image**     | `randn`, `zeros`, `resize`, `crop`, `normalize`, `denormalize`         |
| **Diffusion** | `timesteps`, `euler_step`, `ddpm_step`, `ddim_step`, `cfg_guidance`    |
| **Audio**     | `load`, `resample`, `mel_spectrogram`, `stft`, `istft`                 |
| **Text**      | `tokenize`, `decode`, `sample_greedy`, `sample_top_p`, `init_kv_cache` |

### Registry Pattern

```rust
pub trait BuiltinOp: Send + Sync {
    fn name(&self) -> &'static str;
    fn execute(&self, ctx: &ExecutionContext, inputs: &[Value], params: &Params) -> Result<Vec<Value>>;
    fn validate(&self, inputs: &[Value], params: &Params) -> Result<()>;
    fn infer_shapes(&self, input_shapes: &[Vec<usize>], params: &Params) -> Result<Vec<Vec<usize>>>;
}
```

---

## 6. Output Handlers

### Handler Types

| Type       | Formats                | Use Case              |
| ---------- | ---------------------- | --------------------- |
| **Image**  | PNG, JPEG, WebP        | SD, image classifiers |
| **Audio**  | WAV, MP3, FLAC         | Whisper, music gen    |
| **Text**   | Plain, JSON, streaming | LLMs, NLP             |
| **Tensor** | NPZ, safetensors       | Raw export            |

### Image Handler Config

```toml
[pipeline.execution.output_handlers.image]
type = "image"
source = "decoded_image"
format = "rgb"              # rgb, rgba, grayscale
layout = "NCHW"             # NCHW, NHWC
value_range = "neg_one_one" # neg_one_one, zero_one, zero_255
output_format = "png"       # png, jpeg, webp
width = 512
height = 512
```

### Text Handler Config (Streaming LLM)

```toml
[pipeline.execution.output_handlers.text]
type = "text"
source = "generated_tokens"
tokenizer = "microsoft/phi-2"
format = "plain"
skip_special_tokens = true
streaming = true
```

---

## 7. Execution Engine

### Pipeline Executor Flow

```
1. Parse ExecutionConfig from TOML/YAML
2. Load compiled components from .pipe file
3. Initialize ExecutionContext with user inputs
4. For each stage:
   a. Evaluate condition (if present)
   b. Resolve input expressions
   c. Execute (Model/Builtin/Loop)
   d. Store outputs in context
5. Process outputs through handlers
6. Return results
```

### Streaming Mode (Bounded Memory)

- Load one component at a time
- Memory-map weights file (zero-copy)
- Preserve boundary tensors between components
- Peak memory = max(component_size) + boundaries

---

## 8. Core Types

```rust
pub enum Value {
    Tensor(TensorBuffer),
    Float(f64),
    Int(i64),
    String(String),
    Bool(bool),
    Array(Vec<Value>),
    Map(HashMap<String, Value>),
    Null,
}

pub struct TensorBuffer {
    pub data: Arc<Vec<f32>>,
    pub shape: Vec<usize>,
    pub dtype: DType,
}

pub enum DType { Float32, Float16, Int32, Int64, Int8, UInt8, Bool }
```

---

## 9. Example Configs (Simplified)

### Image Classifier (Minimal)

```toml
# 4 lines total!
[[models]]
path = "resnet50.onnx"

[[stages]]
model = "resnet50"
```

### Stable Diffusion

```toml
name = "sd-turbo"

[inputs]
seed = { type = "int", default = -1 }
width = 512
height = 512
num_steps = 4

[[models]]
path = "text_encoder/model.onnx"
[[models]]
path = "unet/model.onnx"
[[models]]
path = "vae_decoder/model.onnx"

[[stages]]
type = "tokenize"
text = "prompt"

[[stages]]
model = "text_encoder"

[[stages]]
type = "randn"
shape = [1, 4, "height/8", "width/8"]

[[stages]]
loop = "num_steps"
  [[stages.stages]]
  model = "unet"
  sample = "latents"
  timestep = "timesteps[i]"

  [[stages.stages]]
  type = "euler_step"

[[stages]]
model = "vae_decoder"
latent_sample = "latents / 0.18215"
```

### Whisper

```toml
name = "whisper"

[inputs]
audio_path = { type = "string" }
max_tokens = 448

[[models]]
path = "encoder.onnx"
[[models]]
path = "decoder.onnx"

[[stages]]
type = "audio.load"
path = "audio_path"
sample_rate = 16000

[[stages]]
type = "mel_spectrogram"
n_mels = 80

[[stages]]
model = "encoder"

[[stages]]
type = "init_sequence"
tokens = [50258, 50259]  # <|startoftranscript|><|en|>

[[stages]]
loop = "max_tokens"
break_on = "next_token == 50257"
  [[stages.stages]]
  model = "decoder"

  [[stages.stages]]
  type = "sample_greedy"
  logits = "logits[:, -1, :]"

  [[stages.stages]]
  type = "append"
  token = "next_token"

[[stages]]
type = "decode"
tokenizer = "whisper"
```

### LLM (Phi-2)

```toml
name = "phi-2"

[inputs]
max_tokens = 128
temperature = 0.7
top_p = 0.95

[[models]]
path = "phi-2.onnx"

[[stages]]
type = "tokenize"
tokenizer = "microsoft/phi-2"
text = "prompt"

[[stages]]
type = "init_kv_cache"
layers = 32
heads = 32

[[stages]]
loop = "max_tokens"
break_on = "next_token == eos"
  [[stages.stages]]
  model = "phi-2"
  input_ids = "i == 0 ? tokens : next_token"
  past_key_values = "cache"

  [[stages.stages]]
  type = "sample_top_p"
  temperature = "temperature"
  top_p = "top_p"

[[stages]]
type = "decode"
tokenizer = "microsoft/phi-2"

[output_handlers.text]
streaming = true
```

---

## 10. Implementation Phases

### Phase 1: Core Infrastructure

- [ ] Project structure and build system
- [ ] Core types: `Value`, `TensorBuffer`, `Error`
- [ ] Expression lexer, parser, AST
- [ ] Expression evaluator with functions
- [ ] Config parser (TOML/YAML)

### Phase 2: Compilation Engine

- [ ] ONNX loader
- [ ] IR representation
- [ ] Basic optimization passes
- [ ] CPU backend
- [ ] Output format (.pipe + .weights)

### Phase 3: Runtime Engine

- [ ] Pipeline executor
- [ ] Execution context
- [ ] Stage dispatch (Model/Builtin/Loop)
- [ ] Memory management
- [ ] Streaming execution mode

### Phase 4: Built-in Operations

- [ ] Common ops (math, tensor manipulation)
- [ ] Image ops (randn, resize, normalize)
- [ ] Diffusion ops (schedulers, timesteps)
- [ ] Audio ops (mel spectrogram, STFT)
- [ ] Text ops (tokenize, decode, sampling)

### Phase 5: Output Handlers

- [ ] Image handler (PNG, JPEG, WebP)
- [ ] Audio handler (WAV, MP3)
- [ ] Text handler (plain, streaming)
- [ ] Tensor handler (NPZ, safetensors)

### Phase 6: CLI and Polish

- [ ] CLI commands: compile, run, info, validate
- [ ] Progress reporting
- [ ] Error messages
- [ ] Documentation

### Phase 7: Advanced Features

- [ ] GPU backends (CUDA, Metal)
- [ ] Quantization support
- [ ] Operator fusion
- [ ] Performance profiling

---

## 11. Critical Files to Create

| File                                        | Purpose                   |
| ------------------------------------------- | ------------------------- |
| `crates/onnx-pipe-core/src/value.rs`        | Runtime value types       |
| `crates/onnx-pipe-expr/src/parser.rs`       | Expression parser         |
| `crates/onnx-pipe-expr/src/evaluator.rs`    | Expression evaluation     |
| `crates/onnx-pipe-config/src/schema.rs`     | Config schema definitions |
| `crates/onnx-pipe-runtime/src/executor.rs`  | Pipeline executor         |
| `crates/onnx-pipe-builtins/src/registry.rs` | Built-in op registry      |
| `crates/onnx-pipe-handlers/src/image.rs`    | Image output handler      |

---

## 12. Key Design Decisions

1. **Separate compilation and execution configs** - Compilation is model-specific, execution is pipeline-specific
2. **Expression system is limited** - Not a full language, just enough for config flexibility
3. **Streaming by default for large models** - Memory-bounded execution is critical
4. **Trait-based extensibility** - BuiltinOp and OutputHandler traits for plugins
5. **Zero-copy where possible** - Memory-mapped weights, Arc<Vec<f32>> for tensors
6. **No model-specific code** - Everything driven by config, works with any ONNX model

---

## 13. Getting Started Instructions

When starting implementation:

1. **Phase 1**: Create the workspace with all 8 crates, implement core Value types and Error handling
2. **Phase 2**: Build the expression lexer/parser/evaluator with tests for the grammar
3. **Phase 3**: Implement TOML/YAML config parsing with the simplified schema
4. **Phase 4**: Create the pipeline executor shell that can run builtin-only configs
5. **Phase 5**: Add ONNX loading and model execution support
6. **Phase 6**: Implement the most critical builtins (tokenize, randn, scheduler steps)
7. **Phase 7**: Add output handlers (image first, then audio/text)

Start with Phase 1 by creating the Cargo workspace and the onnx-pipe-core crate with Value, TensorBuffer, DType, and Error types.
