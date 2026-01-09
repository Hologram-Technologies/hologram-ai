# T5 End-to-End Debugging - Continuation Prompt

## Status Summary

**Goal**: Run T5 text generation pipeline end-to-end with hologram backend.

**Current State**: Encoder produces NaN values causing crash. Two bugs were fixed, one remains.

---

## Bugs Fixed

### 1. UnaryOp Neg/Abs Incorrectly Mapped to Binary ELEM_MUL Kernel

**Location (old)**: `/hologram/crates/compiler/src/pipeline.rs` lines 764-766

**Problem**: `UnaryOp::Neg` and `UnaryOp::Abs` were mapped to `KernelId::ELEM_MUL` (a binary kernel requiring 2 inputs), but unary operations only have 1 input.

**Fix Applied**:
1. Added `UNARY_NEG` (0x0509) and `UNARY_ABS` (0x050A) to `/hologram/crates/backend/src/plan.rs`
2. Changed mapping in pipeline.rs:
   ```rust
   UnaryOp::Neg => KernelId::UNARY_NEG,
   UnaryOp::Abs => KernelId::UNARY_ABS,
   ```
3. Added `neg_kernel_f32` and `abs_kernel_f32` implementations in `/hologram/crates/backend/src/cpu/kernels.rs`
4. Added dispatch entries in the transcendental kernel table

**Status**: ✅ Fixed

---

### 2. LayerNorm Epsilon Parameter Not Passed to Kernel

**Location (old)**: `/hologram/crates/compiler/src/pipeline.rs` ~line 2100

**Problem**: The `params.extra[0]` field was not being set for LayerNorm/RMSNorm operations, causing the kernel to read epsilon=0.

**Fix Applied**:
```rust
// For LayerNorm
if let Some(OpNode::LayerNorm { epsilon, normalized_shape }) = graph.get_op(node_idx) {
    let norm_size: usize = normalized_shape.iter().product::<usize>().max(1);
    params.dims[0] = 128;  // batch_size
    params.dims[1] = norm_size;
    params.extra[0] = epsilon.to_bits() as usize;
}

// Similar for RMSNorm
```

**Status**: ✅ Fixed (but may not apply to T5 since T5 uses decomposed RMS norm via Add/Sqrt/Div)

---

## Remaining Bug: NaN Values in T5 Encoder

### Symptoms

Running:
```bash
cargo run --release -- run --config configs/t5-generate.toml
```

Produces:
```
reshape_kernel_f32: first_10=[12.062562, 10.732231, ...]  ← Good initial values
reshape_kernel_f32: first_10=[0.0, 0.0, 0.0, ...]        ← Zeros (padding positions)
reshape_kernel_f32: first_10=[NaN, NaN, NaN, ...]        ← NaN appears
reshape_kernel_f32: first_10=[inf, inf, inf, ...]        ← Then inf
```

Then crashes with SIGSEGV (exit code 139).

### Root Cause Analysis

T5 does NOT use a dedicated LayerNorm kernel. It uses **decomposed RMS normalization**:

T5 ONNX operations for RMS norm:
1. `ReduceMean(x²)` → variance
2. `Add(variance, epsilon)` → variance + ε (epsilon ≈ 1e-6 as ONNX constant)
3. `Sqrt(variance + ε)` → standard deviation
4. `Div(x, std)` → normalized output
5. `Mul(normalized, gamma)` → scaled output

**The problem**: When `x` is all zeros (padding positions):
- `mean(x²) = 0`
- If the epsilon constant is not correctly handled: `sqrt(0 + 0) = 0`
- `x / 0 = NaN` (0/0 is NaN for floats)

### Investigation Needed

1. **Check epsilon constant handling**: How are ONNX constants being embedded in the compiled plan?
2. **Trace the Add operation**: Is the epsilon constant being correctly added to the variance?
3. **Check Div kernel**: Does it handle division by very small numbers?

### T5 ONNX Operations

The T5 encoder uses these operations (no LayerNorm op):
```
Abs, Add, Cast, Concat, Constant, ConstantOfShape, Div, Gather, Greater,
Less, Log, MatMul, Min, Mul, Pow, Range, ReduceMean, Relu, Reshape,
Shape, Softmax, Sqrt, Sub, Transpose, Unsqueeze, Where
```

---

## File Mappings (Old → New Structure)

After restructure to hologram-ai workspace per `/workspace/specs/plans/24-gguf-safetensors-support.md`:

| Old Path | New Path |
|----------|----------|
| `/workspace/src/core/parser.rs` | `/workspace/crates/hologram-ai-onnx/src/parser.rs` |
| `/workspace/src/core/translator.rs` | `/workspace/crates/hologram-ai-onnx/src/translator.rs` |
| `/workspace/src/translators/` | `/workspace/crates/hologram-ai-onnx/src/translators/` |
| `/workspace/src/cli/run.rs` | `/workspace/crates/hologram-ai/src/cli/run.rs` |
| `/workspace/src/runtime/` | `/workspace/crates/hologram-ai/src/runtime/` |
| `/workspace/configs/t5-generate.toml` | `/workspace/configs/t5-generate.toml` (unchanged) |
| `/workspace/models/t5-small/` | `/workspace/models/t5-small/` (unchanged) |

Hologram files (unchanged, external dependency):
- `/hologram/crates/compiler/src/pipeline.rs`
- `/hologram/crates/backend/src/cpu/kernels.rs`
- `/hologram/crates/backend/src/plan.rs`

---

## Reproduction Steps

1. Ensure T5 models exist:
   ```bash
   ls models/t5-small/encoder_model.onnx
   ls models/t5-small/decoder_model.onnx
   ls models/t5-small/tokenizer.json
   ```

2. Compile models (after any hologram changes):
   ```bash
   cargo run --release -- compile models/t5-small/encoder_model.onnx -o models/t5-small/compiled/encoder.holo
   cargo run --release -- compile models/t5-small/decoder_model.onnx -o models/t5-small/compiled/decoder.holo
   ```

3. Run pipeline:
   ```bash
   cargo run --release -- run --config configs/t5-generate.toml
   ```

4. Observe NaN values in log output and SIGSEGV crash.

---

## Config File Reference

`configs/t5-generate.toml`:
```toml
name = "T5 Text Generation"
description = "End-to-end text generation with T5"

[inputs]
prompt = "tell me a joke"

[tokenizer]
type = "sentencepiece"
vocab_path = "models/t5-small/tokenizer.json"
max_length = 512
pad_token_id = 0
eos_token_id = 1

[models.encoder]
path = "../models/t5-small/encoder_model.onnx"
precompiled = "../models/t5-small/compiled/encoder.holo"

[models.decoder]
path = "../models/t5-small/decoder_model.onnx"
precompiled = "../models/t5-small/compiled/decoder.holo"

[[stages]]
type = "builtin"
builtin = "tokenize"
outputs = ["input_ids", "attention_mask"]
# ... etc
```

---

## Next Steps to Debug

1. **Add debug logging to Div kernel** in `/hologram/crates/backend/src/cpu/kernels.rs`:
   - Log when divisor is very small (< 1e-6)
   - Log when result is NaN or inf

2. **Trace constant handling** in `/hologram/crates/compiler/src/pipeline.rs`:
   - How are ONNX Constant nodes converted to BufferRef::Constant?
   - Are small float constants (like 1e-6 epsilon) being preserved correctly?

3. **Add numerical stability** to division:
   ```rust
   // In div_kernel_f32:
   let divisor = b[i];
   if divisor.abs() < 1e-12 {
       out[i] = 0.0;  // or clamp to safe value
   } else {
       out[i] = a[i] / divisor;
   }
   ```

4. **Compare with PyTorch/ONNX Runtime**:
   - Run same T5 model in ONNX Runtime
   - Compare intermediate tensor values at each layer
   - Identify exact point of divergence

---

## Useful Debug Commands

```bash
# Run with full trace
RUST_LOG=hologram_backend=debug cargo run --release -- run --config configs/t5-generate.toml 2>&1 | tee debug.log

# Search for NaN appearance
grep -n "NaN\|inf" debug.log | head -20

# Check operation that produces first NaN
grep -B5 "first_10=\[NaN" debug.log | head -20
```

---

## PRIORITY: Debug Constant Handling in Add Operation

### The Core Problem

T5's RMS normalization is decomposed into primitive ops. The epsilon constant (1e-6) must be added to the variance before taking sqrt:

```
variance = ReduceMean(x²)           # Returns 0 for padding positions
safe_var = Add(variance, epsilon)   # Should be 0 + 1e-6 = 1e-6
std = Sqrt(safe_var)                # Should be sqrt(1e-6) ≈ 0.001
normalized = Div(x, std)            # Should be 0 / 0.001 = 0, NOT NaN
```

**But we're getting NaN**, which means `safe_var = 0` (epsilon not added), so `std = 0`, and `0/0 = NaN`.

### Where to Look in Hologram

#### 1. Constant Embedding in Compiler

**File**: `/hologram/crates/compiler/src/pipeline.rs`

Search for how `Constant` nodes become `BufferRef::Constant`:

```rust
// Look for code that handles OpNode::Constant or similar
// Check if small float values (1e-6) are being correctly serialized
```

Key questions:
- Are constants being stored as f32 or f64?
- Is there any truncation happening for very small values?
- How is the constant offset/size calculated?

#### 2. Add Kernel Implementation

**File**: `/hologram/crates/backend/src/cpu/kernels.rs`

Find the Add kernel (likely `add_kernel_f32` or element-wise add):

```rust
// Add debug logging:
pub fn add_kernel_f32(inputs: &[*const u8], outputs: &[*mut u8], params: &KernelParams) -> Result<(), BackendError> {
    let a = unsafe { std::slice::from_raw_parts(inputs[0] as *const f32, size) };
    let b = unsafe { std::slice::from_raw_parts(inputs[1] as *const f32, size) };
    let out = unsafe { std::slice::from_raw_parts_mut(outputs[0] as *mut f32, size) };

    // DEBUG: Log when one input is very small (likely epsilon)
    let b_max = b.iter().cloned().fold(0.0f32, f32::max);
    let b_min = b.iter().cloned().fold(f32::MAX, f32::min);
    if b_max < 1e-3 && b_max > 0.0 {
        eprintln!("ADD: Small constant detected: min={}, max={}", b_min, b_max);
        eprintln!("ADD: First input values: {:?}", &a[..10.min(a.len())]);
    }

    for i in 0..size {
        out[i] = a[i] + b[i];
    }
    Ok(())
}
```

#### 3. Constant Buffer Resolution

**File**: `/hologram/crates/backend/src/executor.rs`

Check how `BufferRef::Constant { offset, size }` is resolved to actual data:

```rust
// Look for resolve_buffer_ref or similar function
// Verify that:
// 1. offset is correct
// 2. size matches expected bytes (4 for f32)
// 3. Data is being read as f32, not misinterpreted
```

### Specific Debugging Steps

#### Step 1: Log All Add Operations with Small Constants

In `/hologram/crates/backend/src/cpu/kernels.rs`, find the add kernel and add:

```rust
// At start of add kernel:
let input_b_first = unsafe { *(inputs[1] as *const f32) };
if input_b_first.abs() < 1e-3 && input_b_first != 0.0 {
    eprintln!("[ADD DEBUG] Small constant input: {}", input_b_first);
    eprintln!("[ADD DEBUG] Params: dims={:?}", params.dims);
}
```

#### Step 2: Check Constant Data at Compile Time

In `/hologram/crates/compiler/src/pipeline.rs`, when processing constants:

```rust
// When serializing a Constant node:
if let Some(OpNode::Constant { data, .. }) = graph.get_op(node_idx) {
    let f32_vals: Vec<f32> = data.chunks(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    if f32_vals.iter().any(|&v| v.abs() < 1e-3 && v != 0.0) {
        eprintln!("[CONST DEBUG] Small constant values: {:?}", &f32_vals[..10.min(f32_vals.len())]);
    }
}
```

#### Step 3: Verify Constant Is Being Used

Add logging to the executor when resolving constants:

```rust
// In resolve_buffer_ref for Constant case:
BufferRef::Constant { offset, size } => {
    let ptr = constant_base.add(offset);
    // Debug: read and log the value
    if size == 4 {
        let val = unsafe { *(ptr as *const f32) };
        if val.abs() < 1e-3 && val != 0.0 {
            eprintln!("[RESOLVE DEBUG] Small constant at offset {}: {}", offset, val);
        }
    }
    ptr
}
```

### Expected vs Actual Behavior

**Expected**:
```
[ADD DEBUG] Small constant input: 1e-6 (or similar)
[ADD] Output after adding epsilon: min=1e-6, values_near_zero=0
[SQRT] Input range: [1e-6, ...]
[DIV] Divisor range: [0.001, ...] (sqrt of 1e-6)
```

**Actual (buggy)**:
```
[ADD] No small constant detected (epsilon missing!)
[SQRT] Input range: [0.0, ...]
[DIV] Divisor = 0 → NaN
```

### Possible Root Causes

1. **Constant node not being translated**: The ONNX `Constant` op with epsilon value might not be making it into the hologram IR.

2. **Broadcasting issue**: Epsilon might be a scalar (shape []) but variance is [batch, seq_len]. The Add might not be broadcasting correctly.

3. **Wrong constant offset**: The `BufferRef::Constant { offset, size }` might point to wrong data.

4. **Byte order issue**: Epsilon might be stored as big-endian but read as little-endian, corrupting the value.

5. **Size mismatch**: If epsilon is stored as f64 but read as f32, or vice versa.

### Quick Test: Force Epsilon in Sqrt Kernel

As a temporary workaround to verify the diagnosis, modify the sqrt kernel:

```rust
// In sqrt_kernel_f32:
pub fn sqrt_kernel_f32(...) {
    for i in 0..size {
        // Clamp input to minimum epsilon before sqrt
        let safe_val = input[i].max(1e-12);
        output[i] = safe_val.sqrt();
    }
}
```

If this fixes the NaN issue, it confirms that epsilon is not being added in the Add operation.

---

## Summary for Continuation

After restructure:
1. T5 models and configs remain in same locations
2. ONNX code moves to `crates/hologram-ai-onnx/`
3. Runtime/CLI moves to `crates/hologram-ai/`
4. Hologram dependency unchanged
5. The NaN bug is in hologram's handling of small constants or division, not in the ONNX translation layer
6. Focus debugging on the Div kernel and constant embedding in hologram compiler

---

## Chat Continuation Prompt

Copy-paste this into a new chat session to continue debugging:

```
## Continuation Prompt: T5 NaN Debugging in Hologram

### Context

We're debugging T5 text generation on the hologram backend. Two bugs were fixed:
1. **UnaryOp Neg/Abs mapped to binary ELEM_MUL** - Fixed with UNARY_NEG/UNARY_ABS kernels
2. **LayerNorm epsilon not passed** - Fixed by setting params.extra[0]

### Current Bug: NaN in Decomposed RMS Norm

T5 uses decomposed RMS normalization (not a dedicated LayerNorm op):

    ReduceMean(x²) → Add(variance, epsilon) → Sqrt → Div

For padding positions (zeros), variance=0. The epsilon constant (1e-6) should make
sqrt(0 + 1e-6) ≈ 0.001, but instead we get sqrt(0) = 0, causing 0/0 = NaN.

**The epsilon constant is not being correctly added.**

### Observed Output

    reshape_kernel_f32: first_10=[12.06, 10.73, ...]  ← Good
    reshape_kernel_f32: first_10=[0.0, 0.0, ...]      ← Padding zeros
    reduce_mean_kernel_f32: ...                        ← variance=0
    reshape_kernel_f32: first_10=[NaN, NaN, ...]      ← NaN appears!

### Files to Debug in Hologram

1. `/hologram/crates/compiler/src/pipeline.rs` - constant embedding
2. `/hologram/crates/backend/src/cpu/kernels.rs` - add kernel, sqrt kernel
3. `/hologram/crates/backend/src/executor.rs` - BufferRef::Constant resolution

### Likely Root Causes

- Scalar constant not broadcasting correctly in Add
- Wrong offset in BufferRef::Constant
- Constant not translated from ONNX to hologram IR

### Quick Fix to Verify

Clamp sqrt input: `output[i] = input[i].max(1e-12).sqrt()` - if this fixes NaN,
confirms epsilon not being added.

### Reproduction

    cargo run --release -- run --config configs/t5-generate.toml

### Reference Doc

/workspace/specs/plans/25-t5-nan-debugging.md has full details.
```
