# Plan 33: Transformer Operations for T5 Support

## Status: RESOLVED

T5-small encoder-decoder generation is fully working as of commit `2e28e0d`. The model produces coherent text for translation, summarization, and question answering tasks.

### Verified Capabilities

- **Translation**: "translate English to German: Hello, how are you?" → "Hallo, wie sind Sie?"
- **Question Answering**: "What is the capital of France?" → "Paris"
- **Summarization**: Produces coherent summaries of input text

## Resolution Summary

All required operations are now supported through a combination of:
1. **Runtime execution** - hologram backend supports the math operations
2. **Compile-time constant folding** - Shape computation chains fold to constants

### Operation Status (Updated)

| Operation           | Count | Status                        |
| ------------------- | ----- | ----------------------------- |
| Constant            | 156   | Supported                     |
| MatMul              | 48    | Supported                     |
| Add                 | 35    | Supported                     |
| Unsqueeze           | 30    | Supported + constant folding  |
| Mul                 | 29    | Supported                     |
| Concat              | 25    | Supported + constant folding  |
| Reshape             | 25    | Supported                     |
| Transpose           | 25    | Supported                     |
| Cast                | 24    | Supported + constant folding  |
| Div                 | 15    | Supported                     |
| Pow                 | 13    | Supported (runtime)           |
| ReduceMean          | 13    | Supported                     |
| Sqrt                | 13    | Supported (runtime)           |
| Gather              | 10    | Supported + constant folding  |
| Shape               | 9     | Constant-folded               |
| Relu                | 6     | Supported                     |
| Softmax             | 6     | Supported                     |
| Sub                 | 2     | Supported                     |
| Abs                 | 1     | Supported (runtime)           |
| ConstantOfShape     | 1     | Constant-folded               |
| Greater             | 1     | Constant-folded               |
| Less                | 1     | Constant-folded               |
| Log                 | 1     | Supported (runtime)           |
| Min (element-wise)  | 1     | Supported (runtime)           |
| Range               | 1     | Constant-folded               |
| Where               | 1     | Constant-folded               |

## Key Implementation Details

### Position Bias Bucket Computation (Constant Folding)

T5's relative position bias uses this chain to compute attention bucket indices:
```
Range → Unsqueeze → Sub → Abs → Less → Where → Add → Gather
```

All operations in this chain are constant-folded at compile time because:
1. `Range` generates position indices [0, 1, 2, ..., seq_len-1]
2. The entire computation depends only on sequence length (known at compile time)
3. Each op propagates constants forward

**Implementation files:**
- [comparison.rs](../../crates/hologram-ai-onnx/src/ops/comparison.rs) - `WhereOp::try_fold()` with 3-input broadcasting
- [gather.rs](../../crates/hologram-ai-onnx/src/ops/gather.rs) - Multi-dimensional index support for [512, 512] indices
- [range.rs](../../crates/hologram-ai-onnx/src/ops/range.rs) - Sequence generation with step
- [cast.rs](../../crates/hologram-ai-onnx/src/ops/cast.rs) - Type conversion constant folding

### LayerNorm Decomposition (Runtime)

T5's RMSNorm uses this chain:
```
ReduceMean → Sub → Pow → ReduceMean → Add → Sqrt → Div → Mul → Add
```

These operations execute at runtime via hologram backend kernels:
- `Pow`, `Sqrt`, `Log`, `Abs` - Element-wise math
- `ReduceMean` - Reduction operation
- `Add`, `Sub`, `Mul`, `Div` - Binary arithmetic

---

## Original Analysis (Historical)

### Context

Hologram currently supports CNN architectures (ResNet18 compiles and runs successfully), but transformer models like T5 require additional operations in hologram's `OpKind` enum.

This document specifies the operations needed to support T5 and similar transformer architectures.

### T5-small Encoder Analysis

Analyzed the T5-small encoder model (492 operations total).

### Required Operations

#### 1. Element-wise Math Functions (High Priority)

Used extensively in LayerNormalization decomposition (`ReduceMean → Sub → Pow → ReduceMean → Add → Sqrt → Div → Mul → Add`):

```rust
/// Element-wise power: output[i] = base[i] ^ exponent[i]
Pow,

/// Element-wise square root: output[i] = sqrt(input[i])
Sqrt,

/// Element-wise natural logarithm: output[i] = ln(input[i])
Log,

/// Element-wise exponential: output[i] = e^input[i]
Exp,

/// Element-wise absolute value: output[i] = |input[i]|
Abs,
```

#### 2. Comparison Operations (Medium Priority)

Used for attention masking. Output dtype is `Bool`:

```rust
/// Element-wise greater than: output[i] = (a[i] > b[i])
Greater,

/// Element-wise less than: output[i] = (a[i] < b[i])
Less,

/// Element-wise equality: output[i] = (a[i] == b[i])
Equal,
```

#### 3. Conditional Selection (High Priority)

Critical for attention masking in transformers:

```rust
/// Conditional element selection: output[i] = condition[i] ? x[i] : y[i]
/// Input 0: condition tensor (Bool)
/// Input 1: x tensor (values when true)
/// Input 2: y tensor (values when false)
Where,
```

#### 4. Element-wise Min/Max (Low Priority)

Distinct from reduction operations - operates on two tensors:

```rust
/// Element-wise minimum: output[i] = min(a[i], b[i])
ElemMin,

/// Element-wise maximum: output[i] = max(a[i], b[i])
ElemMax,
```

#### 5. Shape/Constant Generation

These can potentially be constant-folded at compile time when inputs are static:

```rust
/// Returns the shape of input tensor as 1D int64 tensor
Shape,

/// Generates sequence [start, limit) with step
Range,

/// Creates tensor of given shape filled with a constant value
ConstantOfShape { value: f32 },
```

### Alternative: Compile-Time Constant Folding

For `Shape`, `Range`, and `ConstantOfShape`, if all inputs are compile-time constants, the frontend could fold these to `Constant` nodes during graph translation. This requires:

1. Tracking which nodes are constants during graph building
2. Evaluating these ops at compile time when inputs are constant
3. Emitting the result as a new `Constant` node

This approach avoids adding runtime operations for what are essentially compile-time computations.

**Note: This approach was successfully implemented and is the primary mechanism for T5 position bias computation.**

### Backend Implementation Notes

#### CPU Backend Kernels

For the CPU backend, these operations have straightforward SIMD implementations:

```rust
// Pow: Can use libm or approximate with exp(y * ln(x))
// Sqrt: _mm256_sqrt_ps / vsqrtq_f32
// Log: Can use libm or polynomial approximation
// Exp: Can use libm or polynomial approximation
// Abs: _mm256_andnot_ps with sign mask / vabsq_f32
// Greater/Less: _mm256_cmp_ps / vcgtq_f32, vcltq_f32
// Where: _mm256_blendv_ps / vbslq_f32
```

#### Shape Inference

All new operations have straightforward shape inference:
- Unary ops (`Sqrt`, `Log`, `Exp`, `Abs`): output shape = input shape
- Binary ops (`Pow`, `Greater`, `Less`, `ElemMin`, `ElemMax`): broadcast shapes
- `Where`: broadcast all three input shapes
- `Shape`: output is 1D with length = input rank
- `Range`: output is 1D with length = ceil((limit - start) / delta)
- `ConstantOfShape`: output shape = input shape tensor values

### Testing

Each operation should have tests for:
1. Scalar inputs
2. 1D vectors
3. Multi-dimensional tensors
4. Broadcasting behavior (for binary ops)
5. Edge cases (negative inputs for Sqrt/Log, zero for Pow, etc.)

## References

- T5 Paper: https://arxiv.org/abs/1910.10683
