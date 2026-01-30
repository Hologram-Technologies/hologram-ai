# Hologram Compiler Bug: Corrupted dims Array in BackendOp

## Issue Summary

The hologram compiler is generating corrupted `dims` arrays in `BackendOp.params.dims` during T5 decoder compilation. The corrupted values cause runtime buffer overflow errors, blocking T5 text generation.

**Corrupted dims observed**: `[65536, 65536, 65536, 0]` where 65536 = 0x10000 (appears to be uninitialized/sentinel value)

## Quick Reproduction

```bash
# 1. Compile T5 decoder ONNX model
cargo run --release -p hologram-ai -- compile \
  /workspace/models/t5-small/decoder_model.onnx \
  --output /tmp/decoder.holo

# 2. Run inference (fails at OP[97])
cargo run --release -p hologram-ai -- run \
  --config examples/T5/t5.toml \
  --prompt "translate English to French: Hello"
```

**Error**:
```
ERROR: Kernel execution failed at OP[97] kernel=KernelId(769)
dims=[65536, 65536, 65536, 0]
InvalidConfiguration: input[1] size 262144 bytes exceeds workspace region
'workspace_22' allocation of 2048 bytes
```

## Evidence

**Expected**: `dims[0] = 65536` is correct (1D tensor with 65536 elements)
**Actual**: `dims = [65536, 65536, 65536, 0]` - elements 1-3 are corrupted
**Impact**: Backend calculates buffer size as 65536³ instead of 65536¹

Multiple operations show this pattern:
- Gather operation (KernelId 769) at OP[97]: `[65536, 65536, 65536, 0]` ← **Failure point**
- Activation operations throughout: `[65536, 0, 0, 0]`

## Root Cause Hypothesis

The `BackendOp.params.dims: [usize; 4]` array appears to not be fully initialized:

1. **dims[0]** = 65536 (correct tensor size)
2. **dims[1], dims[2], dims[3]** = uninitialized or incorrectly defaulted

Likely location: `/hologram/crates/compiler/src/pipeline/mod.rs` in `build_plan_ops()` where BackendOp is created.

## Suggested Investigation

```bash
# 1. Find where dims arrays are initialized
grep -r "\.dims = " /hologram/crates/compiler/src/pipeline/
grep -r "65536" /hologram/crates/compiler/

# 2. Check Gather operation (KernelId 769) dims handling
grep -A 20 "KernelId::GATHER" /hologram/crates/compiler/src/

# 3. Add logging to track dims values during compilation
# In build_plan_ops():
tracing::info!("OP[{}] kernel={:?} dims={:?}", op_idx, kernel_id, dims);
```

## Questions for Hologram Team

1. Where in the compiler pipeline are `BackendOp.params.dims` values set?
2. Is there a default value of `65536` used anywhere?
3. Should dims always be fully initialized to `[dim0, dim1, dim2, dim3]` or `[size, 1, 1, 1]`?
4. Are there validation checks that dims elements are non-zero and reasonable?

## Impact

- ✅ **T5 encoder**: Works correctly
- ❌ **T5 decoder**: Compiles but fails at runtime (OP[97])
- ❌ **T5 text generation**: Completely blocked

This likely affects all large transformer models (BERT, GPT, T5, LLaMA, etc.) that use Gather operations with symbolic shapes.

## Additional Context

**Full bug report**: `/workspace/specs/hologram-compiler-dims-bug-report.md`

**Previous workspace allocation investigation**: We initially thought this was a workspace sizing issue, but after extensive debugging, we've traced it to corrupted dims values being written during compilation.

**Test model**: T5-small decoder from Hugging Face (897 ONNX nodes, dynamic shapes)

## Request

Can you help identify where in the compiler pipeline the dims array should be properly initialized? We're happy to test any patches or provide additional debugging output.

---

**Contact**: hologram-ai integration team
**Date**: 2026-01-27
