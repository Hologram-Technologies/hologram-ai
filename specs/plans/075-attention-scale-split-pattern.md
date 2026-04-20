# Plan 075: Fix M>1 Prefill — Transpose + Sin/Cos Swap

## Context

Qwen2-0.5B compiled and run with M>1 prompt produces wrong output. M=1
(single token) matches ORT perfectly. Root cause investigation revealed
TWO bugs, not the originally hypothesized scale issue.

## Bug 1: InlineTranspose 0-sentinel passthrough (FIXED)

**Commit:** `833b7f5` in hologram base.

When `InlineTranspose` had a 0-sentinel in its compiled `input_shape`,
`compiled_elems = 0` caused the transpose to fall through to a passthrough
(identity copy). For M=1 this was invisible (transposing dim=1 is a no-op)
but for M>1 the flat memory layout requires an actual transpose.

**Fix:** When `compiled_elems == 0`, resolve the 0-sentinel from
`input_elems / product_of_nonzero_dims`. For shape `[32, 0]` with 160
input elements, the 0-dim resolves to 5.

**Effect:** RoPE frequency computation now correct. Output improved from
complete gibberish to real words. TinyLlama regression-free.

## Bug 2: Sin/Cos output swap (OPEN)

After the Transpose fix, the RoPE `Concat` produces correct position ×
frequency values (row 0 = all zeros for position 0). But the downstream
Sin and Cos nodes produce **swapped outputs**:

- Instruction 16 (labeled "Sin", output 318): produces `[1, 1, 1, ...]` = cos(0)
- Instruction 17 (labeled "Cos", output 319): produces `[0, 0, 0, ...]` = sin(0)

The Sin kernel computes cos and vice versa. Downstream Mul nodes that
expect `Q × cos + rotate(Q) × sin` instead get `Q × sin + rotate(Q) × cos`,
corrupting the RoPE rotation.

### Investigation

The swap could be caused by:

1. **Reshape nodes between Concat and Sin/Cos** (instructions 21-22) that
   reorder buffer indices, causing the wrong data to reach each kernel
2. **Graph wiring issue** during lowering — output tensor IDs 318/319 are
   assigned to the wrong downstream consumers
3. **ONNX import** mapping Sin/Cos ops to the wrong tensor IDs

### Diagnosis approach

1. Check the ONNX graph: which tensor name does the Sin node output vs the
   Cos node? Verify the ONNX importer preserves the mapping.
2. Check the lowering: do the AiOp::Sin and AiOp::Cos nodes get assigned
   the correct output tensor IDs?
3. Check the Reshape nodes between Concat and Sin/Cos — are they introducing
   a layout change that effectively swaps the data?
4. Add an ORT conformance test: `sin(x)` and `cos(x)` for a small tensor,
   verify hologram's dispatch produces matching results.

### Fix

Once the swap source is identified, the fix is one of:
- Swap the output tensor ID assignment in the lowering
- Fix the Reshape handling between Concat and Sin/Cos
- Swap the Sin/Cos kernel implementations (unlikely — they're just `f32::sin/cos`)

## Verification

After both fixes:
- Qwen2 M=5 "The capital of France is" should produce " Paris" as top-1
  (ORT reference: token 12095)
- Qwen2 M=1 should remain correct
- TinyLlama should remain correct (already verified for Bug 1 fix)

## Key Files

| File | Bug | Status |
|------|-----|--------|
| `hologram/crates/hologram-exec/src/tape.rs:1282` | Bug 1 (Transpose) | FIXED |
| `hologram-ai-common/src/lower/` or `hologram-ai-onnx/src/` | Bug 2 (Sin/Cos) | OPEN |
| `hologram-ai/tests/qwen2_e2e.rs` | Conformance test | Needs M=5 logit check |
