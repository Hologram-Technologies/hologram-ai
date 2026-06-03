# int8 Weight Quantization (Baseline) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compile-time per-channel symmetric int8 quantization of linear/matmul weights, producing a ~4× smaller `.holo` that runs (near-losslessly) through the already-wired `Dequantize→MatMul` (`matmul_dequant`) path.

**Architecture:** A `no_std` encoder in `hologram-ai-quant` turns an f32 `[k,n]` weight into i8 bytes + per-column f32 scales. A graph pass in `hologram-ai-common` (run between optimization and lowering, gated on `QuantStrategy::Int8`) rewrites each MatMul whose B operand is an f32 constant: it swaps the constant for an i8 constant and inserts an `AiOp::Dequantize { axis: 1 }` feeding B. The existing lowering emits `OpKind::Dequantize` (per-channel) and the runtime fuses it to `MatMulDequant` — which already runs on the SIMD/NEON/wasm matmul kernel. No backend kernel changes.

**Tech Stack:** Rust, `hologram-ai-common` (AiGraph IR + lowering), `hologram-ai-quant` (`no_std` quant), `hologram` (reused dequant kernels/fusion). Tests via `cargo test` and wasmtime.

Spec: `docs/superpowers/specs/2026-06-03-int8-weight-quantization-design.md`.

---

## File Structure

- `hologram-ai/crates/hologram-ai-quant/src/encode.rs` — **NEW**. Pure `encode_int8_per_channel` + unit tests.
- `hologram-ai/crates/hologram-ai-quant/src/lib.rs` — **MODIFY**. `pub mod encode;` + re-export.
- `hologram-ai/crates/hologram-ai-common/src/lower/quantize.rs` — **NEW**. `quantize_weights` graph pass + unit tests.
- `hologram-ai/crates/hologram-ai-common/src/lower/mod.rs` — **MODIFY**. `pub mod quantize;` + re-export; add `Int8`/`Int4` to `QuantStrategy`.
- `hologram-ai/crates/hologram-ai-common/src/lower/builder.rs` — **MODIFY** (only if `QuantStrategy` lives here): add `Int8`/`Int4` variants.
- `hologram-ai/crates/hologram-ai-common/Cargo.toml` — **MODIFY**. Add `hologram-ai-quant` dependency.
- `hologram-ai/crates/hologram-ai/src/compiler.rs` — **MODIFY**. Call `quantize_weights` between `pipeline.run()` (`:293`) and `lower()` (`:414`).
- `hologram-ai/crates/hologram-ai/src/cli.rs` — **MODIFY**. `parse_quant`: accept `none`/`int8`/`int4`.
- `hologram-ai/crates/hologram-ai/tests/int8_accuracy.rs` — **NEW**. f32-vs-int8 cosine ≥ 0.999 on a fixture.

---

## Task 0: De-risk — confirm the fusion assumption holds today

The whole plan relies on an AiGraph `Dequantize→MatMul` fusing to `MatMulDequant`. An existing test already builds exactly that graph. Confirm it passes before building anything.

**Files:** Read-only: `hologram-ai/crates/hologram-ai/tests/quantized_weight_memory.rs`.

- [ ] **Step 1: Run the existing fusion/accuracy test**

Run: `cargo test -p hologram-ai --test quantized_weight_memory -- --nocapture`
Expected: PASS, including the `dequant_matmul_fused_count() == 1` assertion. This proves an AiGraph `Dequantize{axis:-1}`+`MatMul` fuses and matches f32. (Our pass emits the same shape with a per-channel `axis: 1`.)

If it FAILS: stop and investigate the fusion before proceeding — the rest of the plan assumes it works.

---

## Task 1: int8 per-channel encoder (`hologram-ai-quant`)

**Files:**
- Create: `hologram-ai/crates/hologram-ai-quant/src/encode.rs`
- Modify: `hologram-ai/crates/hologram-ai-quant/src/lib.rs`
- Test: in `encode.rs` (`#[cfg(test)]`)

- [ ] **Step 1: Write the failing tests**

Create `hologram-ai/crates/hologram-ai-quant/src/encode.rs`:

```rust
//! f32 → per-channel symmetric int8 weight encoder (the inverse of the dequant
//! unpackers in this crate). `no_std`; pure arithmetic, no IR dependency.

use alloc::vec;
use alloc::vec::Vec;

/// Per-channel symmetric int8 encoding of a row-major `[k, n]` weight.
///
/// One scale per **column** (output channel `j`):
/// `scale_j = max_i |W[i,j]| / 127`. Returns `(q, scales)` where `q` is the
/// row-major `[k,n]` i8 weight and `scales` has length `n`. A column of all
/// zeros gets `scale = 1.0` so dequant reproduces zeros exactly.
pub fn encode_int8_per_channel(w: &[f32], k: usize, n: usize) -> (Vec<i8>, Vec<f32>) {
    assert_eq!(w.len(), k * n, "weight length must equal k*n");
    let mut scales = vec![1.0f32; n];
    for (j, scale) in scales.iter_mut().enumerate() {
        let mut amax = 0.0f32;
        for i in 0..k {
            let a = w[i * n + j].abs();
            if a > amax {
                amax = a;
            }
        }
        if amax > 0.0 {
            *scale = amax / 127.0;
        }
    }
    let mut q = vec![0i8; k * n];
    for i in 0..k {
        for j in 0..n {
            let v = (w[i * n + j] / scales[j]).round();
            let clamped = v.clamp(-127.0, 127.0);
            q[i * n + j] = clamped as i8;
        }
    }
    (q, scales)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_within_half_scale() {
        let (k, n) = (5usize, 3usize);
        let w: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.13 - 0.7).collect();
        let (q, scales) = encode_int8_per_channel(&w, k, n);
        assert_eq!(scales.len(), n);
        assert_eq!(q.len(), k * n);
        for i in 0..k {
            for j in 0..n {
                let deq = q[i * n + j] as f32 * scales[j];
                assert!(
                    (deq - w[i * n + j]).abs() <= scales[j] / 2.0 + 1e-6,
                    "elem ({i},{j}): deq {deq} vs {}",
                    w[i * n + j]
                );
            }
        }
    }

    #[test]
    fn zero_column_scale_one_and_exact_zero() {
        let (k, n) = (3usize, 2usize);
        // Column 1 is all zeros.
        let w = vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0];
        let (q, scales) = encode_int8_per_channel(&w, k, n);
        assert_eq!(scales[1], 1.0);
        for i in 0..k {
            assert_eq!(q[i * n + 1], 0);
        }
    }

    #[test]
    fn max_abs_maps_to_127() {
        // A column whose max-abs element should hit ±127 after rounding.
        let (k, n) = (2usize, 1usize);
        let w = vec![0.5f32, -1.0];
        let (q, scales) = encode_int8_per_channel(&w, k, n);
        assert_eq!(scales[0], 1.0 / 127.0);
        assert_eq!(q[1], -127);
    }
}
```

- [ ] **Step 2: Wire the module**

Modify `hologram-ai/crates/hologram-ai-quant/src/lib.rs` — add after the existing `pub mod` lines:

```rust
pub mod encode;
pub use encode::encode_int8_per_channel;
```

- [ ] **Step 3: Run tests to verify they pass**

Run: `cargo test -p hologram-ai-quant encode`
Expected: 3 tests PASS.

- [ ] **Step 4: Commit**

```bash
git add crates/hologram-ai-quant/src/encode.rs crates/hologram-ai-quant/src/lib.rs
git commit -m "feat(quant): per-channel symmetric int8 weight encoder"
```

---

## Task 2: `QuantStrategy::Int8`/`Int4` variants + CLI flag rename

**Files:**
- Modify: `hologram-ai/crates/hologram-ai-common/src/lower/builder.rs` (the `QuantStrategy` enum)
- Modify: `hologram-ai/crates/hologram-ai/src/cli.rs:122` (`parse_quant`)

- [ ] **Step 1: Add Int8/Int4 variants**

In `builder.rs`, extend the enum (keep existing variants to avoid breaking other matches):

```rust
pub enum QuantStrategy {
    None,
    Auto,
    Q4_0,
    Q8_0,
    Q2_0,
    /// Per-channel symmetric int8 weight quantization (this plan).
    Int8,
    /// Per-group int4 (not yet implemented; accepted by the parser, rejected
    /// by the quantize pass until the int4 spec lands).
    Int4,
}
```

- [ ] **Step 2: Update the CLI parser + help text**

In `cli.rs`, replace `parse_quant` and the arg help:

```rust
/// Weight quantization scheme: `none`/`f32`, `int8`, `int4`.
#[arg(long, value_name = "SCHEME")]
quantize: Option<String>,
```

```rust
fn parse_quant(s: Option<&str>) -> anyhow::Result<QuantStrategy> {
    Ok(match s.map(|s| s.to_ascii_lowercase()).as_deref() {
        None | Some("none") | Some("f32") => QuantStrategy::None,
        Some("int8") => QuantStrategy::Int8,
        Some("int4") => QuantStrategy::Int4,
        Some(other) => {
            anyhow::bail!("unknown quantization scheme {other:?} (expected none/int8/int4)")
        }
    })
}
```

- [ ] **Step 3: Build to verify it compiles**

Run: `cargo build -p hologram-ai`
Expected: builds clean (the new variants are additive; `parse_quant` no longer references the q*_0 strings).

- [ ] **Step 4: Commit**

```bash
git add crates/hologram-ai-common/src/lower/builder.rs crates/hologram-ai/src/cli.rs
git commit -m "feat(quant): add Int8/Int4 QuantStrategy + rename --quantize values"
```

---

## Task 3: `quantize_weights` graph pass (`hologram-ai-common`)

**Files:**
- Create: `hologram-ai/crates/hologram-ai-common/src/lower/quantize.rs`
- Modify: `hologram-ai/crates/hologram-ai-common/src/lower/mod.rs` (`pub mod quantize;`)
- Modify: `hologram-ai/crates/hologram-ai-common/Cargo.toml` (add `hologram-ai-quant`)
- Test: in `quantize.rs` (`#[cfg(test)]`)

- [ ] **Step 1: Add the crate dependency**

In `hologram-ai/crates/hologram-ai-common/Cargo.toml`, under `[dependencies]`:

```toml
hologram-ai-quant = { path = "../hologram-ai-quant" }
```

- [ ] **Step 2: Write the failing test**

Create `hologram-ai/crates/hologram-ai-common/src/lower/quantize.rs`:

```rust
//! Compile-time weight quantization pass: rewrites MatMul f32-weight constants
//! into i8 + per-channel scale + a `Dequantize` node. Gated on `QuantStrategy`.

use crate::ir::{shape_from_concrete, AiGraph, AiNode, AiOp, AiParam, DType, TensorId, TensorInfo};
use crate::lower::QuantStrategy;
use alloc::sync::Arc;
use anyhow::{bail, Result};
use hologram_ai_quant::encode_int8_per_channel;

/// Read a tensor's concrete 2D dims `[k, n]`, or `None` if not rank-2 concrete.
fn concrete_2d(info: &TensorInfo) -> Option<(usize, usize)> {
    let k = info.shape.get(0)?.as_concrete()?;
    let n = info.shape.get(1)?.as_concrete()?;
    if info.shape.get(2).is_some() {
        return None; // higher rank — skip (batched matmul weight)
    }
    Some((k as usize, n as usize))
}

/// Rewrite each MatMul whose B (weight) operand is an inline f32 rank-2 constant
/// into `Dequantize(i8_weight, per_column_scale) → MatMul`. No-op unless
/// `strategy == Int8`. `Int4` is rejected until the int4 spec lands.
pub fn quantize_weights(graph: &mut AiGraph, strategy: QuantStrategy) -> Result<()> {
    match strategy {
        QuantStrategy::Int8 => {}
        QuantStrategy::Int4 => bail!("int4 quantization is not yet implemented"),
        _ => return Ok(()),
    }

    let mut next_tid: TensorId = graph
        .tensor_info
        .keys()
        .chain(graph.params.keys())
        .copied()
        .max()
        .unwrap_or(0)
        + 1;
    let mut next_nid = graph.nodes.iter().map(|n| n.id).max().unwrap_or(0) + 1;

    let mut new_nodes: Vec<AiNode> = Vec::new();
    let mut dead_params: Vec<TensorId> = Vec::new();
    let mut changed = false;

    for idx in 0..graph.nodes.len() {
        if !matches!(graph.nodes[idx].op, AiOp::MatMul) {
            continue;
        }
        let b_tid = match graph.nodes[idx].inputs.get(1).copied() {
            Some(t) => t,
            None => continue,
        };
        let (data, info) = match graph.params.get(&b_tid) {
            Some(AiParam::Inline { data, info }) => (data.clone(), info.clone()),
            _ => continue, // mmap or non-constant B: skip in this baseline
        };
        if info.logical_dtype != DType::F32 {
            continue;
        }
        let (k, n) = match concrete_2d(&info) {
            Some(kn) => kn,
            None => continue,
        };
        if data.len() != k * n * 4 {
            continue;
        }

        let wf: Vec<f32> = data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let (q, scales) = encode_int8_per_channel(&wf, k, n);

        let wq_tid = next_tid;
        let scale_tid = next_tid + 1;
        let deq_tid = next_tid + 2;
        next_tid += 3;

        // i8 weight constant [k, n].
        let q_bytes: Vec<u8> = q.iter().map(|&v| v as u8).collect();
        let wq_info = TensorInfo::new(DType::INT8, info.shape.clone());
        graph
            .params
            .insert(wq_tid, AiParam::Inline { data: Arc::new(q_bytes), info: wq_info.clone() });
        graph.tensor_info.insert(wq_tid, wq_info);

        // Per-column scale constant [n] f32.
        let scale_bytes: Vec<u8> = scales.iter().flat_map(|v| v.to_le_bytes()).collect();
        let scale_info = TensorInfo::new(DType::F32, shape_from_concrete(&[n as u64]));
        graph.params.insert(
            scale_tid,
            AiParam::Inline { data: Arc::new(scale_bytes), info: scale_info.clone() },
        );
        graph.tensor_info.insert(scale_tid, scale_info);

        // Dequant output tensor [k, n] f32.
        graph
            .tensor_info
            .insert(deq_tid, TensorInfo::new(DType::F32, info.shape.clone()));

        // Dequantize node (zero-point omitted ⇒ lowering fills zeros), axis=1
        // (per output column).
        new_nodes.push(AiNode::new(
            next_nid,
            AiOp::Dequantize { axis: 1 },
            vec![wq_tid, scale_tid],
            vec![deq_tid],
        ));
        next_nid += 1;

        // Rewire MatMul B → dequant output; retire the f32 weight constant.
        graph.nodes[idx].inputs[1] = deq_tid;
        dead_params.push(b_tid);
        changed = true;
    }

    graph.nodes.extend(new_nodes);
    for t in dead_params {
        graph.params.remove(&t);
        graph.tensor_info.remove(&t);
    }
    if changed {
        graph.invalidate_topo_cache();
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f32_weight_matmul_graph() -> AiGraph {
        // X[1,4] · W[4,2] ; W is an inline f32 constant.
        let mut g = AiGraph::default();
        g.name = "qtest".into();
        // tids: 0 = X (input), 1 = W (f32 const), 2 = Y (matmul out)
        g.tensor_info
            .insert(0, TensorInfo::new(DType::F32, shape_from_concrete(&[1, 4])));
        let w: Vec<f32> = vec![0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8];
        let w_bytes: Vec<u8> = w.iter().flat_map(|v| v.to_le_bytes()).collect();
        let w_info = TensorInfo::new(DType::F32, shape_from_concrete(&[4, 2]));
        g.params.insert(1, AiParam::inline(w_bytes, w_info.clone()));
        g.tensor_info.insert(1, w_info);
        g.tensor_info
            .insert(2, TensorInfo::new(DType::F32, shape_from_concrete(&[1, 2])));
        g.nodes.push(AiNode::new(0, AiOp::MatMul, vec![0, 1], vec![2]));
        g.inputs = vec![0];
        g.outputs = vec![2];
        g
    }

    #[test]
    fn none_is_noop() {
        let mut g = f32_weight_matmul_graph();
        let before = g.nodes.len();
        quantize_weights(&mut g, QuantStrategy::None).unwrap();
        assert_eq!(g.nodes.len(), before);
        assert!(matches!(g.params.get(&1), Some(AiParam::Inline { .. })));
    }

    #[test]
    fn int8_rewrites_matmul_weight() {
        let mut g = f32_weight_matmul_graph();
        quantize_weights(&mut g, QuantStrategy::Int8).unwrap();

        // A Dequantize node was added.
        let deq = g
            .nodes
            .iter()
            .find(|n| matches!(n.op, AiOp::Dequantize { .. }))
            .expect("dequant node inserted");
        // Its weight operand is now an i8 constant of the original shape.
        let wq_tid = deq.inputs[0];
        match g.params.get(&wq_tid) {
            Some(AiParam::Inline { info, data }) => {
                assert_eq!(info.logical_dtype, DType::INT8);
                assert_eq!(data.len(), 4 * 2); // 1 byte/elem
            }
            _ => panic!("i8 weight constant missing"),
        }
        // Scale operand is an f32 vector of length n=2.
        let scale_tid = deq.inputs[1];
        match g.params.get(&scale_tid) {
            Some(AiParam::Inline { info, data }) => {
                assert_eq!(info.logical_dtype, DType::F32);
                assert_eq!(data.len(), 2 * 4);
            }
            _ => panic!("scale constant missing"),
        }
        // The MatMul's B now points at the dequant output; the old f32 const is gone.
        let mm = g.nodes.iter().find(|n| matches!(n.op, AiOp::MatMul)).unwrap();
        assert_eq!(mm.inputs[1], deq.outputs[0]);
        assert!(g.params.get(&1).is_none(), "old f32 weight retired");
    }

    #[test]
    fn int4_errors() {
        let mut g = f32_weight_matmul_graph();
        assert!(quantize_weights(&mut g, QuantStrategy::Int4).is_err());
    }
}
```

- [ ] **Step 3: Wire the module**

In `hologram-ai/crates/hologram-ai-common/src/lower/mod.rs`, add:

```rust
pub mod quantize;
pub use quantize::quantize_weights;
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p hologram-ai-common quantize`
Expected: `none_is_noop`, `int8_rewrites_matmul_weight`, `int4_errors` PASS.
(If a constructor/accessor name differs — e.g. `Shape::get`/`as_concrete` or `AiGraph::default` — fix to the real API surfaced by the compile error; the test bodies remain.)

- [ ] **Step 5: Commit**

```bash
git add crates/hologram-ai-common/src/lower/quantize.rs crates/hologram-ai-common/src/lower/mod.rs crates/hologram-ai-common/Cargo.toml
git commit -m "feat(quant): quantize_weights graph pass (per-channel int8)"
```

---

## Task 4: Wire the pass into the compile pipeline

**Files:**
- Modify: `hologram-ai/crates/hologram-ai/src/compiler.rs` (between `:293` and `:414`)

- [ ] **Step 1: Call the pass after optimization, before lowering**

In `compiler.rs`, immediately after the optimization pipeline runs (the `let mut ai_graph = pipeline.run(ai_graph)...` at ~line 293), insert:

```rust
        // Step 3b — compile-time weight quantization (no-op unless requested).
        hologram_ai_common::lower::quantize_weights(&mut ai_graph, compiler.quant_strategy)
            .context("weight quantization pass failed")?;
```

(`ai_graph` is already `mut`. `compiler.quant_strategy` is the `ModelCompiler` field. Ensure `quantize_weights` is imported or call it fully-qualified as shown.)

- [ ] **Step 2: Build**

Run: `cargo build -p hologram-ai`
Expected: clean build.

- [ ] **Step 3: Commit**

```bash
git add crates/hologram-ai/src/compiler.rs
git commit -m "feat(quant): run quantize_weights between optimize and lower"
```

---

## Task 5: End-to-end accuracy gate (f32 vs int8, cosine ≥ 0.999)

**Files:**
- Create: `hologram-ai/crates/hologram-ai/tests/int8_accuracy.rs`

This mirrors `tests/quantized_weight_memory.rs` but drives int8 through the *pass* (compile the same f32 graph twice: once `QuantStrategy::None`, once `Int8`) and compares output logits.

- [ ] **Step 1: Write the test**

Create `hologram-ai/crates/hologram-ai/tests/int8_accuracy.rs`:

```rust
//! Accuracy gate: a small linear model compiled f32 vs int8 (via the
//! quantize_weights pass) must agree to logit cosine ≥ 0.999.

use std::collections::HashMap;

use hologram_ai::compiler::{ModelCompiler, ModelSource};
use hologram_ai::runner::HoloRunner;
use hologram_ai_common::ir::{shape_from_concrete, AiGraph, AiNode, AiOp, AiParam, DType, TensorInfo};
use hologram_ai_common::lower::QuantStrategy;

const K: usize = 64;
const N: usize = 48;

fn linear_graph() -> AiGraph {
    // Y[1,N] = X[1,K] · W[K,N], W an inline f32 constant with varied magnitudes.
    let mut g = AiGraph::default();
    g.name = "int8_acc".into();
    g.tensor_info
        .insert(0, TensorInfo::new(DType::F32, shape_from_concrete(&[1, K as u64])));
    let w: Vec<f32> = (0..K * N)
        .map(|i| ((i % 23) as f32 - 11.0) * 0.05 + ((i % 7) as f32) * 0.01)
        .collect();
    let w_bytes: Vec<u8> = w.iter().flat_map(|v| v.to_le_bytes()).collect();
    let w_info = TensorInfo::new(DType::F32, shape_from_concrete(&[K as u64, N as u64]));
    g.params.insert(1, AiParam::inline(w_bytes, w_info.clone()));
    g.tensor_info.insert(1, w_info);
    g.tensor_info
        .insert(2, TensorInfo::new(DType::F32, shape_from_concrete(&[1, N as u64])));
    g.nodes.push(AiNode::new(0, AiOp::MatMul, vec![0, 1], vec![2]));
    g.inputs = vec![0];
    g.outputs = vec![2];
    g
}

fn run(strategy: QuantStrategy, x: &[f32]) -> (HoloRunner, Vec<f32>) {
    let compiler = ModelCompiler {
        quant_strategy: strategy,
        ..ModelCompiler::default()
    };
    let archive = compiler
        .compile(ModelSource::AiGraph(linear_graph()))
        .expect("compile failed");
    let mut runner = HoloRunner::from_bytes(archive.bytes).expect("load failed");
    let x_bytes: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
    let out = runner.execute(&[&x_bytes]).expect("execute failed");
    let y: Vec<f32> = out[0]
        .bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    (runner, y)
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb)
}

#[test]
fn int8_matches_f32_within_cosine_999() {
    let x: Vec<f32> = (0..K).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();
    let (_f, f_out) = run(QuantStrategy::None, &x);
    let (q_runner, q_out) = run(QuantStrategy::Int8, &x);

    // The dequant→matmul must have fused (weight stays packed int8).
    assert_eq!(
        q_runner.dequant_matmul_fused_count(),
        1,
        "int8 weight must fuse into MatMulDequant"
    );

    let cos = cosine(&f_out, &q_out);
    let max_abs: f32 = f_out
        .iter()
        .zip(&q_out)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    assert!(
        cos >= 0.999,
        "logit cosine {cos} < 0.999 (max abs delta {max_abs})"
    );
}
```

- [ ] **Step 2: Run the test**

Run: `cargo test -p hologram-ai --test int8_accuracy -- --nocapture`
Expected: PASS — `dequant_matmul_fused_count() == 1` and cosine ≥ 0.999.
(If `dequant_matmul_fused_count`/`HoloRunner::from_bytes`/`ModelSource::AiGraph` names differ from `quantized_weight_memory.rs`, copy the exact ones from that file.)

- [ ] **Step 3: Commit**

```bash
git add crates/hologram-ai/tests/int8_accuracy.rs
git commit -m "test(quant): int8 vs f32 accuracy gate (logit cosine >= 0.999)"
```

---

## Task 6: wasm correctness of the quantized path

**Files:** none new — runs the Task 5 test under wasmtime.

- [ ] **Step 1: Run the accuracy test on wasm32-wasip1 under wasmtime**

The dequant-matmul path is `no_std`/wasm-compatible. Confirm the quantized fixture is correct on the primary target. (Uses the rustup toolchain because Homebrew rust lacks wasm std; wasmtime installed earlier.)

Run:
```bash
TC=/Users/auser/.rustup/toolchains/stable-aarch64-apple-darwin/bin
RUSTC=$TC/rustc CARGO_TARGET_DIR=target-wasm-test \
  RUSTFLAGS="-Ctarget-feature=+simd128" \
  CARGO_TARGET_WASM32_WASIP1_RUNNER=wasmtime \
  $TC/cargo test -p hologram-ai --test int8_accuracy --target wasm32-wasip1 -- --test-threads=1
```
Expected: PASS under wasmtime. If `hologram-ai`'s test deps pull `std`-only crates that don't build for wasip1, fall back to running the `hologram-ai-common` pass unit tests (Task 3) on wasm instead, and note the gap.

- [ ] **Step 2: Commit (if any harness tweaks were needed)**

```bash
git commit -am "test(quant): verify int8 path under wasmtime" --allow-empty
```

---

## Task 7: Manual TinyLlama validation (documented, not CI)

**Files:**
- Modify: the spec's "Manual E2E" note, or add `docs/superpowers/specs/...` follow-up notes with the measured numbers.

- [ ] **Step 1: Compile TinyLlama int8 and record size + accuracy**

(Heavy AI-stack build; run on demand, not in CI.)
```bash
cd /Users/auser/work/uor/hologram/hologram-ai
cargo run --release -- compile models/TinyLlama-1.1B-Chat-v1.0/model.onnx \
  --quantize int8 -o /tmp/tinyllama-int8.holo
ls -la /tmp/tinyllama-int8.holo   # expect ~4x smaller than the f32 .holo
```
Then compare generated text / logits vs the f32 `.holo` on a fixed prompt and record cosine + the size reduction in the spec's results section.

- [ ] **Step 2: Commit the recorded results**

```bash
git commit -am "docs(quant): record TinyLlama int8 size + accuracy" --allow-empty
```

---

## Self-Review Notes

- **Spec coverage:** encoder (§ Unit: weight encoder → Task 1); pass + per-channel axis (§ Unit: pass → Tasks 3–4); flag rename (§ Decisions 4 → Task 2); accuracy gate cosine ≥ 0.999 (§ Accuracy → Task 5); wasm (§ Accuracy 3 → Task 6); manual TinyLlama (§ Accuracy 4 → Task 7); de-risk fusion first (§ Risks 1 → Task 0). Out-of-scope items (int4, embeddings, fused-int) are explicitly *not* tasked; Int4 is wired only to a clear error (Task 3).
- **Type consistency:** `encode_int8_per_channel(&[f32], usize, usize) -> (Vec<i8>, Vec<f32>)` used identically in Tasks 1 and 3. `AiOp::Dequantize { axis: i64 }`, `AiParam::Inline { data: Arc<Vec<u8>>, info }` / `AiParam::inline(Vec<u8>, TensorInfo)`, `TensorInfo::new(DType, Shape)`, `shape_from_concrete(&[u64])`, `graph.invalidate_topo_cache()` — all per the extracted API.
- **Known soft spots (resolved via TDD compile, not placeholders):** exact `Shape` accessor (`get`/`as_concrete`) and `HoloRunner`/`ModelSource` method names are taken from existing tests (`quantized_weight_memory.rs`, `rules/mod.rs`); if any differ, the failing compile names the correction while the logic stays.
