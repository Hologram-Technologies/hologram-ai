//! Custom op handler factory functions for AI ops with no native hologram counterpart.
//!
//! Each function returns a `hologram::CustomHandler` (an `Arc` closure) that receives
//! byte-slice inputs and returns output bytes. Compute is done in f32 internally.

use std::sync::Arc;
use hologram::CustomHandler;

// ── f32 arithmetic with broadcasting ──────────────────────────────────────
//
// These replace native PrimOps (which do byte-level Z/256Z ring arithmetic)
// with proper f32 arithmetic and broadcasting support for AI inference.

/// Inputs: [a (f32), b (f32)]  Output: [a + b (f32)] with broadcasting.
pub fn add_handler() -> CustomHandler {
    Arc::new(|inputs, _| {
        let a = bytemuck::cast_slice::<u8, f32>(inputs[0]);
        let b = bytemuck::cast_slice::<u8, f32>(inputs[1]);
        let out: Vec<f32> = a.iter().zip(b.iter().cycle())
            .map(|(&x, &y)| x + y)
            .collect();
        Ok(bytemuck::cast_slice(&out).to_vec())
    })
}

/// Inputs: [a (f32), b (f32)]  Output: [a - b (f32)] with broadcasting.
pub fn sub_handler() -> CustomHandler {
    Arc::new(|inputs, _| {
        let a = bytemuck::cast_slice::<u8, f32>(inputs[0]);
        let b = bytemuck::cast_slice::<u8, f32>(inputs[1]);
        let out: Vec<f32> = a.iter().zip(b.iter().cycle())
            .map(|(&x, &y)| x - y)
            .collect();
        Ok(bytemuck::cast_slice(&out).to_vec())
    })
}

/// Inputs: [a (f32), b (f32)]  Output: [a * b (f32)] with broadcasting.
pub fn mul_handler() -> CustomHandler {
    Arc::new(|inputs, _| {
        let a = bytemuck::cast_slice::<u8, f32>(inputs[0]);
        let b = bytemuck::cast_slice::<u8, f32>(inputs[1]);
        let out: Vec<f32> = a.iter().zip(b.iter().cycle())
            .map(|(&x, &y)| x * y)
            .collect();
        Ok(bytemuck::cast_slice(&out).to_vec())
    })
}

/// Inputs: [x (f32)]  Output: [-x (f32)]
pub fn neg_handler() -> CustomHandler {
    Arc::new(|inputs, _| {
        let x = bytemuck::cast_slice::<u8, f32>(inputs[0]);
        let out: Vec<f32> = x.iter().map(|v| -v).collect();
        Ok(bytemuck::cast_slice(&out).to_vec())
    })
}

// ── f32 activations ──────────────────────────────────────────────────────
//
// These replace native LutOps (byte-level LUT tables) with proper f32
// implementations for AI inference.

pub fn relu_handler() -> CustomHandler {
    Arc::new(|inputs, _| {
        let x = bytemuck::cast_slice::<u8, f32>(inputs[0]);
        let out: Vec<f32> = x.iter().map(|&v| v.max(0.0)).collect();
        Ok(bytemuck::cast_slice(&out).to_vec())
    })
}

pub fn gelu_handler() -> CustomHandler {
    Arc::new(|inputs, _| {
        let x = bytemuck::cast_slice::<u8, f32>(inputs[0]);
        let out: Vec<f32> = x.iter().map(|&v| {
            0.5 * v * (1.0 + (std::f32::consts::FRAC_2_SQRT_PI * (v + 0.044715 * v * v * v) * std::f32::consts::FRAC_1_SQRT_2).tanh())
        }).collect();
        Ok(bytemuck::cast_slice(&out).to_vec())
    })
}

pub fn silu_handler() -> CustomHandler {
    Arc::new(|inputs, _| {
        let x = bytemuck::cast_slice::<u8, f32>(inputs[0]);
        let out: Vec<f32> = x.iter().map(|&v| v / (1.0 + (-v).exp())).collect();
        Ok(bytemuck::cast_slice(&out).to_vec())
    })
}

pub fn tanh_handler() -> CustomHandler {
    Arc::new(|inputs, _| {
        let x = bytemuck::cast_slice::<u8, f32>(inputs[0]);
        let out: Vec<f32> = x.iter().map(|v| v.tanh()).collect();
        Ok(bytemuck::cast_slice(&out).to_vec())
    })
}

pub fn sigmoid_handler() -> CustomHandler {
    Arc::new(|inputs, _| {
        let x = bytemuck::cast_slice::<u8, f32>(inputs[0]);
        let out: Vec<f32> = x.iter().map(|&v| 1.0 / (1.0 + (-v).exp())).collect();
        Ok(bytemuck::cast_slice(&out).to_vec())
    })
}

pub fn exp_handler() -> CustomHandler {
    Arc::new(|inputs, _| {
        let x = bytemuck::cast_slice::<u8, f32>(inputs[0]);
        let out: Vec<f32> = x.iter().map(|v| v.exp()).collect();
        Ok(bytemuck::cast_slice(&out).to_vec())
    })
}

pub fn log_handler() -> CustomHandler {
    Arc::new(|inputs, _| {
        let x = bytemuck::cast_slice::<u8, f32>(inputs[0]);
        let out: Vec<f32> = x.iter().map(|v| v.ln()).collect();
        Ok(bytemuck::cast_slice(&out).to_vec())
    })
}

pub fn sqrt_handler() -> CustomHandler {
    Arc::new(|inputs, _| {
        let x = bytemuck::cast_slice::<u8, f32>(inputs[0]);
        let out: Vec<f32> = x.iter().map(|v| v.sqrt()).collect();
        Ok(bytemuck::cast_slice(&out).to_vec())
    })
}

pub fn abs_handler() -> CustomHandler {
    Arc::new(|inputs, _| {
        let x = bytemuck::cast_slice::<u8, f32>(inputs[0]);
        let out: Vec<f32> = x.iter().map(|v| v.abs()).collect();
        Ok(bytemuck::cast_slice(&out).to_vec())
    })
}

pub fn cos_handler() -> CustomHandler {
    Arc::new(|inputs, _| {
        let x = bytemuck::cast_slice::<u8, f32>(inputs[0]);
        let out: Vec<f32> = x.iter().map(|v| v.cos()).collect();
        Ok(bytemuck::cast_slice(&out).to_vec())
    })
}

pub fn sin_handler() -> CustomHandler {
    Arc::new(|inputs, _| {
        let x = bytemuck::cast_slice::<u8, f32>(inputs[0]);
        let out: Vec<f32> = x.iter().map(|v| v.sin()).collect();
        Ok(bytemuck::cast_slice(&out).to_vec())
    })
}

// ── RmsNorm ────────────────────────────────────────────────────────────────

/// Inputs: [x (f32), weight (f32)]  Output: [y (f32)]
pub fn rms_norm_handler(epsilon: f32) -> CustomHandler {
    Arc::new(move |inputs, _| {
        let x      = bytemuck::cast_slice::<u8, f32>(inputs[0]);
        let weight = bytemuck::cast_slice::<u8, f32>(inputs[1]);
        let n = x.len();
        let rms = (x.iter().map(|v| v * v).sum::<f32>() / n as f32 + epsilon).sqrt();
        let out: Vec<f32> = x.iter().zip(weight.iter().cycle())
            .map(|(xi, wi)| xi / rms * wi)
            .collect();
        Ok(bytemuck::cast_slice(&out).to_vec())
    })
}

// ── LayerNorm ──────────────────────────────────────────────────────────────

/// Inputs: [x (f32), weight (f32), bias (f32)]  Output: [y (f32)]
pub fn layer_norm_handler(epsilon: f32) -> CustomHandler {
    Arc::new(move |inputs, _| {
        let x      = bytemuck::cast_slice::<u8, f32>(inputs[0]);
        let weight = bytemuck::cast_slice::<u8, f32>(inputs[1]);
        let bias   = bytemuck::cast_slice::<u8, f32>(inputs[2]);
        let n = x.len() as f32;
        let mean = x.iter().sum::<f32>() / n;
        let var  = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
        let std  = (var + epsilon).sqrt();
        let out: Vec<f32> = x.iter().enumerate()
            .map(|(i, xi)| (xi - mean) / std * weight[i % weight.len()] + bias[i % bias.len()])
            .collect();
        Ok(bytemuck::cast_slice(&out).to_vec())
    })
}

// ── Softmax ────────────────────────────────────────────────────────────────

/// Inputs: [x (f32)]  Output: [y (f32)]
pub fn softmax_handler(_axis: i64) -> CustomHandler {
    Arc::new(|inputs, _| {
        let x = bytemuck::cast_slice::<u8, f32>(inputs[0]);
        let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = x.iter().map(|v| (v - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let out: Vec<f32> = exps.iter().map(|e| e / sum).collect();
        Ok(bytemuck::cast_slice(&out).to_vec())
    })
}

// ── Embed ──────────────────────────────────────────────────────────────────

/// Inputs: [token_ids (u32), embedding_table (f32)]  Output: [embeddings (f32)]
pub fn embed_handler() -> CustomHandler {
    Arc::new(|inputs, _| {
        let ids   = bytemuck::cast_slice::<u8, u32>(inputs[0]);
        let table = bytemuck::cast_slice::<u8, f32>(inputs[1]);
        if ids.is_empty() || table.is_empty() {
            return Ok(Vec::new());
        }
        let max_id = ids.iter().copied().max().unwrap_or(0) as usize;
        let dim = table.len() / (max_id + 1).max(1);
        let mut out = vec![0.0f32; ids.len() * dim];
        for (i, &id) in ids.iter().enumerate() {
            let src = id as usize * dim;
            let dst = i * dim;
            if src + dim <= table.len() {
                out[dst..dst + dim].copy_from_slice(&table[src..src + dim]);
            }
        }
        Ok(bytemuck::cast_slice(&out).to_vec())
    })
}

// ── Dequantize ─────────────────────────────────────────────────────────────

/// Inputs: [q4_0_bytes]  Output: [f32 values]
pub fn dequant_handler() -> CustomHandler {
    Arc::new(|inputs, _| {
        let floats = hologram_ai_quant::dequant_q4_0(inputs[0]);
        Ok(bytemuck::cast_slice(&floats).to_vec())
    })
}

// ── Reshape / Transpose ────────────────────────────────────────────────────

/// Sprint 001: shape is metadata only — pass bytes through unchanged.
pub fn reshape_handler() -> CustomHandler {
    Arc::new(|inputs, _| Ok(inputs[0].to_vec()))
}

// ── Cast ───────────────────────────────────────────────────────────────────

/// Sprint 001: identity cast (same dtype → same bytes).
pub fn cast_handler() -> CustomHandler {
    Arc::new(|inputs, _| Ok(inputs[0].to_vec()))
}

// ── Concat ─────────────────────────────────────────────────────────────────

/// Concatenate all inputs byte-wise (flat layout — Phase 2 handles axes).
pub fn concat_handler() -> CustomHandler {
    Arc::new(|inputs, _| {
        let total: usize = inputs.iter().map(|i| i.len()).sum();
        let mut out = Vec::with_capacity(total);
        for inp in inputs {
            out.extend_from_slice(inp);
        }
        Ok(out)
    })
}

// ── FusedSwiGLU ────────────────────────────────────────────────────────────

/// Inputs: [gate (f32), up (f32)]  Output: [silu(gate) * up (f32)]
pub fn swiglu_handler() -> CustomHandler {
    Arc::new(|inputs, _| {
        let gate = bytemuck::cast_slice::<u8, f32>(inputs[0]);
        let up   = bytemuck::cast_slice::<u8, f32>(inputs[1]);
        let out: Vec<f32> = gate.iter().zip(up.iter())
            .map(|(&g, &u)| g / (1.0 + (-g).exp()) * u)
            .collect();
        Ok(bytemuck::cast_slice(&out).to_vec())
    })
}

// ── RotaryEmbedding ────────────────────────────────────────────────────────

/// Sprint 001 stub: passes input through; full RoPE in Phase 2.
pub fn rope_handler(_base: f32, _dim: u32) -> CustomHandler {
    Arc::new(|inputs, _| Ok(inputs[0].to_vec()))
}

// ── Gather ─────────────────────────────────────────────────────────────────

/// Inputs: [data (f32, shape [n_rows, row_size]), indices (i64)]  Output: [gathered (f32)]
///
/// Implements ONNX Gather with axis=0: selects rows from `data` by `indices`.
pub fn gather_handler(row_size: usize) -> CustomHandler {
    Arc::new(move |inputs, _| {
        if row_size == 0 {
            return Ok(Vec::new());
        }
        let data    = bytemuck::cast_slice::<u8, f32>(inputs[0]);
        let indices = bytemuck::cast_slice::<u8, i64>(inputs[1]);
        let n_rows  = data.len() / row_size;
        let mut out = vec![0.0f32; indices.len() * row_size];
        for (i, &idx) in indices.iter().enumerate() {
            let row = idx.rem_euclid(n_rows as i64) as usize;
            let src = row * row_size;
            let dst = i   * row_size;
            out[dst..dst + row_size].copy_from_slice(&data[src..src + row_size]);
        }
        Ok(bytemuck::cast_slice(&out).to_vec())
    })
}

// ── MatMul ─────────────────────────────────────────────────────────────────

/// Inputs: [A (f32, [m, inner]), B (f32, [inner, n_cols])]  Output: [C (f32, [m, n_cols])]
///
/// Naive 2-D matrix multiply. Optional third input (Gemm bias) is added if present.
pub fn matmul_handler(inner: usize, n_cols: usize) -> CustomHandler {
    Arc::new(move |inputs, _| {
        if inner == 0 || n_cols == 0 {
            return Ok(Vec::new());
        }
        let a = bytemuck::cast_slice::<u8, f32>(inputs[0]);
        let b = bytemuck::cast_slice::<u8, f32>(inputs[1]);
        let m = a.len() / inner;
        let mut out = vec![0.0f32; m * n_cols];
        for i in 0..m {
            for k in 0..inner {
                let a_val = a[i * inner + k];
                for j in 0..n_cols {
                    out[i * n_cols + j] += a_val * b[k * n_cols + j];
                }
            }
        }
        if let Some(bias_bytes) = inputs.get(2) {
            let bias = bytemuck::cast_slice::<u8, f32>(bias_bytes);
            for i in 0..m {
                for j in 0..n_cols {
                    out[i * n_cols + j] += bias[j % bias.len()];
                }
            }
        }
        Ok(bytemuck::cast_slice(&out).to_vec())
    })
}

// ── Shape ─────────────────────────────────────────────────────────────────

/// Inputs: [x]  Output: [shape as i64 values]
/// Sprint 001 stub: returns a single-element shape [n_elements].
pub fn shape_handler() -> CustomHandler {
    Arc::new(|inputs, _| {
        let n = inputs[0].len() as i64 / 4; // assume f32
        Ok(bytemuck::cast_slice(&[n]).to_vec())
    })
}

// ── Where ─────────────────────────────────────────────────────────────────

/// Inputs: [condition (bool/u8), x (f32), y (f32)]  Output: [selected (f32)]
pub fn where_handler() -> CustomHandler {
    Arc::new(|inputs, _| {
        let cond = inputs[0];
        let x    = bytemuck::cast_slice::<u8, f32>(inputs[1]);
        let y    = bytemuck::cast_slice::<u8, f32>(inputs[2]);
        let out: Vec<f32> = (0..x.len().max(y.len()))
            .map(|i| {
                let c = cond.get(i % cond.len()).copied().unwrap_or(0);
                if c != 0 { x[i % x.len()] } else { y[i % y.len()] }
            })
            .collect();
        Ok(bytemuck::cast_slice(&out).to_vec())
    })
}

// ── Range ─────────────────────────────────────────────────────────────────

/// Inputs: [start (f32), limit (f32), delta (f32)]  Output: [range (f32)]
pub fn range_handler() -> CustomHandler {
    Arc::new(|inputs, _| {
        let start = bytemuck::cast_slice::<u8, f32>(inputs[0]);
        let limit = bytemuck::cast_slice::<u8, f32>(inputs[1]);
        let delta = bytemuck::cast_slice::<u8, f32>(inputs[2]);
        let s = start.first().copied().unwrap_or(0.0);
        let l = limit.first().copied().unwrap_or(0.0);
        let d = delta.first().copied().unwrap_or(1.0);
        if d == 0.0 { return Ok(Vec::new()); }
        let n = ((l - s) / d).ceil().max(0.0) as usize;
        let out: Vec<f32> = (0..n).map(|i| s + i as f32 * d).collect();
        Ok(bytemuck::cast_slice(&out).to_vec())
    })
}

// ── GatherND ──────────────────────────────────────────────────────────────

/// Inputs: [data, indices]  Output: [gathered]
/// Sprint 001 stub: passes data through (full N-D gather in Phase 2).
pub fn gather_nd_handler() -> CustomHandler {
    Arc::new(|inputs, _| Ok(inputs[0].to_vec()))
}

// ── IsNaN ─────────────────────────────────────────────────────────────────

/// Inputs: [x (f32)]  Output: [mask (u8, 0 or 1)]
pub fn isnan_handler() -> CustomHandler {
    Arc::new(|inputs, _| {
        let x = bytemuck::cast_slice::<u8, f32>(inputs[0]);
        let out: Vec<u8> = x.iter().map(|v| u8::from(v.is_nan())).collect();
        Ok(out)
    })
}

// ── Flatten ───────────────────────────────────────────────────────────────

/// Sprint 001 stub: passes bytes through unchanged (metadata-only reshape).
pub fn flatten_handler() -> CustomHandler {
    Arc::new(|inputs, _| Ok(inputs[0].to_vec()))
}

// ── Div ──────────────────────────────────────────────────────────────────

/// Inputs: [a (f32), b (f32)]  Output: [a / b (f32)]
pub fn div_handler() -> CustomHandler {
    Arc::new(|inputs, _| {
        let a = bytemuck::cast_slice::<u8, f32>(inputs[0]);
        let b = bytemuck::cast_slice::<u8, f32>(inputs[1]);
        let out: Vec<f32> = a.iter().zip(b.iter().cycle())
            .map(|(&x, &y)| x / y)
            .collect();
        Ok(bytemuck::cast_slice(&out).to_vec())
    })
}

// ── Pow ──────────────────────────────────────────────────────────────────

/// Inputs: [base (f32), exp (f32)]  Output: [base^exp (f32)]
pub fn pow_handler() -> CustomHandler {
    Arc::new(|inputs, _| {
        let base = bytemuck::cast_slice::<u8, f32>(inputs[0]);
        let exp  = bytemuck::cast_slice::<u8, f32>(inputs[1]);
        let out: Vec<f32> = base.iter().zip(exp.iter().cycle())
            .map(|(&b, &e)| b.powf(e))
            .collect();
        Ok(bytemuck::cast_slice(&out).to_vec())
    })
}

// ── Mod ──────────────────────────────────────────────────────────────────

/// Inputs: [a (f32), b (f32)]  Output: [a % b (f32)]
pub fn mod_handler() -> CustomHandler {
    Arc::new(|inputs, _| {
        let a = bytemuck::cast_slice::<u8, f32>(inputs[0]);
        let b = bytemuck::cast_slice::<u8, f32>(inputs[1]);
        let out: Vec<f32> = a.iter().zip(b.iter().cycle())
            .map(|(&x, &y)| x % y)
            .collect();
        Ok(bytemuck::cast_slice(&out).to_vec())
    })
}

// ── Min / Max (elementwise binary) ──────────────────────────────────────

/// Inputs: [a (f32), b (f32)]  Output: [min(a,b) (f32)]
pub fn min_handler() -> CustomHandler {
    Arc::new(|inputs, _| {
        let a = bytemuck::cast_slice::<u8, f32>(inputs[0]);
        let b = bytemuck::cast_slice::<u8, f32>(inputs[1]);
        let out: Vec<f32> = a.iter().zip(b.iter().cycle())
            .map(|(&x, &y)| x.min(y))
            .collect();
        Ok(bytemuck::cast_slice(&out).to_vec())
    })
}

/// Inputs: [a (f32), b (f32)]  Output: [max(a,b) (f32)]
pub fn max_handler() -> CustomHandler {
    Arc::new(|inputs, _| {
        let a = bytemuck::cast_slice::<u8, f32>(inputs[0]);
        let b = bytemuck::cast_slice::<u8, f32>(inputs[1]);
        let out: Vec<f32> = a.iter().zip(b.iter().cycle())
            .map(|(&x, &y)| x.max(y))
            .collect();
        Ok(bytemuck::cast_slice(&out).to_vec())
    })
}

// ── Boolean ops ──────────────────────────────────────────────────────────

/// Inputs: [a (u8), b (u8)]  Output: [a & b (u8)]
pub fn and_handler() -> CustomHandler {
    Arc::new(|inputs, _| {
        let out: Vec<u8> = inputs[0].iter().zip(inputs[1].iter().cycle())
            .map(|(&a, &b)| u8::from(a != 0 && b != 0))
            .collect();
        Ok(out)
    })
}

/// Inputs: [a (u8), b (u8)]  Output: [a | b (u8)]
pub fn or_handler() -> CustomHandler {
    Arc::new(|inputs, _| {
        let out: Vec<u8> = inputs[0].iter().zip(inputs[1].iter().cycle())
            .map(|(&a, &b)| u8::from(a != 0 || b != 0))
            .collect();
        Ok(out)
    })
}

/// Inputs: [a (u8), b (u8)]  Output: [a ^ b (u8)]
pub fn xor_handler() -> CustomHandler {
    Arc::new(|inputs, _| {
        let out: Vec<u8> = inputs[0].iter().zip(inputs[1].iter().cycle())
            .map(|(&a, &b)| u8::from((a != 0) ^ (b != 0)))
            .collect();
        Ok(out)
    })
}

/// Inputs: [a (u8)]  Output: [!a (u8)]
pub fn not_handler() -> CustomHandler {
    Arc::new(|inputs, _| {
        let out: Vec<u8> = inputs[0].iter().map(|&a| u8::from(a == 0)).collect();
        Ok(out)
    })
}

// ── Comparisons ──────────────────────────────────────────────────────────

/// Inputs: [a (f32), b (f32)]  Output: [a == b (u8)]
pub fn equal_handler() -> CustomHandler {
    Arc::new(|inputs, _| {
        let a = bytemuck::cast_slice::<u8, f32>(inputs[0]);
        let b = bytemuck::cast_slice::<u8, f32>(inputs[1]);
        let out: Vec<u8> = a.iter().zip(b.iter().cycle())
            .map(|(&x, &y)| u8::from(x == y))
            .collect();
        Ok(out)
    })
}

/// Inputs: [a (f32), b (f32)]  Output: [a < b (u8)]
pub fn less_handler() -> CustomHandler {
    Arc::new(|inputs, _| {
        let a = bytemuck::cast_slice::<u8, f32>(inputs[0]);
        let b = bytemuck::cast_slice::<u8, f32>(inputs[1]);
        let out: Vec<u8> = a.iter().zip(b.iter().cycle())
            .map(|(&x, &y)| u8::from(x < y))
            .collect();
        Ok(out)
    })
}

/// Inputs: [a (f32), b (f32)]  Output: [a <= b (u8)]
pub fn less_or_equal_handler() -> CustomHandler {
    Arc::new(|inputs, _| {
        let a = bytemuck::cast_slice::<u8, f32>(inputs[0]);
        let b = bytemuck::cast_slice::<u8, f32>(inputs[1]);
        let out: Vec<u8> = a.iter().zip(b.iter().cycle())
            .map(|(&x, &y)| u8::from(x <= y))
            .collect();
        Ok(out)
    })
}

/// Inputs: [a (f32), b (f32)]  Output: [a > b (u8)]
pub fn greater_handler() -> CustomHandler {
    Arc::new(|inputs, _| {
        let a = bytemuck::cast_slice::<u8, f32>(inputs[0]);
        let b = bytemuck::cast_slice::<u8, f32>(inputs[1]);
        let out: Vec<u8> = a.iter().zip(b.iter().cycle())
            .map(|(&x, &y)| u8::from(x > y))
            .collect();
        Ok(out)
    })
}

/// Inputs: [a (f32), b (f32)]  Output: [a >= b (u8)]
pub fn greater_or_equal_handler() -> CustomHandler {
    Arc::new(|inputs, _| {
        let a = bytemuck::cast_slice::<u8, f32>(inputs[0]);
        let b = bytemuck::cast_slice::<u8, f32>(inputs[1]);
        let out: Vec<u8> = a.iter().zip(b.iter().cycle())
            .map(|(&x, &y)| u8::from(x >= y))
            .collect();
        Ok(out)
    })
}

// ── Unary math (custom) ──────────────────────────────────────────────────

/// Inputs: [x (f32)]  Output: [1/x (f32)]
pub fn reciprocal_handler() -> CustomHandler {
    Arc::new(|inputs, _| {
        let x = bytemuck::cast_slice::<u8, f32>(inputs[0]);
        let out: Vec<f32> = x.iter().map(|v| 1.0 / v).collect();
        Ok(bytemuck::cast_slice(&out).to_vec())
    })
}

/// Inputs: [x (f32)]  Output: [sign(x) (f32)]
pub fn sign_handler() -> CustomHandler {
    Arc::new(|inputs, _| {
        let x = bytemuck::cast_slice::<u8, f32>(inputs[0]);
        let out: Vec<f32> = x.iter().map(|v| v.signum()).collect();
        Ok(bytemuck::cast_slice(&out).to_vec())
    })
}

/// Inputs: [x (f32)]  Output: [floor(x) (f32)]
pub fn floor_handler() -> CustomHandler {
    Arc::new(|inputs, _| {
        let x = bytemuck::cast_slice::<u8, f32>(inputs[0]);
        let out: Vec<f32> = x.iter().map(|v| v.floor()).collect();
        Ok(bytemuck::cast_slice(&out).to_vec())
    })
}

/// Inputs: [x (f32)]  Output: [ceil(x) (f32)]
pub fn ceil_handler() -> CustomHandler {
    Arc::new(|inputs, _| {
        let x = bytemuck::cast_slice::<u8, f32>(inputs[0]);
        let out: Vec<f32> = x.iter().map(|v| v.ceil()).collect();
        Ok(bytemuck::cast_slice(&out).to_vec())
    })
}

/// Inputs: [x (f32)]  Output: [round(x) (f32)]
pub fn round_handler() -> CustomHandler {
    Arc::new(|inputs, _| {
        let x = bytemuck::cast_slice::<u8, f32>(inputs[0]);
        let out: Vec<f32> = x.iter().map(|v| v.round()).collect();
        Ok(bytemuck::cast_slice(&out).to_vec())
    })
}

/// Inputs: [x (f32)]  Output: [clamp(x, min, max) (f32)]
/// Sprint 001: no-op clip (min/max from attrs not wired yet).
pub fn clip_handler() -> CustomHandler {
    Arc::new(|inputs, _| Ok(inputs[0].to_vec()))
}

/// Inputs: [x (f32)]  Output: [erf(x) (f32)]
pub fn erf_handler() -> CustomHandler {
    Arc::new(|inputs, _| {
        let x = bytemuck::cast_slice::<u8, f32>(inputs[0]);
        // Abramowitz & Stegun approximation
        let out: Vec<f32> = x.iter().map(|&v| {
            let sign = v.signum();
            let a = v.abs();
            let t = 1.0 / (1.0 + 0.3275911 * a);
            let poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
            sign * (1.0 - poly * (-a * a).exp())
        }).collect();
        Ok(bytemuck::cast_slice(&out).to_vec())
    })
}

// ── Reductions ──────────────────────────────────────────────────────────

/// Inputs: [x (f32)]  Output: [sum (f32)]
/// Sprint 001: reduces over all elements (ignores axes).
pub fn reduce_sum_handler() -> CustomHandler {
    Arc::new(|inputs, _| {
        let x = bytemuck::cast_slice::<u8, f32>(inputs[0]);
        let s: f32 = x.iter().sum();
        Ok(bytemuck::cast_slice(&[s]).to_vec())
    })
}

/// Inputs: [x (f32)]  Output: [mean (f32)]
/// Sprint 001: reduces over all elements (ignores axes).
pub fn reduce_mean_handler() -> CustomHandler {
    Arc::new(|inputs, _| {
        let x = bytemuck::cast_slice::<u8, f32>(inputs[0]);
        let n = x.len() as f32;
        let m = if n > 0.0 { x.iter().sum::<f32>() / n } else { 0.0 };
        Ok(bytemuck::cast_slice(&[m]).to_vec())
    })
}

/// Inputs: [x (f32)]  Output: [max (f32)]
/// Sprint 001: reduces over all elements (ignores axes).
pub fn reduce_max_handler() -> CustomHandler {
    Arc::new(|inputs, _| {
        let x = bytemuck::cast_slice::<u8, f32>(inputs[0]);
        let m = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        Ok(bytemuck::cast_slice(&[m]).to_vec())
    })
}

/// Inputs: [x (f32)]  Output: [min (f32)]
/// Sprint 001: reduces over all elements (ignores axes).
pub fn reduce_min_handler() -> CustomHandler {
    Arc::new(|inputs, _| {
        let x = bytemuck::cast_slice::<u8, f32>(inputs[0]);
        let m = x.iter().cloned().fold(f32::INFINITY, f32::min);
        Ok(bytemuck::cast_slice(&[m]).to_vec())
    })
}

// ── LogSoftmax ──────────────────────────────────────────────────────────

/// Inputs: [x (f32)]  Output: [log_softmax(x) (f32)]
pub fn log_softmax_handler() -> CustomHandler {
    Arc::new(|inputs, _| {
        let x = bytemuck::cast_slice::<u8, f32>(inputs[0]);
        let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = x.iter().map(|v| (v - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let log_sum = sum.ln();
        let out: Vec<f32> = x.iter().map(|v| v - max - log_sum).collect();
        Ok(bytemuck::cast_slice(&out).to_vec())
    })
}

// ── Attention ──────────────────────────────────────────────────────────────

/// Scaled dot-product attention. Inputs: [Q, K, V (f32)]  Output: [out (f32)]
pub fn attention_handler(head_dim: u32, scale: f32, causal: bool) -> CustomHandler {
    Arc::new(move |inputs, _| {
        let q = bytemuck::cast_slice::<u8, f32>(inputs[0]);
        let k = bytemuck::cast_slice::<u8, f32>(inputs[1]);
        let v = bytemuck::cast_slice::<u8, f32>(inputs[2]);
        let d = head_dim as usize;
        if d == 0 {
            return Ok(Vec::new());
        }
        let seq_q = q.len() / d;
        let seq_k = k.len() / d;
        let mut out = vec![0.0f32; q.len()];

        for qi in 0..seq_q {
            let q_row = &q[qi * d..(qi + 1) * d];
            let mut scores: Vec<f32> = (0..seq_k).map(|ki| {
                let k_row = &k[ki * d..(ki + 1) * d];
                q_row.iter().zip(k_row).map(|(a, b)| a * b).sum::<f32>() * scale
            }).collect();

            if causal {
                for score in &mut scores[(qi + 1)..seq_k] {
                    *score = f32::NEG_INFINITY;
                }
            }

            let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
            let sum: f32 = exps.iter().sum();
            let attn: Vec<f32> = exps.iter().map(|e| e / sum).collect();

            let out_row = &mut out[qi * d..(qi + 1) * d];
            for dim_i in 0..d {
                out_row[dim_i] = (0..seq_k).map(|ki| attn[ki] * v[ki * d + dim_i]).sum();
            }
        }
        Ok(bytemuck::cast_slice(&out).to_vec())
    })
}
