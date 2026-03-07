//! Custom op handler factory functions for AI ops with no native hologram counterpart.
//!
//! Each function returns a `hologram::CustomHandler` (an `Arc` closure) that receives
//! byte-slice inputs and returns output bytes. Compute is done in f32 internally.

use std::sync::Arc;
use hologram::CustomHandler;

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
                for ki in (qi + 1)..seq_k {
                    scores[ki] = f32::NEG_INFINITY;
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
