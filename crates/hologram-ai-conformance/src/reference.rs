//! Pure-Rust reference implementations for complex ops.
//!
//! These serve as the "ground truth" for ops that don't have simple
//! hand-computed expected values. Each function implements the math
//! spec for an op without any optimization tricks — correctness over speed.

#![allow(clippy::too_many_arguments)]

use std::f32::consts::PI;

/// Reference Softmax: exp(x_i - max) / sum(exp(x_j - max))
pub fn softmax(input: &[f32], size: usize) -> Vec<f32> {
    let mut out = input.to_vec();
    for row in out.chunks_mut(size) {
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for v in row.iter_mut() {
            *v = (*v - max).exp();
            sum += *v;
        }
        for v in row.iter_mut() {
            *v /= sum;
        }
    }
    out
}

/// Reference LogSoftmax: x_i - max - log(sum(exp(x_j - max)))
pub fn log_softmax(input: &[f32], size: usize) -> Vec<f32> {
    let mut out = input.to_vec();
    for row in out.chunks_mut(size) {
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let log_sum_exp: f32 = row.iter().map(|v| (v - max).exp()).sum::<f32>().ln();
        for v in row.iter_mut() {
            *v = *v - max - log_sum_exp;
        }
    }
    out
}

/// Reference RmsNorm: x / rms(x) * weight, where rms = sqrt(mean(x^2) + eps)
pub fn rms_norm(input: &[f32], weight: &[f32], size: usize, epsilon: f32) -> Vec<f32> {
    let mut out = input.to_vec();
    for row in out.chunks_mut(size) {
        let ms: f32 = row.iter().map(|v| v * v).sum::<f32>() / size as f32;
        let rms = (ms + epsilon).sqrt();
        for (v, &w) in row.iter_mut().zip(weight.iter()) {
            *v = (*v / rms) * w;
        }
    }
    out
}

/// Reference LayerNorm: (x - mean) / sqrt(var + eps) * weight + bias
pub fn layer_norm(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    size: usize,
    epsilon: f32,
) -> Vec<f32> {
    let mut out = input.to_vec();
    for row in out.chunks_mut(size) {
        let mean: f32 = row.iter().sum::<f32>() / size as f32;
        let var: f32 = row.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / size as f32;
        let std = (var + epsilon).sqrt();
        for (i, v) in row.iter_mut().enumerate() {
            *v = ((*v - mean) / std) * weight[i] + bias[i];
        }
    }
    out
}

/// Reference MatMul: [m, k] x [k, n] -> [m, n]
pub fn matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            out[i * n + j] = sum;
        }
    }
    out
}

/// Reference Gemm: alpha * op(A) @ op(B) + beta * C
pub fn gemm(
    a: &[f32],
    b: &[f32],
    c: Option<&[f32]>,
    m: usize,
    k: usize,
    n: usize,
    alpha: f32,
    beta: f32,
    trans_a: bool,
    trans_b: bool,
) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                let a_val = if trans_a { a[p * m + i] } else { a[i * k + p] };
                let b_val = if trans_b { b[j * k + p] } else { b[p * n + j] };
                sum += a_val * b_val;
            }
            out[i * n + j] = alpha * sum;
            if let Some(c) = c {
                out[i * n + j] += beta * c[i * n + j];
            }
        }
    }
    out
}

/// Reference GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
pub fn gelu(x: f32) -> f32 {
    let k = (2.0f32 / PI).sqrt();
    0.5 * x * (1.0 + (k * (x + 0.044715 * x * x * x)).tanh())
}

/// Reference SiLU: x * sigmoid(x)
pub fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

/// Reference sigmoid: 1 / (1 + exp(-x))
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Reference FusedSwiGLU: silu(gate) * up
pub fn fused_swiglu(gate: &[f32], up: &[f32]) -> Vec<f32> {
    gate.iter()
        .zip(up.iter())
        .map(|(&g, &u)| silu(g) * u)
        .collect()
}

/// Reference RoPE (interleaved convention): pairs (0,1), (2,3), ...
pub fn rotary_embedding(
    x: &[f32],
    dim: usize,
    base: f32,
    n_heads: usize,
    start_pos: usize,
) -> Vec<f32> {
    let half = dim / 2;
    let n_heads = n_heads.max(1);
    let mut out = x.to_vec();
    for (chunk_idx, chunk) in out.chunks_mut(dim).enumerate() {
        let token_pos = chunk_idx / n_heads;
        let pos = (start_pos + token_pos) as f32;
        for i in 0..half {
            let freq = 1.0 / base.powf(2.0 * i as f32 / dim as f32);
            let angle = pos * freq;
            let cos_a = angle.cos();
            let sin_a = angle.sin();
            let x0 = chunk[2 * i];
            let x1 = chunk[2 * i + 1];
            chunk[2 * i] = x0 * cos_a - x1 * sin_a;
            chunk[2 * i + 1] = x0 * sin_a + x1 * cos_a;
        }
    }
    out
}

/// Reference Attention: Q @ K^T * scale -> softmax -> @ V
pub fn attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    head_dim: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    scale: f32,
    causal: bool,
) -> Vec<f32> {
    let total_q = q.len() / head_dim;
    let seq_q = total_q / num_q_heads;
    let total_kv = k.len() / head_dim;
    let seq_kv = total_kv / num_kv_heads;
    let heads_per_group = num_q_heads / num_kv_heads;

    let mut output = vec![0.0f32; q.len()];

    for h in 0..num_q_heads {
        let kv_h = h / heads_per_group;

        for t in 0..seq_q {
            // Compute Q @ K^T for this position
            let mut scores: Vec<f32> = (0..seq_kv)
                .map(|s| {
                    if causal && s > t {
                        return f32::NEG_INFINITY;
                    }
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        let q_idx = h * seq_q * head_dim + t * head_dim + d;
                        let k_idx = kv_h * seq_kv * head_dim + s * head_dim + d;
                        dot += q[q_idx] * k[k_idx];
                    }
                    dot * scale
                })
                .collect();

            // Softmax
            let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for s in scores.iter_mut() {
                *s = (*s - max_score).exp();
                sum += *s;
            }
            for s in scores.iter_mut() {
                *s /= sum;
            }

            // Weighted sum of V
            for d in 0..head_dim {
                let mut val = 0.0f32;
                for (s, &score) in scores.iter().enumerate() {
                    let v_idx = kv_h * seq_kv * head_dim + s * head_dim + d;
                    val += score * v[v_idx];
                }
                let out_idx = h * seq_q * head_dim + t * head_dim + d;
                output[out_idx] = val;
            }
        }
    }

    output
}

/// Reference ReduceSum along last `size` elements.
pub fn reduce_sum(input: &[f32], size: usize) -> Vec<f32> {
    input
        .chunks(size)
        .map(|chunk| chunk.iter().sum())
        .collect()
}

/// Reference ReduceMean along last `size` elements.
pub fn reduce_mean(input: &[f32], size: usize) -> Vec<f32> {
    input
        .chunks(size)
        .map(|chunk| chunk.iter().sum::<f32>() / size as f32)
        .collect()
}

/// Reference ReduceMax along last `size` elements.
pub fn reduce_max(input: &[f32], size: usize) -> Vec<f32> {
    input
        .chunks(size)
        .map(|chunk| chunk.iter().copied().fold(f32::NEG_INFINITY, f32::max))
        .collect()
}

/// Reference ReduceMin along last `size` elements.
pub fn reduce_min(input: &[f32], size: usize) -> Vec<f32> {
    input
        .chunks(size)
        .map(|chunk| chunk.iter().copied().fold(f32::INFINITY, f32::min))
        .collect()
}

/// Reference ReduceProd along last `size` elements.
pub fn reduce_prod(input: &[f32], size: usize) -> Vec<f32> {
    input
        .chunks(size)
        .map(|chunk| chunk.iter().product())
        .collect()
}

/// Reference Conv2d (NCHW format, no dilation, no groups).
pub fn conv2d_simple(
    input: &[f32],
    kernel: &[f32],
    in_h: usize,
    in_w: usize,
    k_h: usize,
    k_w: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
) -> Vec<f32> {
    let out_h = (in_h + 2 * pad_h - k_h) / stride_h + 1;
    let out_w = (in_w + 2 * pad_w - k_w) / stride_w + 1;
    let mut output = vec![0.0f32; out_h * out_w];

    for oh in 0..out_h {
        for ow in 0..out_w {
            let mut sum = 0.0f32;
            for kh in 0..k_h {
                for kw in 0..k_w {
                    let ih = oh * stride_h + kh;
                    let iw = ow * stride_w + kw;
                    let ih = ih as isize - pad_h as isize;
                    let iw = iw as isize - pad_w as isize;
                    if ih >= 0 && ih < in_h as isize && iw >= 0 && iw < in_w as isize {
                        sum += input[ih as usize * in_w + iw as usize]
                            * kernel[kh * k_w + kw];
                    }
                }
            }
            output[oh * out_w + ow] = sum;
        }
    }
    output
}
