//! Per-model decode metrics for the wasm worker pool (ADR-0018), with UNITS.
//!
//! Built as the `hologram-backend` example the bench script injects, so it can
//! drive the substrate's int8 GEMV (`matmul_i8_pc_omajor`) and worker pool
//! (`wasm_pool`) directly. Run under wasmtime (wasm32-wasip1-threads): std
//! threads drive the SAME atomics fork-join the browser web workers do, so the
//! decode-throughput scaling is representative of the browser.
//!
//! It composes each metric from the model's REAL decode step — every int8
//! projection GEMV (M=1) summed across layers + the LM head — for real chat
//! models, SERIAL vs POOLED, and reports:
//!   - decode latency          ms / token   (lower = better)
//!   - decode throughput       tokens / s   (= 1000 / latency)
//!   - decode memory bandwidth GB / s        (int8 weights streamed per token)
//!   - speedup                 pooled tok/s ÷ serial tok/s
//!   - TTFT (prefill)          ms           (@ a fixed prompt length; the pool
//!                                            does NOT accelerate this — the
//!                                            substrate runs prefill, m>1, on a
//!                                            serial cache-blocked kernel)
//!
//! Attention (QK^T / softmax / scores·V) is O(seq·head_dim), not a weight GEMV,
//! and is not pool-parallelised; it is excluded from the weight-GEMV sum (small
//! at chat context lengths) — these are weight-bandwidth-bound decode figures.

use hologram_backend::cpu::simd::matmul_i8_pc_omajor;
use hologram_backend::cpu::wasm_pool;
use std::time::Instant;

struct Model {
    name: &'static str,
    hidden: usize,
    interm: usize,
    layers: usize,
    heads: usize,
    kv_heads: usize,
    vocab: usize,
}

// Real HF configs (config.json) for models a user actually chats with.
const MODELS: &[Model] = &[
    Model { name: "Qwen2.5-0.5B", hidden: 896,  interm: 4864,  layers: 24, heads: 14, kv_heads: 2, vocab: 151936 },
    Model { name: "Llama-3.2-1B", hidden: 2048, interm: 8192,  layers: 16, heads: 32, kv_heads: 8, vocab: 128256 },
    Model { name: "Qwen2.5-1.5B", hidden: 1536, interm: 8960,  layers: 28, heads: 12, kv_heads: 2, vocab: 151936 },
    Model { name: "Qwen2.5-3B",   hidden: 2048, interm: 11008, layers: 36, heads: 16, kv_heads: 2, vocab: 151936 },
    Model { name: "Llama-3.2-3B", hidden: 3072, interm: 8192,  layers: 28, heads: 24, kv_heads: 8, vocab: 128256 },
    Model { name: "Phi-3-mini-3.8B", hidden: 3072, interm: 8192, layers: 32, heads: 32, kv_heads: 32, vocab: 32064 },
    Model { name: "Qwen2.5-7B",   hidden: 3584, interm: 18944, layers: 28, heads: 28, kv_heads: 4, vocab: 152064 },
    Model { name: "Llama-3.1-8B", hidden: 4096, interm: 14336, layers: 32, heads: 32, kv_heads: 8, vocab: 128256 },
];

const PROMPT_LEN: usize = 128; // prefill length for the TTFT figure

// Time one int8 GEMV (M rows × K × N), returning microseconds per call. Iters
// auto-scale to keep each measurement ~a few ms of work regardless of size.
fn time_matmul(m: usize, k: usize, n: usize) -> f64 {
    let a = vec![0.03f32; m * k];
    let bq = vec![1i8; k * n];
    let scales = vec![0.01f32; n];
    let mut out = vec![0f32; m * n];
    matmul_i8_pc_omajor(&a, &bq, &scales, &mut out, m, k, n); // warm
    // u64: on wasm32 `usize` is 32-bit and m·k·n overflows for large GEMMs.
    let iters = (30_000_000u64 / (m as u64 * k as u64 * n as u64)).clamp(3, 500) as usize;
    let t0 = Instant::now();
    for _ in 0..iters {
        matmul_i8_pc_omajor(&a, &bq, &scales, &mut out, m, k, n);
        std::hint::black_box(&out);
    }
    t0.elapsed().as_secs_f64() * 1e6 / iters as f64
}

// The int8 weight bytes streamed per decode step (≈ the model's resident params).
// All in u64 — on wasm32 `usize` is 32-bit and these products overflow it.
fn decode_weight_bytes(md: &Model) -> u64 {
    let h = md.hidden as u64;
    let head_dim = md.hidden / md.heads;
    let kv = (md.kv_heads * head_dim) as u64;
    let interm = md.interm as u64;
    let per_layer = 2 * h * h        // Q, O
        + 2 * h * kv                  // K, V
        + 2 * h * interm              // gate, up
        + interm * h; // down
    md.layers as u64 * per_layer + h * md.vocab as u64
}

// Sum the decode-step GEMV time (µs/token): every projection across all layers
// plus the LM head, all at M=1.
fn decode_us(md: &Model) -> f64 {
    let head_dim = md.hidden / md.heads;
    let kv = md.kv_heads * head_dim;
    let t_qo = time_matmul(1, md.hidden, md.hidden);
    let t_kv = time_matmul(1, md.hidden, kv);
    let t_gu = time_matmul(1, md.hidden, md.interm);
    let t_dn = time_matmul(1, md.interm, md.hidden);
    let t_head = time_matmul(1, md.hidden, md.vocab);
    md.layers as f64 * (2.0 * t_qo + 2.0 * t_kv + 2.0 * t_gu + t_dn) + t_head
}

// Prefill time (µs) for a P-token prompt: the projections as M=P GEMMs (the
// substrate's m>1 kernel — NOT pooled), plus the head for the last position
// (M=1). This is the compute behind time-to-first-token.
fn prefill_us(md: &Model, p: usize) -> f64 {
    let head_dim = md.hidden / md.heads;
    let kv = md.kv_heads * head_dim;
    let t_qo = time_matmul(p, md.hidden, md.hidden);
    let t_kv = time_matmul(p, md.hidden, kv);
    let t_gu = time_matmul(p, md.hidden, md.interm);
    let t_dn = time_matmul(p, md.interm, md.hidden);
    let t_head = time_matmul(1, md.hidden, md.vocab);
    md.layers as f64 * (2.0 * t_qo + 2.0 * t_kv + 2.0 * t_gu + t_dn) + t_head
}

fn params_b(md: &Model) -> f64 {
    decode_weight_bytes(md) as f64 / 1e9 // ~params in billions (int8 = 1 byte/param)
}

fn main() {
    let workers: u32 = std::env::var("POOL_WORKERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| {
            std::thread::available_parallelism().map(|n| n.get() as u32 - 1).unwrap_or(3)
        });

    println!("# Per-model decode metrics — wasm worker pool (ADR-0018)");
    println!("# participants = {} workers + main = {}; prompt_len = {}\n", workers, workers + 1, PROMPT_LEN);

    // Serial baseline (0 workers registered → matmul runs the serial kernel).
    let mut serial = Vec::new();
    for md in MODELS {
        serial.push((decode_us(md), prefill_us(md, PROMPT_LEN)));
    }

    // Bring the pool up and re-measure BOTH decode (m=1) and prefill (m>1). As of
    // substrate v0.8.2 the m>1 GEMM pools too, so prefill/TTFT parallelises.
    let _h: Vec<_> = (0..workers)
        .map(|i| std::thread::spawn(move || wasm_pool::hologram_worker_run(i)))
        .collect();
    while wasm_pool::hologram_pool_workers() < workers {
        std::thread::yield_now();
    }
    let mut pooled = Vec::new();
    for md in MODELS {
        pooled.push((decode_us(md), prefill_us(md, PROMPT_LEN)));
    }
    wasm_pool::hologram_pool_shutdown();

    // decode tok/s (serial → pooled) and TTFT ms (serial → pooled). Both pool now.
    println!(
        "{:<17} {:>5} | {:>8} {:>8} {:>7} | {:>8} {:>8} {:>7}",
        "model", "~GB", "dec tok/s", "→pooled", "sp", "TTFT ms", "→pooled", "sp"
    );
    for (i, md) in MODELS.iter().enumerate() {
        let (sdec, spf) = serial[i];
        let (pdec, ppf) = pooled[i];
        let s_toks = 1e6 / sdec;
        let p_toks = 1e6 / pdec;
        let s_ttft = (spf + sdec) / 1e3;
        let p_ttft = (ppf + pdec) / 1e3;
        println!(
            "{:<17} {:>5.1} | {:>8.2} {:>8.2} {:>5.2}x | {:>8.0} {:>8.0} {:>5.2}x",
            md.name, params_b(md),
            s_toks, p_toks, p_toks / s_toks,
            s_ttft, p_ttft, s_ttft / p_ttft
        );
    }
    println!("\n# decode = per-token steady-state (tok/s); TTFT = prefill (m={PROMPT_LEN} GEMM) + 1 decode step (ms).");
    println!("# v0.8.2 pools BOTH: decode (m=1) AND prefill (m>1) → TTFT now parallelises. sp = pooled speedup.");
}
