//! Per-model decode metrics for the wasm worker pool (ADR-0018), with UNITS.
//!
//! Built as the `hologram-backend` example the bench script injects, so it can
//! drive the substrate's int8 GEMV (`matmul_i8_pc_omajor`), the f32 attention
//! matmul (`matmul_f32_blocked`), and the worker pool (`wasm_pool`) directly.
//! Run under wasmtime (wasm32-wasip1-threads): std threads drive the SAME atomics
//! fork-join the browser web workers do, so the scaling is representative.
//!
//! Two views:
//!
//!  1. **Per-model decode + TTFT** (serial → pooled), the ADR-0018 headline:
//!     - decode latency ms/token, throughput tok/s, speedup;
//!     - TTFT (prefill, m=P GEMM) ms, serial → pooled.
//!
//!  2. **Decode-step DECOMPOSITION at chat context lengths** — the honest
//!     Amdahl picture. The weight GEMV pools; the rest of the per-token step does
//!     NOT (substrate decode map, f031e8b): the exploded attention block
//!     (QK^T / softmax / P·V, per kv-group per layer), the KV-cache recopy
//!     (Concat/Transpose read+write the whole cache each step), and sampling.
//!     Attention COMPUTE ∝ layers·hidden·L, so as context L grows the serial
//!     work overtakes the (pooled) weight GEMV and the pool speedup erodes. This
//!     view quantifies that per model and per context length — the ceiling the
//!     GEMV pool alone cannot lift.
//!
//! Attention matmuls only cross the f32 pool threshold (m·k·n ≥ 2^23) at LONG
//! context; softmax, KV-recopy and sampling never pool. We measure attention
//! both serial and pooled to show exactly where it starts to benefit.

use hologram_backend::cpu::simd::{matmul_f32_blocked, matmul_i8_pc_omajor};
use hologram_backend::cpu::wasm_pool;
use std::hint::black_box;
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
// Context (KV-cache) lengths for the decode-step decomposition. Chat scale: a
// short turn to a long-context session. The pool cap is the wasm32 4 GiB space,
// not any of these — they only vary the SERIAL attention/KV cost per token.
const CONTEXTS: &[usize] = &[128, 512, 2048, 8192, 32768];
// Decomposition is shown for a size spread (physics generalises across the rest).
const DECOMP_MODELS: &[&str] = &["Qwen2.5-0.5B", "Qwen2.5-1.5B", "Qwen2.5-3B", "Qwen2.5-7B"];

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
        black_box(&out);
    }
    t0.elapsed().as_secs_f64() * 1e6 / iters as f64
}

// Time one f32 matmul (M × K × N) — the attention QK^T / P·V kernel. Serial when
// no pool workers are registered; pools (m·k·n ≥ 2^23) when they are.
fn time_matmul_f32(m: usize, k: usize, n: usize) -> f64 {
    if m == 0 || k == 0 || n == 0 {
        return 0.0;
    }
    let a = vec![0.02f32; m * k];
    let b = vec![0.02f32; k * n];
    let mut out = vec![0f32; m * n];
    let mut scratch = Vec::new();
    matmul_f32_blocked(&a, &b, &mut out, m, k, n, &mut scratch); // warm
    let iters = (30_000_000u64 / (m as u64 * k as u64 * n as u64).max(1)).clamp(3, 500) as usize;
    let t0 = Instant::now();
    for _ in 0..iters {
        matmul_f32_blocked(&a, &b, &mut out, m, k, n, &mut scratch);
        black_box(&out);
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

// Sum the decode-step weight-GEMV time (µs/token): every projection across all
// layers plus the LM head, all at M=1. Serial or pooled per the registered pool.
fn decode_gemv_us(md: &Model) -> f64 {
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
// substrate's m>1 kernel — pooled on v0.8.2), plus the head for the last position
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

// ---- Decode-step NON-GEMV components (serial in the current impl) ----

// Attention QK^T + P·V (f32), summed over kv-groups × layers, at context L.
// Per group: QK^T = (g_q × head_dim)·(head_dim × L); P·V = (g_q × L)·(L × head_dim),
// where g_q = heads/kv_heads query rows share one KV head. Total attention COMPUTE
// = 2·layers·hidden·L MACs/token — grows with context, unlike the fixed weight GEMV.
fn attn_matmul_us(md: &Model, l: usize) -> f64 {
    let head_dim = md.hidden / md.heads;
    let g_q = (md.heads / md.kv_heads).max(1);
    let t_qk = time_matmul_f32(g_q, head_dim, l);
    let t_pv = time_matmul_f32(g_q, l, head_dim);
    md.layers as f64 * md.kv_heads as f64 * (t_qk + t_pv)
}

// Softmax over the attention scores: `heads` query rows, each a serial max/exp/sum
// over L keys, × layers. Never pools.
fn softmax_us(md: &Model, l: usize) -> f64 {
    let rows = md.heads;
    let mut scores = vec![0.1f32; rows * l];
    let run = |s: &mut [f32]| {
        for r in s.chunks_mut(l) {
            let mut mx = f32::MIN;
            for &x in r.iter() {
                if x > mx {
                    mx = x;
                }
            }
            let mut sum = 0f32;
            for x in r.iter_mut() {
                *x = (*x - mx).exp();
                sum += *x;
            }
            let inv = 1.0 / sum;
            for x in r.iter_mut() {
                *x *= inv;
            }
        }
    };
    run(&mut scores); // warm
    let iters = (60_000_000u64 / (rows as u64 * l as u64).max(1)).clamp(3, 200) as usize;
    let t0 = Instant::now();
    for _ in 0..iters {
        run(&mut scores);
        black_box(&scores);
    }
    md.layers as f64 * t0.elapsed().as_secs_f64() * 1e6 / iters as f64
}

// KV-cache recopy (Concat past∥new + Transpose): the decode graph reads and
// rewrites the whole K and V cache each step. Modeled as a memcpy of the KV
// bytes (f32), × layers — a memory-bandwidth cost that grows with context.
fn kv_copy_us(md: &Model, l: usize) -> f64 {
    let head_dim = md.hidden / md.heads;
    let elems = 2 * l * md.kv_heads * head_dim; // K and V, f32
    if elems == 0 {
        return 0.0;
    }
    let src = vec![1.0f32; elems];
    let mut dst = vec![0f32; elems];
    dst.copy_from_slice(&src); // warm
    let iters = (400_000_000u64 / elems as u64).clamp(3, 500) as usize;
    let t0 = Instant::now();
    for _ in 0..iters {
        dst.copy_from_slice(&src);
        black_box(&dst);
    }
    md.layers as f64 * t0.elapsed().as_secs_f64() * 1e6 / iters as f64
}

// Greedy sampling: argmax over the vocab logits — one serial O(vocab) scan/token.
fn argmax_us(vocab: usize) -> f64 {
    let logits: Vec<f32> = (0..vocab).map(|i| ((i as f32) * 0.0007).sin()).collect();
    let run = |l: &[f32]| {
        let mut bi = 0usize;
        let mut bv = f32::MIN;
        for (i, &x) in l.iter().enumerate() {
            if x > bv {
                bv = x;
                bi = i;
            }
        }
        bi
    };
    black_box(run(&logits));
    let iters = 200usize;
    let t0 = Instant::now();
    for _ in 0..iters {
        black_box(run(&logits));
    }
    t0.elapsed().as_secs_f64() * 1e6 / iters as f64
}

// Temperature/top-k sampling as CURRENTLY implemented: a full O(vocab·log vocab)
// sort of the logits (generate.rs). Reported separately — it is the cost that a
// partial top-k selection would remove.
fn sort_sample_us(vocab: usize) -> f64 {
    let base: Vec<f32> = (0..vocab).map(|i| ((i as f32) * 0.0011).sin()).collect();
    let run = |b: &[f32]| {
        let mut idx: Vec<u32> = (0..b.len() as u32).collect();
        idx.sort_unstable_by(|&a, &c| {
            b[c as usize].partial_cmp(&b[a as usize]).unwrap_or(std::cmp::Ordering::Equal)
        });
        idx[0]
    };
    black_box(run(&base));
    let iters = 20usize;
    let t0 = Instant::now();
    for _ in 0..iters {
        black_box(run(&base));
    }
    t0.elapsed().as_secs_f64() * 1e6 / iters as f64
}

fn params_b(md: &Model) -> f64 {
    decode_weight_bytes(md) as f64 / 1e9 // ~params in billions (int8 = 1 byte/param)
}

fn find(name: &str) -> &'static Model {
    MODELS.iter().find(|m| m.name == name).unwrap()
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

    // ---- SERIAL phase (0 workers): baseline GEMV + all non-GEMV components ----
    let mut serial_gemv = Vec::new();
    let mut serial_prefill = Vec::new();
    for md in MODELS {
        serial_gemv.push(decode_gemv_us(md));
        serial_prefill.push(prefill_us(md, PROMPT_LEN));
    }
    // Non-GEMV decode components (measured serial — the current impl never pools
    // softmax/KV-copy/sampling; attention pools only above 2^23, measured below).
    // Indexed [model][context].
    let mut attn_s = vec![vec![0f64; CONTEXTS.len()]; MODELS.len()];
    let mut softmax = vec![vec![0f64; CONTEXTS.len()]; MODELS.len()];
    let mut kvcopy = vec![vec![0f64; CONTEXTS.len()]; MODELS.len()];
    for (i, md) in MODELS.iter().enumerate() {
        if !DECOMP_MODELS.contains(&md.name) {
            continue;
        }
        for (j, &l) in CONTEXTS.iter().enumerate() {
            attn_s[i][j] = attn_matmul_us(md, l);
            softmax[i][j] = softmax_us(md, l);
            kvcopy[i][j] = kv_copy_us(md, l);
        }
    }
    // Sampling per distinct vocab (serial argmax; sort is the temperature cost).
    let argmax: Vec<f64> = MODELS.iter().map(|m| argmax_us(m.vocab)).collect();
    let sortsample: Vec<f64> = MODELS.iter().map(|m| sort_sample_us(m.vocab)).collect();

    // ---- POOLED phase: bring the pool up, re-measure GEMV, prefill, attention ----
    let _h: Vec<_> = (0..workers)
        .map(|i| std::thread::spawn(move || wasm_pool::hologram_worker_run(i)))
        .collect();
    while wasm_pool::hologram_pool_workers() < workers {
        std::thread::yield_now();
    }
    let mut pooled_gemv = Vec::new();
    let mut pooled_prefill = Vec::new();
    for md in MODELS {
        pooled_gemv.push(decode_gemv_us(md));
        pooled_prefill.push(prefill_us(md, PROMPT_LEN));
    }
    let mut attn_p = vec![vec![0f64; CONTEXTS.len()]; MODELS.len()];
    for (i, md) in MODELS.iter().enumerate() {
        if !DECOMP_MODELS.contains(&md.name) {
            continue;
        }
        for (j, &l) in CONTEXTS.iter().enumerate() {
            attn_p[i][j] = attn_matmul_us(md, l);
        }
    }
    wasm_pool::hologram_pool_shutdown();

    // ===== View 1: decode tok/s + TTFT (serial → pooled) =====
    println!("## Weight-GEMV decode + prefill/TTFT (serial → pooled)");
    println!(
        "{:<17} {:>5} | {:>8} {:>8} {:>6} | {:>8} {:>8} {:>6}",
        "model", "~GB", "dec tk/s", "→pooled", "sp", "TTFT ms", "→pooled", "sp"
    );
    for (i, md) in MODELS.iter().enumerate() {
        let (sdec, spf) = (serial_gemv[i], serial_prefill[i]);
        let (pdec, ppf) = (pooled_gemv[i], pooled_prefill[i]);
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

    // ===== View 2: decode-step decomposition at chat context lengths =====
    println!("\n## Decode-step decomposition (µs/token) vs context length L");
    println!("# GEMV pools; attention pools only at large L; softmax/KVcopy/sample never pool.");
    println!("# step_p = pooled GEMV + (pooled attn) + softmax + KVcopy + argmax  (the real per-token time).");
    println!(
        "{:<15} {:>7} | {:>8} {:>8} {:>8} {:>8} {:>7} | {:>8} {:>7} {:>6}",
        "model @ L", "L", "gemv_p", "attn", "softmax", "KVcopy", "smpl", "step_p ms", "tok/s", "poolsp"
    );
    for name in DECOMP_MODELS {
        let md = find(name);
        let i = MODELS.iter().position(|m| m.name == *name).unwrap();
        for (j, &l) in CONTEXTS.iter().enumerate() {
            let gemv_s = serial_gemv[i];
            let gemv_p = pooled_gemv[i];
            let a_s = attn_s[i][j];
            let a_p = attn_p[i][j].min(a_s); // pool never hurts; guard noise
            let sm = softmax[i][j];
            let kv = kvcopy[i][j];
            let smpl = argmax[i];
            let serial_rest = a_s + sm + kv + smpl;
            let pooled_rest = a_p + sm + kv + smpl;
            let step_s = gemv_s + serial_rest;
            let step_p = gemv_p + pooled_rest;
            let toks = 1e6 / step_p;
            let poolsp = step_s / step_p;
            println!(
                "{:<15} {:>7} | {:>8.0} {:>8.0} {:>8.0} {:>8.0} {:>7.0} | {:>8.2} {:>7.1} {:>5.2}x",
                if j == 0 { md.name } else { "" },
                l, gemv_p, a_p, sm, kv, smpl, step_p / 1e3, toks, poolsp
            );
        }
    }

    // Temperature-sampling sort cost (the O(vocab·log vocab) that a partial top-k
    // would remove), shown per distinct vocab.
    println!("\n## Sampling cost per token (serial)");
    println!("{:<17} {:>8} {:>12} {:>14}", "model", "vocab", "argmax µs", "full-sort µs");
    for (i, md) in MODELS.iter().enumerate() {
        if !DECOMP_MODELS.contains(&md.name) {
            continue;
        }
        println!("{:<17} {:>8} {:>12.1} {:>14.1}", md.name, md.vocab, argmax[i], sortsample[i]);
    }

    println!("\n# Reading it: as L grows, attn+softmax+KVcopy (serial) overtake the pooled");
    println!("# GEMV, so poolsp → 1 and tok/s falls. The GEMV pool alone cannot hold");
    println!("# throughput at chat context lengths — attention/KV and sampling are the ceiling.");
}
