//! Size the per-token cost of BLAKE3-re-hashing the carried K/V cache on the
//! byte `execute` path — the cost the kernel-level `pool-bench` never measures.
//!
//! Our decode driver (`decode.rs`) binds `past_k`/`past_v` as whole host byte
//! buffers each token and runs them through `LmSession::execute`, whose
//! substrate impl content-addresses every input via `address_bytes` (BLAKE3).
//! The per-port input cache always misses on decode (the K/V grows each step),
//! so the *entire* carried K/V region is re-hashed every token. The addressed
//! path (`execute_addressed`, already exported on `HoloRunner`) binds by
//! κ-label with no hashing — this bench sizes what that would save.
//!
//! The hashed region per token is `2 (K and V) · layers · kv_heads · bucket ·
//! head_dim · 4 B` (see `DecodeGeometry::kv_row_bytes` / `kv_buffer_bytes`).
//! At a realized context length `L`, `bucket ≈ L`. We hash a buffer of that
//! size with the same BLAKE3 the substrate uses and report ms/token.
//!
//! This measures the SIMD `blake3` crate natively; the deployed wasm32 target
//! (no AVX) hashes slower, so every number here is an OPTIMISTIC lower bound on
//! the deployed per-token tax. Run: `cargo run -q --release --example
//! kv_rehash_cost` (CARGO_TARGET_DIR=/tmp/hai-target).

use std::time::Instant;

/// A real decoder geometry (from the model's own HF config).
struct Geom {
    name: &'static str,
    layers: usize,
    kv_heads: usize,
    head_dim: usize,
}

impl Geom {
    /// Carried K/V bytes hashed per token at realized length `l` (`bucket ≈ l`):
    /// `2 · layers · kv_heads · l · head_dim · 4`.
    fn kv_bytes(&self, l: usize) -> usize {
        2 * self.layers * self.kv_heads * l * self.head_dim * 4
    }
}

fn main() {
    // Warm the SIMD path + get a throughput figure on THIS machine, so the
    // ms/token below are grounded in a measured rate, not an assumed one.
    let probe = vec![0u8; 64 * 1024 * 1024];
    let t = Instant::now();
    let reps = 8;
    let mut sink = 0u8;
    for _ in 0..reps {
        sink ^= blake3::hash(&probe).as_bytes()[0];
    }
    let secs = t.elapsed().as_secs_f64();
    let gbps = (probe.len() as f64 * reps as f64) / secs / 1e9;
    std::hint::black_box(sink);
    println!("BLAKE3 (SIMD, native) throughput on this host: {gbps:.2} GB/s\n");

    let models = [
        Geom {
            name: "Qwen2.5-0.5B",
            layers: 24,
            kv_heads: 2,
            head_dim: 64,
        },
        Geom {
            name: "Qwen2.5-1.5B",
            layers: 28,
            kv_heads: 2,
            head_dim: 128,
        },
        Geom {
            name: "Qwen2.5-7B",
            layers: 28,
            kv_heads: 4,
            head_dim: 128,
        },
    ];
    let lengths = [128usize, 2048, 8192, 32768];

    for m in &models {
        println!(
            "{} ({} layers, {} kv-heads, {} head-dim)",
            m.name, m.layers, m.kv_heads, m.head_dim
        );
        println!("  {:>7} | {:>10} | {:>14}", "L", "KV MB", "re-hash ms/tok");
        for &l in &lengths {
            let bytes = m.kv_bytes(l);
            let buf = vec![0u8; bytes];
            // Median of a few reps: this is the per-token hash the byte path pays.
            let mut best = f64::INFINITY;
            for _ in 0..3 {
                let t = Instant::now();
                let h = blake3::hash(&buf);
                std::hint::black_box(h.as_bytes()[0]);
                best = best.min(t.elapsed().as_secs_f64());
            }
            let mb = bytes as f64 / (1024.0 * 1024.0);
            println!("  {:>7} | {:>10.1} | {:>14.2}", l, mb, best * 1e3);
        }
        println!();
    }

    println!(
        "Reading: this per-token cost is ADDED to the substrate step time the\n\
         kernel pool-bench reports; it is O(L), paid every token, and is\n\
         removed BYTE-IDENTICALLY by binding the resident K/V by κ-label\n\
         (execute_addressed) instead of re-hashing its bytes. Deployed wasm32\n\
         hashes slower than this native SIMD figure, so the real tax is larger."
    );
}
