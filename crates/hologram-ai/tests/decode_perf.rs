//! Decode-path throughput & latency benchmark with in-repo-overhead attribution.
//!
//! Drives the REAL staged int8 decode pipeline (resident, chunked int8 head)
//! over synthetic weights — no download — with CHANGING positions each step (so
//! nothing is elided that a real generation would not elide). It reports:
//!   * TTFT: window compile + first-token stage materialization.
//!   * Steady-state per-token latency / throughput (all stages resident).
//!   * Attribution: what fraction of a token is the substrate forward vs the
//!     in-repo per-token overhead (the port-info clones + attention-mask build).
//!
//! `#[ignore]` — it builds a mid-sized model, too heavy for the default lane;
//! run it explicitly:
//!   cargo test -p hologram-ai --test decode_perf --release -- --ignored --nocapture

mod common;

use common::families::{dummy_bf16_bytes, Dims, FamilyScale, LLAMA};

use std::collections::HashMap;
use std::hint::black_box;
use std::num::NonZeroU64;
use std::time::Instant;

use hologram_ai::engine::LmSession;
use hologram_ai::materialize::DirKappaStore;
use hologram_ai::quantized::{crystallize_quantized, crystallize_quantized_range};
use hologram_ai::staged::{head_quant_chunks, quantizable_weights, GrowableStagedSession};
use hologram_ai::DecodeSession;
use hologram_ai_common::lower::{quant_key, QuantMap};
use hologram_ai_common::DType;

/// A mid-sized bench scale: realistic WIDTH (so the substrate forward is
/// non-trivial) at a modest depth (so the model builds in seconds). Parametric
/// — no term is a model's magic number; `head_dim · heads == hidden`.
fn bench_dims() -> Dims {
    Dims {
        hidden_size: 1024,
        layers: 8,
        num_attention_heads: 16,
        num_key_value_heads: 4,
        head_dim: 64,
        intermediate_size: 2752,
        vocab_size: 16_384, // large enough to chunk the head
        max_position_embeddings: 4096,
        rope_theta: 10_000.0,
        rms_norm_eps: 1e-6,
    }
}

const PROMPT_LEN: usize = 24;
const STEPS: usize = 64;

fn argmax(row: &[f32]) -> usize {
    row.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

#[test]
#[ignore = "benchmark — run explicitly with --ignored --release --nocapture"]
fn decode_throughput_latency_and_overhead_attribution() {
    let scale = FamilyScale::new(LLAMA, bench_dims());
    let config_json = scale.config_json();
    let (keys, shapes): (Vec<String>, Vec<Vec<u64>>) = scale.manifest().into_iter().unzip();
    let dtypes = vec![DType::BF16; keys.len()];
    let one = NonZeroU64::new(1).expect("1 is non-zero");

    // ── build the int8 κ-store (projections + chunked head) ──────────────────
    let build_t = Instant::now();
    let store_dir = std::env::temp_dir().join(format!("hai-perf-{}", std::process::id()));
    std::fs::create_dir_all(&store_dir).expect("temp dir");
    let mut dir = DirKappaStore::new(&store_dir);
    let kappas: Vec<String> = keys
        .iter()
        .zip(&shapes)
        .map(|(name, dims)| dir.insert(&dummy_bf16_bytes(name, dims)).expect("persist"))
        .collect();
    let idx_of: HashMap<&str, usize> = kappas
        .iter()
        .enumerate()
        .map(|(i, k)| (k.as_str(), i))
        .collect();
    let mut quant = QuantMap::new();
    for wide in
        &quantizable_weights(&config_json, &keys, &kappas, &shapes, &dtypes, None, one).unwrap()
    {
        let i = idx_of[wide.as_str()];
        let e =
            crystallize_quantized(&mut dir, wide, DType::BF16, shapes[i][0], shapes[i][1]).unwrap();
        quant.insert(wide.clone(), e);
    }
    let head_targets =
        head_quant_chunks(&config_json, &keys, &kappas, &shapes, &dtypes, None, one).unwrap();
    for t in &head_targets {
        let e = crystallize_quantized_range(
            &mut dir,
            &t.kappa,
            t.offset,
            t.len,
            DType::BF16,
            t.out_features,
            t.in_features,
        )
        .unwrap();
        quant.insert(quant_key(&t.kappa, Some((t.offset, t.len))), e);
    }
    let build_ms = build_t.elapsed().as_secs_f64() * 1e3;

    let mut session = GrowableStagedSession::new(
        config_json.clone(),
        keys.clone(),
        kappas.clone(),
        shapes.clone(),
        dtypes.clone(),
        None,
        one,
        Box::new(DirKappaStore::new(&store_dir)),
    )
    .unwrap();
    session.set_residency_budget(1 << 32); // ample — the whole model stays resident
    session.set_bound_by_footprint(true);
    session.set_quant_map(quant);

    // ── TTFT: window compile + first-token materialization ───────────────────
    let ctx = scale.dims.max_position_embeddings as usize;
    let compile_t = Instant::now();
    // Size the bucket to hold prompt + all steps so the steady-state loop never
    // grows — growth is a separate, one-off cost, not the per-token figure.
    let step = session.decode_runner_for(PROMPT_LEN + STEPS + 8).unwrap();
    let stage_count = step.stage_count();
    let compile_ms = compile_t.elapsed().as_secs_f64() * 1e3;

    let mut decode = DecodeSession::new(step, scale.dims.rope_theta as f32, ctx as u64).unwrap();
    let prompt: Vec<i64> = (0..PROMPT_LEN as i64)
        .map(|i| (i * 7 + 1) % scale.dims.vocab_size as i64)
        .collect();

    // Warm: materialize every stage once (so prefill comparisons below are pure
    // compute, not materialization).
    let ttft_t = Instant::now();
    decode.feed(&prompt).unwrap();
    let ttft_ms = ttft_t.elapsed().as_secs_f64() * 1e3;
    let mats_after_prefill = decode.runner().materialization_count();

    // PREFILL COST: stepping (M=1 × PROMPT_LEN, resident) vs a chunked-prefill
    // SEEDER (batched M=chunk). This is what the seeder-install gate decides.
    decode.reset();
    let pstep_t = Instant::now();
    decode.feed(&prompt).unwrap(); // no seeder → PROMPT_LEN M=1 passes
    let prefill_step_ms = pstep_t.elapsed().as_secs_f64() * 1e3;

    decode.reset();
    let seed_bucket = decode.geometry().bucket;
    let seed_chunk = (hologram_ai::engine::geometric_window(1, ctx) as u64).min(seed_bucket as u64);
    let seeder = session
        .chunk_runner_for(PROMPT_LEN + STEPS + 8, seed_chunk)
        .unwrap();
    decode.set_seeder(seeder).unwrap();
    let pseed_t = Instant::now();
    let mut row = decode.feed(&prompt).unwrap(); // seeder → ceil(PROMPT_LEN/chunk) batched passes (+ materialize)
    let prefill_seed_ms = pseed_t.elapsed().as_secs_f64() * 1e3;
    let prefill_speedup = prefill_step_ms / prefill_seed_ms.max(1e-9);

    // ── steady-state per-token (all stages resident) ─────────────────────────
    let mut next = argmax(&row) as i64;
    // one warm step to settle
    row = decode.step(next).unwrap();
    next = argmax(&row) as i64;
    let mats_before = decode.runner().materialization_count();
    let steady_t = Instant::now();
    for _ in 0..STEPS {
        row = decode.step(next).unwrap();
        next = argmax(&row) as i64;
        black_box(&row);
    }
    let steady = steady_t.elapsed();
    let per_token_us = steady.as_secs_f64() * 1e6 / STEPS as f64;
    let mats_added = decode.runner().materialization_count() - mats_before;
    let tok_s = 1e6 / per_token_us;

    // ── in-repo per-token OVERHEAD, measured directly ────────────────────────
    // (1) the port-info clones DecodeState rebuilds every pass (input+output).
    let runner = decode.runner();
    let clone_iters = 2000u32;
    let clone_t = Instant::now();
    for _ in 0..clone_iters {
        black_box(runner.input_port_info());
        black_box(runner.output_port_info());
    }
    let clone_us = clone_t.elapsed().as_secs_f64() * 1e6 / clone_iters as f64;

    // (2) the decode mask DecodeState rebuilds every step (g·(bucket+1) f32 +
    //     the byte image), sized to the CURRENT geometry.
    let geom = decode.geometry();
    let g = geom.heads / geom.kv_heads;
    let span = geom.bucket + 1;
    let mask_iters = 2000u32;
    let mask_t = Instant::now();
    for _ in 0..mask_iters {
        let mut mask = vec![0.0f32; g * span];
        for (i, slot) in mask.iter_mut().enumerate() {
            if i % 3 == 0 {
                *slot = -1e9;
            }
        }
        let bytes: Vec<u8> = mask.iter().flat_map(|v| v.to_le_bytes()).collect();
        black_box(bytes);
    }
    let mask_us = mask_t.elapsed().as_secs_f64() * 1e6 / mask_iters as f64;

    let overhead_us = clone_us + mask_us;
    let overhead_pct = 100.0 * overhead_us / per_token_us;

    eprintln!("\n================ decode-path benchmark (native, int8, resident) ================");
    eprintln!(
        "model: hidden {} · {} layers · vocab {} · {} heads / {} kv · {} stages ({} head chunk(s))",
        scale.dims.hidden_size,
        scale.dims.layers,
        scale.dims.vocab_size,
        scale.dims.num_attention_heads,
        scale.dims.num_key_value_heads,
        stage_count,
        head_targets.len()
    );
    eprintln!("-- one-time / TTFT --------------------------------------------------------------");
    eprintln!("  build int8 κ-store (download-time, once/model) : {build_ms:>9.1} ms");
    eprintln!("  compile decode window (weightless)            : {compile_ms:>9.1} ms");
    eprintln!("  TTFT = prefill {PROMPT_LEN} tok + materialize {stage_count} stages : {ttft_ms:>9.1} ms  ({mats_after_prefill} materializations)");
    eprintln!(
        "  prefill {PROMPT_LEN} tok by STEPPING (M=1, resident)     : {prefill_step_ms:>9.1} ms"
    );
    eprintln!("  prefill {PROMPT_LEN} tok by SEEDER  (batched M=chunk)    : {prefill_seed_ms:>9.1} ms   ({prefill_speedup:.1}x faster prefill)");
    eprintln!("-- steady-state per token (all resident) ----------------------------------------");
    eprintln!("  per token                                     : {per_token_us:>9.1} µs   ({tok_s:.1} tok/s)");
    eprintln!(
        "  re-materializations across {STEPS} steps          : {mats_added:>9}   (0 = no thrash)"
    );
    eprintln!("-- in-repo per-token overhead (the fixable part) --------------------------------");
    eprintln!("  port-info clones (input+output)               : {clone_us:>9.3} µs");
    eprintln!(
        "  decode-mask build (g={g}, bucket={})            : {mask_us:>9.3} µs",
        geom.bucket
    );
    eprintln!("  overhead total                                : {overhead_us:>9.3} µs   ({overhead_pct:.2}% of a token)");
    eprintln!("================================================================================\n");

    assert_eq!(
        mats_added, 0,
        "steady state must not re-materialize (resident)"
    );
    assert!(per_token_us > 0.0);

    // NO assertion on `prefill_seed_ms < prefill_step_ms`. There used to be one,
    // carrying the claim "the seeder must be installed for prefill regardless of
    // residency". Substrate v0.8.0 falsified it, on this machine and in the
    // browser alike:
    //
    //             per token   prefill-by-stepping   prefill-by-seeder
    //   v0.7.2   456,946 µs         10,931 ms             1,088 ms   seeder 10.0x
    //   v0.8.0    19,702 µs            462 ms               941 ms   seeder  0.5x
    //
    // v0.8.0 dispatches the fused per-channel W8A32 decode GEMV on x86 (and on
    // wasm32+simd128, which we build with), but only at `m ≤ FUSED_W8A32_MAX_M`.
    // Stepping runs at `m = 1` and takes it — 23.6x faster. The seeder runs at
    // `m = chunk = 64`, misses it, and barely moved. So batching now TRADES the
    // fused kernel for weight-stream amortization, and wins only when the weights
    // must actually be streamed.
    //
    // Which strategy wins is therefore a property of the substrate's dispatch
    // gates and of residency — not a constant, and not something a benchmark may
    // assert as a law. Asserting the inequality either way would fit this test to
    // one machine, one model size, and one substrate version. It reports instead;
    // the policy belongs where the choice is made, with its own witness.
    //
    // What IS a law: stepping the prompt and stepping in steady state are the same
    // `m = 1` call, so their per-token costs must agree. That pins the measurement
    // itself without pinning the strategy.
    let step_per_tok_us = prefill_step_ms * 1_000.0 / PROMPT_LEN as f64;
    let ratio = step_per_tok_us / per_token_us;
    assert!(
        (0.5..=2.0).contains(&ratio),
        "prefill-by-stepping ({step_per_tok_us:.0} µs/tok) and steady-state decode \
         ({per_token_us:.0} µs/tok) are the same M=1 call and must cost the same \
         within a factor of two; ratio {ratio:.2} means one of them is not taking \
         the path we think it is"
    );
    // The in-repo per-token overhead is a rounding error against substrate
    // compute — the decode per-token levers are the substrate's, not ours.
    assert!(
        overhead_pct < 5.0,
        "in-repo per-token overhead {overhead_pct:.2}% unexpectedly large — investigate"
    );
}
