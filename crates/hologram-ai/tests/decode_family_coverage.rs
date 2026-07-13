//! Parametric family DECODE coverage (integration V&V).
//!
//! The compilation suite proves the parametric registry BUILDS the graph for
//! every faithful decoder family (Llama, Qwen2, Mistral, Phi3). This test goes
//! further: it drives the FULL browser decode pipeline — int8-quantized staged
//! decode, a footprint-bounded shared-ledger session, a step runner PLUS a
//! chunked-prefill seeder, `feed` + `step` generation — end to end for EACH
//! family, and asserts each one actually decodes: finite, non-empty, and
//! reproducible logits/tokens. A break in any single family's decode path
//! (fused qkv/gate-up, q/k/v bias, tied vs untied head, GQA splice) fails the
//! build, so "arbitrary models" is verified across the popular formats, not
//! assumed and not fit to one.
//!
//! Parametric in BOTH the family (the registry's faithful set) and the scale
//! (`Dims::MODEST`, small enough to run in the default suite in seconds). No
//! term is specialized to a particular model or size.

mod common;

use common::families::{dummy_bf16_bytes, is_norm, Dims, FamilyScale, FAITHFUL_FAMILIES};

use std::collections::HashMap;
use std::num::NonZeroU64;

use hologram_ai::materialize::DirKappaStore;
use hologram_ai::quantized::{
    crystallize_quantized_range_tier, crystallize_quantized_tier, QuantTier,
};
use hologram_ai::staged::{head_quant_chunks, quantizable_weights, GrowableStagedSession};
use hologram_ai::{DecodeSession, LmSession, RopeSpec};
use hologram_ai_common::lower::{quant_key, QuantMap};
use hologram_ai_common::DType;

/// A prompt of arbitrary in-vocabulary token ids (values are immaterial — the
/// SHAPES drive the pipeline; these only need to be `< vocab_size`).
const PROMPT: [i64; 6] = [1, 2, 3, 5, 8, 13];
const GEN_STEPS: usize = 4;

fn argmax(row: &[f32]) -> usize {
    row.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Build the full int8-quantized staged decode session for `scale` and generate
/// `GEN_STEPS` greedy tokens after prefilling `PROMPT`. Returns the last logit
/// row and the generated token ids. Mirrors the browser's decode flow: a
/// footprint-bounded shared-ledger session, a step runner (chunk 1), and a
/// chunked-prefill seeder sharing the ledger.
fn decode_family(scale: &FamilyScale) -> (Vec<f32>, Vec<i64>) {
    decode_family_q(scale, true)
}

/// As [`decode_family`], but `quantize` selects the weight tier: `true` is the
/// int8 derived-artifact path (the browser's deployed path — since v0.8.1 that is
/// fused output-major **W8A8**), `false` keeps the wide bf16 weights as the
/// full-precision reference. Same κ-store, same prompt, same decode; only the
/// weight tier differs, so a token divergence between the two is exactly the
/// quantization error the deployed path introduces.
fn decode_family_q(scale: &FamilyScale, quantize: bool) -> (Vec<f32>, Vec<i64>) {
    decode_family_tier(
        scale,
        if quantize {
            Some(QuantTier::Int8)
        } else {
            None
        },
    )
}

/// [`decode_family_q`] parametric in the quant tier: `None` = full-precision bf16
/// reference, `Some(Int8)` / `Some(Int4)` = the derived-artifact tier. Every
/// eligible projection + head chunk is crystallized at `tier` and the whole
/// transformer decodes through it — the browser's exact staged path, per family.
fn decode_family_tier(scale: &FamilyScale, tier: Option<QuantTier>) -> (Vec<f32>, Vec<i64>) {
    let (mut rows, tokens) = decode_family_rows(scale, tier, None);
    (rows.pop().expect("at least the prefill row"), tokens)
}

/// The core run, returning EVERY logit row (prefill + one per generated step;
/// `GEN_STEPS + 1` rows) and the token ids consumed. `force` teacher-forces the
/// generation on a given token stream instead of the greedy argmax — the
/// apples-to-apples instrument for comparing two weight tiers: forcing the
/// reference tier's tokens through the quantized tier compares logits at every
/// position under a SHARED context. (Autonomous generation amplifies a single
/// knife-edge argmax flip — a legitimate quantization outcome on a near-tied
/// logit pair — into full context divergence, after which logit comparisons
/// measure the divergence, not the quantization.)
fn decode_family_rows(
    scale: &FamilyScale,
    tier: Option<QuantTier>,
    force: Option<&[i64]>,
) -> (Vec<Vec<f32>>, Vec<i64>) {
    decode_family_rows_at(scale, tier, force, 1 << 30) // 1 GiB holds this modest model
}

/// [`decode_family_rows`] with an explicit stage-residency budget — the
/// instrument for the eviction-pressure witness (a tiny budget forces every
/// stage to drop and re-materialize each window).
fn decode_family_rows_at(
    scale: &FamilyScale,
    tier: Option<QuantTier>,
    force: Option<&[i64]>,
    budget: u64,
) -> (Vec<Vec<f32>>, Vec<i64>) {
    let quantize = tier.is_some();
    let quant_tier = tier.unwrap_or(QuantTier::Int8);
    let arch = scale.arch();
    let config_json = scale.config_json();
    let (keys, shapes): (Vec<String>, Vec<Vec<u64>>) = scale.manifest().into_iter().unzip();
    let dtypes = vec![DType::BF16; keys.len()];
    let one = NonZeroU64::new(1).expect("1 is non-zero");

    // κ-store: the wide bf16 weights (the download byte set). The directory must
    // be unique per *invocation*, not per (arch, pid): `decode_family_q` is now
    // called several times for the same arch (the quantized run, the bf16
    // reference, the reproducibility re-run), and cargo runs this binary's tests
    // concurrently — a shared `hai-fam-{arch}-{pid}` dir means two decodes writing
    // and reading one κ-store at once, which corrupts both. A per-call counter
    // disambiguates. (v0.8.1's own ffi commit fixed this same collision class.)
    static STORE_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let seq = STORE_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let store_dir =
        std::env::temp_dir().join(format!("hai-fam-{arch}-{}-{seq}", std::process::id()));
    std::fs::create_dir_all(&store_dir).expect("creating κ-store temp dir");
    let mut dir = DirKappaStore::new(&store_dir);
    let mut kappas: Vec<String> = Vec::with_capacity(keys.len());
    for (name, dims) in keys.iter().zip(&shapes) {
        let kappa = dir
            .insert(&dummy_bf16_bytes(name, dims))
            .unwrap_or_else(|e| panic!("{arch}: persisting weight {name}: {e:#}"));
        kappas.push(kappa);
    }

    // int8 derived-artifact tier (the browser's quantized path): each eligible
    // wide projection is crystallized into a matmul-ready per-channel artifact.
    let idx_of: HashMap<&str, usize> = kappas
        .iter()
        .enumerate()
        .map(|(i, k)| (k.as_str(), i))
        .collect();
    let eligible = if quantize {
        quantizable_weights(&config_json, &keys, &kappas, &shapes, &dtypes, None, one)
            .unwrap_or_else(|e| panic!("{arch}: quantizable_weights: {e:#}"))
    } else {
        Vec::new() // full-precision reference: no weight is quantized
    };
    let mut quant = QuantMap::new();
    for wide in &eligible {
        let i = idx_of[wide.as_str()];
        let (out, inf) = (shapes[i][0], shapes[i][1]);
        assert!(
            !is_norm(&keys[i]),
            "{arch}: a norm weight is not a quantizable projection"
        );
        let entry = crystallize_quantized_tier(&mut dir, wide, DType::BF16, quant_tier, out, inf)
            .unwrap_or_else(|e| {
                panic!("{arch}: crystallizing {quant_tier:?} artifact for {wide}: {e:#}")
            });
        quant.insert(wide.clone(), entry);
    }

    // The LM head joins the int8 tier too (row `quantized-transit`, chunked
    // head): each vocab-row chunk of a large head gets its OWN per-chunk
    // artifact, keyed by (κ, range), so a chunked head is a dequant-fused int8
    // matmul instead of a bf16 matmul whose whole-panel F32 image thrashes
    // residency. A no-op where the head is a single chunk (small vocab).
    let head_targets = if quantize {
        head_quant_chunks(&config_json, &keys, &kappas, &shapes, &dtypes, None, one)
            .unwrap_or_else(|e| panic!("{arch}: head_quant_chunks: {e:#}"))
    } else {
        Vec::new()
    };
    for target in head_targets {
        let entry = crystallize_quantized_range_tier(
            &mut dir,
            &target.kappa,
            target.offset,
            target.len,
            DType::BF16,
            quant_tier,
            target.out_features,
            target.in_features,
        )
        .unwrap_or_else(|e| panic!("{arch}: crystallizing head-chunk artifact: {e:#}"));
        quant.insert(
            quant_key(&target.kappa, Some((target.offset, target.len))),
            entry,
        );
    }

    // The footprint-bounded shared-ledger session (the wasm residency model).
    // The budget is generous for this modest scale — the point here is decode
    // CORRECTNESS per family; the shared-ledger bound itself is witnessed by the
    // stage-residency-cache scenarios and the family memory sweep.
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
    .unwrap_or_else(|e| panic!("{arch}: growable session: {e:#}"));
    session.set_residency_budget(budget);
    session.set_bound_by_footprint(true);
    if quantize {
        session.set_quant_map(quant);
    }

    // Step runner (chunk 1) + chunked-prefill seeder, sharing the ledger.
    let ctx = scale.dims.max_position_embeddings as usize;
    let want = PROMPT.len();
    let bucket = hologram_ai::engine::geometric_window(want.max(1), ctx);
    let chunk = (hologram_ai::engine::geometric_window(1, ctx) as u64).min(bucket as u64);

    let step = session
        .decode_runner_for(want)
        .unwrap_or_else(|e| panic!("{arch}: decode step runner: {e:#}"));
    let mut decode = DecodeSession::new(
        step,
        RopeSpec::plain(scale.dims.rope_theta as f32),
        ctx as u64,
    )
    .unwrap_or_else(|e| panic!("{arch}: decode session: {e:#}"));
    if chunk >= 2 {
        let seeder = session
            .chunk_runner_for(want, chunk)
            .unwrap_or_else(|e| panic!("{arch}: prefill seeder: {e:#}"));
        decode
            .set_seeder(seeder)
            .unwrap_or_else(|e| panic!("{arch}: installing seeder: {e:#}"));
    }

    let mut row = decode
        .feed(&PROMPT)
        .unwrap_or_else(|e| panic!("{arch}: prefill (feed): {e:#}"));
    let mut rows = Vec::with_capacity(GEN_STEPS + 1);
    let mut tokens = Vec::with_capacity(GEN_STEPS);
    for s in 0..GEN_STEPS {
        rows.push(row.clone());
        let next = match force {
            Some(f) => f[s],
            None => argmax(&row) as i64,
        };
        tokens.push(next);
        row = decode
            .step(next)
            .unwrap_or_else(|e| panic!("{arch}: decode step {s}: {e:#}"));
    }
    rows.push(row);
    (rows, tokens)
}

/// Every faithful decoder family DECODES end to end: the full int8-quantized
/// staged decode pipeline compiles, prefills, and generates finite, non-empty,
/// reproducible output for Llama, Qwen2, Mistral, and Phi3 alike.
#[test]
fn decode_generates_at_production_head_dim_int8_staged() {
    // The committed fixture runs at head_dim 16 and `MODEST` at head_dim 64;
    // the browser's real-model hang appears only at the PRODUCTION head_dim
    // (128 = 1024/8, the Qwen/Llama value). This is the native (seconds)
    // reproduction of the deployed path: the int8-quantized STAGED decode
    // pipeline, prefilled and generated, at head_dim 128 for every family.
    for layout in FAITHFUL_FAMILIES {
        let scale = FamilyScale::new(*layout, Dims::DEEP);
        let arch = scale.arch();
        assert_eq!(scale.dims.head_dim, 128, "{arch}: production head_dim");

        let (logits, tokens) = decode_family(&scale);
        assert_eq!(
            logits.len(),
            scale.dims.vocab_size as usize,
            "{arch}: logits width must equal the vocabulary at head_dim 128"
        );
        assert!(
            logits.iter().all(|x| x.is_finite()),
            "{arch}: head_dim-128 decode produced non-finite logits"
        );
        assert!(
            tokens.iter().all(|&t| (t as u64) < scale.dims.vocab_size),
            "{arch}: a head_dim-128 generated token is out of vocabulary range"
        );
        let (_, again) = decode_family(&scale);
        assert_eq!(
            tokens, again,
            "{arch}: head_dim-128 decode is not reproducible"
        );
    }
}

#[test]
fn decode_generates_for_every_supported_family() {
    assert!(
        FAITHFUL_FAMILIES.len() >= 4,
        "the coverage frontier must include the popular formats"
    );
    for layout in FAITHFUL_FAMILIES {
        let scale = FamilyScale::new(*layout, Dims::MODEST);
        let arch = scale.arch();

        let (logits, tokens) = decode_family(&scale);

        assert_eq!(
            logits.len(),
            scale.dims.vocab_size as usize,
            "{arch}: logits width must equal the vocabulary"
        );
        assert!(
            logits.iter().all(|x| x.is_finite()),
            "{arch}: decode produced non-finite logits (the forward is numerically broken for this layout)"
        );
        assert_eq!(
            tokens.len(),
            GEN_STEPS,
            "{arch}: decode produced too few tokens"
        );
        assert!(
            tokens.iter().all(|&t| (t as u64) < scale.dims.vocab_size),
            "{arch}: a generated token is out of vocabulary range"
        );

        // Reproducibility: the same inputs decode to the same tokens (a broken
        // splice / residency path often shows as run-to-run drift).
        let (_, again) = decode_family(&scale);
        assert_eq!(
            tokens, again,
            "{arch}: decode is not reproducible run-to-run"
        );

        eprintln!("[family-coverage] {arch}: decoded {tokens:?} (finite, reproducible)");
    }
}

/// Cosine similarity of two logit rows. `1.0` = identical direction; argmax and
/// the whole ranking are preserved as it approaches 1.
fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        dot += f64::from(*x) * f64::from(*y);
        na += f64::from(*x).powi(2);
        nb += f64::from(*y).powi(2);
    }
    dot / (na.sqrt() * nb.sqrt()).max(1e-30)
}

/// **The V&V migration for turning on W8A8.** Since substrate v0.8.1 the deployed
/// int8 tier is the fused output-major **W8A8** decode GEMV — it quantizes the
/// activation per token, so it is a *different function* from the W8A32 it
/// replaced, and no byte-exact transcript can gate it. The honest replacement is
/// accuracy agreement against full precision: quantizing to int8-W8A8 must not
/// change what the model generates.
///
/// So decode each family twice over the same κ-store, same prompt — once through
/// the int8-W8A8 tier (the browser's deployed path) and once with the wide bf16
/// weights (full precision) — and require the quantized path to track it. The
/// numbers are MEASURED, not hoped: across all four families the greedy token
/// sequences agree exactly and the first-logit cosine is 0.9994–0.9996. The
/// asserts sit conservatively under that (argmax of the first token must agree;
/// cosine ≥ 0.99), so a real regression that flips the top token or tanks the
/// logit direction fails here, while a borderline late-token flip on some future
/// fixture does not make the suite brittle.
///
/// This is also the FIRST CI gate on the int8 κ-materialized decode path at all:
/// before this, its only coverage was the `#[ignore]` `decode_perf` benchmark and
/// the browser. W8A8's *kernel* numerics are pinned separately and exactly by
/// `omajor_w8a8_substrate_contract::w8a8_reproduces_the_exact_integer_oracle_which_w8a32_cannot`;
/// this witnesses the *integration* — the whole transformer plus LM head under
/// per-token activation quantization.
#[test]
fn int8_w8a8_decode_tracks_full_precision_for_every_family() {
    for layout in FAITHFUL_FAMILIES {
        let scale = FamilyScale::new(*layout, Dims::MODEST);
        let arch = scale.arch();

        // bf16 generates autonomously; int8 is TEACHER-FORCED on the bf16
        // tokens, so every position's logits are compared under a SHARED
        // context. This measures what the gate claims — per-position numeric
        // tracking of the quantized forward — at every generation depth.
        // (The previous methodology compared only the final row of two
        // autonomous runs: one knife-edge argmax flip on a near-tied logit
        // pair — a legitimate int8 outcome, observed on Phi3 when the fused
        // v0.9.0 attention's equally-valid f32 schedule moved a tie by an
        // ulp — diverged the contexts and collapsed the cosine to 0.82,
        // gating on sequence luck rather than quantization quality.)
        let (f_rows, f_tokens) = decode_family_rows(&scale, None, None);
        let (q_rows, _) = decode_family_rows(&scale, Some(QuantTier::Int8), Some(&f_tokens));
        assert_eq!(q_rows.len(), f_rows.len());

        let cosines: Vec<f64> = q_rows
            .iter()
            .zip(&f_rows)
            .map(|(q, f)| cosine(q, f))
            .collect();
        // Would int8 have picked the same greedy token at each shared-context
        // position? Reported for visibility; near-tie flips are legal.
        let agree = q_rows
            .iter()
            .zip(&f_rows)
            .filter(|(q, f)| argmax(q) == argmax(f))
            .count();
        eprintln!(
            "[w8a8-accuracy] {arch}: bf16={f_tokens:?} argmax_agree={agree}/{} \
             per-position cos={cosines:?}",
            q_rows.len()
        );

        assert_eq!(
            argmax(&q_rows[0]),
            argmax(&f_rows[0]),
            "{arch}: int8-W8A8 flipped the FIRST predicted token vs full-precision \
             bf16 — the deployed quantized path no longer tracks the model"
        );
        for (i, cos) in cosines.iter().enumerate() {
            assert!(
                *cos >= 0.99,
                "{arch}: int8-W8A8 logit cosine {cos:.6} at position {i} (shared \
                 context) fell below 0.99 — quantization is degrading the forward, \
                 not just rounding it"
            );
        }
    }
}

/// The int4 sibling: the WHOLE transformer + LM head decodes through the int4
/// derived-artifact tier, per family — not just one GEMV. int4 is a per-channel,
/// size-first tier that is MATERIALLY lossier than int8 (≈16% vs ≈1% per-GEMV,
/// stacked across ~200 projections and the head over many layers), so the bound
/// is deliberately loose and STATED: int4 must (1) compile + decode a full
/// multi-layer model without panic, (2) produce finite, non-degenerate logits,
/// and (3) still correlate with the bf16 reference (cosine ≥ 0.80) — a real,
/// measured quality floor for the tier, NOT a claim that int4 tracks int8.
/// Fails-without: an int4 stage that fails to compile/decode, or logits that go
/// to garbage (cosine collapses / non-finite).
#[test]
fn int4_decode_tracks_bf16_within_a_stated_loose_bound_for_every_family() {
    for layout in FAITHFUL_FAMILIES {
        let scale = FamilyScale::new(*layout, Dims::MODEST);
        let arch = scale.arch();

        // Same shared-context methodology as the int8 gate: int4 is
        // teacher-forced on the bf16 tokens, and every position's logits are
        // compared under the same context.
        let (f_rows, f_tokens) = decode_family_rows(&scale, None, None);
        let (q_rows, _) = decode_family_rows(&scale, Some(QuantTier::Int4), Some(&f_tokens));
        let cosines: Vec<f64> = q_rows
            .iter()
            .zip(&f_rows)
            .map(|(q, f)| cosine(q, f))
            .collect();
        eprintln!("[int4-accuracy] {arch}: bf16={f_tokens:?} per-position cos_vs_bf16={cosines:?}");

        assert!(
            q_rows.iter().flatten().all(|v| v.is_finite()),
            "{arch}: int4 decode produced non-finite logits — the tier broke the forward"
        );
        assert!(
            q_rows.iter().flatten().any(|&v| v != 0.0),
            "{arch}: int4 decode produced all-zero logits — degenerate"
        );
        // MEASURED (shared-context, per position): per-channel int4 cosine is
        // ≈0.96–0.99 vs bf16 here, an order of magnitude looser than int8's
        // ≈0.9995 on the SAME synthetic weights — the ~16% per-GEMV error
        // compounds across layers + the LM head. (The earlier ≈0.6–0.7 figure
        // was measured post-divergence — autonomous sequences amplified by
        // near-tie flips — and overstated the numeric error.) int4 remains the
        // materially lossier tier: near-tied tokens flip far more readily, so
        // sequences diverge fast even though per-position tracking stays high.
        // This floor is a regression TRIPWIRE (a decode bug collapses cosine
        // toward 0), NOT a quality endorsement.
        for (i, cos) in cosines.iter().enumerate() {
            assert!(
                *cos >= 0.45,
                "{arch}: int4 logit cosine {cos:.6} at position {i} fell below the \
                 regression floor (0.45) — the staged int4 pipeline is decoding wrong \
                 (a bug), not merely coarsely. int4 is lossy (≈0.6–0.7 expected) but a \
                 collapse toward 0 is a defect"
            );
        }
    }
}

/// **The resident-carry eviction witness (ADR-0019 increment 3b, staged).**
/// The same decode, twice per family: a generous budget (stages stay resident,
/// the K/V carry lives as κ-labels inside each stage session) versus a 1-byte
/// budget (NO stage is ever admitted — every stage session drops after every
/// window, so every carried cache must be BANKED into the runner's shadow at
/// eviction and re-ingested on the next walk, every step). The two runs must
/// agree bit-for-bit: same kernels, same values — only the carry vehicle
/// differs (labels vs banked bytes).
///
/// Fails-without: remove the eviction banking and the dropped stage takes the
/// carried truth with it — the next window binds stale host bytes and the
/// sequences diverge (or the walk errors on a missing carry).
#[test]
fn resident_kv_decode_is_identical_under_stage_eviction_pressure() {
    for layout in FAITHFUL_FAMILIES {
        let scale = FamilyScale::new(*layout, Dims::MODEST);
        let arch = scale.arch();

        let (resident_rows, resident_tokens) =
            decode_family_rows_at(&scale, Some(QuantTier::Int8), None, 1 << 30);
        let (windowed_rows, windowed_tokens) =
            decode_family_rows_at(&scale, Some(QuantTier::Int8), None, 1);

        assert_eq!(
            resident_tokens, windowed_tokens,
            "{arch}: eviction pressure changed the decoded tokens — the carried K/V \
             truth was lost (or corrupted) across a stage drop"
        );
        for (i, (r, w)) in resident_rows.iter().zip(&windowed_rows).enumerate() {
            assert_eq!(
                r.len(),
                w.len(),
                "{arch}: logit row {i} width differs under eviction pressure"
            );
            for (j, (a, b)) in r.iter().zip(w.iter()).enumerate() {
                assert!(
                    a.to_bits() == b.to_bits(),
                    "{arch}: logit row {i} cell {j} differs under eviction pressure \
                     ({a} vs {b}) — the banked carry did not round-trip bit-exactly"
                );
            }
        }
        eprintln!("[eviction-carry] {arch}: resident == strict-windowed, bit-for-bit");
    }
}

/// **The staged carry is LIVE, not a dark gate.** Drives one family's
/// `StagedRunner` directly through the `LmSession` surface: a resident walk
/// must (a) return `None` for every carried-cache output (nothing
/// materialized — the whole point), (b) return the SAME bytes as the byte walk
/// for every materialized output, and (c) hold a non-empty carry whose banked
/// bytes equal the byte walk's cache outputs bit-for-bit. Fails-without: if
/// the staged override were missing (trait default), every output would come
/// back `Some` and the carry would be empty — (a) and (c) both fail.
#[test]
fn staged_runner_carries_kv_resident_and_banks_bitwise() {
    let scale = FamilyScale::new(FAITHFUL_FAMILIES[0], Dims::MODEST);
    let arch = scale.arch();
    let config_json = scale.config_json();
    let (keys, shapes): (Vec<String>, Vec<Vec<u64>>) = scale.manifest().into_iter().unzip();
    let dtypes = vec![DType::BF16; keys.len()];
    let one = NonZeroU64::new(1).expect("1 is non-zero");
    let store_dir = std::env::temp_dir().join(format!("hai-carry-{arch}-{}", std::process::id()));
    std::fs::create_dir_all(&store_dir).expect("κ-store dir");
    let dir = DirKappaStore::new(&store_dir);
    let kappas: Vec<String> = keys
        .iter()
        .zip(&shapes)
        .map(|(name, dims)| dir.insert(&dummy_bf16_bytes(name, dims)).expect("weight"))
        .collect();
    let mut session = GrowableStagedSession::new(
        config_json,
        keys,
        kappas,
        shapes,
        dtypes,
        None,
        one,
        Box::new(DirKappaStore::new(&store_dir)),
    )
    .expect("growable session");
    session.set_residency_budget(1 << 30);
    let mut runner = session
        .decode_runner_for(PROMPT.len())
        .expect("step runner");

    // Deterministic inputs by port dtype — the carry mechanics are the subject,
    // so the values only need to be finite and identical across both walks.
    let ports = runner.input_port_info();
    let bufs: Vec<Vec<u8>> = ports
        .iter()
        .map(|p| match p.dtype {
            5 => (0..p.element_count)
                .flat_map(|_| 1i64.to_le_bytes())
                .collect(),
            4 => (0..p.element_count)
                .flat_map(|_| 0i32.to_le_bytes())
                .collect(),
            _ => (0..p.element_count)
                .flat_map(|i| (((i % 7) as f32) * 0.01).to_le_bytes())
                .collect(),
        })
        .collect();
    let refs: Vec<&[u8]> = bufs.iter().map(|b| b.as_slice()).collect();

    let byte_walk = LmSession::execute(&mut runner, &refs).expect("byte walk");
    let out_ports = LmSession::output_port_info(&runner);
    let resident_walk = runner
        .execute_kv_resident(&refs, false)
        .expect("resident walk");

    let mut carried_ports = 0usize;
    for ((port, byte_out), res_out) in out_ports.iter().zip(&byte_walk).zip(&resident_walk) {
        let is_kv = port.name.starts_with("k_new_") || port.name.starts_with("v_new_");
        match res_out {
            None => {
                assert!(
                    is_kv,
                    "{arch}: non-cache output `{}` was not materialized",
                    port.name
                );
                carried_ports += 1;
            }
            Some(out) => {
                assert!(
                    !is_kv,
                    "{arch}: cache output `{}` was materialized on the resident walk",
                    port.name
                );
                assert_eq!(
                    out.bytes, byte_out.bytes,
                    "{arch}: `{}` differs resident vs byte",
                    port.name
                );
            }
        }
    }
    assert!(
        carried_ports > 0,
        "{arch}: the staged resident walk carried NOTHING — the override is not live"
    );

    // The banked truth equals the byte walk's cache outputs, bit for bit.
    let carry = runner.take_kv_carry().expect("carry materializes");
    assert_eq!(
        carry.len(),
        carried_ports,
        "{arch}: carry entries != carried outputs"
    );
    for (name, bytes) in carry {
        let out_name = name
            .strip_prefix("past_k_")
            .map(|l| format!("k_new_{l}"))
            .or_else(|| name.strip_prefix("past_v_").map(|l| format!("v_new_{l}")))
            .unwrap_or_else(|| panic!("{arch}: unexpected carry key `{name}`"));
        let idx = out_ports
            .iter()
            .position(|p| p.name == out_name)
            .unwrap_or_else(|| panic!("{arch}: no output port `{out_name}`"));
        assert_eq!(
            bytes, byte_walk[idx].bytes,
            "{arch}: banked carry `{name}` != byte walk `{out_name}`"
        );
    }
    eprintln!("[staged-carry] {arch}: {carried_ports} caches carried resident, banked bitwise");
}
