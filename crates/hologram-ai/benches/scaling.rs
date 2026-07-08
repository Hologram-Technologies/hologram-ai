//! Model-size scaling benchmarks (V&V class PV).
//!
//! hologram-ai is bound by the same performance contract as hologram, whose
//! thesis is *perf is content addressing, not micro-opt*: a κ-label memo hit
//! returns a cached result O(1) in graph size, and matmul throughput holds its
//! efficiency across scale. These benches mirror hologram's own matmul sweep
//! (64 / 128 / 256 / 512) through the full hologram-ai pipeline and validate
//! both axes:
//!
//! - `matmul_compile/{n}` — compile (model → `.holo`) cost vs size.
//! - `matmul_cold/{n}` — forward with *novel* inputs each iter (full recompute);
//!   the matmul throughput-vs-size curve.
//! - `matmul_reuse_hit/{n}` — forward on *fixed* κ-labels (content-addressed
//!   memo hit); should be ~flat in n (O(1) reuse).
//! - `imported_forward` — a real imported model (`mini_transformer.onnx`),
//!   real-world verification, not just synthetic ops.
//!
//! No size is special-cased and no dimension is clamped — the sweep exists to
//! prove no arbitrary limit throttles a larger model (see `tests/perf_floor.rs`
//! for the asserted floor). Run: `cargo bench -p hologram-ai --bench scaling`.

use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput};
use hologram_ai::{HoloRunner, ModelCompiler, ModelSource};
use hologram_ai_conformance::ort_runner::onnx_builder;

const SIZES: &[usize] = &[64, 128, 256, 512];

fn compile(model: Vec<u8>) -> HoloRunner {
    let archive = ModelCompiler::default()
        .compile(ModelSource::OnnxBytes {
            model_bytes: model,
            external_data: None,
        })
        .expect("compile failed");
    HoloRunner::from_bytes(archive.bytes).expect("load failed")
}

/// Zeroed input buffers sized to the model's ports, plus a mutable copy used to
/// perturb bytes (forcing a novel κ-label → real recompute) for the cold path.
fn zeroed_inputs(runner: &HoloRunner) -> Vec<Vec<u8>> {
    runner
        .input_byte_sizes()
        .iter()
        .map(|&n| vec![0u8; n])
        .collect()
}

fn bench_matmul_compile(c: &mut Criterion) {
    let mut g = c.benchmark_group("matmul_compile");
    for &n in SIZES {
        let model = onnx_builder::matmul(n, n, n);
        g.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| compile(model.clone()));
        });
    }
    g.finish();
}

fn bench_matmul_cold(c: &mut Criterion) {
    let mut g = c.benchmark_group("matmul_cold");
    for &n in SIZES {
        // 2·n³ flops per n×n×n matmul.
        g.throughput(Throughput::Elements((2 * n * n * n) as u64));
        let mut runner = compile(onnx_builder::matmul(n, n, n));
        let base = zeroed_inputs(&runner);
        let mut seed = 0u8;
        g.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter_batched(
                || {
                    // Perturb one byte of each input so its content address —
                    // and thus the result — is novel: forces a real recompute,
                    // never a memo hit. (Wraps; the value is irrelevant.)
                    seed = seed.wrapping_add(1);
                    let mut ins = base.clone();
                    for buf in ins.iter_mut() {
                        if let Some(first) = buf.first_mut() {
                            *first = seed;
                        }
                    }
                    ins
                },
                |ins| {
                    let refs: Vec<&[u8]> = ins.iter().map(|v| v.as_slice()).collect();
                    runner.execute(&refs).expect("execute failed");
                },
                BatchSize::SmallInput,
            );
        });
    }
    g.finish();
}

fn bench_matmul_reuse_hit(c: &mut Criterion) {
    let mut g = c.benchmark_group("matmul_reuse_hit");
    for &n in SIZES {
        let mut runner = compile(onnx_builder::matmul(n, n, n));
        let inputs = zeroed_inputs(&runner);
        // Intern once; the labels are fixed, so every call after the first is a
        // whole-graph memo hit (cached output labels, no compute, no copy).
        let labels: Vec<_> = inputs.iter().map(|v| runner.intern_input(v)).collect();
        runner.execute_addressed(&labels).expect("warm memo");
        g.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| runner.execute_addressed(&labels).expect("reuse hit"));
        });
    }
    g.finish();
}

fn bench_imported_forward(c: &mut Criterion) {
    // Real-world verification: a complete imported transformer, not a synthetic
    // single op. Skips cleanly if the fixture is absent.
    let Some(model) = hologram_ai_conformance::ort_runner::fixtures::load("mini_transformer")
    else {
        return;
    };
    let mut runner = {
        let archive = ModelCompiler {
            seq_len_override: Some(64),
            ..Default::default()
        }
        .compile(ModelSource::OnnxBytes {
            model_bytes: model,
            external_data: None,
        })
        .expect("compile failed");
        HoloRunner::from_bytes(archive.bytes).expect("load failed")
    };
    let inputs = zeroed_inputs(&runner);
    let refs: Vec<&[u8]> = inputs.iter().map(|v| v.as_slice()).collect();
    let mut g = c.benchmark_group("imported_forward");
    g.bench_function("mini_transformer_seq64", |b| {
        b.iter(|| runner.execute(&refs).expect("execute failed"));
    });
    g.finish();
}

/// Batch-dimension sweep for a fixed weight — the decode-shape efficiency
/// witness. A decode step is a SINGLE-position (`M=1`) matmul: one activation
/// vector times each weight matrix, a GEMV — the thinnest shape. Prefill (and a
/// speculative/chunked verify pass) batches `M=C` positions into a GEMM that
/// reuses each weight across `C` rows. The substrate matmul is a GEMM kernel:
/// efficient batched, poor at `M=1`.
///
/// This sweeps `M` over a fixed `[K,N]` weight and reports output-element
/// throughput. Only the activation is perturbed each iter (novel κ → real
/// recompute); the weight κ is stable (resident, as in real decode). The fixed
/// per-`execute` cost is constant in `M`, so it amortizes as `M` grows — the
/// rising elem/s curve is therefore the kernel's own batching headroom, not
/// host overhead. The `M=1` point is the decode floor; the climb to `M=C` is
/// exactly the wall-clock a batched-verify decode (row `positional-cones` /
/// speculative decode) would convert. Reported, never asserted (machine-
/// relative). Run: `cargo bench -p hologram-ai --bench scaling -- decode_shape`.
const DECODE_BATCH: &[usize] = &[1, 2, 4, 8, 16, 32];

fn bench_decode_shape(c: &mut Criterion) {
    let (k, n) = (2048usize, 2048usize);
    let mut g = c.benchmark_group("decode_shape");
    for &m in DECODE_BATCH {
        // Output elements produced per pass: M·N. Criterion emits elem/s, so the
        // curve across M is per-output efficiency — flat would mean pure
        // kernel-bound, rising means the GEMM path recovers as it batches.
        g.throughput(Throughput::Elements((m * n) as u64));
        let mut runner = compile(onnx_builder::matmul(m, k, n));
        let base = zeroed_inputs(&runner);
        let mut seed = 0u8;
        g.bench_with_input(BenchmarkId::from_parameter(m), &m, |b, _| {
            b.iter_batched(
                || {
                    // Perturb ONLY the activation (first input): a novel κ forces
                    // a real recompute while the weight stays resident — the
                    // decode pattern (fixed weights, changing position).
                    seed = seed.wrapping_add(1);
                    let mut ins = base.clone();
                    if let Some(first) = ins.first_mut().and_then(|a| a.first_mut()) {
                        *first = seed;
                    }
                    ins
                },
                |ins| {
                    let refs: Vec<&[u8]> = ins.iter().map(|v| v.as_slice()).collect();
                    runner.execute(&refs).expect("execute failed");
                },
                BatchSize::SmallInput,
            );
        });
    }
    g.finish();
}

criterion_group!(
    benches,
    bench_matmul_compile,
    bench_matmul_cold,
    bench_matmul_reuse_hit,
    bench_decode_shape,
    bench_imported_forward
);
criterion_main!(benches);
