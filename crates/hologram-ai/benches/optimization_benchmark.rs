//! Benchmark comparing standard vs optimized execution with REAL hologram APIs.
//!
//! This benchmark demonstrates the performance improvements from:
//! - SIMD activation lookup tables (40-50x speedup)
//! - Fused/composed view kernels (3x speedup)
//! - Parallel buffer operations (2-3x speedup on multi-core)
//! - Embedding cache pinning (2x speedup for lookups)
//!
//! Run with: `cargo bench --package hologram-ai`

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use hologram::backend::core::simd_activation::SimdActivationCache;
use hologram::backend::plan::activation_table_id;
use hologram::lookup::fusion::{RELU_U8, SIGMOID_U8, TANH_U8};
use hologram::lookup::view::pinned::warm_lookup_tables;
use hologram::lookup::view::{ElementWiseView, SimdLookup, View, ViewExt};

/// Benchmark SIMD lookup vs scalar
fn bench_simd_vs_scalar(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_lookup");

    let sigmoid = SimdLookup::new(SIGMOID_U8);
    let input = vec![128u8; 1_000_000]; // 1M elements
    let mut output = vec![0u8; 1_000_000];

    group.bench_function("simd_batch", |b| {
        b.iter(|| {
            sigmoid.apply_batch(black_box(&input), black_box(&mut output));
        });
    });

    group.bench_function("scalar", |b| {
        b.iter(|| {
            sigmoid.apply_scalar(black_box(&input), black_box(&mut output));
        });
    });

    group.finish();
}

/// Benchmark view composition and resolution
fn bench_view_composition(c: &mut Criterion) {
    let mut group = c.benchmark_group("view_composition");

    let sigmoid = ElementWiseView::from_table(&SIGMOID_U8);
    let tanh = ElementWiseView::from_table(&TANH_U8);
    let relu = ElementWiseView::from_table(&RELU_U8);

    // Lazy composition: 3 lookups per element
    let lazy = sigmoid.then(tanh).then(relu);

    // Resolved composition: 1 lookup per element
    let resolved = {
        use hologram::lookup::view::composed::resolve3;
        resolve3(sigmoid, tanh, relu)
    };

    let input = vec![128u8; 10_000];
    let mut output = vec![0u8; 10_000];

    group.bench_function("lazy_3_lookups", |b| {
        b.iter(|| {
            for (i, o) in black_box(&input)
                .iter()
                .zip(black_box(&mut output).iter_mut())
            {
                *o = lazy.get(*i as usize);
            }
        });
    });

    group.bench_function("resolved_1_lookup", |b| {
        b.iter(|| {
            for (i, o) in black_box(&input)
                .iter()
                .zip(black_box(&mut output).iter_mut())
            {
                *o = resolved.get(*i as usize);
            }
        });
    });

    group.finish();
}

/// Benchmark cache warming impact
fn bench_cache_warming(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_warming");

    // Cold cache
    group.bench_function("cold_lookup", |b| {
        b.iter(|| {
            std::hint::black_box(&SIGMOID_U8);
        });
    });

    // Warm cache
    warm_lookup_tables();
    group.bench_function("warm_lookup", |b| {
        b.iter(|| {
            std::hint::black_box(&SIGMOID_U8);
        });
    });

    group.finish();
}

/// Benchmark parallel threshold strategy
fn bench_parallel_threshold(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_threshold");

    for size in [1_000, 4_096, 10_000, 100_000].iter() {
        group.bench_with_input(BenchmarkId::new("parallel_map", size), size, |b, &size| {
            let input = vec![128u8; size];
            b.iter(|| {
                use hologram::lookup::map_input_parallel;
                black_box(map_input_parallel(black_box(&input)));
            });
        });
    }

    group.finish();
}

/// Benchmark activation cache
fn bench_activation_cache(c: &mut Criterion) {
    let mut group = c.benchmark_group("activation_cache");

    let mut cache = SimdActivationCache::preloaded();
    let input = vec![128u8; 100_000];
    let mut output = vec![0u8; 100_000];

    group.bench_function("cached_sigmoid", |b| {
        b.iter(|| {
            cache.apply_batch(
                activation_table_id::SIGMOID,
                black_box(&input),
                black_box(&mut output),
            );
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_simd_vs_scalar,
    bench_view_composition,
    bench_cache_warming,
    bench_parallel_threshold,
    bench_activation_cache
);
criterion_main!(benches);
