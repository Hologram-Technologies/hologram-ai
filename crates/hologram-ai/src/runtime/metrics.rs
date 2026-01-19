//! Performance metrics for parallel view optimizations.
//!
//! This module provides infrastructure for tracking and reporting performance
//! metrics related to SIMD activations, composed views, parallel execution,
//! and embedding cache performance.
//!
//! # Usage
//!
//! ```ignore
//! use hologram_ai::runtime::metrics::PerformanceMetrics;
//!
//! let mut metrics = PerformanceMetrics::new();
//!
//! // Track operations
//! metrics.record_simd_op();
//! metrics.record_parallel_level();
//! metrics.record_cache_hit();
//!
//! // Generate report
//! println!("{}", metrics.report());
//! ```

use std::time::{Duration, Instant};

/// Performance metrics for parallel view system optimizations.
///
/// Tracks utilization of:
/// - SIMD activation lookups
/// - Composed view fusion
/// - Parallel execution levels
/// - Embedding cache hits/misses
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Number of operations using SIMD path
    pub simd_ops: usize,
    /// Number of operations using scalar path
    pub scalar_ops: usize,
    /// Number of composed view operations
    pub composed_view_ops: usize,
    /// Number of parallel execution levels
    pub parallel_levels: usize,
    /// Number of sequential execution levels
    pub sequential_levels: usize,
    /// Number of embedding cache hits
    pub cache_hits: usize,
    /// Number of embedding cache misses
    pub cache_misses: usize,
    /// Total execution time in microseconds
    pub total_time_us: u64,
    /// Timestamp of last reset
    start_time: Instant,
}

impl PerformanceMetrics {
    /// Create a new metrics tracker.
    pub fn new() -> Self {
        Self {
            simd_ops: 0,
            scalar_ops: 0,
            composed_view_ops: 0,
            parallel_levels: 0,
            sequential_levels: 0,
            cache_hits: 0,
            cache_misses: 0,
            total_time_us: 0,
            start_time: Instant::now(),
        }
    }

    /// Record a SIMD operation.
    #[inline]
    pub fn record_simd_op(&mut self) {
        self.simd_ops += 1;
    }

    /// Record a scalar operation.
    #[inline]
    pub fn record_scalar_op(&mut self) {
        self.scalar_ops += 1;
    }

    /// Record a composed view operation.
    #[inline]
    pub fn record_composed_view_op(&mut self) {
        self.composed_view_ops += 1;
    }

    /// Record a parallel execution level.
    #[inline]
    pub fn record_parallel_level(&mut self) {
        self.parallel_levels += 1;
    }

    /// Record a sequential execution level.
    #[inline]
    pub fn record_sequential_level(&mut self) {
        self.sequential_levels += 1;
    }

    /// Record an embedding cache hit.
    #[inline]
    pub fn record_cache_hit(&mut self) {
        self.cache_hits += 1;
    }

    /// Record an embedding cache miss.
    #[inline]
    pub fn record_cache_miss(&mut self) {
        self.cache_misses += 1;
    }

    /// Update total execution time.
    pub fn update_execution_time(&mut self) {
        self.total_time_us = self.start_time.elapsed().as_micros() as u64;
    }

    /// Set execution time from duration.
    pub fn set_execution_time(&mut self, duration: Duration) {
        self.total_time_us = duration.as_micros() as u64;
    }

    /// Calculate SIMD utilization percentage.
    ///
    /// Returns the percentage of activation operations using SIMD path.
    pub fn simd_utilization(&self) -> f64 {
        let total = self.simd_ops + self.scalar_ops;
        if total == 0 {
            0.0
        } else {
            (self.simd_ops as f64 / total as f64) * 100.0
        }
    }

    /// Calculate composed view utilization percentage.
    ///
    /// Returns the percentage of operations using composed views.
    pub fn composed_view_utilization(&self) -> f64 {
        let total_ops = self.simd_ops + self.scalar_ops + self.composed_view_ops;
        if total_ops == 0 {
            0.0
        } else {
            (self.composed_view_ops as f64 / total_ops as f64) * 100.0
        }
    }

    /// Calculate parallel execution percentage.
    ///
    /// Returns the percentage of execution levels that were parallelized.
    pub fn parallel_utilization(&self) -> f64 {
        let total = self.parallel_levels + self.sequential_levels;
        if total == 0 {
            0.0
        } else {
            (self.parallel_levels as f64 / total as f64) * 100.0
        }
    }

    /// Calculate cache hit rate percentage.
    ///
    /// Returns the percentage of embedding lookups that were cache hits.
    pub fn cache_hit_rate(&self) -> f64 {
        let total = self.cache_hits + self.cache_misses;
        if total == 0 {
            0.0
        } else {
            (self.cache_hits as f64 / total as f64) * 100.0
        }
    }

    /// Calculate throughput in operations per second.
    pub fn ops_per_second(&self) -> f64 {
        let total_ops = self.simd_ops + self.scalar_ops + self.composed_view_ops;
        if self.total_time_us == 0 {
            0.0
        } else {
            (total_ops as f64 / self.total_time_us as f64) * 1_000_000.0
        }
    }

    /// Generate a formatted performance report.
    ///
    /// Returns a multi-line string with all metrics formatted for display.
    pub fn report(&self) -> String {
        let mut lines = vec!["=== Hologram Parallel View Performance Metrics ===".to_string()];

        // SIMD metrics
        lines.push(format!(
            "\nSIMD Activations: {:.1}% utilization ({} SIMD ops, {} scalar ops)",
            self.simd_utilization(),
            self.simd_ops,
            self.scalar_ops
        ));

        // Composed view metrics
        if self.composed_view_ops > 0 {
            lines.push(format!(
                "Composed Views: {:.1}% utilization ({} fused ops)",
                self.composed_view_utilization(),
                self.composed_view_ops
            ));
        }

        // Parallel execution metrics
        lines.push(format!(
            "Parallel Execution: {:.1}% parallelized ({} parallel levels, {} sequential)",
            self.parallel_utilization(),
            self.parallel_levels,
            self.sequential_levels
        ));

        // Cache metrics
        if self.cache_hits + self.cache_misses > 0 {
            lines.push(format!(
                "Embedding Cache: {:.1}% hit rate ({}/{} lookups)",
                self.cache_hit_rate(),
                self.cache_hits,
                self.cache_hits + self.cache_misses
            ));
        }

        // Timing metrics
        lines.push(format!(
            "\nTotal Time: {:.2} ms ({:.0} ops/sec)",
            self.total_time_us as f64 / 1000.0,
            self.ops_per_second()
        ));

        lines.push("=".repeat(50));

        lines.join("\n")
    }

    /// Generate a compact one-line summary.
    pub fn summary(&self) -> String {
        format!(
            "SIMD: {:.0}%, Composed: {:.0}%, Parallel: {:.0}%, Cache: {:.0}%, {:.2}ms",
            self.simd_utilization(),
            self.composed_view_utilization(),
            self.parallel_utilization(),
            self.cache_hit_rate(),
            self.total_time_us as f64 / 1000.0
        )
    }

    /// Reset all metrics to zero.
    pub fn reset(&mut self) {
        self.simd_ops = 0;
        self.scalar_ops = 0;
        self.composed_view_ops = 0;
        self.parallel_levels = 0;
        self.sequential_levels = 0;
        self.cache_hits = 0;
        self.cache_misses = 0;
        self.total_time_us = 0;
        self.start_time = Instant::now();
    }

    /// Merge metrics from another tracker.
    ///
    /// Useful for combining metrics from multiple inference runs.
    pub fn merge(&mut self, other: &PerformanceMetrics) {
        self.simd_ops += other.simd_ops;
        self.scalar_ops += other.scalar_ops;
        self.composed_view_ops += other.composed_view_ops;
        self.parallel_levels += other.parallel_levels;
        self.sequential_levels += other.sequential_levels;
        self.cache_hits += other.cache_hits;
        self.cache_misses += other.cache_misses;
        self.total_time_us += other.total_time_us;
    }
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for constructing performance metrics with specific values.
///
/// Useful for testing and simulation.
#[derive(Debug, Default)]
pub struct MetricsBuilder {
    simd_ops: usize,
    scalar_ops: usize,
    composed_view_ops: usize,
    parallel_levels: usize,
    sequential_levels: usize,
    cache_hits: usize,
    cache_misses: usize,
    total_time_us: u64,
}

impl MetricsBuilder {
    /// Create a new metrics builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set SIMD operations count.
    pub fn simd_ops(mut self, count: usize) -> Self {
        self.simd_ops = count;
        self
    }

    /// Set scalar operations count.
    pub fn scalar_ops(mut self, count: usize) -> Self {
        self.scalar_ops = count;
        self
    }

    /// Set composed view operations count.
    pub fn composed_view_ops(mut self, count: usize) -> Self {
        self.composed_view_ops = count;
        self
    }

    /// Set parallel levels count.
    pub fn parallel_levels(mut self, count: usize) -> Self {
        self.parallel_levels = count;
        self
    }

    /// Set sequential levels count.
    pub fn sequential_levels(mut self, count: usize) -> Self {
        self.sequential_levels = count;
        self
    }

    /// Set cache hits count.
    pub fn cache_hits(mut self, count: usize) -> Self {
        self.cache_hits = count;
        self
    }

    /// Set cache misses count.
    pub fn cache_misses(mut self, count: usize) -> Self {
        self.cache_misses = count;
        self
    }

    /// Set total execution time in microseconds.
    pub fn total_time_us(mut self, time: u64) -> Self {
        self.total_time_us = time;
        self
    }

    /// Build the performance metrics.
    pub fn build(self) -> PerformanceMetrics {
        PerformanceMetrics {
            simd_ops: self.simd_ops,
            scalar_ops: self.scalar_ops,
            composed_view_ops: self.composed_view_ops,
            parallel_levels: self.parallel_levels,
            sequential_levels: self.sequential_levels,
            cache_hits: self.cache_hits,
            cache_misses: self.cache_misses,
            total_time_us: self.total_time_us,
            start_time: Instant::now(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_new() {
        let metrics = PerformanceMetrics::new();
        assert_eq!(metrics.simd_ops, 0);
        assert_eq!(metrics.scalar_ops, 0);
        assert_eq!(metrics.total_time_us, 0);
    }

    #[test]
    fn test_record_operations() {
        let mut metrics = PerformanceMetrics::new();

        metrics.record_simd_op();
        metrics.record_simd_op();
        metrics.record_scalar_op();
        metrics.record_composed_view_op();

        assert_eq!(metrics.simd_ops, 2);
        assert_eq!(metrics.scalar_ops, 1);
        assert_eq!(metrics.composed_view_ops, 1);
    }

    #[test]
    fn test_simd_utilization() {
        let mut metrics = PerformanceMetrics::new();

        // 80% SIMD utilization (8 SIMD, 2 scalar)
        for _ in 0..8 {
            metrics.record_simd_op();
        }
        for _ in 0..2 {
            metrics.record_scalar_op();
        }

        assert_eq!(metrics.simd_utilization(), 80.0);
    }

    #[test]
    fn test_parallel_utilization() {
        let mut metrics = PerformanceMetrics::new();

        // 60% parallel (3 parallel, 2 sequential)
        metrics.record_parallel_level();
        metrics.record_parallel_level();
        metrics.record_parallel_level();
        metrics.record_sequential_level();
        metrics.record_sequential_level();

        assert_eq!(metrics.parallel_utilization(), 60.0);
    }

    #[test]
    fn test_cache_hit_rate() {
        let mut metrics = PerformanceMetrics::new();

        // 75% hit rate (30 hits, 10 misses)
        for _ in 0..30 {
            metrics.record_cache_hit();
        }
        for _ in 0..10 {
            metrics.record_cache_miss();
        }

        assert_eq!(metrics.cache_hit_rate(), 75.0);
    }

    #[test]
    fn test_composed_view_utilization() {
        let mut metrics = PerformanceMetrics::new();

        metrics.record_simd_op();
        metrics.record_simd_op();
        metrics.record_composed_view_op();
        metrics.record_composed_view_op();
        metrics.record_composed_view_op();

        // 3 out of 5 = 60%
        assert_eq!(metrics.composed_view_utilization(), 60.0);
    }

    #[test]
    fn test_ops_per_second() {
        let mut metrics = PerformanceMetrics::new();

        metrics.record_simd_op();
        metrics.record_simd_op();
        metrics.record_scalar_op();
        metrics.total_time_us = 1000; // 1ms

        // 3 ops / 1ms = 3000 ops/sec
        assert_eq!(metrics.ops_per_second(), 3000.0);
    }

    #[test]
    fn test_report_format() {
        let mut metrics = PerformanceMetrics::new();

        metrics.record_simd_op();
        metrics.record_scalar_op();
        metrics.record_parallel_level();
        metrics.total_time_us = 5000; // 5ms

        let report = metrics.report();
        assert!(report.contains("SIMD Activations"));
        assert!(report.contains("50.0%")); // 1 SIMD / 2 total
        assert!(report.contains("Parallel Execution"));
        assert!(report.contains("5.00 ms"));
    }

    #[test]
    fn test_summary_format() {
        let mut metrics = PerformanceMetrics::new();

        metrics.record_simd_op();
        metrics.record_scalar_op();
        metrics.total_time_us = 2500; // 2.5ms

        let summary = metrics.summary();
        assert!(summary.contains("SIMD: 50%"));
        assert!(summary.contains("2.50ms"));
    }

    #[test]
    fn test_reset() {
        let mut metrics = PerformanceMetrics::new();

        metrics.record_simd_op();
        metrics.record_parallel_level();
        metrics.record_cache_hit();

        metrics.reset();

        assert_eq!(metrics.simd_ops, 0);
        assert_eq!(metrics.parallel_levels, 0);
        assert_eq!(metrics.cache_hits, 0);
    }

    #[test]
    fn test_merge() {
        let mut metrics1 = PerformanceMetrics::new();
        metrics1.record_simd_op();
        metrics1.record_simd_op();
        metrics1.total_time_us = 1000;

        let mut metrics2 = PerformanceMetrics::new();
        metrics2.record_simd_op();
        metrics2.record_scalar_op();
        metrics2.total_time_us = 2000;

        metrics1.merge(&metrics2);

        assert_eq!(metrics1.simd_ops, 3);
        assert_eq!(metrics1.scalar_ops, 1);
        assert_eq!(metrics1.total_time_us, 3000);
    }

    #[test]
    fn test_metrics_builder() {
        let metrics = MetricsBuilder::new()
            .simd_ops(100)
            .scalar_ops(20)
            .parallel_levels(10)
            .cache_hits(80)
            .cache_misses(20)
            .total_time_us(50000)
            .build();

        assert_eq!(metrics.simd_ops, 100);
        assert_eq!(metrics.scalar_ops, 20);
        assert_eq!(metrics.parallel_levels, 10);
        assert_eq!(metrics.cache_hit_rate(), 80.0);
        // 100 SIMD / 120 total = 83.33%
        assert!((metrics.simd_utilization() - 83.33333333333333).abs() < 0.0001);
    }

    #[test]
    fn test_zero_division_safety() {
        let metrics = PerformanceMetrics::new();

        // Should not panic with no operations
        assert_eq!(metrics.simd_utilization(), 0.0);
        assert_eq!(metrics.parallel_utilization(), 0.0);
        assert_eq!(metrics.cache_hit_rate(), 0.0);
        assert_eq!(metrics.ops_per_second(), 0.0);
    }
}
