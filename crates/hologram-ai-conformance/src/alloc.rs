//! Counting-allocator harness for the structural V&V axis.
//!
//! Enforces class **ZA** (zero runtime heap allocation): a full prefill+decode
//! inference call must perform no heap allocation after warm-up, and lowering
//! must allocate a bounded, input-independent amount. See `CONFORMANCE.md`.
//!
//! This module is deliberately self-contained (no hologram dependency) so it
//! is usable the moment the runtime core builds against hologram 0.5.0.
//!
//! # Usage
//!
//! Install the allocator at the top of the test binary, then assert around the
//! hot path:
//!
//! ```ignore
//! use hologram_ai_conformance::alloc::{CountingAllocator, assert_no_alloc};
//!
//! #[global_allocator]
//! static GLOBAL: CountingAllocator = CountingAllocator::new();
//!
//! #[test]
//! fn za1_decode_step_is_alloc_free() {
//!     let mut session = warm_up_session();          // allocations allowed here
//!     assert_no_alloc("ZA-1 decode step", || {
//!         session.decode_one_token();                // must allocate nothing
//!     });
//! }
//! ```

use core::alloc::{GlobalAlloc, Layout};
use core::sync::atomic::{AtomicUsize, Ordering};
use std::alloc::System;

/// A global allocator that forwards to the system allocator while counting
/// allocations and total bytes requested. Counters are process-global and
/// cheap to read; reset them around the region under test.
pub struct CountingAllocator {
    inner: System,
}

static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);
static ALLOC_BYTES: AtomicUsize = AtomicUsize::new(0);
static REALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);

impl CountingAllocator {
    pub const fn new() -> Self {
        Self { inner: System }
    }
}

impl Default for CountingAllocator {
    fn default() -> Self {
        Self::new()
    }
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        ALLOC_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        self.inner.alloc(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        self.inner.dealloc(ptr, layout)
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        REALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        ALLOC_BYTES.fetch_add(new_size.saturating_sub(layout.size()), Ordering::Relaxed);
        self.inner.realloc(ptr, layout, new_size)
    }
}

/// A snapshot of the global allocation counters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AllocStats {
    pub allocations: usize,
    pub reallocations: usize,
    pub bytes: usize,
}

impl AllocStats {
    /// True when no heap activity occurred (the ZA condition).
    pub fn is_quiet(&self) -> bool {
        self.allocations == 0 && self.reallocations == 0
    }
}

/// Read the current counters.
pub fn snapshot() -> AllocStats {
    AllocStats {
        allocations: ALLOC_COUNT.load(Ordering::Relaxed),
        reallocations: REALLOC_COUNT.load(Ordering::Relaxed),
        bytes: ALLOC_BYTES.load(Ordering::Relaxed),
    }
}

/// Zero the counters. Call immediately before the region under test.
pub fn reset() {
    ALLOC_COUNT.store(0, Ordering::Relaxed);
    ALLOC_BYTES.store(0, Ordering::Relaxed);
    REALLOC_COUNT.store(0, Ordering::Relaxed);
}

/// Run `f` and return the allocations it caused.
pub fn measure<R>(f: impl FnOnce() -> R) -> (R, AllocStats) {
    let before = snapshot();
    let out = f();
    let after = snapshot();
    let delta = AllocStats {
        allocations: after.allocations - before.allocations,
        reallocations: after.reallocations - before.reallocations,
        bytes: after.bytes - before.bytes,
    };
    (out, delta)
}

/// Assert that `f` performs no heap allocation (ZA-1). Panics with the observed
/// counts (and the supplied label) on violation.
///
/// Requires `CountingAllocator` to be installed as the `#[global_allocator]`;
/// otherwise the counters never move and this trivially passes — call
/// [`assert_allocator_installed`] once in the suite to guard against that.
pub fn assert_no_alloc<R>(label: &str, f: impl FnOnce() -> R) -> R {
    let (out, delta) = measure(f);
    assert!(
        delta.is_quiet(),
        "{label}: expected zero runtime heap allocation, observed \
         {} alloc(s), {} realloc(s), {} byte(s)",
        delta.allocations,
        delta.reallocations,
        delta.bytes,
    );
    out
}

/// Assert that `f` allocates at most `max_allocations` times (ZA-2: bounded,
/// input-independent allocation during graph lowering).
pub fn assert_alloc_bounded<R>(label: &str, max_allocations: usize, f: impl FnOnce() -> R) -> R {
    let (out, delta) = measure(f);
    assert!(
        delta.allocations <= max_allocations,
        "{label}: expected at most {max_allocations} allocation(s), observed {}",
        delta.allocations,
    );
    out
}

/// Sanity guard: verify the counting allocator is actually wired in as the
/// global allocator, so [`assert_no_alloc`] cannot pass vacuously.
pub fn assert_allocator_installed() {
    // `core::hint::black_box` around both the input and result keeps a
    // release-mode optimizer (which would otherwise see an unused
    // capacity-64 Vec and elide the heap allocation entirely) from
    // making this check vacuous.
    let (v, delta) = measure(|| {
        let mut v = Vec::<u8>::with_capacity(core::hint::black_box(64));
        v.push(core::hint::black_box(0));
        v
    });
    let _ = core::hint::black_box(v);
    assert!(
        delta.allocations > 0,
        "CountingAllocator is not installed as #[global_allocator]; \
         ZA assertions would pass vacuously"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    // The harness's own tests install the allocator for this test binary.
    #[global_allocator]
    static GLOBAL: CountingAllocator = CountingAllocator::new();

    #[test]
    fn allocator_is_installed() {
        assert_allocator_installed();
    }

    #[test]
    fn measure_counts_an_allocation() {
        let (_v, delta) = measure(|| Vec::<u64>::with_capacity(128));
        assert!(delta.allocations >= 1);
        assert!(delta.bytes >= 128 * core::mem::size_of::<u64>());
    }

    #[test]
    fn no_alloc_holds_for_stack_work() {
        assert_no_alloc("stack arithmetic", || {
            let mut acc = 0u64;
            for i in 0..1000u64 {
                acc = acc.wrapping_add(i);
            }
            core::hint::black_box(acc);
        });
    }

    #[test]
    #[should_panic(expected = "zero runtime heap allocation")]
    fn no_alloc_catches_an_allocation() {
        assert_no_alloc("heap vec", || {
            let v: Vec<u8> = Vec::with_capacity(4096);
            core::hint::black_box(&v);
        });
    }
}
