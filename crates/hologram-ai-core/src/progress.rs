//! Progress reporting and cancellation.

use alloc::string::String;

/// A structured progress event from acquisition or compilation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProgressEvent {
    /// Coarse stage (e.g. `resolve`, `download`, `verify`, `observe`,
    /// `cover`, `score`, `bundle`, `archive`).
    pub stage: String,
    /// Completion within the stage, 0–100. `None` = indeterminate.
    pub percent: Option<u8>,
    /// Human-readable detail. Never contains credentials.
    pub detail: String,
}

/// Receiver of structured progress events.
pub trait ProgressSink {
    fn on_progress(&mut self, event: ProgressEvent);
}

/// A no-op progress sink.
pub struct NullProgressSink;

impl ProgressSink for NullProgressSink {
    fn on_progress(&mut self, _event: ProgressEvent) {}
}

impl<F: FnMut(ProgressEvent)> ProgressSink for F {
    fn on_progress(&mut self, event: ProgressEvent) {
        self(event)
    }
}

/// Cooperative cancellation token, safe to share across threads.
///
/// Clone-cheap handle over an atomic flag; engines and providers poll it at
/// step boundaries and return [`crate::ErrorCategory::Cancelled`].
#[derive(Debug, Clone, Default)]
pub struct CancellationToken {
    flag: alloc::sync::Arc<core::sync::atomic::AtomicBool>,
}

impl CancellationToken {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn cancel(&self) {
        self.flag.store(true, core::sync::atomic::Ordering::Relaxed);
    }

    pub fn is_cancelled(&self) -> bool {
        self.flag.load(core::sync::atomic::Ordering::Relaxed)
    }
}
