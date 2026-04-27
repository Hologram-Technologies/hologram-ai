//! Ring buffer for tracing events plus a subscriber layer that fills it.
//!
//! The buffer is read by the `recent_logs` IPC command. Subprocess stdout
//! is also piped through this buffer (see `commands::process_runner`), so
//! the Logs screen reflects both in-process tracing and CLI output.

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::SystemTime;

use parking_lot::RwLock;
use serde::Serialize;
use tracing::field::{Field, Visit};
use tracing::{Event, Level, Subscriber};
use tracing_subscriber::layer::{Context, Layer};
use tracing_subscriber::prelude::*;

const CAPACITY: usize = 4096;

#[derive(Clone, Debug, Serialize)]
pub struct LogEntry {
    pub timestamp_ms: u64,
    pub level: String,
    pub target: String,
    pub message: String,
}

pub struct LogBuffer {
    inner: RwLock<VecDeque<LogEntry>>,
}

impl LogBuffer {
    fn new() -> Self {
        Self {
            inner: RwLock::new(VecDeque::with_capacity(CAPACITY)),
        }
    }

    pub fn push(&self, entry: LogEntry) {
        let mut q = self.inner.write();
        if q.len() == CAPACITY {
            q.pop_front();
        }
        q.push_back(entry);
    }

    pub fn snapshot(&self, since_idx: usize) -> (Vec<LogEntry>, usize) {
        let q = self.inner.read();
        // We expose monotonically-increasing indices to the frontend so it
        // can request only entries newer than what it last saw.
        let total_seen = q.len() + q.front().map(|_| 0).unwrap_or(0);
        let from = since_idx.min(q.len());
        let entries: Vec<LogEntry> = q.iter().skip(from).cloned().collect();
        (entries, total_seen)
    }

    pub fn clear(&self) {
        self.inner.write().clear();
    }
}

struct BufferLayer {
    buf: Arc<LogBuffer>,
}

struct MessageVisitor {
    message: String,
}

impl Visit for MessageVisitor {
    fn record_str(&mut self, field: &Field, value: &str) {
        if field.name() == "message" {
            self.message.push_str(value);
        } else {
            self.append_kv(field.name(), value);
        }
    }

    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        if field.name() == "message" {
            self.message.push_str(&format!("{value:?}"));
        } else {
            self.append_kv(field.name(), &format!("{value:?}"));
        }
    }
}

impl MessageVisitor {
    fn append_kv(&mut self, key: &str, value: &str) {
        if !self.message.is_empty() {
            self.message.push(' ');
        }
        self.message.push_str(key);
        self.message.push('=');
        self.message.push_str(value);
    }
}

impl<S: Subscriber> Layer<S> for BufferLayer {
    fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
        let mut visitor = MessageVisitor {
            message: String::new(),
        };
        event.record(&mut visitor);
        let level = match *event.metadata().level() {
            Level::ERROR => "error",
            Level::WARN => "warn",
            Level::INFO => "info",
            Level::DEBUG => "debug",
            Level::TRACE => "trace",
        };
        self.buf.push(LogEntry {
            timestamp_ms: SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0),
            level: level.to_string(),
            target: event.metadata().target().to_string(),
            message: visitor.message,
        });
    }
}

/// Install the global tracing subscriber and return the shared buffer.
///
/// Combines a stderr formatter (for terminal visibility) with the in-memory
/// ring buffer that the Logs screen reads from.
pub fn install_subscriber() -> Arc<LogBuffer> {
    let buf = Arc::new(LogBuffer::new());
    let env_filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));

    let layer = BufferLayer { buf: buf.clone() };
    let _ = tracing_subscriber::registry()
        .with(env_filter)
        .with(tracing_subscriber::fmt::layer().with_target(false))
        .with(layer)
        .try_init();
    buf
}
