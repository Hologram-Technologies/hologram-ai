use std::sync::Arc;

use parking_lot::Mutex;
use tokio::sync::oneshot;

use crate::log_buffer::LogBuffer;

/// Shared app state held in Tauri's `State` extractor.
pub struct AppState {
    pub logs: Arc<LogBuffer>,
    /// Cancellation handle for the currently running generation, if any.
    /// Sending on the channel signals the generation task to terminate
    /// its child subprocess.
    pub active_generation: Mutex<Option<oneshot::Sender<()>>>,
}

impl AppState {
    pub fn new(logs: Arc<LogBuffer>) -> Self {
        Self {
            logs,
            active_generation: Mutex::new(None),
        }
    }
}
