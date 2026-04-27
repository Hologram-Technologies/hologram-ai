//! Log buffer access for the Logs screen.

use serde::Serialize;
use tauri::State;

use crate::log_buffer::LogEntry;
use crate::state::AppState;

#[derive(Serialize)]
pub struct LogsResponse {
    pub entries: Vec<LogEntry>,
    pub next_index: usize,
}

#[tauri::command]
pub fn recent_logs(state: State<'_, AppState>, since: usize) -> LogsResponse {
    let (entries, next_index) = state.logs.snapshot(since);
    LogsResponse {
        entries,
        next_index,
    }
}

#[tauri::command]
pub fn clear_logs(state: State<'_, AppState>) {
    state.logs.clear();
}
