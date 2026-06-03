//! Subprocess helper: spawn a child, stream its stdout/stderr line-by-line
//! into both the log buffer and a Tauri event channel, and signal-cancel
//! it when requested.

use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Arc;
use std::time::SystemTime;

use serde::Serialize;
use tauri::{AppHandle, Emitter};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::{mpsc, oneshot};

use crate::log_buffer::{LogBuffer, LogEntry};

#[derive(Clone, Debug, Serialize)]
pub struct ProcessLine {
    pub stream: &'static str, // "stdout" or "stderr"
    pub line: String,
}

/// Spawn `bin args...` with the supplied working directory. Each line of
/// stdout/stderr is appended to `logs` and emitted as `event_name` to the
/// frontend. The returned channel is signalled with the final exit status.
///
/// `cancel` is a oneshot that, when received, kills the child process.
pub async fn spawn_streaming(
    app: AppHandle,
    bin: PathBuf,
    args: Vec<String>,
    cwd: PathBuf,
    logs: Arc<LogBuffer>,
    event_name: &'static str,
    mut cancel: oneshot::Receiver<()>,
) -> anyhow::Result<i32> {
    let mut child: Child = Command::new(&bin)
        .args(&args)
        .current_dir(&cwd)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| anyhow::anyhow!("spawning {}: {e}", bin.display()))?;

    let stdout = child.stdout.take().expect("piped stdout");
    let stderr = child.stderr.take().expect("piped stderr");

    let (tx, mut rx) = mpsc::channel::<ProcessLine>(256);

    spawn_reader(stdout, "stdout", tx.clone());
    spawn_reader(stderr, "stderr", tx);

    // Pump child output to logs + frontend until the channel closes.
    let app_for_pump = app.clone();
    let logs_for_pump = logs.clone();
    let event_name_owned: &'static str = event_name;
    let pump = tokio::spawn(async move {
        while let Some(line) = rx.recv().await {
            // The CLI emits informational tracing on stderr — both streams
            // are surfaced as "info" in the buffer; the Logs screen shows
            // them inline with their target.
            logs_for_pump.push(LogEntry {
                timestamp_ms: SystemTime::now()
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .map(|d| d.as_millis() as u64)
                    .unwrap_or(0),
                level: "info".to_string(),
                target: "hologram-ai".to_string(),
                message: line.line.clone(),
            });
            let _ = app_for_pump.emit(event_name_owned, &line);
        }
    });

    let exit_code = tokio::select! {
        status = child.wait() => {
            status.map(|s| s.code().unwrap_or(-1)).unwrap_or(-1)
        }
        _ = &mut cancel => {
            // Graceful kill — child cleanup is best-effort.
            let _ = child.start_kill();
            let _ = child.wait().await;
            130 // conventional cancelled-by-signal exit code
        }
    };

    let _ = pump.await;
    Ok(exit_code)
}

fn spawn_reader<R: tokio::io::AsyncRead + Unpin + Send + 'static>(
    reader: R,
    stream: &'static str,
    tx: mpsc::Sender<ProcessLine>,
) {
    tokio::spawn(async move {
        let mut buf = BufReader::new(reader).lines();
        while let Ok(Some(line)) = buf.next_line().await {
            if tx.send(ProcessLine { stream, line }).await.is_err() {
                break;
            }
        }
    });
}
