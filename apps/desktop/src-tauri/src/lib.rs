//! hologram-ai-desktop Tauri app — thin wrapper around the `hologram-ai` CLI.
//!
//! All inference and compilation runs as subprocesses of the existing
//! `hologram-ai` binary. The Tauri app owns: subprocess lifecycle, token
//! streaming, log buffering, and three screens (models, chat, logs).

mod commands;
mod known_models;
mod log_buffer;
mod paths;
mod state;

use state::AppState;

pub fn run() {
    let log_buf = log_buffer::install_subscriber();

    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .manage(AppState::new(log_buf))
        .invoke_handler(tauri::generate_handler![
            commands::models::list_known_models,
            commands::models::list_compiled_archives,
            commands::models::download_known_model,
            commands::models::compile_known_model,
            commands::models::workspace_paths,
            commands::chat::generate,
            commands::chat::cancel_generation,
            commands::logs::recent_logs,
            commands::logs::clear_logs,
        ])
        .run(tauri::generate_context!())
        .expect("error while running hologram-ai-desktop");
}
