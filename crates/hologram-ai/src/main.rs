//! Unified hologram-ai CLI.
//!
//! This is the main entry point for the hologram-ai command-line tool.
//! It provides commands for compiling, running, and inspecting AI models
//! from various formats.

fn main() -> anyhow::Result<()> {
    hologram_ai::cli::run()
}
