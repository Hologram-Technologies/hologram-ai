//! CLI entry point for hologram-ai.

use clap::Parser;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "hologram-ai", about = "AI model inference via hologram runtime")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(clap::Subcommand)]
enum Command {
    /// Run inference on a model file.
    Run {
        /// Path to ONNX or GGUF model.
        #[arg(short, long)]
        model: PathBuf,
        /// Input token IDs (comma-separated).
        #[arg(short, long, value_delimiter = ',')]
        tokens: Vec<u32>,
    },
    /// Print model metadata without running inference.
    Info {
        #[arg(short, long)]
        model: PathBuf,
    },
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber_init();
    let cli = Cli::parse();

    match cli.command {
        Command::Run { model, tokens } => {
            use hologram_ai::{session::{ModelCompiler, InferenceSession, ModelSource, CompileOptions}};
            use std::sync::Arc;

            let source = model_source_from_path(&model)?;
            let compiled = ModelCompiler::compile(source, CompileOptions::default())?;
            let mut sess = InferenceSession::new(Arc::new(compiled));
            let logits = sess.run(&tokens)?;
            println!("logits shape: [{}]", logits.len());
        }
        Command::Info { model } => {
            use hologram_ai::session::{ModelCompiler, ModelSource, CompileOptions};

            let source = model_source_from_path(&model)?;
            let compiled = ModelCompiler::compile(source, CompileOptions::default())?;
            let m = &compiled.metadata;
            println!("arch:        {}", m.arch);
            println!("vocab_size:  {}", m.vocab_size);
            println!("context_len: {}", m.context_len);
            println!("n_layers:    {}", m.n_layers);
            println!("n_embd:      {}", m.n_embd);
        }
    }

    Ok(())
}

fn model_source_from_path(path: &std::path::Path) -> anyhow::Result<ModelSource> {
    use hologram_ai::session::ModelSource;
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    match ext {
        "onnx" => Ok(ModelSource::OnnxPath(path.to_owned())),
        "gguf" => Ok(ModelSource::GgufPath(path.to_owned())),
        other  => anyhow::bail!("unsupported model extension: '.{other}'"),
    }
}

fn tracing_subscriber_init() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .try_init();
}
