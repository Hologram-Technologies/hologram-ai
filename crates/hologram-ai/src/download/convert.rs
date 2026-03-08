use std::path::{Path, PathBuf};
use std::process::Command;

pub struct ConversionResult {
    pub model_path: PathBuf,
    pub companion_files: Vec<PathBuf>,
}

const COMPANION_NAMES: &[&str] = &[
    "tokenizer.json",
    "config.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
];

// ── ONNX conversion via torch.onnx.export ────────────────────────────────────

/// Inline Python script that uses torch.onnx.export to convert a HuggingFace
/// model to ONNX format. Written to a temp file and executed inside the venv.
const ONNX_EXPORT_SCRIPT: &str = r#"
import sys, os, warnings, logging
model_id = sys.argv[1]
output_dir = sys.argv[2]
os.makedirs(output_dir, exist_ok=True)

# Suppress noisy warnings from torch/onnx/transformers
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

from transformers import AutoModel, AutoTokenizer, AutoConfig
import torch

print(f"Loading model {model_id}...", file=sys.stderr)
config = AutoConfig.from_pretrained(model_id)
config.use_cache = False  # avoid DynamicCache in outputs
model = AutoModel.from_pretrained(model_id, config=config, torch_dtype=torch.float32)

# Save tokenizer and config
try:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.save_pretrained(output_dir)
except Exception as e:
    print(f"Warning: could not save tokenizer: {e}", file=sys.stderr)

config.save_pretrained(output_dir)

# Build dummy inputs
seq_len = 8
dummy_ids = torch.ones(1, seq_len, dtype=torch.long)
dummy_mask = torch.ones(1, seq_len, dtype=torch.long)

input_names = ["input_ids", "attention_mask"]
output_names = ["last_hidden_state"]
dynamic_axes = {
    "input_ids": {0: "batch_size", 1: "sequence_length"},
    "attention_mask": {0: "batch_size", 1: "sequence_length"},
    "last_hidden_state": {0: "batch_size", 1: "sequence_length"},
}

onnx_path = os.path.join(output_dir, "model.onnx")
print(f"Exporting to {onnx_path}...", file=sys.stderr)

model.eval()
with torch.no_grad():
    torch.onnx.export(
        model,
        (dummy_ids, dummy_mask),
        onnx_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=18,
    )

print(f"Exported ONNX model to {onnx_path}", file=sys.stderr)
"#;

pub fn convert_to_onnx(
    model_id: &str,
    output_dir: &Path,
    keep_venv: bool,
) -> anyhow::Result<ConversionResult> {
    let python = find_python()?;

    let tmp_dir = tempfile::tempdir()?;
    let venv_path = tmp_dir.path().join("hologram-ai-conv");

    // Create virtualenv
    eprintln!("Creating Python virtualenv for ONNX conversion...");
    run_cmd(
        Command::new(&python).args(["-m", "venv"]).arg(&venv_path),
        "Failed to create Python virtualenv",
    )?;

    // Install dependencies
    let pip = venv_bin(&venv_path, "pip");
    eprintln!("Installing conversion dependencies (this may take a few minutes)...");
    run_cmd(
        Command::new(&pip).args([
            "install",
            "transformers",
            "torch",
            "onnx",
            "onnxscript",
            "sentencepiece",
            "protobuf",
        ]),
        "Failed to install Python dependencies",
    )?;

    // Write export script to temp file and run it
    let script_path = tmp_dir.path().join("export_onnx.py");
    std::fs::write(&script_path, ONNX_EXPORT_SCRIPT)?;

    std::fs::create_dir_all(output_dir)?;
    let venv_python = venv_bin(&venv_path, "python");
    eprintln!("Converting {model_id} to ONNX...");
    run_cmd(
        Command::new(&venv_python)
            .arg(&script_path)
            .arg(model_id)
            .arg(output_dir),
        "ONNX conversion failed. Check model compatibility.",
    )?;

    // Collect output files
    let mut result = ConversionResult {
        model_path: PathBuf::new(),
        companion_files: Vec::new(),
    };
    for entry in std::fs::read_dir(output_dir)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().to_string();
        if name.ends_with(".onnx") {
            result.model_path = entry.path();
        } else if COMPANION_NAMES.contains(&name.as_str()) {
            result.companion_files.push(entry.path());
        }
    }
    if result.model_path.as_os_str().is_empty() {
        anyhow::bail!("Conversion produced no .onnx file");
    }

    finish_venv(tmp_dir, &venv_path, keep_venv);
    Ok(result)
}

// ── GGUF conversion via llama.cpp convert script ─────────────────────────────

pub fn convert_to_gguf(
    model_id: &str,
    output_dir: &Path,
    quantization: Option<&str>,
    keep_venv: bool,
) -> anyhow::Result<ConversionResult> {
    let python = find_python()?;

    let tmp_dir = tempfile::tempdir()?;
    let venv_path = tmp_dir.path().join("hologram-ai-conv");
    let gguf_output_dir = tmp_dir.path().join("gguf-output");
    std::fs::create_dir_all(&gguf_output_dir)?;

    // Create virtualenv
    eprintln!("Creating Python virtualenv for GGUF conversion...");
    run_cmd(
        Command::new(&python).args(["-m", "venv"]).arg(&venv_path),
        "Failed to create Python virtualenv",
    )?;

    // Install dependencies (llama-cpp-python ships convert_hf_to_gguf.py)
    let pip = venv_bin(&venv_path, "pip");
    eprintln!("Installing conversion dependencies (this may take a few minutes)...");
    run_cmd(
        Command::new(&pip).args([
            "install",
            "torch",
            "transformers",
            "sentencepiece",
            "protobuf",
            "gguf",
        ]),
        "Failed to install Python dependencies",
    )?;

    // Download the convert script from llama.cpp
    let convert_script = tmp_dir.path().join("convert_hf_to_gguf.py");
    let venv_python = venv_bin(&venv_path, "python");
    eprintln!("Downloading llama.cpp convert script...");
    run_cmd(
        Command::new(&venv_python).args([
            "-c",
            &format!(
                "import urllib.request; urllib.request.urlretrieve(\
                 'https://raw.githubusercontent.com/ggerganov/llama.cpp/master/convert_hf_to_gguf.py', \
                 '{}')",
                convert_script.display()
            ),
        ]),
        "Failed to download convert_hf_to_gguf.py",
    )?;

    // Determine output filename
    let model_name = model_id.split('/').next_back().unwrap_or(model_id);
    let quant_suffix = quantization.unwrap_or("F16");
    let out_file = gguf_output_dir.join(format!("{model_name}-{quant_suffix}.gguf"));

    // Run conversion
    eprintln!("Converting {model_id} to GGUF...");
    let mut cmd = Command::new(&venv_python);
    cmd.arg(&convert_script)
        .args(["--model", model_id])
        .arg("--outfile")
        .arg(&out_file);

    if let Some(q) = quantization {
        cmd.args(["--outtype", q]);
    }

    run_cmd(
        &mut cmd,
        "GGUF conversion failed. Check model compatibility.",
    )?;

    // Copy output files
    std::fs::create_dir_all(output_dir)?;
    let dest = output_dir.join(out_file.file_name().unwrap());
    std::fs::copy(&out_file, &dest)?;

    // Try to also download companion files (tokenizer, config) via transformers
    let companion_script = format!(
        "from transformers import AutoTokenizer; \
         t = AutoTokenizer.from_pretrained('{}'); \
         t.save_pretrained('{}')",
        model_id,
        output_dir.display()
    );
    // Best-effort — don't fail if tokenizer download doesn't work
    let _ = Command::new(&venv_python)
        .args(["-c", &companion_script])
        .status();

    let mut companion_files = Vec::new();
    for name in COMPANION_NAMES {
        let p = output_dir.join(name);
        if p.exists() {
            companion_files.push(p);
        }
    }

    finish_venv(tmp_dir, &venv_path, keep_venv);

    Ok(ConversionResult {
        model_path: dest,
        companion_files,
    })
}

// ── Helpers ──────────────────────────────────────────────────────────────────

fn run_cmd(cmd: &mut Command, fail_msg: &str) -> anyhow::Result<()> {
    let status = cmd.status()?;
    if !status.success() {
        anyhow::bail!("{fail_msg}");
    }
    Ok(())
}

pub fn find_python() -> anyhow::Result<PathBuf> {
    for name in ["python3", "python"] {
        if let Ok(path) = which::which(name) {
            let output = Command::new(&path).args(["--version"]).output()?;
            let version = String::from_utf8_lossy(&output.stdout);
            if let Some(minor) = parse_python_minor(&version) {
                if minor >= 10 {
                    return Ok(path);
                }
            }
        }
    }
    anyhow::bail!(
        "python3 >= 3.10 required for model conversion. Install Python 3.10+ or use a format \
         that already exists on HuggingFace."
    )
}

fn parse_python_minor(version_str: &str) -> Option<u32> {
    // "Python 3.12.1" → 12
    let rest = version_str.trim().strip_prefix("Python ")?;
    let mut parts = rest.split('.');
    let _major = parts.next()?.parse::<u32>().ok()?;
    let minor = parts.next()?.parse::<u32>().ok()?;
    Some(minor)
}

fn venv_bin(venv: &Path, name: &str) -> PathBuf {
    if cfg!(windows) {
        venv.join("Scripts").join(name)
    } else {
        venv.join("bin").join(name)
    }
}

fn finish_venv(tmp_dir: tempfile::TempDir, venv_path: &Path, keep: bool) {
    if keep {
        let display = venv_path.display().to_string();
        std::mem::forget(tmp_dir);
        eprintln!("Virtualenv preserved at: {display}");
    } else {
        eprintln!("Cleaned up conversion virtualenv.");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_python_version_312() {
        assert_eq!(parse_python_minor("Python 3.12.1"), Some(12));
    }

    #[test]
    fn parse_python_version_310() {
        assert_eq!(parse_python_minor("Python 3.10.0"), Some(10));
    }

    #[test]
    fn parse_python_version_39() {
        assert_eq!(parse_python_minor("Python 3.9.7"), Some(9));
    }

    #[test]
    fn parse_python_version_garbage() {
        assert_eq!(parse_python_minor("not a version"), None);
    }
}
