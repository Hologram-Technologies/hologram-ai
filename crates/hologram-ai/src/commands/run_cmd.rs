//! `hologram-ai run` — execute a compiled `.holo` archive.
//!
//! Loads the archive into a [`HoloRunner`] (a thin `InferenceSession` wrapper)
//! and runs a forward pass over caller-supplied input buffers. The UOR-native
//! runtime needs no KV-cache, shape projection, or host config — the compiled
//! archive carries concrete shapes and a schedule, and content-addressed
//! elision handles repeated computation (architecture §5.3, §7).

use anyhow::{Context as _, Result};
use clap::Args;
use std::path::PathBuf;

use crate::runner::HoloRunner;

/// Arguments for the `run` subcommand.
#[derive(Args, Debug)]
pub struct RunArgs {
    /// Path to the `.holo` file to execute.
    pub file: PathBuf,
    /// Input values as `INDEX:HEX` pairs (e.g. `--input 0:deadbeef`).
    #[arg(long = "input", value_name = "INDEX:HEX")]
    pub inputs: Vec<String>,
    /// Input from file as `INDEX:PATH` pairs (e.g. `--input-file 0:input.bin`).
    #[arg(long = "input-file", value_name = "INDEX:PATH")]
    pub input_files: Vec<String>,
    /// Print output bytes as hex (otherwise only shapes/sizes are printed).
    #[arg(long)]
    pub verbose: bool,
}

/// Execute the `run` subcommand.
pub fn execute(args: RunArgs) -> Result<()> {
    let mut runner = HoloRunner::from_path(&args.file, None)
        .with_context(|| format!("loading model {:?}", args.file))?;

    let n_inputs = runner.input_count();
    println!(
        "Loaded {:?}: {} input(s), {} output(s)",
        args.file,
        n_inputs,
        runner.output_count()
    );

    // Collect input byte-buffers indexed by graph-input position.
    let mut slots: Vec<Option<Vec<u8>>> = vec![None; n_inputs];
    for pair in &args.inputs {
        let (idx, bytes) = parse_hex_input(pair)?;
        store_slot(&mut slots, idx, bytes)?;
    }
    for pair in &args.input_files {
        let (idx, path) = split_index_path(pair)?;
        let bytes = std::fs::read(&path).with_context(|| format!("reading input file {path:?}"))?;
        store_slot(&mut slots, idx, bytes)?;
    }

    let missing: Vec<usize> = slots
        .iter()
        .enumerate()
        .filter_map(|(i, s)| s.is_none().then_some(i))
        .collect();
    if !missing.is_empty() {
        anyhow::bail!(
            "missing input(s) {missing:?}; supply each graph input via --input INDEX:HEX or --input-file INDEX:PATH"
        );
    }

    let owned: Vec<Vec<u8>> = slots.into_iter().map(|s| s.unwrap()).collect();
    let refs: Vec<&[u8]> = owned.iter().map(|v| v.as_slice()).collect();

    let outputs = runner.execute(&refs).context("inference failed")?;

    for (i, out) in outputs.iter().enumerate() {
        if args.verbose {
            println!(
                "output[{i}] ({} bytes): {}",
                out.bytes.len(),
                hex(&out.bytes)
            );
        } else {
            println!("output[{i}]: {} bytes", out.bytes.len());
        }
    }
    Ok(())
}

// ── input parsing ─────────────────────────────────────────────────────────────

fn store_slot(slots: &mut [Option<Vec<u8>>], idx: usize, bytes: Vec<u8>) -> Result<()> {
    let n = slots.len();
    let slot = slots
        .get_mut(idx)
        .with_context(|| format!("input index {idx} out of range (model has {n} inputs)"))?;
    *slot = Some(bytes);
    Ok(())
}

/// Parse an `INDEX:HEX` pair.
fn parse_hex_input(s: &str) -> Result<(usize, Vec<u8>)> {
    let (idx, hex_str) = s
        .split_once(':')
        .with_context(|| format!("malformed --input {s:?}; expected INDEX:HEX"))?;
    let idx: usize = idx
        .trim()
        .parse()
        .with_context(|| format!("bad input index in {s:?}"))?;
    Ok((
        idx,
        decode_hex(hex_str.trim()).map_err(|e| anyhow::anyhow!("{e} in {s:?}"))?,
    ))
}

/// Parse an `INDEX:PATH` pair.
fn split_index_path(s: &str) -> Result<(usize, PathBuf)> {
    let (idx, path) = s
        .split_once(':')
        .with_context(|| format!("malformed --input-file {s:?}; expected INDEX:PATH"))?;
    let idx: usize = idx
        .trim()
        .parse()
        .with_context(|| format!("bad input index in {s:?}"))?;
    Ok((idx, PathBuf::from(path.trim())))
}

fn decode_hex(s: &str) -> Result<Vec<u8>, String> {
    let s = s.strip_prefix("0x").unwrap_or(s);
    if !s.len().is_multiple_of(2) {
        return Err("hex string has odd length".into());
    }
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).map_err(|e| e.to_string()))
        .collect()
}

fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}
