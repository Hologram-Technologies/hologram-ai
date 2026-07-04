//! Workspace automation (`cargo run -p xtask -- <command>`).
//!
//! - `oracle-verify` — every committed oracle artifact matches its
//!   `model/oracles.toml` sha256 (offline).
//! - `pin-check`     — every pinned upstream (git revisions, HF model
//!   revisions, release tags) still exists upstream (online).
//! - `report`        — emit the conformance ledger from the dictionary.
//! - `gen-fixture`   — regenerate the hermetic handshake-tiny model + its
//!   committed deterministic references (oracle `journey-reference`).

#![forbid(unsafe_code)]

mod fixture;

use anyhow::{bail, Context, Result};
use hologram_ai_model::{Model, Tier};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

fn root() -> PathBuf {
    hologram_ai_model::workspace_root()
}

pub(crate) fn sha256_file(path: &Path) -> Result<String> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let mut h = Sha256::new();
    h.update(&bytes);
    Ok(hex_of(&h.finalize()))
}

fn hex_of(bytes: &[u8]) -> String {
    use std::fmt::Write;
    bytes.iter().fold(String::new(), |mut s, b| {
        write!(s, "{b:02x}").expect("writing to a String cannot fail");
        s
    })
}

fn main() -> Result<()> {
    let cmd = std::env::args().nth(1).unwrap_or_default();
    match cmd.as_str() {
        "oracle-verify" => oracle_verify(),
        "pin-check" => pin_check(),
        "report" => report(),
        "gen-fixture" => fixture::gen_fixture(),
        other => {
            bail!("unknown command `{other}`; use: oracle-verify | pin-check | report | gen-fixture")
        }
    }
}

/// Verify every committed oracle artifact against its recorded sha256.
fn oracle_verify() -> Result<()> {
    let model = Model::load()?;
    let root = root();
    let mut checked = 0u32;
    for oracle in &model.oracles {
        for artifact in &oracle.artifacts {
            let path = root.join(&artifact.path);
            let got = sha256_file(&path)?;
            if got != artifact.sha256 {
                bail!(
                    "oracle `{}` artifact `{}` sha256 mismatch: {got} != {} (manifest)",
                    oracle.id,
                    artifact.path,
                    artifact.sha256
                );
            }
            println!("ok   {:<22} {}  {}…", oracle.id, artifact.path, &got[..12]);
            checked += 1;
        }
    }
    println!("oracle-verify: {checked} artifact(s) verified against model/oracles.toml");
    Ok(())
}

/// Confirm every pinned upstream still exists (git revision pins only; release
/// tags and HF revisions are exercised by the suites that download them).
fn pin_check() -> Result<()> {
    let model = Model::load()?;
    let mut checked = 0u32;
    for oracle in &model.oracles {
        if oracle.pin.is_empty() {
            continue;
        }
        match oracle.kind.as_str() {
            "substrate-witness" => {
                git_pin_is_reachable(&oracle.source, &oracle.pin)
                    .with_context(|| format!("oracle `{}`", oracle.id))?;
                println!("ok   {:<22} git {} reachable", oracle.id, &oracle.pin[..12]);
                checked += 1;
            }
            "live-authority" if oracle.source.contains("huggingface.co") => {
                hf_revision_exists(&oracle.source, &oracle.pin)
                    .with_context(|| format!("oracle `{}`", oracle.id))?;
                println!("ok   {:<22} HF revision {} live", oracle.id, &oracle.pin[..12]);
                checked += 1;
            }
            "live-authority" if oracle.source.contains("github.com") => {
                git_tag_is_live(&oracle.source, &oracle.pin)
                    .with_context(|| format!("oracle `{}`", oracle.id))?;
                println!("ok   {:<22} tag {} live", oracle.id, oracle.pin);
                checked += 1;
            }
            _ => {}
        }
    }
    println!("pin-check: {checked} pin(s) confirmed live upstream");
    Ok(())
}

/// A pinned git commit must be an ancestor of a live ref (fetchable).
fn git_pin_is_reachable(source: &str, pin: &str) -> Result<()> {
    // `git fetch --depth 1 <url> <sha>` succeeds exactly when the commit is
    // fetchable upstream — works for any reachable commit, not only ref tips.
    let tmp = std::env::temp_dir().join(format!("hai-pin-check-{}", std::process::id()));
    std::fs::create_dir_all(&tmp)?;
    let run = |args: &[&str]| -> Result<bool> {
        Ok(std::process::Command::new("git")
            .current_dir(&tmp)
            .args(args)
            .output()
            .context("running git")?
            .status
            .success())
    };
    if !run(&["init", "-q"])? {
        bail!("git init failed");
    }
    let ok = run(&["fetch", "--depth", "1", source, pin])?;
    std::fs::remove_dir_all(&tmp).ok();
    if !ok {
        bail!("pinned commit {pin} is not fetchable from {source}");
    }
    Ok(())
}

/// A pinned tag must be listed by `git ls-remote --tags`.
fn git_tag_is_live(source: &str, tag: &str) -> Result<()> {
    let out = std::process::Command::new("git")
        .args(["ls-remote", "--tags", source, tag])
        .output()
        .with_context(|| format!("git ls-remote {source}"))?;
    if !out.status.success() || out.stdout.is_empty() {
        bail!("tag {tag} not found at {source}");
    }
    Ok(())
}

/// A pinned HF revision must resolve through the Hub API.
fn hf_revision_exists(source: &str, revision: &str) -> Result<()> {
    let repo = source
        .strip_prefix("https://huggingface.co/")
        .with_context(|| format!("not an HF repo url: {source}"))?;
    let url = format!("https://huggingface.co/api/models/{repo}/revision/{revision}");
    let status = std::process::Command::new("curl")
        .args(["-fsSL", "-o", "/dev/null", &url])
        .status()
        .context("running curl")?;
    if !status.success() {
        bail!("HF revision {revision} not found for {repo}");
    }
    Ok(())
}

/// Emit the conformance ledger: every dictionary row, its status, tier,
/// stage, executor, and oracles.
fn report() -> Result<()> {
    let model = Model::load()?;
    let audit = hologram_ai_model::honesty::audit(&model, &root())?;

    println!("# Conformance ledger\n");
    println!(
        "{} rows — {} gating suites, {} measured targets ({} rust / {} browser)\n",
        model.rows.len(),
        audit.suites,
        audit.targets,
        audit.rust_rows,
        audit.browser_rows
    );
    println!("| row | stage | status | tier | executor | oracles |");
    println!("|---|---|---|---|---|---|");
    for row in &model.rows {
        println!(
            "| {} | {} | {} | {} | {:?} | {} |",
            row.id,
            row.stage,
            row.status,
            match row.tier {
                Tier::Suite => "suite",
                Tier::Target => "target",
            },
            row.executor,
            row.oracles.join(", ")
        );
    }
    Ok(())
}
