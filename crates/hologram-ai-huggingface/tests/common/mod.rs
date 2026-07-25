//! Shared helpers for the hermetic test suite: temp dirs, fixture copies,
//! and fake `hf` executables. No test touches the network.

#![allow(dead_code)]

use std::fs;
use std::path::{Path, PathBuf};
use std::process;
use std::sync::atomic::{AtomicU32, Ordering};

static COUNTER: AtomicU32 = AtomicU32::new(0);

/// Unique temporary directory, removed on drop (best effort).
pub struct TempDir {
    pub path: PathBuf,
}

impl TempDir {
    pub fn new(label: &str) -> Self {
        let path = std::env::temp_dir().join(format!(
            "holo-hf-{label}-{}-{}",
            process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&path).expect("create temp dir");
        Self { path }
    }

    pub fn join(&self, rel: &str) -> PathBuf {
        self.path.join(rel)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

/// Absolute path of the hermetic HF-style fixture at `oracles/fixture`.
pub fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../oracles/fixture")
}

/// Copy all fixture files (flat) into `dest`.
pub fn copy_fixture(dest: &Path) {
    fs::create_dir_all(dest).expect("create dest");
    for entry in fs::read_dir(fixture_dir()).expect("read fixture") {
        let entry = entry.expect("fixture entry");
        let name = entry.file_name();
        fs::copy(entry.path(), dest.join(&name)).expect("copy fixture file");
    }
}

/// Write a fake `hf` executable with `body` as the script source and return
/// its path. `__FIXTURE__` in the body is replaced with the fixture path.
#[cfg(unix)]
pub fn write_fake_hf(dir: &Path, body: &str) -> PathBuf {
    use std::os::unix::fs::PermissionsExt;
    let script = body.replace("__FIXTURE__", &fixture_dir().to_string_lossy());
    let path = dir.join("hf");
    fs::write(&path, script).expect("write fake hf");
    fs::set_permissions(&path, fs::Permissions::from_mode(0o755)).expect("chmod fake hf");
    path
}

/// Fake `hf` that copies the fixture into `--local-dir` and exits 0.
pub const FAKE_HF_COPY: &str = r#"#!/bin/sh
dest=""
prev=""
for arg in "$@"; do
  if [ "$prev" = "--local-dir" ]; then dest="$arg"; fi
  prev="$arg"
done
if [ -z "$dest" ]; then echo "fake-hf: missing --local-dir" >&2; exit 2; fi
mkdir -p "$dest"
cp __FIXTURE__/config.json \
   __FIXTURE__/generation_config.json \
   __FIXTURE__/model.safetensors \
   __FIXTURE__/tokenizer.json \
   __FIXTURE__/tokenizer_config.json \
   __FIXTURE__/reference-transcript.json \
   "$dest"/
"#;

/// Fake `hf` that also records hf-style resolved-commit metadata for
/// `config.json` (40-hex sha `__SHA__`) before copying the fixture.
pub const FAKE_HF_COPY_WITH_METADATA: &str = r#"#!/bin/sh
dest=""
prev=""
for arg in "$@"; do
  if [ "$prev" = "--local-dir" ]; then dest="$arg"; fi
  prev="$arg"
done
if [ -z "$dest" ]; then echo "fake-hf: missing --local-dir" >&2; exit 2; fi
mkdir -p "$dest/.cache/huggingface"
printf '%s\n%s\n%s\n' "__SHA__" "fake-etag" "0" > "$dest/.cache/huggingface/config.json.metadata"
cp __FIXTURE__/config.json \
   __FIXTURE__/generation_config.json \
   __FIXTURE__/model.safetensors \
   __FIXTURE__/tokenizer.json \
   __FIXTURE__/tokenizer_config.json \
   __FIXTURE__/reference-transcript.json \
   "$dest"/
"#;

/// Fake `hf` that fails with a 401 and echoes the token on stderr (to prove
/// the provider redacts it from error messages).
pub const FAKE_HF_AUTH_FAILURE: &str = r#"#!/bin/sh
echo "401 Unauthorized: bad token $HF_TOKEN for repo" >&2
exit 1
"#;

/// Fake `hf` that requires the token via the environment and refuses to run
/// if it appears anywhere in argv.
pub const FAKE_HF_TOKEN_ENV_CHECK: &str = r#"#!/bin/sh
if [ -z "$HF_TOKEN" ]; then echo "fake-hf: HF_TOKEN not set" >&2; exit 5; fi
for arg in "$@"; do
  if [ "$arg" = "$HF_TOKEN" ]; then echo "fake-hf: token leaked into argv" >&2; exit 6; fi
done
dest=""
prev=""
for arg in "$@"; do
  if [ "$prev" = "--local-dir" ]; then dest="$arg"; fi
  prev="$arg"
done
if [ -z "$dest" ]; then echo "fake-hf: missing --local-dir" >&2; exit 2; fi
mkdir -p "$dest"
cp __FIXTURE__/config.json \
   __FIXTURE__/generation_config.json \
   __FIXTURE__/model.safetensors \
   __FIXTURE__/tokenizer.json \
   __FIXTURE__/tokenizer_config.json \
   __FIXTURE__/reference-transcript.json \
   "$dest"/
"#;

/// Fake `hf` that sleeps before copying (for cancellation-during-download).
pub const FAKE_HF_SLOW: &str = r#"#!/bin/sh
sleep 5
dest=""
prev=""
for arg in "$@"; do
  if [ "$prev" = "--local-dir" ]; then dest="$arg"; fi
  prev="$arg"
done
if [ -z "$dest" ]; then echo "fake-hf: missing --local-dir" >&2; exit 2; fi
mkdir -p "$dest"
cp __FIXTURE__/config.json \
   __FIXTURE__/generation_config.json \
   __FIXTURE__/model.safetensors \
   __FIXTURE__/tokenizer.json \
   __FIXTURE__/tokenizer_config.json \
   __FIXTURE__/reference-transcript.json \
   "$dest"/
"#;

/// A syntactically valid pinned sha for tests.
pub const TEST_SHA: &str = "0123456789abcdef0123456789abcdef01234567";

/// A second valid sha (for resolved-revision tests).
pub const TEST_SHA_RESOLVED: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

/// A syntactically valid repository id for tests.
pub const TEST_REPO: &str = "acme/fixture-model";
