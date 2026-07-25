//! Local-directory provider: fixture acceptance, digest determinism, and
//! rejection of unsupported layouts, traversal, and symlinks escaping root.

mod common;

use std::fs;

use hologram_ai_core::{CancellationToken, ErrorCategory, NullProgressSink};
use hologram_ai_huggingface::{
    LocalDirectoryProvider, ModelSourceProvider, Revision, SourceRequest,
};

use common::{copy_fixture, fixture_dir, TempDir};

fn acquire_local(
    path: &std::path::Path,
) -> hologram_ai_core::AiResult<hologram_ai_huggingface::AcquiredSource> {
    let provider = LocalDirectoryProvider::new();
    let request = SourceRequest::local_dir(path);
    provider.acquire(&request, &mut NullProgressSink, &CancellationToken::new())
}

#[test]
fn fixture_acquires_with_deterministic_digest() {
    let first = acquire_local(&fixture_dir()).expect("fixture must acquire");
    assert_eq!(first.repository, None);
    assert_eq!(first.resolved_revision, None);
    assert_eq!(first.cache_key, None);
    assert!(!first.from_cache);
    assert_eq!(
        first.source_dir,
        fs::canonicalize(fixture_dir()).expect("canonical fixture"),
        "source_dir is the canonicalized path"
    );

    let second = acquire_local(&fixture_dir()).expect("fixture must acquire again");
    assert_eq!(
        first.source_digest, second.source_digest,
        "source_digest must be deterministic"
    );
    assert_ne!(first.source_digest, [0u8; 32]);
}

#[test]
fn missing_tokenizer_is_unsupported_model() {
    let temp = TempDir::new("missing-tokenizer");
    let dir = temp.join("model");
    copy_fixture(&dir);
    fs::remove_file(dir.join("tokenizer.json")).expect("remove tokenizer");

    let err = acquire_local(&dir).expect_err("must be rejected");
    assert_eq!(err.category(), ErrorCategory::UnsupportedModel);
    assert!(
        err.message().contains("tokenizer.json"),
        "message should name the missing file: {}",
        err.message()
    );
}

#[test]
fn missing_weights_is_unsupported_model() {
    let temp = TempDir::new("missing-weights");
    let dir = temp.join("model");
    copy_fixture(&dir);
    fs::remove_file(dir.join("model.safetensors")).expect("remove weights");

    let err = acquire_local(&dir).expect_err("must be rejected");
    assert_eq!(err.category(), ErrorCategory::UnsupportedModel);
    assert!(
        err.message().contains("weights"),
        "message should mention weights: {}",
        err.message()
    );
}

#[test]
fn sharded_weights_are_accepted() {
    let temp = TempDir::new("sharded");
    let dir = temp.join("model");
    copy_fixture(&dir);
    fs::rename(
        dir.join("model.safetensors"),
        dir.join("model-00001-of-00001.safetensors"),
    )
    .expect("rename to shard name");
    acquire_local(&dir).expect("sharded weights must be accepted");
}

#[test]
fn missing_config_is_unsupported_model() {
    let temp = TempDir::new("missing-config");
    let dir = temp.join("model");
    copy_fixture(&dir);
    fs::remove_file(dir.join("config.json")).expect("remove config");

    let err = acquire_local(&dir).expect_err("must be rejected");
    assert_eq!(err.category(), ErrorCategory::UnsupportedModel);
}

#[test]
fn missing_directory_is_invalid_argument() {
    let temp = TempDir::new("missing-dir");
    let err = acquire_local(&temp.join("does-not-exist")).expect_err("must be rejected");
    assert_eq!(err.category(), ErrorCategory::InvalidArgument);
}

#[cfg(unix)]
#[test]
fn symlink_escaping_root_is_rejected() {
    let temp = TempDir::new("symlink-escape");
    let outside = temp.join("outside");
    fs::create_dir_all(&outside).expect("create outside dir");
    fs::write(outside.join("secret.json"), b"{}").expect("write outside file");

    let dir = temp.join("model");
    copy_fixture(&dir);
    std::os::unix::fs::symlink(outside.join("secret.json"), dir.join("evil.json"))
        .expect("create symlink");

    let err = acquire_local(&dir).expect_err("escaping symlink must be rejected");
    assert_eq!(err.category(), ErrorCategory::InvalidArgument);
    assert!(
        err.message().contains("escapes"),
        "message should explain the escape: {}",
        err.message()
    );
}

#[cfg(unix)]
#[test]
fn symlinked_directory_is_rejected() {
    let temp = TempDir::new("symlink-dir");
    let dir = temp.join("model");
    copy_fixture(&dir);
    std::os::unix::fs::symlink(&dir, dir.join("self-link")).expect("create dir symlink");

    let err = acquire_local(&dir).expect_err("symlinked dir must be rejected");
    assert_eq!(err.category(), ErrorCategory::InvalidArgument);
}

#[test]
fn hf_repo_request_is_invalid_argument_for_local_provider() {
    let provider = LocalDirectoryProvider::new();
    let request = SourceRequest::hf_repo(common::TEST_REPO, Revision::pinned(common::TEST_SHA));
    let err = provider
        .acquire(&request, &mut NullProgressSink, &CancellationToken::new())
        .expect_err("wrong request kind must be rejected");
    assert_eq!(err.category(), ErrorCategory::InvalidArgument);
}

#[test]
fn cancellation_is_honoured() {
    let token = CancellationToken::new();
    token.cancel();
    let provider = LocalDirectoryProvider::new();
    let request = SourceRequest::local_dir(fixture_dir());
    let err = provider
        .acquire(&request, &mut NullProgressSink, &token)
        .expect_err("pre-cancelled token must abort");
    assert_eq!(err.category(), ErrorCategory::Cancelled);
}
