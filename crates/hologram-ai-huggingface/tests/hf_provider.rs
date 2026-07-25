//! HuggingFaceProvider: cache population/hits, staging semantics, offline
//! mode, concurrency, cancellation, credentials — all hermetic via a fake
//! `hf` executable (a POSIX sh script injected with `with_hf_executable`;
//! the default `hf` lookup via PATH is never exercised).

mod common;

use std::fs;
use std::sync::Arc;
use std::time::Duration;

use hologram_ai_core::{CancellationToken, ErrorCategory, NullProgressSink};
use hologram_ai_huggingface::{
    cache_key, HfCredentials, HuggingFaceProvider, LocalDirectoryProvider, ModelSourceProvider,
    Revision, SourceRequest,
};

use common::{
    copy_fixture, fixture_dir, write_fake_hf, TempDir, FAKE_HF_AUTH_FAILURE, FAKE_HF_COPY,
    FAKE_HF_COPY_WITH_METADATA, FAKE_HF_SLOW, FAKE_HF_TOKEN_ENV_CHECK, TEST_REPO, TEST_SHA,
    TEST_SHA_RESOLVED,
};

fn pinned_request() -> SourceRequest {
    SourceRequest::hf_repo(TEST_REPO, Revision::pinned(TEST_SHA))
}

fn entry_dir(cache: &std::path::Path) -> std::path::PathBuf {
    cache.join("hf").join(cache_key(TEST_REPO, TEST_SHA))
}

#[test]
fn acquire_populates_cache_with_complete_marker() {
    let temp = TempDir::new("populate");
    let hf = write_fake_hf(&temp.path, FAKE_HF_COPY);
    let cache = temp.join("cache");
    let provider = HuggingFaceProvider::new(&cache).with_hf_executable(&hf);

    let source = provider
        .acquire(
            &pinned_request(),
            &mut NullProgressSink,
            &CancellationToken::new(),
        )
        .expect("acquire must succeed");

    let entry = entry_dir(&cache);
    assert_eq!(source.source_dir, entry);
    assert!(
        entry.join("COMPLETE").is_file(),
        "COMPLETE marker must exist"
    );
    assert!(entry.join("config.json").is_file());
    assert!(entry.join("model.safetensors").is_file());
    assert!(!source.from_cache);
    assert_eq!(source.repository.as_deref(), Some(TEST_REPO));
    assert_eq!(source.resolved_revision.as_deref(), Some(TEST_SHA));
    assert_eq!(
        source.cache_key.as_deref(),
        Some(cache_key(TEST_REPO, TEST_SHA).as_str())
    );
    assert!(
        !cache
            .join("hf")
            .join(format!(".staging-{}", cache_key(TEST_REPO, TEST_SHA)))
            .exists(),
        "staging must be promoted, not left behind"
    );
    assert!(
        !cache
            .join("hf")
            .join(format!(".lock-{}", cache_key(TEST_REPO, TEST_SHA)))
            .exists(),
        "lock must be released"
    );

    // The cached copy must digest identically to the local fixture.
    let local = LocalDirectoryProvider::new()
        .acquire(
            &SourceRequest::local_dir(fixture_dir()),
            &mut NullProgressSink,
            &CancellationToken::new(),
        )
        .expect("local fixture acquire");
    assert_eq!(source.source_digest, local.source_digest);
}

#[test]
fn second_acquire_is_a_verified_cache_hit() {
    let temp = TempDir::new("hit");
    let hf = write_fake_hf(&temp.path, FAKE_HF_COPY);
    let cache = temp.join("cache");
    let provider = HuggingFaceProvider::new(&cache).with_hf_executable(&hf);
    provider
        .acquire(
            &pinned_request(),
            &mut NullProgressSink,
            &CancellationToken::new(),
        )
        .expect("first acquire");

    // Point at a nonexistent executable: a hit must never spawn `hf`.
    let offline_exec_provider =
        HuggingFaceProvider::new(&cache).with_hf_executable("/nonexistent/hf-binary");
    let source = offline_exec_provider
        .acquire(
            &pinned_request(),
            &mut NullProgressSink,
            &CancellationToken::new(),
        )
        .expect("cache hit must not need hf");
    assert!(source.from_cache);
    assert_eq!(source.resolved_revision.as_deref(), Some(TEST_SHA));
}

#[test]
fn staging_directory_is_never_a_cache_hit() {
    let temp = TempDir::new("staging-not-hit");
    let cache = temp.join("cache");
    let key = cache_key(TEST_REPO, TEST_SHA);
    // Fabricate a complete-looking staging dir, manifest and all.
    let staging = cache.join("hf").join(format!(".staging-{key}"));
    copy_fixture(&staging);
    fs::write(staging.join("COMPLETE"), "fake manifest\n").expect("write marker");

    let provider = HuggingFaceProvider::new(&cache).with_offline(true);
    let err = provider
        .acquire(
            &pinned_request(),
            &mut NullProgressSink,
            &CancellationToken::new(),
        )
        .expect_err("staging must not count as a hit");
    assert_eq!(err.category(), ErrorCategory::Download);
    assert!(err.message().contains("offline"), "{}", err.message());
}

#[test]
fn offline_hit_and_miss() {
    let temp = TempDir::new("offline");
    let hf = write_fake_hf(&temp.path, FAKE_HF_COPY);
    let cache = temp.join("cache");

    // Populate once online.
    HuggingFaceProvider::new(&cache)
        .with_hf_executable(&hf)
        .acquire(
            &pinned_request(),
            &mut NullProgressSink,
            &CancellationToken::new(),
        )
        .expect("online populate");

    // Offline hit.
    let offline = HuggingFaceProvider::new(&cache).with_offline(true);
    let hit = offline
        .acquire(
            &pinned_request(),
            &mut NullProgressSink,
            &CancellationToken::new(),
        )
        .expect("offline hit must succeed");
    assert!(hit.from_cache);

    // Offline miss (different repo → different key).
    let miss = offline
        .acquire(
            &SourceRequest::hf_repo("acme/other-model", Revision::pinned(TEST_SHA)),
            &mut NullProgressSink,
            &CancellationToken::new(),
        )
        .expect_err("offline miss must fail");
    assert_eq!(miss.category(), ErrorCategory::Download);
    assert!(miss.message().contains("offline"), "{}", miss.message());
}

#[test]
fn offline_resolve_is_rejected() {
    let temp = TempDir::new("offline-resolve");
    let provider = HuggingFaceProvider::new(temp.join("cache")).with_offline(true);
    let err = provider
        .acquire(
            &SourceRequest::hf_repo(TEST_REPO, Revision::resolve("main")),
            &mut NullProgressSink,
            &CancellationToken::new(),
        )
        .expect_err("offline resolve must fail");
    assert_eq!(err.category(), ErrorCategory::Download);
    assert!(
        err.message().contains("Revision::Pinned"),
        "{}",
        err.message()
    );
}

#[test]
fn concurrent_same_key_acquisition_is_consistent() {
    let temp = TempDir::new("concurrent");
    let hf = write_fake_hf(&temp.path, FAKE_HF_COPY);
    let cache = temp.join("cache");
    let provider = Arc::new(HuggingFaceProvider::new(&cache).with_hf_executable(&hf));

    let mut handles = Vec::new();
    for _ in 0..4 {
        let provider = Arc::clone(&provider);
        handles.push(std::thread::spawn(move || {
            provider
                .acquire(
                    &pinned_request(),
                    &mut NullProgressSink,
                    &CancellationToken::new(),
                )
                .expect("concurrent acquire")
        }));
    }
    let results: Vec<_> = handles
        .into_iter()
        .map(|h| h.join().expect("join"))
        .collect();

    let first_digest = results[0].source_digest;
    for source in &results {
        assert_eq!(source.source_digest, first_digest);
        assert_eq!(source.source_dir, entry_dir(&cache));
    }
    assert!(
        results.iter().any(|s| !s.from_cache),
        "at least one thread must have populated the cache"
    );
    let entry = entry_dir(&cache);
    assert!(entry.join("COMPLETE").is_file(), "entry must be complete");
    assert!(
        entry.join("tokenizer.json").is_file(),
        "entry must be intact"
    );
}

#[test]
fn cancellation_before_spawn_aborts() {
    let temp = TempDir::new("cancel-before");
    let hf = write_fake_hf(&temp.path, FAKE_HF_COPY);
    let cache = temp.join("cache");
    let provider = HuggingFaceProvider::new(&cache).with_hf_executable(&hf);
    let token = CancellationToken::new();
    token.cancel();

    let err = provider
        .acquire(&pinned_request(), &mut NullProgressSink, &token)
        .expect_err("pre-cancelled token must abort");
    assert_eq!(err.category(), ErrorCategory::Cancelled);

    let key = cache_key(TEST_REPO, TEST_SHA);
    assert!(
        !cache.join("hf").join(&key).exists(),
        "no cache entry may be created"
    );
    assert!(
        !cache.join("hf").join(format!(".staging-{key}")).exists(),
        "hf must never have been spawned"
    );
}

#[test]
fn cancellation_during_download_leaves_staging_for_resume() {
    let temp = TempDir::new("cancel-during");
    let hf = write_fake_hf(&temp.path, FAKE_HF_SLOW);
    let cache = temp.join("cache");
    let provider = HuggingFaceProvider::new(&cache).with_hf_executable(&hf);
    let token = CancellationToken::new();
    let canceller = token.clone();
    std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(200));
        canceller.cancel();
    });

    let err = provider
        .acquire(&pinned_request(), &mut NullProgressSink, &token)
        .expect_err("cancelled download must abort");
    assert_eq!(err.category(), ErrorCategory::Cancelled);

    let key = cache_key(TEST_REPO, TEST_SHA);
    assert!(
        cache.join("hf").join(format!(".staging-{key}")).is_dir(),
        "staging must be preserved so a retry resumes"
    );
    assert!(
        !cache.join("hf").join(&key).exists(),
        "no complete entry may appear"
    );
}

#[test]
fn token_is_passed_via_env_never_argv() {
    let temp = TempDir::new("token-env");
    let hf = write_fake_hf(&temp.path, FAKE_HF_TOKEN_ENV_CHECK);
    let provider = HuggingFaceProvider::new(temp.join("cache"))
        .with_hf_executable(&hf)
        .with_credentials(HfCredentials::new("hf_test_token_abc123"));
    provider
        .acquire(
            &pinned_request(),
            &mut NullProgressSink,
            &CancellationToken::new(),
        )
        .expect("fake hf exits non-zero unless the token arrived via env only");
}

#[test]
fn token_is_redacted_from_errors() {
    let temp = TempDir::new("token-redact");
    let hf = write_fake_hf(&temp.path, FAKE_HF_AUTH_FAILURE);
    let token = "hf_secret_token_xyz789";
    let provider = HuggingFaceProvider::new(temp.join("cache"))
        .with_hf_executable(&hf)
        .with_credentials(HfCredentials::new(token));

    let err = provider
        .acquire(
            &pinned_request(),
            &mut NullProgressSink,
            &CancellationToken::new(),
        )
        .expect_err("auth failure must surface");
    assert_eq!(err.category(), ErrorCategory::Authentication);
    assert!(
        !err.message().contains(token),
        "token must be scrubbed from the error: {}",
        err.message()
    );
    assert!(err.message().contains("[REDACTED]"), "{}", err.message());
}

#[test]
fn missing_hf_executable_is_source_acquisition() {
    let temp = TempDir::new("no-hf");
    let provider =
        HuggingFaceProvider::new(temp.join("cache")).with_hf_executable("/nonexistent/hf-binary");
    let err = provider
        .acquire(
            &pinned_request(),
            &mut NullProgressSink,
            &CancellationToken::new(),
        )
        .expect_err("missing hf must fail");
    assert_eq!(err.category(), ErrorCategory::SourceAcquisition);
    assert!(err.message().contains("not found"), "{}", err.message());
}

#[test]
fn invalid_repository_is_rejected_before_any_io() {
    let temp = TempDir::new("invalid-repo");
    let hf = write_fake_hf(&temp.path, FAKE_HF_COPY);
    let provider = HuggingFaceProvider::new(temp.join("cache")).with_hf_executable(&hf);
    let err = provider
        .acquire(
            &SourceRequest::hf_repo("bad repo; rm -rf /", Revision::pinned(TEST_SHA)),
            &mut NullProgressSink,
            &CancellationToken::new(),
        )
        .expect_err("invalid repository must be rejected");
    assert_eq!(err.category(), ErrorCategory::InvalidArgument);
}

#[test]
fn resolve_records_resolved_sha_from_metadata() {
    let temp = TempDir::new("resolve-sha");
    let hf = write_fake_hf(
        &temp.path,
        &FAKE_HF_COPY_WITH_METADATA.replace("__SHA__", TEST_SHA_RESOLVED),
    );
    let cache = temp.join("cache");
    let provider = HuggingFaceProvider::new(&cache).with_hf_executable(&hf);

    let source = provider
        .acquire(
            &SourceRequest::hf_repo(TEST_REPO, Revision::resolve("main")),
            &mut NullProgressSink,
            &CancellationToken::new(),
        )
        .expect("resolve acquire must succeed");
    assert_eq!(source.resolved_revision.as_deref(), Some(TEST_SHA_RESOLVED));
    // The cache entry is keyed by the *resolved* sha, not the branch name.
    assert_eq!(
        source.source_dir,
        cache
            .join("hf")
            .join(cache_key(TEST_REPO, TEST_SHA_RESOLVED))
    );
    assert!(!source.from_cache);
}

#[test]
fn resolve_without_metadata_asks_for_a_pin() {
    let temp = TempDir::new("resolve-nometa");
    let hf = write_fake_hf(&temp.path, FAKE_HF_COPY); // writes no .cache metadata
    let provider = HuggingFaceProvider::new(temp.join("cache")).with_hf_executable(&hf);

    let err = provider
        .acquire(
            &SourceRequest::hf_repo(TEST_REPO, Revision::resolve("main")),
            &mut NullProgressSink,
            &CancellationToken::new(),
        )
        .expect_err("unresolvable revision must fail");
    assert_eq!(err.category(), ErrorCategory::SourceAcquisition);
    assert!(
        err.message().contains("Revision::Pinned"),
        "{}",
        err.message()
    );
}

#[test]
fn local_dir_request_is_invalid_argument_for_hf_provider() {
    let temp = TempDir::new("wrong-kind");
    let provider = HuggingFaceProvider::new(temp.join("cache"));
    let err = provider
        .acquire(
            &SourceRequest::local_dir(fixture_dir()),
            &mut NullProgressSink,
            &CancellationToken::new(),
        )
        .expect_err("wrong request kind must be rejected");
    assert_eq!(err.category(), ErrorCategory::InvalidArgument);
}

#[test]
fn stale_lock_is_broken_and_acquired() {
    let temp = TempDir::new("stale-lock");
    let hf = write_fake_hf(&temp.path, FAKE_HF_COPY);
    let cache = temp.join("cache");
    let key = cache_key(TEST_REPO, TEST_SHA);
    let hf_root = cache.join("hf");
    fs::create_dir_all(&hf_root).expect("create hf root");
    fs::write(
        hf_root.join(format!(".lock-{key}")),
        "pid=999999 created=1\n",
    )
    .expect("plant stale lock");

    // TTL of zero: any pre-existing lock is older than the TTL ⇒ stale.
    let provider = HuggingFaceProvider::new(&cache)
        .with_hf_executable(&hf)
        .with_lock_ttl(Duration::ZERO);
    let source = provider
        .acquire(
            &pinned_request(),
            &mut NullProgressSink,
            &CancellationToken::new(),
        )
        .expect("stale lock must be broken, acquisition must proceed");
    assert!(!source.from_cache);
    assert!(
        !hf_root.join(format!(".lock-{key}")).exists(),
        "lock must be released afterwards"
    );
}

#[test]
fn held_lock_times_out_with_cache_error() {
    let temp = TempDir::new("held-lock");
    let hf = write_fake_hf(&temp.path, FAKE_HF_COPY);
    let cache = temp.join("cache");
    let key = cache_key(TEST_REPO, TEST_SHA);
    let hf_root = cache.join("hf");
    fs::create_dir_all(&hf_root).expect("create hf root");
    fs::write(hf_root.join(format!(".lock-{key}")), "pid=1 created=now\n")
        .expect("plant fresh lock");

    let provider = HuggingFaceProvider::new(&cache)
        .with_hf_executable(&hf)
        .with_lock_ttl(Duration::from_secs(3600))
        .with_lock_timeout(Duration::from_millis(100));
    let err = provider
        .acquire(
            &pinned_request(),
            &mut NullProgressSink,
            &CancellationToken::new(),
        )
        .expect_err("a fresh lock must not be broken");
    assert_eq!(err.category(), ErrorCategory::Cache);
    assert!(err.message().contains("timed out"), "{}", err.message());
}

#[test]
fn progress_reports_resolve_download_verify_stages() {
    use std::sync::Mutex;
    let temp = TempDir::new("progress");
    let hf = write_fake_hf(&temp.path, FAKE_HF_COPY);
    let provider = HuggingFaceProvider::new(temp.join("cache")).with_hf_executable(&hf);
    let stages = Arc::new(Mutex::new(Vec::new()));
    let seen = Arc::clone(&stages);
    let mut sink = move |event: hologram_ai_core::ProgressEvent| {
        seen.lock().expect("lock").push(event.stage);
    };
    provider
        .acquire(&pinned_request(), &mut sink, &CancellationToken::new())
        .expect("acquire");
    let stages = stages.lock().expect("lock");
    assert!(stages.iter().any(|s| s == "resolve"), "{stages:?}");
    assert!(stages.iter().any(|s| s == "download"), "{stages:?}");
    assert!(stages.iter().any(|s| s == "verify"), "{stages:?}");
}
