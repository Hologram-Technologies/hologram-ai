//! Validation matrix, cache-key determinism, argv construction, redaction.

mod common;

use std::ffi::OsString;
use std::path::Path;

use hologram_ai_core::ErrorCategory;
use hologram_ai_huggingface::{
    build_download_args, cache_key, redact, validate_repository_id, validate_revision, Revision,
};

#[test]
fn repository_id_validation_matrix() {
    let valid = [
        "owner/repo",
        "a/b",
        "meta-llama/Llama-3.2-1B-Instruct",
        "a-b_c.d/e-f_g.h",
        "0x/9z",
    ];
    for repo in valid {
        assert!(
            validate_repository_id(repo).is_ok(),
            "expected `{repo}` to be valid"
        );
    }
    let invalid = [
        "",
        "owner",
        "owner/",
        "/repo",
        "a/b/c",
        "owner//repo",
        "own er/repo",
        "owner/../x",
        "a..b/repo",
        "-owner/repo",
        ".owner/repo",
        "owner/repo-",
        "owner/repo.",
        "owner/repo;rm -rf /",
        "owner/$(whoami)",
        "owner/re`po`",
        "owner/repo|cat",
        "ownér/repo",
    ];
    for repo in invalid {
        let err =
            validate_repository_id(repo).expect_err(&format!("expected `{repo}` to be rejected"));
        assert_eq!(err.category(), ErrorCategory::InvalidArgument, "`{repo}`");
    }
}

#[test]
fn revision_validation_matrix() {
    assert!(validate_revision(&Revision::pinned(common::TEST_SHA)).is_ok());
    assert!(validate_revision(&Revision::resolve("main")).is_ok());
    assert!(validate_revision(&Revision::resolve("v1.0.0")).is_ok());
    assert!(validate_revision(&Revision::resolve("feature/branch-x")).is_ok());

    let bad_pinned = [
        "",
        "abc",
        "0123456789abcdef0123456789abcdef0123456", // 39 chars
        "0123456789abcdef0123456789abcdef012345678", // 42 chars
        "0123456789ABCDEF0123456789abcdef01234567", // uppercase
        "g123456789abcdef0123456789abcdef01234567", // non-hex
    ];
    for sha in bad_pinned {
        let err = validate_revision(&Revision::pinned(sha))
            .expect_err(&format!("expected pinned `{sha}` to be rejected"));
        assert_eq!(err.category(), ErrorCategory::InvalidArgument, "`{sha}`");
    }

    let bad_resolve = [
        "", "../x", "a..b", "a//b", "-bad", ".bad", "/bad", "bad/", "a b", "a;b", "a$(b)",
    ];
    for name in bad_resolve {
        let err = validate_revision(&Revision::resolve(name))
            .expect_err(&format!("expected resolve `{name}` to be rejected"));
        assert_eq!(err.category(), ErrorCategory::InvalidArgument, "`{name}`");
    }
}

#[test]
fn cache_key_is_deterministic_and_sensitive() {
    let key_a = cache_key(common::TEST_REPO, common::TEST_SHA);
    let key_b = cache_key(common::TEST_REPO, common::TEST_SHA);
    assert_eq!(key_a, key_b, "same inputs must give the same key");
    assert_eq!(key_a.len(), 64, "key is lowercase hex of a 32-byte digest");
    assert!(key_a.chars().all(|c| c.is_ascii_hexdigit()));
    assert_ne!(
        key_a,
        cache_key("other/repo", common::TEST_SHA),
        "repository must feed the key"
    );
    assert_ne!(
        key_a,
        cache_key(common::TEST_REPO, common::TEST_SHA_RESOLVED),
        "revision must feed the key"
    );
}

#[test]
fn download_args_are_exact_and_shell_free() {
    let argv = build_download_args("owner/repo", common::TEST_SHA, Path::new("/some/dest dir"));
    let expected: Vec<OsString> = vec![
        OsString::from("download"),
        OsString::from("owner/repo"),
        OsString::from("--revision"),
        OsString::from(common::TEST_SHA),
        OsString::from("--local-dir"),
        OsString::from("/some/dest dir"),
        OsString::from("--quiet"),
    ];
    assert_eq!(argv, expected);
}

#[test]
fn download_args_keep_shell_metacharacters_inert() {
    // A hostile value stays a single opaque argv element; no shell ever sees
    // it, so nothing can be interpolated.
    let hostile = "owner/repo; rm -rf /";
    let argv = build_download_args(hostile, "main", Path::new("/dest"));
    assert_eq!(argv[1], OsString::from(hostile));
    assert_eq!(argv.len(), 7);
}

#[test]
fn redact_removes_all_token_occurrences() {
    assert_eq!(
        redact("secret123", "failed: token secret123 leaked secret123"),
        "failed: token [REDACTED] leaked [REDACTED]"
    );
    // Substring occurrences inside larger words are scrubbed too (safe side).
    assert_eq!(redact("tok", "no token here"), "no [REDACTED]en here");
    assert_eq!(
        redact("", "unchanged"),
        "unchanged",
        "empty token is a no-op"
    );
    assert_eq!(redact("abc", ""), "");
}
