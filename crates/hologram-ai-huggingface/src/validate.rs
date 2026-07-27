//! Conservative input validation, cache keys, and token redaction.

use hologram_ai_core::{AiError, AiResult, ErrorCategory};

use crate::types::Revision;

/// Validate a Hugging Face repository id of the form `owner/repo`.
///
/// Conservative rules: exactly two `/`-separated segments, each non-empty,
/// using only `[A-Za-z0-9._-]`, containing no `..`, and starting and ending
/// with an alphanumeric character. Anything else is
/// [`ErrorCategory::InvalidArgument`].
pub fn validate_repository_id(repository: &str) -> AiResult<()> {
    let invalid = |reason: &str| {
        AiError::new(
            ErrorCategory::InvalidArgument,
            format!("invalid repository id `{repository}`: {reason}"),
        )
    };
    let segments: Vec<&str> = repository.split('/').collect();
    if segments.len() != 2 {
        return Err(invalid("expected exactly the `owner/repo` form"));
    }
    for segment in segments {
        if segment.is_empty() {
            return Err(invalid("empty path segment"));
        }
        if segment.len() > 96 {
            return Err(invalid("path segment too long (max 96 chars)"));
        }
        if segment.contains("..") {
            return Err(invalid("`..` is not allowed"));
        }
        // '\0' is not alphanumeric, so an empty segment is rejected here too.
        let first = segment.chars().next().unwrap_or_default();
        let last = segment.chars().last().unwrap_or_default();
        if !first.is_ascii_alphanumeric() || !last.is_ascii_alphanumeric() {
            return Err(invalid(
                "segments must start and end with an alphanumeric character",
            ));
        }
        if !segment
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
        {
            return Err(invalid("only [A-Za-z0-9._-] characters are allowed"));
        }
    }
    Ok(())
}

/// Validate a [`Revision`].
///
/// `Pinned` must be a full 40-character lowercase hex commit sha. `Resolve`
/// accepts a conservative branch/tag charset (`[A-Za-z0-9._/-]`, no `..`, no
/// leading `/`, `-`, or `.`, no trailing `/`, max 200 chars).
pub fn validate_revision(revision: &Revision) -> AiResult<()> {
    match revision {
        Revision::Pinned(sha) => {
            if is_full_sha(sha) {
                Ok(())
            } else {
                Err(AiError::new(
                    ErrorCategory::InvalidArgument,
                    "pinned revision must be a full 40-character lowercase hex \
                     commit sha",
                ))
            }
        }
        Revision::Resolve(name) => {
            let invalid = |reason: &str| {
                AiError::new(
                    ErrorCategory::InvalidArgument,
                    format!("invalid revision name `{name}`: {reason}"),
                )
            };
            if name.is_empty() {
                return Err(invalid("empty revision name"));
            }
            if name.len() > 200 {
                return Err(invalid("revision name too long (max 200 chars)"));
            }
            if name.contains("..") || name.contains("//") {
                return Err(invalid("`..` and `//` are not allowed"));
            }
            if name.starts_with(['/', '-', '.']) || name.ends_with('/') {
                return Err(invalid(
                    "must not start with `/`, `-`, or `.`, or end with `/`",
                ));
            }
            if !name
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-' | '/'))
            {
                return Err(invalid("only [A-Za-z0-9._/-] characters are allowed"));
            }
            Ok(())
        }
    }
}

/// Content-addressed cache key for a repository at a resolved revision:
/// `hex(blake3(repository || 0x00 || resolved_revision))`.
pub fn cache_key(repository: &str, resolved_revision: &str) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(repository.as_bytes());
    hasher.update(&[0]);
    hasher.update(resolved_revision.as_bytes());
    hex_lower(hasher.finalize().as_bytes())
}

/// Replace every occurrence of `token` in `text` with `[REDACTED]`.
///
/// Used to scrub credentials out of error messages and progress details. An
/// empty token is a no-op (replacing it would mangle the text).
pub fn redact(token: &str, text: &str) -> String {
    if token.is_empty() {
        return text.to_string();
    }
    text.replace(token, "[REDACTED]")
}

/// Whether `s` is a full 40-character lowercase hex commit sha.
pub(crate) fn is_full_sha(s: &str) -> bool {
    s.len() == 40
        && s.bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
}

/// Lowercase hex encoding of `bytes`.
pub(crate) fn hex_lower(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0x0f) as usize] as char);
    }
    out
}
