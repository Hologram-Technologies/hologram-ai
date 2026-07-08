//! Speculative decoding (dictionary row `speculative-decode`): the parametric,
//! zero-weight drafter.
//!
//! A decode step is the substrate matmul kernel's worst shape (`M = 1`); the
//! `decode_shape` bench measures an `M = K` pass at a fraction of the wall-clock
//! of `K` single steps. Speculative decoding exploits that: DRAFT `K`
//! continuation tokens cheaply, VERIFY them all in one `M = K` pass (the verify
//! head, [`crate::decode::DecodeSession::verify`]), and ACCEPT the longest
//! prefix the model would itself have produced. For greedy (temperature 0) the
//! result is byte-identical to single-step decode — a pure speedup whose size is
//! the draft's acceptance rate.
//!
//! The drafter here is **prompt-lookup** (a.k.a. LLM n-gram): the next tokens
//! are guessed by finding the most recent earlier occurrence of the current
//! suffix in the realized sequence and copying what followed it. No draft model,
//! no training, no weights — parametric over any model and any input. It shines
//! on structured/repetitive text (code, JSON, retrieval, format echoing) and
//! simply returns nothing on novel text, so the caller falls back to one plain
//! decode step — drafting is never worse than not drafting, because every
//! drafted token is verified and only the model's own tokens are accepted.

/// Draft up to `max_draft` continuation tokens for `seq` by prompt-lookup.
///
/// For the largest suffix length `g ∈ [1, ngram_max]` that occurs earlier in
/// `seq`, take the most recent such earlier occurrence and copy the up-to-
/// `max_draft` tokens that followed it. A longer matched suffix is a more
/// specific context, so it is tried first (higher acceptance). Returns an empty
/// draft when nothing matches — the signal to fall back to a single decode step.
///
/// This never affects correctness, only speed: every drafted token is verified
/// and a wrong one is rejected. It is a pure function of the realized sequence.
pub fn prompt_lookup_draft(seq: &[i64], ngram_max: usize, max_draft: usize) -> Vec<i64> {
    let n = seq.len();
    if n < 2 || max_draft == 0 || ngram_max == 0 {
        return Vec::new();
    }
    // Longest matching suffix first: a more specific context accepts better.
    let g_max = ngram_max.min(n - 1);
    for g in (1..=g_max).rev() {
        let needle = &seq[n - g..];
        // Most recent earlier occurrence first (skip the trailing suffix itself).
        for start in (0..n - g).rev() {
            if &seq[start..start + g] == needle {
                let from = start + g;
                let take = max_draft.min(n - from);
                if take > 0 {
                    return seq[from..from + take].to_vec();
                }
            }
        }
    }
    Vec::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_or_trivial_sequence_drafts_nothing() {
        assert!(prompt_lookup_draft(&[], 3, 4).is_empty());
        assert!(prompt_lookup_draft(&[7], 3, 4).is_empty());
        // A budget or ngram of zero disables drafting.
        assert!(prompt_lookup_draft(&[1, 2, 3], 0, 4).is_empty());
        assert!(prompt_lookup_draft(&[1, 2, 3], 3, 0).is_empty());
    }

    #[test]
    fn repeated_suffix_drafts_its_earlier_continuation() {
        // "a b c d ... a b" → the suffix `a b` last continued with `c d e`.
        let seq = [10, 20, 30, 40, 99, 10, 20];
        let draft = prompt_lookup_draft(&seq, 3, 3);
        assert_eq!(
            draft,
            vec![30, 40, 99],
            "drafts what followed the earlier `10,20`"
        );
    }

    #[test]
    fn longest_suffix_wins_over_a_shorter_ambiguous_one() {
        // The 2-gram `2 3` occurred once (→ 4); the 1-gram `3` also occurred
        // earlier (→ 4 as well). The longer, more specific match is used.
        let seq = [1, 2, 3, 4, 5, 2, 3];
        let draft = prompt_lookup_draft(&seq, 3, 2);
        assert_eq!(draft, vec![4, 5], "the 2-gram `2,3` context drafts `4,5`");
    }

    #[test]
    fn most_recent_occurrence_is_preferred() {
        // `7` appears after position 0 (→ 1) and position 3 (→ 8). The most
        // recent earlier occurrence (index 3) drafts its follower.
        let seq = [7, 1, 2, 7, 8, 9, 7];
        let draft = prompt_lookup_draft(&seq, 1, 1);
        assert_eq!(draft, vec![8], "the freshest match for `7` drafts `8`");
    }

    #[test]
    fn max_draft_caps_the_length() {
        let seq = [5, 6, 7, 8, 9, 5, 6];
        assert_eq!(prompt_lookup_draft(&seq, 3, 2), vec![7, 8]);
        assert_eq!(prompt_lookup_draft(&seq, 3, 10), vec![7, 8, 9, 5, 6]);
    }

    #[test]
    fn novel_suffix_drafts_nothing() {
        // The trailing `42` never occurred earlier → no draft, fall back to a step.
        let seq = [1, 2, 3, 4, 42];
        assert!(prompt_lookup_draft(&seq, 3, 4).is_empty());
    }
}
