@row:derived-artifact-kappa @stage:S3 @status:build @executor:rust @lane:default
Feature: The known set closes over derivation
  Any artifact computed deterministically from κ inputs has a derived κ and
  is itself content (resource model, Closure). A window's stage archives are
  a deterministic function of (config, κ-manifest, window, partition) —
  `deterministic-compile` witnesses bit-identity — so they persist in the
  derived store under that derivation key and later sessions RESOLVE them
  instead of re-deriving: the warm session pays a verified read, not a
  compile. Soundness is inherited, nothing new: content verifies against its
  recorded κ at load, a corrupted entry evaporates, and the recovery is
  derivation itself — a wrong prior degrades to a compile, never a dead end.

  Background:
    Given the deterministic tiny decoder fixture with its weights in a κ-store

  Scenario: a warm session resolves the derivation instead of re-deriving
    When two sessions with identical inputs generate over a shared derived store
    Then the second session resolves its window from the derived store
    And both completions are identical and non-empty

  Scenario: a corrupted derived entry evaporates and derivation recovers
    When a session generates over a derived store with a corrupted entry
    Then the window is re-derived instead of resolved
    And the completion is unaffected
    And the derived store holds the fresh derivation
