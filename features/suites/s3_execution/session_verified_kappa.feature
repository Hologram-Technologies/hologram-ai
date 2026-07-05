@row:session-verified-kappa @stage:S3 @status:build @executor:rust @lane:default
Feature: A κ verifies once per session, never per traversal
  Verification belongs at trust-boundary crossings: content crosses into a
  session ONCE — that is where canonicalize-then-hash runs. Staged execution
  rematerializes stages every window pass; re-hashing a session-verified κ on
  every traversal would put a full model of hashing on the per-token path,
  buying nothing (the session already established the content). With the
  session verified-κ set, rematerialization is read-only I/O. The set is
  session-scoped by construction: a fresh session re-verifies at first touch,
  so a store corrupted between sessions is rejected loudly, naming the label.
  The prior accelerates; the posterior governs.

  Background:
    Given the deterministic tiny decoder fixture with its weights in a κ-store

  Scenario: rematerialization within a session is read-only resolution
    When a stage materializes twice in one session over a store corrupted between the passes
    Then the second pass succeeds without re-hashing the session-verified content

  Scenario: a fresh session verifies at first touch and fails loud
    When a fresh session materializes a stage over a store corrupted after another session verified it
    Then materialization is rejected naming the corrupted label
