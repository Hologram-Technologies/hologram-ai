@row:content-verified-resolution @stage:S0 @status:build @executor:rust @lane:default
Feature: Content-verified κ resolution
  Resolving κ → bytes re-hashes the bytes: content addressing is the
  integrity check, so a κ-store can never silently hand back the wrong
  content. Content that does not reproduce its κ is rejected with the label;
  a κ the store does not hold is rejected with the label. Validated as a
  construction over the verified κ-addressing (blake3) invariant.

  Scenario: resolved content reproduces its κ
    Given an empty κ-store directory
    When bytes are persisted under their derived κ
    Then resolving that κ returns bytes that re-hash to the same κ

  Scenario: a missing κ is rejected naming the label
    Given an empty κ-store directory
    When a κ that was never persisted is resolved
    Then the resolution fails naming that κ

  Scenario: content that does not reproduce its κ is rejected with the label
    Given a κ-store holding corrupt bytes under a known κ
    When a k-form archive requiring that κ is materialized against the store
    Then materialization fails the integrity check naming the expected κ
