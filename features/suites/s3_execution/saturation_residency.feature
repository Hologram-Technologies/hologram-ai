@row:saturation-residency @stage:S3 @status:build @executor:rust @lane:default
Feature: A failed verification unpins — corruption leaves the cache by the law that admitted it
  The κ-store's retention is derived from resolution state, not assigned by
  policy (resource model, Lifecycle): content enters the cache by verifying
  at a trust boundary, and the one mandatory eviction event is the inverse
  crossing — a verification failure. The failed entry evaporates and
  resolution re-resolves ONCE through the deeper tier (recorded provenance),
  re-verifying before anything executes; a wrong cache degrades to a stream,
  never a dead end. Only the failing entry is unpinned: bound content is
  never evicted by another entry's failure. Without a deeper tier the
  failure stays loud, naming the label — fail closed, recover by rebind.

  Background:
    Given the deterministic tiny decoder fixture with its weights in a κ-store

  Scenario: cache corruption recovers through the provenance tier
    When a stage materializes over a corrupted cache backed by a provenance tier
    Then materialization succeeds on content re-verified from the deeper tier
    And the corrupted cache entry has evaporated
    And every other cached entry is untouched

  Scenario: recovered content must still reproduce its label
    When the provenance tier itself serves corrupted content
    Then materialization is rejected naming the corrupted label
