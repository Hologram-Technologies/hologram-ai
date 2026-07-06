@row:chunked-head @stage:S3 @status:build @executor:rust @lane:default
Feature: No head is too large to execute
  The whole-vocabulary head matmul is a residency assumption, not a law —
  the same assumption layer staging already removed for the model body. The
  head partitions into vocab-row chunks no heavier than a layer stage, each
  chunk binding a BYTE RANGE of the head weight's κ (sub-tensor
  κ-resolution: the κ names, and first-touch verification covers, the whole
  content; the constant holds one verified slice). Verification is the only
  whole-content read: once a session has verified a κ, a ranged binding
  rematerializes through KappaStore::resolve_range — read-only I/O of the
  slice, never the tensor. Row-partitioned matmul
  concatenation is mathematically the whole matmul; the substrate's
  reduction tiling varies with output width, so agreement is witnessed at
  kernel reduction-order tolerance (measured ≤ 4e-7) with EXACT greedy-
  decode parity — and no whole-vocabulary image ever materializes. A head
  within layer granularity is one chunk: the classic head stage, unchanged.

  Background:
    Given a wide-vocabulary decoder fixture with its weights in a κ-store

  Scenario: the head chunks at the pipeline's own stage granularity
    When the wide-vocabulary fixture is compiled as stages
    Then the head partitions into multiple chunk stages bound by κ-ranges
    And every chunk stage stays within the layer-stage granularity

  Scenario: chunked logits agree with the whole head
    When the same token window runs through the chunked stages and the monolithic archive
    Then the chunked logits match the monolithic logits within reduction-order tolerance
    And the greedy choice at every position is identical

  Scenario: chunked generation equals the monolithic completion
    When a greedy completion is generated through the chunked staged session
    Then the chunked completion equals the monolithic completion and every chunk resolved through its κ-range

  Scenario: a verified κ rematerializes moving only its bytes
    When the chunked stages execute twice in one session
    Then every ranged touch of the verified head κ moves only its slice and whole transits stay at one per pass
