@row:chunked-kappa-persisting @stage:S0 @status:verified @executor:rust @lane:default
Feature: Chunked κ-hashing equals the one-shot κ
  Incremental (chunked) κ-hashing of a stream equals the one-shot κ of the
  whole content, for every chunking — the property that lets the download
  worker persist a tensor without ever holding a whole shard in memory.
  Anchored to the official BLAKE3 test vectors: every KAT input, fed through
  the incremental hasher in chunks, must reproduce both the one-shot κ and
  the KAT digest itself.

  Scenario Outline: incremental hashing is chunking-invariant on the BLAKE3 KATs
    Given the official BLAKE3 test vectors
    When every KAT input is κ-hashed incrementally in chunks of <chunk> bytes
    Then every incremental digest equals the one-shot κ of the whole input

    Examples:
      | chunk |
      | 1     |
      | 7     |
      | 65536 |
      | whole |
