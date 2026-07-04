@row:kappa-addressing @stage:S0 @status:verified @executor:rust @lane:default
Feature: κ-addressing reproduces the official BLAKE3 vectors
  A buffer's κ-label is `blake3:<hex>` of its bytes. The hex digest is anchored
  to the official BLAKE3 test vectors (oracles/blake3/test_vectors.json, the
  project's published KATs), and the pipeline's `kappa_of` must agree
  byte-for-byte with the substrate's `holospaces::address` and with the
  reference blake3 hasher on every KAT input.

  Scenario: the official BLAKE3 KATs reproduce through every κ surface
    Given the official BLAKE3 test vectors
    When every KAT input is κ-labeled by the pipeline, the substrate, and the reference hasher
    Then every κ-label equals `blake3:` followed by the KAT digest
    And the pipeline, the substrate, and the reference hasher agree on every label
