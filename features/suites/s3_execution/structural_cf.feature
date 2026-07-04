@row:structural-cf @stage:S3 @status:verified @executor:rust @lane:default
Feature: Structural class CF — only canonical forms cross the boundary
  Only canonical forms cross the hologram boundary: the closed OpKind
  catalog, interned shapes, canonical dtypes. The witness is the
  substrate-contract test `structural_cf`, run in its own isolated release
  process — the runner asserts that process exits green.

  @serial
  Scenario: the CF witness is green in an isolated process
    When the structural witness "structural_cf" runs in an isolated release process
    Then the witness process exits green
