@row:structural-ce @stage:S3 @status:verified @executor:rust @lane:default
Feature: Structural class CE — content-addressed compute elision
  Unchanged compute cones are elided by κ-residency on the pinned substrate:
  a node whose output κ-label is already resident is skipped, not
  recomputed. The witness is the substrate-contract test `structural_ce`,
  run in its own isolated release process — the runner asserts that process
  exits green.

  @serial
  Scenario: the CE witness is green in an isolated process
    When the structural witness "structural_ce" runs in an isolated release process
    Then the witness process exits green
