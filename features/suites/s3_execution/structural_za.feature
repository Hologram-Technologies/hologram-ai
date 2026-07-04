@row:structural-za @stage:S3 @status:verified @executor:rust @lane:default
Feature: Structural class ZA — bounded runtime heap on the decode hot path
  Addressed decode steps allocate zero unbounded runtime heap on the pinned
  substrate: after warm-up, each `execute_addressed` call holds to a tight,
  input-independent allocation budget, and repeated compiles of the same
  graph do not grow. The witness is the substrate-contract test
  `structural_za`, which installs a counting global allocator and therefore
  must run in its own isolated process — the runner asserts that process
  exits green.

  @serial
  Scenario: the ZA witness is green in an isolated process
    When the structural witness "structural_za" runs in an isolated release process
    Then the witness process exits green
