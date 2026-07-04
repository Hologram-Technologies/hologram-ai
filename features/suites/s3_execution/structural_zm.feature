@row:structural-zm @stage:S3 @status:verified @executor:rust @lane:default
Feature: Structural class ZM — zero buffer movement post-load
  Execution moves no buffer bytes after load on the pinned substrate: values
  flow by κ-label, never by copy. The witness is the substrate-contract test
  `structural_zm`, run in its own isolated release process so its
  instrumentation cannot interleave with other tests — the runner asserts
  that process exits green.

  @serial
  Scenario: the ZM witness is green in an isolated process
    When the structural witness "structural_zm" runs in an isolated release process
    Then the witness process exits green
