@row:structural-im @stage:S3 @status:build @executor:rust @lane:default
Feature: Structural class IM — parsing is confined to the import perimeter
  Byte-level format parsing is confined to the import perimeter; nothing
  mid-pipeline parses raw model bytes. The invariant is structural: past the
  importers, model content exists only as canonical graph forms and
  κ-addressed buffers. The witness is the perimeter test `structural_im`,
  run in its own isolated release process — the runner asserts that process
  exits green.

  @serial
  Scenario: the IM witness is green in an isolated process
    When the structural witness "structural_im" runs in an isolated release process
    Then the witness process exits green
