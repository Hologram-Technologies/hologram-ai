@row:network-skip @stage:S1 @status:build @executor:browser
Feature: Known content never re-transits
  The transit floor is the set-difference from the known, where known means
  provenance-recorded κ — not cached bytes. A shard's HTTP ETag is its
  content pin (on the Hub, the blob's own hash): under an identical pin,
  every (range → κ) an earlier stream recorded is still true, so a repeat
  download moves ZERO shard bytes — the κs enter the manifest directly and
  only the unknown runs would transit, coalesced. A changed pin invalidates
  the prior wholesale: the shard streams fully and the prior re-records.
  No skipped byte is trusted: the prior only asserts labels; content
  verifies at first materialization, and a wrong prior unpins and recovers
  through provenance. The prior accelerates; the posterior governs.

  Background:
    Given the app is open in the browser against the hermetic model server

  Scenario: a repeat download under the same pin moves no shard bytes
    When the fixture model is downloaded
    And the model directory is removed but the transit prior survives
    And the fixture model is downloaded again
    Then the repeat download transferred no shard body bytes
    And the model directory records κ provenance for every manifest tensor
    When the user sends handshake message 1
    Then the completion matches reference turn 1

  Scenario: a changed content pin discards the prior
    When the fixture model is downloaded
    And the model directory is removed but the transit prior survives
    And the shard content pin changes
    And the fixture model is downloaded again
    Then the repeat download streamed the shard body
