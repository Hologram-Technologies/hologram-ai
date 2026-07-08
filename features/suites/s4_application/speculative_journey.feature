@row:speculative-journey @stage:S4 @status:build @executor:browser
Feature: Speculative decode runs the deployed staged journey
  Speculative decode (row `speculative-decode`) is a batched shortcut over the
  same greedy path: it drafts the next tokens from the realized sequence's own
  recurrence and verifies them in one M=K pass through a paged/staged verify
  pipeline whose head is parametric over any vocabulary. It only engages on the
  STAGED decode session (the deployed browser path), so this witnesses it there:
  the chat journey streams live, warm across turns, with the verify pipeline
  installed. Byte-for-byte equality with plain decode is witnessed natively (row
  `speculative-decode`, `generate_stream_speculative`); the forced-staging config
  used to reach the staged session trims later turns by the session window (its
  context shrinks), so the browser witness is liveness and warmth, not the
  monolithic reference transcript.

  Background:
    Given the app is open in the browser against the hermetic model server
    And a forced single-layer execution window
    And speculative decode is enabled
    When the fixture model is downloaded

  Scenario: speculative decode streams a warm staged journey
    When the user sends handshake message 1
    And the user sends handshake message 2
    Then the second turn reports a warm session
    And assistant turn 2 streams a non-empty completion
