@row:warm-turn @stage:S4 @status:build @executor:browser
Feature: Later turns run warm — the session survives across sends
  The cross-turn residency the Rust suite witnesses natively must reach the
  user's journey: the generation worker outlives a send, and the staged
  session — compiled window, resident stage sessions (measured admission),
  session verified-κ set, derived-artifact cache — carries to the next turn.
  A warm turn pays decode: no window recompile, no stage rematerialization,
  no re-verification. A cold turn (first send, model switch, after cancel)
  has identical semantics — warmth is a projection, never a meaning.

  Background:
    Given the app is open in the browser against the hermetic model server
    And a forced single-layer execution window
    When the fixture model is downloaded

  Scenario: the second turn reuses the warm session
    When the user sends handshake message 1
    And the user sends handshake message 2
    Then the second turn reports a warm session
    And the second turn materializes no stages
    And assistant turn 2 streams a non-empty completion
