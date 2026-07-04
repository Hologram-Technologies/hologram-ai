@row:chat-handshake @stage:S4 @status:build @executor:browser
Feature: The three-message chat handshake
  The application's contract journey: three user messages, three streamed
  assistant completions, over the materialized fixture model — deterministic
  (temperature 0, fixed seed) and matching the committed reference transcript.

  Background:
    Given the app is open in the browser against the hermetic model server
    And the fixture model is downloaded

  Scenario: three turns complete deterministically
    When the user sends handshake message 1
    Then assistant turn 1 streams a non-empty completion
    When the user sends handshake message 2
    Then assistant turn 2 streams a non-empty completion
    When the user sends handshake message 3
    Then assistant turn 3 streams a non-empty completion
    And the transcript matches the committed reference transcript
