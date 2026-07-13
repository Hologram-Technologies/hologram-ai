@row:model-switch @stage:S4 @status:build @executor:browser
Feature: Switching models mid-session keeps chat streaming
  Two downloaded models are two directories, two archives, two sessions; the
  chat page switches between them without recompiling the world or wedging the
  stream (the route-independent stream store and the residency hand-off are
  what this pins). Switching BACK re-selects the first archive — its download
  is never repeated.

  Background:
    Given the app is open in the browser against the hermetic model server

  Scenario: chat streams across switches between two models
    Given the fixture model is downloaded
    And the second fixture model is downloaded
    When the user sends handshake message 1
    Then assistant turn 1 streams a non-empty completion
    When the user switches the chat to the second fixture model
    And the user sends a chat message on the switched model
    Then assistant turn 2 streams a non-empty completion
    When the user switches the chat back to the first fixture model
    And the user sends a chat message on the switched model
    Then assistant turn 3 streams a non-empty completion
    And no download requests were repeated for the first model
