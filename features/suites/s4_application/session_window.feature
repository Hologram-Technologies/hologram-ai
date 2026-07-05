@row:session-window @stage:S4 @status:build @executor:browser
Feature: The session window — the model's own context is the only limit
  The archive compiles at the model's own max_position_embeddings (staging
  absorbs the memory scaling). A transcript that outgrows the window trims
  oldest-turn-first — whole user/assistant pairs, counted with the model's
  own tokenizer over the fully templated prompt — so the conversation
  continues; it never dead-ends on an arbitrary cap.

  Background:
    Given the app is open in the browser against the hermetic model server
    And the fixture model is downloaded

  Scenario: a transcript outgrowing the context trims oldest-first and continues
    When the user sends handshake message 1
    And the user sends handshake message 2
    And the user sends handshake message 3
    And the user sends a message that overflows the context window
    Then the overflow turn completes without error
    And the overflow prompt omits the oldest turn
