@row:live-model-journey @stage:S4 @status:verified @executor:browser @live
Feature: The live model journey
  The full journey — download, compile, materialize, run, three-message
  handshake — completes in real Chromium against the pinned
  HuggingFaceTB/SmolLM2-135M-Instruct from the live HuggingFace Hub.
  Runs in the scheduled architecture matrix (network + model weights).

  Scenario: SmolLM2-135M-Instruct completes the journey
    Given the app is open in the browser against the live HuggingFace Hub
    When the pinned SmolLM2 model is downloaded
    Then the model reaches the runnable state
    When the user completes the three-message handshake
    Then every assistant turn streams a non-empty completion respecting stop conditions
