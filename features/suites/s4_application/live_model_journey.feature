@row:live-model-journey @stage:S4 @status:verified @executor:browser @live
Feature: The live model journey
  The full journey — download, compile, materialize, run, three-message
  handshake — completes in real Chromium against the pinned
  HuggingFaceTB/SmolLM2-135M-Instruct from the live HuggingFace Hub.

  Thorough real-model verification, run via `pnpm bdd:live` (network + real
  model weights; slow — single-threaded wasm decodes at a few tok/s). The Pages
  DEPLOY is gated by the leaner single-turn real-model probe
  (`bdd/probe-deployed-live.mjs`, wired into `.github/workflows/pages.yml`),
  which downloads this same model through the deploy bundle and asserts a
  coherent, error-free completion before publish. This three-turn journey is
  the deeper local/manual check.

  Scenario: SmolLM2-135M-Instruct completes the journey
    Given the app is open in the browser against the live HuggingFace Hub
    When the pinned SmolLM2 model is downloaded
    Then the model reaches the runnable state
    When the user completes the three-message handshake
    Then every assistant turn streams a non-empty completion respecting stop conditions
