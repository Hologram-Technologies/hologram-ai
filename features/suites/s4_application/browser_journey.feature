@row:browser-journey @stage:S4 @status:build @executor:browser
Feature: The hermetic browser journey
  Download → compile → materialize → run completes in real Chromium against the
  hermetic fixture, exercising the genuine workers, wasm pipeline, and OPFS
  κ-store — no mocks between the UI and the substrate.

  Background:
    Given the app is open in the browser against the hermetic model server

  Scenario: the fixture model reaches the runnable state
    When the fixture model is downloaded
    Then the model directory holds a k-form archive whose κ-map is fully resolvable from OPFS
    And the model is listed as ready to chat

  Scenario: the materialized model runs a forward pass
    When the fixture model is downloaded
    And a single-turn prompt is sent
    Then a non-empty completion streams back
