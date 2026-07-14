@row:deep-model-journey @stage:S4 @status:build @executor:browser
Feature: A real-model-shape checkpoint runs the whole browser journey
  The committed handshake fixture is 2 layers — too shallow to exercise the
  real-model path: many stages, the int8 quantized tier, bucket growth, and
  the multi-stage decode window. A fixture at production head_dim (128), SYNTHESIZED at
  serve time from the same deterministic weight law (zero repo bytes), drives
  that shape hermetically — the class of failure a shallow fixture cannot
  catch (a decode session that hangs or crashes only past N stages / at the
  int8 tier / on the first real generation step).

  Background:
    Given the app is open in the browser against the hermetic model server
    And the deep hermetic fixture model is available

  Scenario: the deep fixture downloads, compiles int8, and completes a turn honestly
    When the deep fixture model is downloaded
    And the user sends a chat message on the deep fixture model
    Then the real-shape turn completes and its assistant reply is committed honestly

  # The DEPLOYED shape the browser never ran: at production head_dim 128 the model
  # STAGES (partitions), and under a weight budget each stage PAGES from the κ-store
  # and EVICTS between decode steps, so the resident K/V carry crosses a dropped-and-
  # rematerialized stage — the exact seam the deployed Qwen2.5-1.5B trapped on. The
  # shallow fixtures exercise staging+eviction only at head_dim 16; this closes the
  # gap at head_dim 128, proving the SHIPPED (legacy) decode survives it end to end.
  Scenario: at production head_dim the deep fixture stages, evicts, and still decodes honestly
    Given the deep fixture is staged under a small weight-paging budget
    When the deep fixture model is downloaded
    And the user sends a chat message on the deep fixture model
    Then the deep fixture compiled to multiple stages and paged its weights under the budget
    And the real-shape turn completes and its assistant reply is committed honestly
