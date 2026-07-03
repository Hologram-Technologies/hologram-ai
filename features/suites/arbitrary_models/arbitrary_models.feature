Feature: Arbitrary Models
  In order to ensure that the hologram-ai compiler supports arbitrary models
  As a developer
  I want to run conformance tests against external authoritative ONNX fixtures

  Scenario: Model manifest instantiation
    Given an arbitrary model name "test_model"
    When the model manifest is instantiated with a holospaces::Kappa for "test_kappa"
    Then the model manifest preserves the holospaces::Kappa

  Scenario: Executing an ONNX fixture
    Given the external authoritative ONNX fixture "mlp"
    When the fixture is compiled and executed via the holographic compiler
    Then the outputs must exactly match the ONNX Runtime authoritative execution
