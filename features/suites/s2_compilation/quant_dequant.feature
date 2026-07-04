@row:quant-dequant @stage:S2 @status:verified @executor:rust @lane:default
Feature: Quantized block dequantization matches the GGML reference
  Q4_0 and Q8_0 dequantization reproduces the GGML reference golden vectors —
  committed, sha256-tracked oracle artifacts under oracles/quant/ — element
  for element. The vectors carry the reference block layout; our
  `hologram-ai-quant` kernels are held to them directly, independent of any
  dispatch path.

  Scenario Outline: <scheme> dequantization reproduces the golden vectors
    Given the committed golden vectors "<file>"
    When every block is dequantized with the <scheme> kernel
    Then every element matches the reference within <tolerance>

    Examples:
      | scheme | file             | tolerance |
      | Q4_0   | q4_0_golden.json | 0.0001    |
      | Q8_0   | q8_0_golden.json | 0.001     |
