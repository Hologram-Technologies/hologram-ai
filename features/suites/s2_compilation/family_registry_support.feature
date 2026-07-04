@row:family-registry-support @stage:S2 @status:verified @executor:rust @lane:default
Feature: Family-registry support, witnessed against pinned published models
  Every registered architecture family is verified against a pinned published
  model — the external authority. The witness streams the repository's real
  config.json and safetensors manifest at the pinned revision (metadata and
  headers only, via ranged requests — never weight bytes), compiles the
  parametric k-form through the real pipeline, and holds the emitted κ-map to
  cover every manifest tensor exactly once. Fused checkpoints (Phi3) bind the
  FUSED tensor names — exactly what the downloader persists under κ.

  Scenario Outline: a pinned authority compiles weightlessly for its family
    Given the streamed metadata of "<repo>" at revision "<revision>" from the Hub
    When the manifest is compiled without weights
    Then the selected family is "<family>"
    And the archive carries a kappa_map
    And the kappa_map names every manifest weight tensor exactly once
    And no weight bytes were fetched

    Examples:
      | family             | repo                                | revision                                 |
      | LlamaForCausalLM   | HuggingFaceTB/SmolLM2-135M-Instruct | 12fd25f77366fa6b3b4b768ec3050bf629380bac |
      | Qwen2ForCausalLM   | Qwen/Qwen2.5-0.5B-Instruct          | 7ae557604adf67be50417f59c2c2f167def9a775 |
      | MistralForCausalLM | mistralai/Mistral-7B-Instruct-v0.3  | c170c708c41dac9275d15a8fff4eca08d52bab71 |
      | Phi3ForCausalLM    | microsoft/phi-4                     | 932b33c0ec9ca189badeb22480721a8de9d0e006 |
