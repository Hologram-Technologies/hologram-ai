@row:hf-model-resolution @stage:S1 @status:verified @executor:rust @lane:default
Feature: HuggingFace model resolution
  Any HuggingFace repo id resolves to its file manifest via the live Hub API
  (`api/models/{id}`), classified into safetensors shards (weights) and
  companion assets (config.json, tokenizer.json, generation_config.json).
  The oracle is the live HuggingFace Hub itself. An unknown repository fails
  loud naming the repository — never a silent empty manifest.

  Scenario: a published repository resolves to a classified manifest
    Given the HuggingFace repository "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    When the file manifest is resolved via the Hub API
    Then the manifest classifies at least one safetensors shard
    And the manifest classifies "config.json" and "tokenizer.json" as companions
    And no file is classified as both shard and companion

  Scenario: an unknown repository fails loud naming the repository
    Given the HuggingFace repository "hologram-ai/this-repo-does-not-exist"
    When the file manifest resolution is attempted
    Then the resolution fails naming the repository
