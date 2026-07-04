@row:tokenizer-parity @stage:S3 @status:verified @executor:rust @lane:default
Feature: Tokenizer parity with the HuggingFace reference
  Encode matches the reference HuggingFace `tokenizers` crate — the canonical
  implementation of tokenizer.json — on the model's own published tokenizer
  at the pinned SmolLM2 revision, and decode round-trips the encoded corpus.
  The tokenizer is the model's published artifact: if it is not on disk it is
  fetched from the pinned revision recorded in the oracle registry.

  Background:
    Given the pinned model's published tokenizer.json

  Scenario: encode matches the reference on a representative corpus
    When the representative corpus is encoded by our tokenizer and the reference
    Then every corpus entry encodes to the reference token ids

  Scenario: decode round-trips the corpus
    When the round-trippable corpus is encoded and decoded by our tokenizer
    Then every entry round-trips to its input text
