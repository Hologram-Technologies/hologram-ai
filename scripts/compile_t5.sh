#!/usr/bin/env bash

# Compile models
cargo run -- compile ./models/t5-small/encoder_model.onnx \
            --output ./models/t5-small/encoder.holo

cargo run -- compile ./models/t5-small/decoder_model.onnx \
            --output ./models/t5-small/decoder.holo

# Run pipeline
# cargo run -- run-pipeline t5-pipeline.holo \
#             --prompt "Tell me a joke" --max-tokens 50
