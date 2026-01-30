#!/usr/bin/env python3
"""Run ONNX shape inference on T5 models and save inferred versions."""

import onnx
from onnx import shape_inference
from pathlib import Path


def infer_and_save(input_path, output_path):
    """Run shape inference and save the result."""
    print(f"\nProcessing: {input_path.name}")
    print(f"  Loading model...")
    model = onnx.load(str(input_path))

    print(f"  Running shape inference...")
    try:
        inferred_model = shape_inference.infer_shapes(model)
        print(f"  Shape inference successful!")

        print(f"  Saving to {output_path.name}...")
        onnx.save(inferred_model, str(output_path))

        # Verify the inferred model
        graph = inferred_model.graph
        tensors_with_shapes = {vi.name for vi in graph.value_info}
        for inp in graph.input:
            tensors_with_shapes.add(inp.name)
        for out in graph.output:
            tensors_with_shapes.add(out.name)

        total_tensors = sum(1 for node in graph.node for output in node.output if output)
        has_shapes = sum(1 for node in graph.node for output in node.output
                        if output and output in tensors_with_shapes)

        ratio = (has_shapes / total_tensors * 100) if total_tensors > 0 else 0
        print(f"  Result: {has_shapes}/{total_tensors} tensors ({ratio:.1f}%) now have shapes")

        return True
    except Exception as e:
        print(f"  ERROR: Shape inference failed: {e}")
        return False


def main():
    """Main function."""
    models_dir = Path("/workspace/models/t5-small")

    encoder_in = models_dir / "encoder_model.onnx"
    encoder_out = models_dir / "encoder_model_inferred.onnx"

    decoder_in = models_dir / "decoder_model.onnx"
    decoder_out = models_dir / "decoder_model_inferred.onnx"

    print("="*60)
    print("Running ONNX Shape Inference on T5 Models")
    print("="*60)

    # Infer shapes for both models
    encoder_ok = infer_and_save(encoder_in, encoder_out)
    decoder_ok = infer_and_save(decoder_in, decoder_out)

    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"Encoder: {'✅ SUCCESS' if encoder_ok else '❌ FAILED'}")
    print(f"  Saved to: {encoder_out}")
    print(f"Decoder: {'✅ SUCCESS' if decoder_ok else '❌ FAILED'}")
    print(f"  Saved to: {decoder_out}")

    if encoder_ok and decoder_ok:
        print("\n✅ All models successfully processed with shape inference!")
        print("\nNext steps:")
        print("1. Compile the *_inferred.onnx models with hologram-ai")
        print("2. Check if workspace allocation errors are resolved")
        print("3. If successful, integrate shape inference into hologram-ai compilation")
    else:
        print("\n❌ Shape inference failed for some models")


if __name__ == "__main__":
    main()
