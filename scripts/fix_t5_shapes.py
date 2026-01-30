#!/usr/bin/env python3
"""Fix T5 ONNX models by running shape inference with concrete input shapes."""

import onnx
from onnx import shape_inference

def infer_shapes_with_concrete_inputs(model_path, output_path, input_shapes):
    """Run ONNX shape inference with concrete input shapes.

    Args:
        model_path: Path to input ONNX model
        output_path: Path to save shape-inferred model
        input_shapes: Dict mapping input names to shape lists
    """
    print(f"Loading {model_path}...")
    model = onnx.load(model_path)

    # Update model inputs with concrete shapes
    print("\nSetting concrete input shapes:")
    for input_tensor in model.graph.input:
        if input_tensor.name in input_shapes:
            shape = input_shapes[input_tensor.name]
            # Clear existing shape
            del input_tensor.type.tensor_type.shape.dim[:]
            # Add concrete dimensions
            for dim in shape:
                input_tensor.type.tensor_type.shape.dim.add().dim_value = dim
            print(f"  {input_tensor.name}: {shape}")

    # Run shape inference
    print("\nRunning shape inference...")
    inferred_model = shape_inference.infer_shapes(
        model,
        check_type=False,
        strict_mode=False,
        data_prop=True
    )

    # Run shape inference a second time to propagate shapes further
    print("Running shape inference (second pass)...")
    inferred_model = shape_inference.infer_shapes(
        inferred_model,
        check_type=False,
        strict_mode=False,
        data_prop=True
    )

    # Count shape quality
    def count_shapes(model):
        useful = sum(
            1 for vi in model.graph.value_info
            if len(vi.type.tensor_type.shape.dim) > 0
            and not any(d.HasField('dim_param') for d in vi.type.tensor_type.shape.dim)
        )
        empty = sum(
            1 for vi in model.graph.value_info
            if len(vi.type.tensor_type.shape.dim) == 0
        )
        symbolic = sum(
            1 for vi in model.graph.value_info
            if len(vi.type.tensor_type.shape.dim) > 0
            and any(d.HasField('dim_param') for d in vi.type.tensor_type.shape.dim)
        )
        return useful, empty, symbolic

    useful, empty, symbolic = count_shapes(inferred_model)
    total = len(inferred_model.graph.value_info)

    print(f"\nShape inference results:")
    print(f"  Total value_info entries: {total}")
    print(f"  Concrete shapes: {useful} ({useful/total*100:.1f}%)")
    print(f"  Empty shapes: {empty} ({empty/total*100:.1f}%)")
    print(f"  Symbolic shapes: {symbolic} ({symbolic/total*100:.1f}%)")

    # Save
    onnx.save(inferred_model, output_path)
    print(f"\nSaved to: {output_path}")

    return inferred_model

if __name__ == "__main__":
    # Fix encoder
    print("=" * 60)
    print("ENCODER")
    print("=" * 60)
    infer_shapes_with_concrete_inputs(
        model_path="/workspace/models/t5-small/encoder_model.onnx",
        output_path="/workspace/models/t5-small/encoder_model_fixed.onnx",
        input_shapes={
            'input_ids': [1, 512],
            'attention_mask': [1, 512]
        }
    )

    # Fix decoder
    print("\n" + "=" * 60)
    print("DECODER")
    print("=" * 60)
    infer_shapes_with_concrete_inputs(
        model_path="/workspace/models/t5-small/decoder_model.onnx",
        output_path="/workspace/models/t5-small/decoder_model_fixed.onnx",
        input_shapes={
            'input_ids': [1, 1],
            'encoder_hidden_states': [1, 512, 512],
            'encoder_attention_mask': [1, 512]
        }
    )

    print("\n" + "=" * 60)
    print("DONE! Models saved with '_fixed.onnx' suffix")
    print("=" * 60)
