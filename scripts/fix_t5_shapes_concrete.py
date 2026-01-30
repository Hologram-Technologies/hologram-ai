#!/usr/bin/env python3
"""Fix T5 ONNX models by running shape inference with CONCRETE batch=1."""

import onnx
from onnx import shape_inference
import onnx.helper as helper

def fix_shapes_concrete(model_path, output_path, input_shapes):
    """Run ONNX shape inference with concrete batch=1 shapes."""
    print(f"Loading {model_path}...")
    model = onnx.load(model_path)

    # Set concrete input shapes (batch=1)
    print("\nSetting CONCRETE input shapes (batch=1):")
    for input_tensor in model.graph.input:
        if input_tensor.name in input_shapes:
            shape = input_shapes[input_tensor.name]
            del input_tensor.type.tensor_type.shape.dim[:]
            for dim in shape:
                input_tensor.type.tensor_type.shape.dim.add().dim_value = dim
            print(f"  {input_tensor.name}: {shape}")

    # Run shape inference multiple times
    print("\nRunning shape inference (pass 1/5)...")
    inferred = shape_inference.infer_shapes(model, check_type=False, strict_mode=False, data_prop=True)

    for i in range(2, 6):
        print(f"Running shape inference (pass {i}/5)...")
        inferred = shape_inference.infer_shapes(inferred, check_type=False, strict_mode=False, data_prop=True)

    # Count shapes
    concrete = sum(1 for vi in inferred.graph.value_info
                   if len(vi.type.tensor_type.shape.dim) > 0
                   and all(d.HasField('dim_value') for d in vi.type.tensor_type.shape.dim))
    empty = sum(1 for vi in inferred.graph.value_info
                if len(vi.type.tensor_type.shape.dim) == 0)
    total = len(inferred.graph.value_info)

    print(f"\nShape inference results:")
    print(f"  Total value_info: {total}")
    print(f"  Concrete shapes: {concrete} ({concrete/total*100:.1f}%)")
    print(f"  Empty shapes: {empty} ({empty/total*100:.1f}%)")

    onnx.save(inferred, output_path)
    print(f"\nSaved to: {output_path}")

if __name__ == "__main__":
    # Encoder with batch=1
    print("=" * 60)
    print("ENCODER (batch=1)")
    print("=" * 60)
    fix_shapes_concrete(
        "/workspace/models/t5-small/encoder_model.onnx",
        "/workspace/models/t5-small/encoder_batch1.onnx",
        {'input_ids': [1, 512], 'attention_mask': [1, 512]}
    )

    # Decoder with batch=1
    print("\n" + "=" * 60)
    print("DECODER (batch=1)")
    print("=" * 60)
    fix_shapes_concrete(
        "/workspace/models/t5-small/decoder_model.onnx",
        "/workspace/models/t5-small/decoder_batch1.onnx",
        {
            'input_ids': [1, 1],
            'encoder_hidden_states': [1, 512, 512],
            'encoder_attention_mask': [1, 512]
        }
    )

    print("\n" + "=" * 60)
    print("DONE! Models saved with '_batch1.onnx' suffix")
    print("=" * 60)
