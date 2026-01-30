#!/usr/bin/env python3
"""Analyze ONNX models to compare shape information between encoder and decoder.

This script helps diagnose why T5 encoder compiles successfully but decoder fails.
It counts how many intermediate tensors have shape information vs missing shapes.
"""

import onnx
from pathlib import Path


def analyze_shapes(model_path, model_name):
    """Analyze shape information in an ONNX model."""
    print(f"\n{'='*60}")
    print(f"Analyzing {model_name}")
    print(f"{'='*60}")

    model = onnx.load(str(model_path))
    graph = model.graph

    # Build a set of all tensors that have shape info in value_info
    tensors_with_shapes = {vi.name for vi in graph.value_info}

    # Also check inputs and outputs (they always have shapes)
    for inp in graph.input:
        tensors_with_shapes.add(inp.name)
    for out in graph.output:
        tensors_with_shapes.add(out.name)

    # Count all intermediate tensors
    has_shapes = 0
    missing_shapes = []

    for node in graph.node:
        for output in node.output:
            if output:  # Skip empty string outputs
                if output in tensors_with_shapes:
                    has_shapes += 1
                else:
                    missing_shapes.append((node.name, node.op_type, output))

    total = has_shapes + len(missing_shapes)
    ratio = (has_shapes / total * 100) if total > 0 else 0

    print(f"Model: {model_path.name}")
    print(f"  Total nodes: {len(graph.node)}")
    print(f"  Total tensors: {total}")
    print(f"  Tensors with shapes: {has_shapes}")
    print(f"  Tensors missing shapes: {len(missing_shapes)}")
    print(f"  Ratio: {ratio:.1f}% have shapes")

    # Show sample of operations missing shapes
    if missing_shapes:
        print(f"\nSample operations missing shape info (first 20):")
        op_type_counts = {}
        for i, (node_name, op_type, tensor) in enumerate(missing_shapes[:20]):
            print(f"  {node_name} ({op_type}) -> {tensor}")
            op_type_counts[op_type] = op_type_counts.get(op_type, 0) + 1

        print(f"\nOperation types missing shapes:")
        for op_type, count in sorted(op_type_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {op_type}: {count}")

    # Operation distribution
    print(f"\nOperation type distribution:")
    op_counts = {}
    for node in graph.node:
        op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1

    for op_type, count in sorted(op_counts.items(), key=lambda x: x[1], reverse=True)[:15]:
        print(f"  {op_type}: {count}")

    return {
        "total_nodes": len(graph.node),
        "total_tensors": total,
        "has_shapes": has_shapes,
        "missing_shapes": len(missing_shapes),
        "ratio": ratio,
        "op_counts": op_counts,
        "missing_by_op": op_type_counts if missing_shapes else {}
    }


def main():
    """Main analysis function."""
    encoder_path = Path("/workspace/models/t5-small/encoder_model.onnx")
    decoder_path = Path("/workspace/models/t5-small/decoder_model.onnx")

    # Check if files exist
    if not encoder_path.exists():
        print(f"ERROR: Encoder model not found at {encoder_path}")
        return
    if not decoder_path.exists():
        print(f"ERROR: Decoder model not found at {decoder_path}")
        return

    # Analyze both models
    encoder_stats = analyze_shapes(encoder_path, "T5 Encoder")
    decoder_stats = analyze_shapes(decoder_path, "T5 Decoder")

    # Summary comparison
    print(f"\n{'='*60}")
    print("Summary Comparison")
    print(f"{'='*60}")
    print(f"                       Encoder    Decoder    Difference")
    print(f"Total nodes:           {encoder_stats['total_nodes']:6d}     {decoder_stats['total_nodes']:6d}     {decoder_stats['total_nodes'] - encoder_stats['total_nodes']:+6d}")
    print(f"Total tensors:         {encoder_stats['total_tensors']:6d}     {decoder_stats['total_tensors']:6d}     {decoder_stats['total_tensors'] - encoder_stats['total_tensors']:+6d}")
    print(f"With shapes:           {encoder_stats['has_shapes']:6d}     {decoder_stats['has_shapes']:6d}     {decoder_stats['has_shapes'] - encoder_stats['has_shapes']:+6d}")
    print(f"Missing shapes:        {encoder_stats['missing_shapes']:6d}     {decoder_stats['missing_shapes']:6d}     {decoder_stats['missing_shapes'] - encoder_stats['missing_shapes']:+6d}")
    print(f"Shape coverage:        {encoder_stats['ratio']:5.1f}%     {decoder_stats['ratio']:5.1f}%     {decoder_stats['ratio'] - encoder_stats['ratio']:+5.1f}%")

    # Check for operation types unique to decoder
    encoder_ops = set(encoder_stats['op_counts'].keys())
    decoder_ops = set(decoder_stats['op_counts'].keys())
    decoder_only_ops = decoder_ops - encoder_ops

    if decoder_only_ops:
        print(f"\nOperation types only in decoder:")
        for op_type in decoder_only_ops:
            print(f"  {op_type}: {decoder_stats['op_counts'][op_type]}")

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
