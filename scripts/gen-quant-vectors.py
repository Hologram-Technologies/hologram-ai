#!/usr/bin/env python3
"""Generate golden quantization test vectors for hologram-ai conformance tests.

These vectors serve as the ground truth for Q4_0 and Q8_0 dequantization.
The math matches ggml's dequantization spec:
  Q4_0: weight = (nibble - 8) * scale    (scale is f16, nibbles are 4-bit unsigned)
  Q8_0: weight = i8_value * scale         (scale is f16, values are signed 8-bit)

Output: JSON files in tests/fixtures/quant/

Requirements:
    pip install numpy
"""

import json
import struct
import os
import numpy as np

FIXTURES = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures", "quant")
os.makedirs(FIXTURES, exist_ok=True)


def f16_to_bytes(val: float) -> bytes:
    """Convert a float to f16 little-endian bytes."""
    return struct.pack("<e", val)


def f16_bits(val: float) -> int:
    """Get the raw u16 bits of an f16 value."""
    return struct.unpack("<H", struct.pack("<e", val))[0]


def dequant_q4_0_block(scale_f16_bits: int, qs: list[int]) -> list[float]:
    """Reference Q4_0 dequantization matching ggml spec."""
    scale = np.float16(0)
    scale_bytes = struct.pack("<H", scale_f16_bits)
    scale = struct.unpack("<e", scale_bytes)[0]

    out = []
    for byte in qs:
        lo = (byte & 0x0F) - 8
        hi = ((byte >> 4) & 0x0F) - 8
        out.append(float(lo * scale))
        out.append(float(hi * scale))
    return out


def dequant_q8_0_block(scale_f16_bits: int, qs: list[int]) -> list[float]:
    """Reference Q8_0 dequantization matching ggml spec."""
    scale_bytes = struct.pack("<H", scale_f16_bits)
    scale = struct.unpack("<e", scale_bytes)[0]

    out = []
    for q in qs:
        # Interpret as signed i8
        if q > 127:
            q -= 256
        out.append(float(q * scale))
    return out


def make_q4_0_block_bytes(scale_bits: int, qs: list[int]) -> list[int]:
    """Build raw Q4_0 block bytes: [scale_lo, scale_hi, qs[0], ..., qs[15]]."""
    return [scale_bits & 0xFF, (scale_bits >> 8) & 0xFF] + qs


def make_q8_0_block_bytes(scale_bits: int, qs: list[int]) -> list[int]:
    """Build raw Q8_0 block bytes: [scale_lo, scale_hi, qs[0], ..., qs[31]]."""
    return [scale_bits & 0xFF, (scale_bits >> 8) & 0xFF] + [q & 0xFF for q in qs]


def gen_q4_0_vectors():
    """Generate Q4_0 golden vectors."""
    vectors = []

    # Vector 1: scale=1.0, varied nibbles
    scale = f16_bits(1.0)
    qs = [0x09, 0xFA, 0x00, 0xFF, 0x88, 0x37, 0xC5, 0x12,
          0xDE, 0x6B, 0x94, 0xA3, 0x78, 0x56, 0xEF, 0x21]
    vectors.append({
        "name": "varied_scale_1",
        "block_bytes": make_q4_0_block_bytes(scale, qs),
        "expected": dequant_q4_0_block(scale, qs),
    })

    # Vector 2: scale=0.5, all same nibbles
    scale = f16_bits(0.5)
    qs = [0x79] * 16  # lo=9-8=1, hi=7-8=-1
    vectors.append({
        "name": "uniform_scale_half",
        "block_bytes": make_q4_0_block_bytes(scale, qs),
        "expected": dequant_q4_0_block(scale, qs),
    })

    # Vector 3: negative scale
    scale = f16_bits(-0.25)
    qs = [0x0F, 0xF0, 0x88, 0x08, 0x80, 0x48, 0xC2, 0x5A,
          0xBE, 0x19, 0x73, 0xD6, 0xAF, 0x64, 0x3C, 0xE7]
    vectors.append({
        "name": "negative_scale",
        "block_bytes": make_q4_0_block_bytes(scale, qs),
        "expected": dequant_q4_0_block(scale, qs),
    })

    # Vector 4: extreme values (scale=max f16 normal)
    scale = f16_bits(65504.0)  # max f16
    qs = [0x0F, 0xF0]  # lo=7, hi=-8 and lo=-8, hi=7
    qs += [0x88] * 14
    vectors.append({
        "name": "extreme_scale",
        "block_bytes": make_q4_0_block_bytes(scale, qs),
        "expected": dequant_q4_0_block(scale, qs),
    })

    # Vector 5: scale=0 (all outputs should be 0)
    scale = 0  # f16(0.0) = 0x0000
    qs = [0xFF] * 16
    vectors.append({
        "name": "zero_scale",
        "block_bytes": make_q4_0_block_bytes(scale, qs),
        "expected": dequant_q4_0_block(scale, qs),
    })

    path = os.path.join(FIXTURES, "q4_0_golden.json")
    with open(path, "w") as f:
        json.dump(vectors, f, indent=2)
    print(f"wrote {path} ({len(vectors)} vectors)")


def gen_q8_0_vectors():
    """Generate Q8_0 golden vectors."""
    vectors = []

    # Vector 1: scale=1.0, varied i8 values
    scale = f16_bits(1.0)
    qs = list(range(-16, 16))  # [-16..15]
    vectors.append({
        "name": "varied_scale_1",
        "block_bytes": make_q8_0_block_bytes(scale, qs),
        "expected": dequant_q8_0_block(scale, qs),
    })

    # Vector 2: scale=0.5, alternating
    scale = f16_bits(0.5)
    qs = [127, -128, 64, -64, 1, -1, 0, 0] * 4
    vectors.append({
        "name": "alternating_scale_half",
        "block_bytes": make_q8_0_block_bytes(scale, qs),
        "expected": dequant_q8_0_block(scale, qs),
    })

    # Vector 3: scale=2.0, extremes
    scale = f16_bits(2.0)
    qs = [127] * 16 + [-128] * 16
    vectors.append({
        "name": "extremes_scale_2",
        "block_bytes": make_q8_0_block_bytes(scale, qs),
        "expected": dequant_q8_0_block(scale, qs),
    })

    # Vector 4: zero scale
    scale = 0
    qs = [127, -128, 42, -99] * 8
    vectors.append({
        "name": "zero_scale",
        "block_bytes": make_q8_0_block_bytes(scale, qs),
        "expected": dequant_q8_0_block(scale, qs),
    })

    path = os.path.join(FIXTURES, "q8_0_golden.json")
    with open(path, "w") as f:
        json.dump(vectors, f, indent=2)
    print(f"wrote {path} ({len(vectors)} vectors)")


if __name__ == "__main__":
    gen_q4_0_vectors()
    gen_q8_0_vectors()
    print("done.")
