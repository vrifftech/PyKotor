"""NSS numeric parsing and fixed-width conversions."""

from __future__ import annotations

import math
import struct


def int32(value: int) -> int:
    """Return ``value`` in the signed 32-bit representation used by NCS."""
    value &= 0xFFFFFFFF
    return value - 0x100000000 if value >= 0x80000000 else value


def float32(value: float) -> float:
    """Round ``value`` to the IEEE-754 single precision used by NCS."""
    try:
        return struct.unpack(">f", struct.pack(">f", float(value)))[0]
    except OverflowError:
        return math.copysign(math.inf, value)


def parse_kotor_float_literal(text: str) -> float:
    """Parse decimal digits with float32 rounding at each arithmetic step."""
    if not text:
        raise ValueError("KotOR float literal cannot be empty")

    sign = float32(-1.0 if text.startswith("-") else 1.0)
    value = float32(0.0)
    fraction_scale = float32(1.0)
    after_decimal = False

    for index, char in enumerate(text):
        if char == "-" and index == 0:
            continue
        if char == ".":
            if after_decimal:
                raise ValueError(f"Invalid KotOR float literal: {text!r}")
            after_decimal = True
            continue
        if not char.isdigit():
            raise ValueError(f"Invalid KotOR float literal: {text!r}")

        digit = ord(char) - ord("0")
        if after_decimal:
            fraction_scale = float32(fraction_scale / float32(10.0))
            contribution = float32(float32(float(digit)) * fraction_scale)
            value = float32(value + contribution)
        else:
            value = float32(value * float32(10.0))
            value = float32(value + float32(float(digit)))

    return float32(value * sign)


def canonicalize_kotor_float_constant(value: float) -> float:
    """Round to float32, format with six decimal places, and reparse."""
    value32 = float32(value)
    if not math.isfinite(value32):
        # Nonfinite constants resolve to zero with their original sign.
        return math.copysign(0.0, value32)
    return parse_kotor_float_literal(f"{value32:.6f}")
