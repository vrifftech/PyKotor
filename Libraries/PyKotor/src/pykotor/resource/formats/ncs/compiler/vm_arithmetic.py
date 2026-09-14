"""Fixed-width arithmetic for the NCS interpreter."""

from __future__ import annotations

from pykotor.resource.formats.ncs.compiler.numeric import float32, int32


# Variable shifts use the low five bits of the count.
_SHIFT_COUNT_MASK = 0x1F


def k2_x86_shift_left(value: int, count: int) -> int:
    """Execute SHLEFTII with a masked count and signed 32-bit result."""
    return int32(int32(value) << (count & _SHIFT_COUNT_MASK))


def k2_x86_shift_right(value: int, count: int) -> int:
    """Execute SHRIGHTII with wrapping negation around negative-value shifts."""
    value = int32(value)
    count &= _SHIFT_COUNT_MASK
    if value < 0:
        negated = int32(-value)
        shifted = negated >> count
        return int32(-shifted)
    return value >> count


def k2_x86_unsigned_shift_right(value: int, count: int) -> int:
    """Execute USHRIGHTII as a sign-extending shift with a masked count."""
    return int32(value) >> (count & _SHIFT_COUNT_MASK)


def divide_vector_float32(vector: tuple[float, float, float], scalar: float) -> tuple[float, float, float]:
    """Divide using a rounded float32 reciprocal followed by component multiplication."""
    scalar = float32(scalar)
    if scalar == 0.0:
        raise ZeroDivisionError("NCS vector division by zero")
    reciprocal = float32(1.0 / scalar)
    x, y, z = vector
    return (
        float32(float32(x) * reciprocal),
        float32(float32(y) * reciprocal),
        float32(float32(z) * reciprocal),
    )
