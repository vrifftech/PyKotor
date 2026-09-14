"""Object literals and typed value comparisons for the NCS interpreter."""
from __future__ import annotations

import struct

from pykotor.resource.formats.ncs.compiler.numeric import float32
from pykotor.resource.formats.ncs.string_encoding import encode_ncs_string


# Runtime handle; distinct from the CONSTO 1 encoding.
OBJECT_INVALID_ID = 0x7F000000


_ASCII_LOWERCASE = bytes.maketrans(b"ABCDEFGHIJKLMNOPQRSTUVWXYZ", b"abcdefghijklmnopqrstuvwxyz")


def ncs_strings_equal(left: str, right: str) -> bool:
    """Compare bytes through the first NUL, folding ASCII A-Z only."""
    left_bytes = encode_ncs_string(left).split(b"\x00", 1)[0]
    right_bytes = encode_ncs_string(right).split(b"\x00", 1)[0]
    return left_bytes.translate(_ASCII_LOWERCASE) == right_bytes.translate(_ASCII_LOWERCASE)


def float32_bits_equal(left: float, right: float) -> bool:
    """Compare binary32 words, distinguishing positive and negative zero."""
    return struct.pack(">f", float32(left)) == struct.pack(">f", float32(right))


def resolve_object_literal(encoded_value: int, executing_object: int) -> int:
    """Resolve CONSTO zero to the owner and all nonzero operands to invalid."""
    return executing_object if encoded_value == 0 else OBJECT_INVALID_ID
