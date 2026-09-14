"""Reversible Windows-1252 encoding with C1 mappings for undefined bytes."""
from __future__ import annotations


def decode_ncs_string(data: bytes) -> str:
    """Decode NCS's one-byte string representation without losing undefined CP1252 bytes."""
    result: list[str] = []
    for byte in data:
        raw = bytes((byte,))
        try:
            result.append(raw.decode("windows-1252"))
        except UnicodeDecodeError:
            result.append(chr(byte))
    return "".join(result)


def encode_ncs_string(value: str) -> bytes:
    """Encode an NCS string, preserving otherwise undefined Windows-1252 bytes."""
    result = bytearray()
    for char in value:
        try:
            result.extend(char.encode("windows-1252"))
        except UnicodeEncodeError:
            codepoint = ord(char)
            if 0 <= codepoint <= 0xFF:
                result.append(codepoint)
            else:
                raise
    return bytes(result)
