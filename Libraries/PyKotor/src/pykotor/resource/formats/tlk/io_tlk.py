from __future__ import annotations

import struct
from typing import TYPE_CHECKING, NamedTuple

from pykotor.common.misc import ResRef
from pykotor.common.stream import ArrayHead
from pykotor.resource.formats.tlk.tlk_data import TLK
from pykotor.resource.type import ResourceReader, ResourceWriter, autoclose

if TYPE_CHECKING:
    from pykotor.common.language import Language
    from pykotor.common.stream import BinaryReader
    from pykotor.resource.type import SOURCE_TYPES, TARGET_TYPES

_FILE_HEADER_SIZE = 20
_ENTRY_SIZES = {"V2.0": 36, "V3.0": 40}


class TLKHeader(NamedTuple):
    version: str
    language_id: int
    string_count: int
    texts_offset: int

    @property
    def entry_size(self) -> int:
        return _ENTRY_SIZES[self.version]

    @classmethod
    def read(cls, reader: BinaryReader) -> TLKHeader:
        reader.seek(0)
        if reader.read_bytes(4) != b"TLK ":
            raise ValueError("Invalid TLK file type.")
        version = reader.read_bytes(4).decode("ascii")
        if version not in _ENTRY_SIZES:
            raise ValueError(f"Unsupported TLK version: {version}")
        header = cls(version, reader.read_uint32(), reader.read_uint32(), reader.read_uint32())
        if _FILE_HEADER_SIZE + header.string_count * header.entry_size > reader.size():
            raise ValueError("TLK entry table exceeds the file size.")
        if header.texts_offset > reader.size():
            raise ValueError("TLK text block is outside the file.")
        return header


class TLKBinaryReader(ResourceReader):
    def __init__(self, source: SOURCE_TYPES, offset: int = 0, size: int = 0, language: Language | None = None):
        super().__init__(source, offset, size)
        self._language = language

    @autoclose
    def load(self, auto_close: bool = True) -> TLK:
        header = TLKHeader.read(self._reader)
        tlk = TLK()
        tlk.version = header.version
        tlk.language_id = header.language_id if self._language is None else int(self._language)
        tlk.resize(header.string_count)
        texts: list[ArrayHead] = []
        for _, entry in tlk:
            entry.flags = self._reader.read_uint32()
            entry.voiceover = ResRef.from_bytes(self._reader.read_bytes(16))
            entry.volume_variance = self._reader.read_uint32()
            entry.pitch_variance = self._reader.read_uint32()
            texts.append(ArrayHead(self._reader.read_uint32(), self._reader.read_uint32()))
            entry.sound_length_bits = self._reader.read_uint32() if header.version == "V3.0" else 0
        for (_, entry), text in zip(tlk, texts):
            if text.length:
                self._reader.seek(header.texts_offset + text.offset)
                data = self._reader.read_bytes(text.length)
            else:
                data = b""
            entry.set_text_bytes(data, tlk.encoding)
        return tlk


class TLKBinaryWriter(ResourceWriter):
    def __init__(self, tlk: TLK, target: TARGET_TYPES):
        # Validate/encode before ResourceWriter opens a path for writing.
        if tlk.version not in _ENTRY_SIZES:
            raise ValueError(f"Unsupported TLK version: {tlk.version}")
        self._texts = [entry.text_bytes(tlk.encoding) for _, entry in tlk]
        self._header = b"TLK " + tlk.version.encode("ascii") + struct.pack(
            "<3I", tlk.language_id, len(tlk), _FILE_HEADER_SIZE + _ENTRY_SIZES[tlk.version] * len(tlk),
        )
        self._entries: list[bytes] = []
        offset = 0
        for (_, entry), text in zip(tlk, self._texts):
            record = struct.pack(
                "<I16s4I", entry.flags, entry.voiceover.to_bytes(), entry.volume_variance,
                entry.pitch_variance, offset, len(text),
            )
            if tlk.version == "V3.0":
                record += struct.pack("<I", entry.sound_length_bits)
            elif entry.sound_length_bits:
                raise ValueError("TLK V2.0 cannot store sound length; select V3.0 before writing it.")
            self._entries.append(record)
            offset += len(text)
        super().__init__(target)

    @autoclose
    def write(self, auto_close: bool = True):
        self._writer.write_bytes(self._header)
        for record in self._entries:
            self._writer.write_bytes(record)
        for text in self._texts:
            self._writer.write_bytes(text)
