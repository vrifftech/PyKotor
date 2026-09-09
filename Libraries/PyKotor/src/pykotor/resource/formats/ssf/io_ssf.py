from __future__ import annotations

from typing import TYPE_CHECKING

from pykotor.resource.formats.ssf.ssf_data import SSF, SSFSound
from pykotor.resource.type import ResourceReader, ResourceWriter, autoclose

if TYPE_CHECKING:
    from pykotor.resource.type import SOURCE_TYPES, TARGET_TYPES


class SSFBinaryReader(ResourceReader):
    def __init__(
        self,
        source: SOURCE_TYPES,
        offset: int = 0,
        size: int = 0,
    ):
        super().__init__(source, offset, size)
        self._ssf: SSF | None = None

    @autoclose
    def load(
        self,
        auto_close: bool = True,
    ) -> SSF:
        self._ssf = SSF()

        file_type = self._reader.read_string(4)
        file_version = self._reader.read_string(4)

        if file_type != "SSF ":
            msg = "Attempted to load an invalid SSF was loaded."
            raise ValueError(msg)

        if file_version != "V1.1":
            msg = "The supplied SSF file version is not supported."
            raise ValueError(msg)

        sounds_offset = self._reader.read_uint32()
        if sounds_offset < 12 or sounds_offset + 28 * 4 > self._size:
            raise ValueError("The SSF data offset must address all 28 sound entries.")

        self._ssf._padding = self._reader.read_bytes(sounds_offset - 12)
        entry_count = min(len(SSFSound), (self._size - sounds_offset) // 4)
        for index in range(entry_count):
            self._ssf.set_data(SSFSound(index), self._reader.read_uint32(max_neg1=True))
        self._ssf._trailing_data = self._reader.read_bytes(self._size - sounds_offset - entry_count * 4)

        return self._ssf


class SSFBinaryWriter(ResourceWriter):
    def __init__(
        self,
        ssf: SSF,
        target: TARGET_TYPES,
    ):
        super().__init__(target)
        self._ssf: SSF = ssf

    @autoclose
    def write(
        self,
        auto_close: bool = True,
    ):
        self._writer.write_string("SSF ")
        self._writer.write_string("V1.1")
        self._writer.write_uint32(12 + len(self._ssf._padding))
        self._writer.write_bytes(self._ssf._padding)

        for index in range(self._ssf._entry_count):
            self._writer.write_uint32(self._ssf.get(SSFSound(index)), max_neg1=True)
        self._writer.write_bytes(self._ssf._trailing_data)
