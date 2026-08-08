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
        self._reader.seek(sounds_offset)

        for sound in SSFSound:
            self._ssf.set_data(sound, self._reader.read_uint32(max_neg1=True))

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
        self._writer.write_uint32(12)

        for sound in SSFSound:
            self._writer.write_uint32(self._ssf.get(sound), max_neg1=True)
