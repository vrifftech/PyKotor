from __future__ import annotations

from typing import TYPE_CHECKING

from pykotor.resource.formats.erf.erf_data import ERF, ERFLocalizedString, ERFType
from pykotor.resource.type import ResourceReader, ResourceType, ResourceWriter, autoclose
from utility.logger_util import RobustRootLogger

if TYPE_CHECKING:
    from pykotor.resource.type import SOURCE_TYPES, TARGET_TYPES


class ERFBinaryReader(ResourceReader):
    def __init__(
        self,
        source: SOURCE_TYPES,
        offset: int = 0,
        size: int = 0,
    ):
        super().__init__(source, offset, size)
        self._erf: ERF | None = None

    @autoclose
    def load(
        self,
        auto_close: bool = True,
    ) -> ERF:
        """Load ERF file.

        Args:
        ----
            self: The ERF object
            auto_close: Whether to close the file after loading

        Returns:
        -------
            ERF: The loaded ERF object

        Processing Logic:
        ----------------
            - Read file header and validate file type and version
            - Read entry count and offsets to keys and resources sections
            - Read keys section into lists of ref, id, type
            - Read resources section into lists of offsets and sizes
            - Seek to each resource and read data into ERF object.
        """
        file_type = self._reader.read_string(4)
        file_version = self._reader.read_string(4)

        if file_version != "V1.0":
            msg = f"ERF version '{file_version}' is unsupported."
            raise ValueError(msg)

        erf_type = next(
            (x for x in ERFType if x.value == file_type),
            None,
        )
        if erf_type is None:
            msg = f"Not a valid ERF file: '{file_type}'"
            raise ValueError(msg)

        self._erf = ERF(erf_type)

        localized_string_count = self._reader.read_uint32()
        localized_string_size = self._reader.read_uint32()
        entry_count = self._reader.read_uint32()
        offset_to_localized_strings = self._reader.read_uint32()
        offset_to_keys = self._reader.read_uint32()
        offset_to_resources = self._reader.read_uint32()
        self._erf.build_year = self._reader.read_uint32()
        self._erf.build_day = self._reader.read_uint32()
        description_strref = self._reader.read_uint32()
        self._erf.description_strref = description_strref
        self._erf.reserved = self._reader.read_bytes(116)
        if description_strref == 0 and file_type == ERFType.MOD.value:
            RobustRootLogger().debug("Assuming this is a SAV file")
            self._erf.is_save_erf = True

        if localized_string_count:
            self._reader.seek(offset_to_localized_strings)
            localized_string_start = self._reader.position()
            for _ in range(localized_string_count):
                language_id = self._reader.read_uint32()
                string_size = self._reader.read_uint32()
                self._erf.localized_strings.append(
                    ERFLocalizedString(language_id, self._reader.read_bytes(string_size)),
                )
            bytes_read = self._reader.position() - localized_string_start
            if bytes_read != localized_string_size:
                RobustRootLogger().warning(
                    "ERF localized-string table size does not match the header: expected %s bytes, read %s bytes.",
                    localized_string_size,
                    bytes_read,
                )

        resrefs: list[str] = []
        resids: list[int] = []
        restypes: list[int] = []
        self._reader.seek(offset_to_keys)
        for _ in range(entry_count):
            resrefs.append(self._reader.read_string(16))
            resids.append(self._reader.read_uint32())
            restypes.append(self._reader.read_uint16())
            self._reader.skip(2)

        resoffsets: list[int] = []
        ressizes: list[int] = []
        self._reader.seek(offset_to_resources)
        for _ in range(entry_count):
            resoffsets.append(self._reader.read_uint32())
            ressizes.append(self._reader.read_uint32())

        for i in range(entry_count):
            self._reader.seek(resoffsets[i])
            resdata = self._reader.read_bytes(ressizes[i])
            self._erf.set_data(resrefs[i], ResourceType.from_id(restypes[i]), resdata)

        return self._erf


class ERFBinaryWriter(ResourceWriter):
    FILE_HEADER_SIZE = 160
    KEY_ELEMENT_SIZE = 24
    RESOURCE_ELEMENT_SIZE = 8

    def __init__(
        self,
        erf: ERF,
        target: TARGET_TYPES,
    ):
        super().__init__(target)
        self.erf: ERF = erf

    @autoclose
    def write(
        self,
        auto_close: bool = True,
    ):
        entry_count = len(self.erf)
        localized_string_size = sum(8 + len(localized_string.data) for localized_string in self.erf.localized_strings)
        offset_to_localized_strings = ERFBinaryWriter.FILE_HEADER_SIZE
        offset_to_keys = offset_to_localized_strings + localized_string_size
        offset_to_resources = offset_to_keys + ERFBinaryWriter.KEY_ELEMENT_SIZE * entry_count

        self._writer.write_string(self.erf.erf_type.value)
        self._writer.write_string("V1.0")
        self._writer.write_uint32(len(self.erf.localized_strings))
        self._writer.write_uint32(localized_string_size)
        self._writer.write_uint32(entry_count)
        self._writer.write_uint32(offset_to_localized_strings)
        self._writer.write_uint32(offset_to_keys)
        self._writer.write_uint32(offset_to_resources)
        self._writer.write_uint32(self.erf.build_year)
        self._writer.write_uint32(self.erf.build_day)
        self._writer.write_uint32(self.erf.description_strref)
        self._writer.write_bytes(self.erf.reserved[:116].ljust(116, b"\0"))

        for localized_string in self.erf.localized_strings:
            self._writer.write_uint32(localized_string.language_id)
            self._writer.write_uint32(len(localized_string.data))
            self._writer.write_bytes(localized_string.data)

        for resid, resource in enumerate(self.erf):
            self._writer.write_string(str(resource.resref), string_length=16)
            self._writer.write_uint32(resid)
            self._writer.write_uint16(resource.restype.type_id)
            self._writer.write_uint16(0)
        data_offset = offset_to_resources + ERFBinaryWriter.RESOURCE_ELEMENT_SIZE * entry_count
        for resource in self.erf:
            self._writer.write_uint32(data_offset)
            self._writer.write_uint32(len(resource.data))
            data_offset += len(resource.data)

        for resource in self.erf:
            self._writer.write_bytes(resource.data)
