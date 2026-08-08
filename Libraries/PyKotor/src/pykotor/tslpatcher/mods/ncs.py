from __future__ import annotations

from typing import TYPE_CHECKING

from pykotor.common.stream import BinaryReader, BinaryWriter
from pykotor.tslpatcher.mods.template import PatcherModifications
from utility.system.path import PurePath, PureWindowsPath

if TYPE_CHECKING:
    from typing_extensions import Literal

    from pykotor.common.misc import Game
    from pykotor.resource.type import SOURCE_TYPES
    from pykotor.tslpatcher.logger import PatchLogger
    from pykotor.tslpatcher.memory import PatcherMemory


class ModificationsNCS(PatcherModifications):
    def __init__(self, filename, replace=None, modifiers=None):
        super().__init__(filename, replace, modifiers)
        self.action: str = "Hack "
        self.hackdata: list[tuple[str, int, int]] = []

    def patch_resource(
        self,
        ncs_source: SOURCE_TYPES,
        memory: PatcherMemory,
        logger: PatchLogger,
        game: Game,
    ) -> bytes | Literal[True]:
        with BinaryReader.from_auto(ncs_source) as reader:
            ncs_bytearray: bytearray = bytearray(reader.read_all())
        self.apply(ncs_bytearray, memory, logger, game)
        return bytes(ncs_bytearray)

    def apply(
        self,
        ncs_bytearray: bytearray,
        memory: PatcherMemory,
        logger: PatchLogger,
        game: Game,
    ):
        is_ncs = PurePath(self.saveas).suffix.lower() == ".ncs"
        with BinaryWriter.to_bytearray(ncs_bytearray) as writer:
            for patch in self.hackdata:
                token_type, offset, token_id_or_value = patch
                if offset >= len(ncs_bytearray):
                    logger.add_warning(
                        f"HACKList {self.sourcefile}: offset {offset:#X} is outside the file; skipping value.",
                    )
                    continue

                value: int
                if token_type.lower() == "strref":
                    memory_strval: int | None = memory.memory_str.get(token_id_or_value)
                    if memory_strval is None:
                        logger.add_warning(
                            f"StrRef{token_id_or_value} was not defined before use in HACKList; writing 0.",
                        )
                        memory_strval = 0
                    value = memory_strval
                elif token_type.lower() == "2damemory":
                    memory_val: str | PureWindowsPath | None = None
                    if memory.memory_2da:
                        highest_token = max(memory.memory_2da)
                        memory_token = (
                            1
                            if token_id_or_value <= 0 or token_id_or_value > highest_token
                            else token_id_or_value
                        )
                        memory_val = memory.memory_2da.get(memory_token)
                    if memory_val is None:
                        logger.add_warning(
                            f"2DAMEMORY{token_id_or_value} was not defined before use in HACKList; skipping value.",
                        )
                        continue
                    if isinstance(memory_val, PureWindowsPath):
                        logger.add_warning(
                            f"2DAMEMORY{token_id_or_value} contains a !FieldPath and cannot be written by HACKList; skipping value.",
                        )
                        continue
                    if not memory_val.isascii() or not memory_val.isdigit():
                        logger.add_warning(
                            f"2DAMEMORY{token_id_or_value} does not contain an unsigned decimal value; skipping value.",
                        )
                        continue
                    value = int(memory_val)
                else:
                    value = token_id_or_value

                if not 0 <= value <= 0x7FFFFFFF:
                    logger.add_warning(
                        f"HACKList {self.sourcefile}: value {value} is outside the supported 32-bit range; skipping value.",
                    )
                    continue

                logger.add_verbose(f"HACKList {self.sourcefile}: writing DWORD {value} at offset {offset:#X}")
                writer.seek(offset)
                writer.write_int32(value, big=is_ncs)

    def pop_tslpatcher_vars(
        self,
        file_section_dict,
        default_destination=PatcherModifications.DEFAULT_DESTINATION,
        default_sourcefolder=".",
    ):
        super().pop_tslpatcher_vars(file_section_dict, default_destination, default_sourcefolder)
        replace_file: bool | str = file_section_dict.pop("ReplaceFile", self.replace_file)
        if isinstance(replace_file, bool):
            self.replace_file = replace_file
        elif replace_file in {"0", "1"}:
            self.replace_file = replace_file == "1"
