from __future__ import annotations

import re

from typing import TYPE_CHECKING, Any

from pykotor.resource.formats.ssf import bytes_ssf
from pykotor.resource.formats.ssf.io_ssf import SSFBinaryReader
from pykotor.tslpatcher.memory import TokenUsage2DA, TokenUsageTLK
from pykotor.tslpatcher.mods.template import PatcherModifications

if TYPE_CHECKING:
    from typing_extensions import Literal

    from pykotor.common.misc import Game
    from pykotor.resource.formats.ssf import SSF, SSFSound
    from pykotor.resource.type import SOURCE_TYPES
    from pykotor.tslpatcher.logger import PatchLogger
    from pykotor.tslpatcher.memory import PatcherMemory, TokenUsage


_ASCII_DIGITS = re.compile(r"^[0-9]+$")
_SIGNED_DECIMAL = re.compile(r"^[+-]?[0-9]+$")
_HEXADECIMAL = re.compile(r"^\$[0-9A-Fa-f]+$")


def _parse_ssf_integer(value: Any) -> int | None:
    text = str(value).strip()
    if _HEXADECIMAL.fullmatch(text):
        return int(text[1:], 16) & 0xFFFFFFFF
    if _SIGNED_DECIMAL.fullmatch(text):
        return int(text, 10) & 0xFFFFFFFF
    return None


class ModifySSF:
    def __init__(self, sound: SSFSound, stringref: TokenUsage | str):
        self.sound = sound
        self.stringref = stringref

    def _resolve(self, memory: PatcherMemory) -> Any:
        if isinstance(self.stringref, TokenUsageTLK):
            return memory.memory_str.get(self.stringref.token_id, 0)

        if isinstance(self.stringref, TokenUsage2DA):
            raw_value = f"2DAMEMORY{self.stringref.token_id}"
            if not memory.memory_2da:
                return raw_value
            token_id = self.stringref.token_id if self.stringref.token_id in memory.memory_2da else 1
            return memory.memory_2da.get(token_id, raw_value)

        raw_value = self.stringref if isinstance(self.stringref, str) else self.stringref.value(memory)
        if raw_value.startswith("2DAMEMORY"):
            if not memory.memory_2da:
                return raw_value
            suffix = raw_value[9:]
            token_id = int(suffix) if _ASCII_DIGITS.fullmatch(suffix) else 1
            if token_id not in memory.memory_2da:
                token_id = 1
            return memory.memory_2da.get(token_id, raw_value)

        return raw_value

    def apply(self, ssf: SSF, memory: PatcherMemory, logger: PatchLogger | None = None) -> bool:
        try:
            value = _parse_ssf_integer(self._resolve(memory))
        except (KeyError, TypeError, ValueError) as exc:
            if logger is not None:
                logger.add_warning(f"Unable to resolve SSF value for {self.sound.name}: {exc}")
            return False
        if value is None:
            if logger is not None:
                logger.add_warning(f"Invalid SSF value for {self.sound.name}; skipping entry.")
            return False
        value = -1 if value == 0xFFFFFFFF else value
        if ssf.get(self.sound) == value:
            return False
        ssf.set_data(self.sound, value)
        return True


class ModificationsSSF(PatcherModifications):
    def __init__(
        self,
        filename: str,
        replace_file: bool,  # noqa: FBT001
        modifiers: list[ModifySSF] | None = None,
    ):
        super().__init__(filename)
        self.replace_file = replace_file
        self.no_replacefile_check = True
        self.modifiers = [] if modifiers is None else modifiers

    def patch_resource(
        self,
        source_ssf: SOURCE_TYPES,
        memory: PatcherMemory,
        logger: PatchLogger,
        game: Game,
    ) -> bytes | Literal[True]:
        ssf: SSF = SSFBinaryReader(source_ssf).load()
        self.apply(ssf, memory, logger, game)
        return bytes_ssf(ssf)

    def apply(
        self,
        ssf: SSF,
        memory: PatcherMemory,
        logger: PatchLogger,
        game: Game,
    ) -> bool:
        changed = False
        for modifier in self.modifiers:
            changed = modifier.apply(ssf, memory, logger) or changed
        return changed
