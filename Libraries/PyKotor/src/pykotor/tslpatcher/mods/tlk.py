from __future__ import annotations

from typing import TYPE_CHECKING

from pykotor.common.misc import ResRef
from pykotor.resource.formats.tlk.io_tlk import TLKBinaryReader
from pykotor.resource.formats.tlk.tlk_auto import bytes_tlk
from pykotor.resource.formats.tlk.tlk_data import TLKEntry
from pykotor.tslpatcher.mods.template import PatcherModifications

if TYPE_CHECKING:
    from typing_extensions import Literal

    from pykotor.common.misc import Game
    from pykotor.resource.formats.tlk import TLK
    from pykotor.resource.type import SOURCE_TYPES
    from pykotor.tslpatcher.logger import PatchLogger
    from pykotor.tslpatcher.memory import PatcherMemory
    from utility.system.path import Path


class ModificationsTLK(PatcherModifications):
    DEFAULT_DESTINATION = "."
    DEFAULT_SOURCEFILE = "append.tlk"
    DEFAULT_SOURCEFILE_F = "appendf.tlk"
    DEFAULT_SAVEAS_FILE = "dialog.tlk"
    DEFAULT_SAVEAS_FILE_F = "dialogf.tlk"

    def __init__(
        self,
        filename: str = DEFAULT_SOURCEFILE,
        replace: bool | None = None,
        modifiers=None,
    ):
        super().__init__(filename)
        self.destination = self.DEFAULT_DESTINATION
        self.modifiers: list[ModifyTLK] = [] if modifiers is None else modifiers
        self.sourcefile_f: str = self.DEFAULT_SOURCEFILE_F  # Polish version of k1
        self.saveas = self.DEFAULT_SAVEAS_FILE
        self.store_memory: bool = True

    def pop_tslpatcher_vars(
        self,
        file_section_dict,
        default_destination=DEFAULT_DESTINATION,
        default_sourcefolder=".",
    ):
        if "!ReplaceFile" in file_section_dict:
            msg = "!ReplaceFile is not supported in [TLKList]"
            raise ValueError(msg)
        if "!OverrideType" in file_section_dict:
            msg = "!OverrideType is not supported in [TLKList]"
            raise ValueError(msg)

        self.sourcefile_f = file_section_dict.pop("!SourceFileF", self.DEFAULT_SOURCEFILE_F)
        super().pop_tslpatcher_vars(file_section_dict, default_destination, default_sourcefolder)

    def patch_resource(
        self,
        source: SOURCE_TYPES,
        memory: PatcherMemory,
        log: PatchLogger,
        game: Game,
    ) -> bytes | Literal[True]:
        dialog: TLK = TLKBinaryReader(source).load()
        self.apply(dialog, memory, log, game)
        return bytes_tlk(dialog)

    def apply(
        self,
        dialog: TLK,
        memory: PatcherMemory,
        log: PatchLogger,
        game: Game,
    ):
        source_cache: dict[str, TLK] = {}
        for modifier in self.modifiers:
            modifier.apply(dialog, memory, source_cache, store_memory=self.store_memory)
            log.complete_patch()


class ModifyTLK:
    def __init__(
        self,
        token_id: int,
        is_replacement: bool = False,  # noqa: FBT001, FBT002
    ):
        self.tlk_filepath: Path | None = None
        self._text: str = ""
        self._sound: ResRef = ResRef.from_blank()
        self._text_set: bool = False
        self._sound_set: bool = False

        self.mod_index: int = token_id
        self.token_id: int = token_id
        self.is_replacement: bool = is_replacement
        self.uses_main_source: bool = False

    @property
    def text(self) -> str:
        return self._text

    @text.setter
    def text(self, value: str):
        self._text = value
        self._text_set = True

    @property
    def sound(self) -> ResRef:
        return self._sound

    @sound.setter
    def sound(self, value: ResRef):
        self._sound = value
        self._sound_set = True

    def apply(
        self,
        dialog: TLK,
        memory: PatcherMemory,
        source_cache: dict[str, TLK] | None = None,
        *,
        store_memory: bool = True,
    ):
        source_entry: TLKEntry | None = self.load(source_cache)
        if self.is_replacement:
            if source_entry is None:
                dialog.replace(
                    self.token_id,
                    self.text if self._text_set else None,
                    self.sound if self._sound_set else None,
                )
            else:
                dialog.replace(self.token_id, source_entry.text, source_entry.voiceover)
            result_index = self.token_id
        else:
            entry = source_entry or TLKEntry(self.text or "", self.sound or ResRef.from_blank())
            result_index = dialog.add_entry(entry)

        if store_memory:
            memory.memory_str[self.token_id] = result_index

    def load(self, source_cache: dict[str, TLK] | None = None) -> TLKEntry | None:
        if self.tlk_filepath is None:
            return None

        cache_key = str(self.tlk_filepath)
        if source_cache is not None and cache_key in source_cache:
            lookup_tlk = source_cache[cache_key]
        else:
            lookup_tlk = TLKBinaryReader(self.tlk_filepath).load()
            if source_cache is not None:
                source_cache[cache_key] = lookup_tlk

        source_entry = lookup_tlk.get(self.mod_index)
        if source_entry is None:
            msg = f"Cannot load nonexistent stringref '{self.mod_index}' from source TLK '{self.tlk_filepath}'"
            raise IndexError(msg)

        entry = source_entry.copy()
        if self._text_set:
            entry.text = self.text
            entry.text_present = bool(self.text)
        else:
            self._text = entry.text
        if self._sound_set:
            entry.voiceover = self.sound
            entry.sound_present = bool(self.sound)
        else:
            self._sound = ResRef(str(entry.voiceover))
        return entry
