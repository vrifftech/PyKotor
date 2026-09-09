from __future__ import annotations

from copy import deepcopy
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


def _load_tlk(
    tlk_filepath: Path,
    source_cache: dict[str, TLK] | None,
) -> TLK:
    cache_key = str(tlk_filepath)
    if source_cache is not None and cache_key in source_cache:
        return source_cache[cache_key]

    tlk = TLKBinaryReader(tlk_filepath).load()
    if source_cache is not None:
        source_cache[cache_key] = tlk
    return tlk


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
        self.modifiers: list[MergeTLK | ModifyTLK] = [] if modifiers is None else modifiers
        self.sourcefile_f: str = self.DEFAULT_SOURCEFILE_F  # Polish version of k1
        self.saveas = self.DEFAULT_SAVEAS_FILE
        self.store_memory: bool = True
        self.female: ModificationsTLK | None = None

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
        if self.female is not None:
            raise ValueError("Paired dialog tables must be patched together with patch_pair().")
        dialog: TLK = TLKBinaryReader(source).load()
        self.apply(dialog, memory, log, game)
        return bytes_tlk(dialog)

    def patch_pair(
        self,
        source: SOURCE_TYPES,
        source_f: SOURCE_TYPES,
        memory: PatcherMemory,
        log: PatchLogger,
        game: Game,
    ) -> tuple[bytes, bytes]:
        """Builds both dialog tables with one index map before publishing any tokens."""
        dialog = TLKBinaryReader(source).load()
        dialog_f = TLKBinaryReader(source_f).load()
        pending_memory = deepcopy(memory)
        self.apply(dialog, pending_memory, log, game, dialog_f=dialog_f)
        output, output_f = bytes_tlk(dialog), bytes_tlk(dialog_f)
        memory.memory_str.update(pending_memory.memory_str)
        return output, output_f

    def apply(
        self,
        dialog: TLK,
        memory: PatcherMemory,
        log: PatchLogger,
        game: Game,
        *,
        dialog_f: TLK | None = None,
    ):
        source_cache: dict[str, TLK] = {}
        for modifier in self.modifiers:
            paired_dialog = dialog_f
            if isinstance(modifier, MergeTLK) and modifier.tlk_filepath_f is None:
                paired_dialog = None
            modifier.apply(
                dialog, memory, source_cache, store_memory=self.store_memory, dialog_f=paired_dialog,
            )


class MergeTLK:
    """Merges a source TLK into the destination and resolves its StrRef mappings."""

    def __init__(
        self,
        tlk_filepath: Path,
    ):
        self.tlk_filepath: Path = tlk_filepath
        self.tlk_filepath_f: Path | None = None
        self.strref_mappings: dict[int, int] = {}

    @property
    def patch_count(self) -> int:
        return len(self.strref_mappings)

    def add_mapping(
        self,
        token_id: int,
        source_stringref: int,
    ) -> None:
        self.strref_mappings[token_id] = source_stringref

    def apply(
        self,
        dialog: TLK,
        memory: PatcherMemory,
        source_cache: dict[str, TLK] | None = None,
        *,
        store_memory: bool = True,
        dialog_f: TLK | None = None,
    ) -> None:
        source = _load_tlk(self.tlk_filepath, source_cache)
        for source_stringref in self.strref_mappings.values():
            if source.get(source_stringref) is None:
                raise IndexError(
                    f"Cannot load nonexistent stringref '{source_stringref}' from source TLK '{self.tlk_filepath}'",
                )
        sources = [source]
        dialogs = [dialog]
        if dialog_f is not None:
            if self.tlk_filepath_f is None:
                raise ValueError("A paired TLK merge requires a female source table.")
            source_f = _load_tlk(self.tlk_filepath_f, source_cache)
            if len(source) != len(source_f):
                raise ValueError("Normal and female source TLKs must contain corresponding rows.")
            sources.append(source_f)
            dialogs.append(dialog_f)

        # Encode all source keys first. An invalid edit cannot leave half a merge.
        source_keys = [
            tuple(self._entry_key(table.entries[i], dest.encoding) for table, dest in zip(sources, dialogs))
            for i in range(len(source))
        ]
        destinations: dict[tuple, int] = {}
        for i in range(min(len(table) for table in dialogs)):
            key = tuple(self._entry_key(table.entries[i], table.encoding) for table in dialogs)
            destinations.setdefault(key, i)

        resolved: dict[int, int] = {}
        for i, key in enumerate(source_keys):
            destination = destinations.get(key)
            if destination is None:
                destination = max(len(table) for table in dialogs)
                for table, source_table in zip(dialogs, sources):
                    _pad_dialog(table, destination)
                    table.add_entry(source_table.entries[i])
                destinations[key] = destination
            resolved[i] = destination
        if store_memory:
            for token_id, source_stringref in self.strref_mappings.items():
                memory.memory_str[token_id] = resolved[source_stringref]

    @staticmethod
    def _entry_key(entry: TLKEntry, encoding: str) -> tuple[bytes, int, bytes, int, int, int]:
        return (
            entry.text_bytes(encoding), entry.flags, entry.voiceover.to_bytes(),
            entry.volume_variance, entry.pitch_variance, entry.sound_length_bits,
        )


def _pad_dialog(dialog: TLK, size: int) -> None:
    """Keeps absent indices skipped rather than creating successful empty lookups."""
    while len(dialog) < size:
        dialog.entries.append(TLKEntry("", ResRef.from_blank(), flags=0x8000))


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

    @property
    def patch_count(self) -> int:
        return 1

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
        dialog_f: TLK | None = None,
    ):
        source_entry = self.load(source_cache)
        dialogs = [dialog] if dialog_f is None else [dialog, dialog_f]
        if self.is_replacement:
            if any(table.get(self.token_id) is None for table in dialogs):
                raise IndexError(f"Cannot replace nonexistent stringref '{self.token_id}'.")
            for table in dialogs:
                table.replace(
                    self.token_id,
                    source_entry.text if source_entry is not None else (self.text if self._text_set else None),
                    source_entry.voiceover if source_entry is not None else (self.sound if self._sound_set else None),
                )
            result_index = self.token_id
        else:
            entry = source_entry if source_entry is not None else TLKEntry(self.text, self.sound)
            # Validate each table's declared encoding before extending either one.
            for table in dialogs:
                entry.text_bytes(table.encoding)
            result_index = max(len(table) for table in dialogs)
            for table in dialogs:
                _pad_dialog(table, result_index)
                table.add_entry(entry)
        if store_memory:
            memory.memory_str[self.token_id] = result_index

    def load(self, source_cache: dict[str, TLK] | None = None) -> TLKEntry | None:
        if self.tlk_filepath is None:
            return None

        source_entry = _load_tlk(self.tlk_filepath, source_cache).get(self.mod_index)
        if source_entry is None:
            msg = f"Cannot load nonexistent stringref '{self.mod_index}' from source TLK '{self.tlk_filepath}'"
            raise IndexError(msg)

        entry = source_entry.copy()
        if self._text_set:
            entry.replace(text=self.text)
        else:
            self._text = entry.text
        if self._sound_set:
            entry.replace(sound_resref=self.sound)
        else:
            self._sound = ResRef.from_bytes(entry.voiceover.to_bytes())
        return entry
