"""This module handles classes relating to working with TLK files."""

from __future__ import annotations

import struct

from itertools import zip_longest
from typing import TYPE_CHECKING

from pykotor.common.language import Language
from pykotor.common.misc import ResRef
from pykotor.resource.type import ResourceType
from utility.string_util import compare_and_format, format_text

if TYPE_CHECKING:
    from collections.abc import Callable


class TLK:
    BINARY_TYPE = ResourceType.TLK

    def __init__(
        self,
        language: Language = Language.ENGLISH,
    ):
        self.entries: list[TLKEntry] = []
        self.language_id: int = int(language)
        self.version: str = "V3.0"

    @property
    def language(self) -> Language:
        if self.language_id in Language._value2member_map_:
            return Language(self.language_id)
        return Language.UNKNOWN

    @language.setter
    def language(self, value: Language | int):
        self.language_id = int(value)

    @staticmethod
    def encoding_for_language(language_id: int) -> str:
        """Uses the declared code page; unknown IDs retain opaque high bytes."""
        if language_id not in Language._value2member_map_:
            return "ascii"
        encoding = Language(language_id).get_encoding()
        return "ascii" if encoding is None else encoding

    @property
    def encoding(self) -> str:
        return self.encoding_for_language(self.language_id)

    def __len__(
        self,
    ) -> int:
        """Returns the number of stored entries."""
        return len(self.entries)

    def __iter__(
        self,
    ):
        """Iterates through the stored entry with each iteration yielding a stringref and the corresponding entry data."""
        yield from enumerate(self.entries)

    def __getitem__(
        self,
        item,
    ):
        """Returns an entry for the specified stringref.

        Args:
        ----
            item: The stringref.

        Raises:
        ------
            IndexError: If the stringref does not exist.

        Returns:
        -------
            The corresponding TLKEntry.
        """
        return self.entries[item] if isinstance(item, int) else NotImplemented

    def get(
        self,
        stringref: int,
    ) -> TLKEntry | None:
        """Returns an entry for the specified stringref if it exists, otherwise returns None.

        Args:
        ----
            stringref: The stringref.

        Returns:
        -------
            The corresponding TLKEntry or None.
        """
        return self.entries[stringref] if 0 <= stringref < len(self) else None

    def add(
        self,
        text: str,
        sound_resref: str = "",
    ) -> int:
        entry = TLKEntry(text, ResRef(sound_resref))
        return self.add_entry(entry)

    def add_entry(
        self,
        entry: TLKEntry,
    ) -> int:
        """Appends a complete TLK entry, preserving all of its metadata."""
        entry = entry.copy()
        self.entries.append(entry)
        return len(self.entries) - 1

    def replace(
        self,
        stringref: int,
        text: str | None = None,
        sound_resref: str | ResRef | None = None,
    ):
        """Replaces selected fields of an entry while preserving its other metadata.

        Args:
        ----
            stringref: The stringref of the entry to be replaced.
            text: The new text for the entry, or ``None`` to leave it unchanged.
            sound_resref: The new sound resref, or ``None`` to leave it unchanged.
        """
        if not 0 <= stringref < len(self.entries):
            msg = f"Cannot replace nonexistent stringref in dialog.tlk: '{stringref}'"
            raise IndexError(msg)

        entry: TLKEntry = self.entries[stringref]
        entry.replace(text, sound_resref)

    def resize(
        self,
        size: int,
    ):
        """Resizes the number of entries to the specified size.

        Args:
        ----
            size: The new number of entries.
        """
        if len(self) > size:
            self.entries = self.entries[:size]
        else:
            self.entries.extend([TLKEntry("", ResRef.from_blank()) for _ in range(len(self), size)])

    def compare(self, other: TLK, log_func: Callable = print) -> bool:
        equal = True
        if self.language_id != other.language_id or self.version != other.version:
            log_func(f"TLK header mismatch: {self.version}/{self.language_id} != {other.version}/{other.language_id}")
            equal = False
        if len(self) != len(other):
            log_func(f"TLK row count mismatch. Old: {len(self)}, New: {len(other)}")
            equal = False
        for stringref, (old_entry, new_entry) in enumerate(zip_longest(self.entries, other.entries)):
            if old_entry is None or new_entry is None:
                equal = False
                continue
            if old_entry != new_entry or old_entry.text_bytes(self.encoding) != new_entry.text_bytes(other.encoding):
                log_func(f"Entry mismatch at stringref: {stringref}")
                if old_entry.text != new_entry.text:
                    log_func(format_text(compare_and_format(old_entry.text, new_entry.text)))
                for field in ("voiceover", "flags", "volume_variance", "pitch_variance", "sound_length_bits"):
                    if getattr(old_entry, field) != getattr(new_entry, field):
                        log_func(f"{field}: {getattr(old_entry, field)!r} != {getattr(new_entry, field)!r}")
                equal = False
        return equal


class TLKEntry:
    def __init__(
        self,
        text: str,
        voiceover: ResRef,
        *,
        flags: int = 0x0007,
        volume_variance: int = 0,
        pitch_variance: int = 0,
        sound_length: float = 0.0,
        sound_length_bits: int | None = None,
    ):
        self._text: str = text
        self._text_bytes: bytes | None = None
        self._text_encoding: str | None = None
        self.voiceover: ResRef = voiceover

        # Presence bits gate text/sound/length lookup; 0x8000 skips this entry.
        self.flags: int = flags
        self.volume_variance: int = volume_variance
        self.pitch_variance: int = pitch_variance
        self._sound_length_bits: int = (
            sound_length_bits
            if sound_length_bits is not None
            else struct.unpack("<I", struct.pack("<f", sound_length))[0]
        )

    @property
    def text(self) -> str:
        return self._text

    @text.setter
    def text(self, value: str):
        if value != self._text:
            self._text_bytes = None
            self._text_encoding = None
        self._text = value

    def set_text_bytes(self, data: bytes, encoding: str) -> None:
        """Decodes once while retaining the original representation for an untouched entry."""
        self._text = data.decode(encoding, "surrogateescape")
        self._text_bytes = bytes(data)
        self._text_encoding = encoding

    def text_bytes(self, encoding: str) -> bytes:
        if self._text_bytes is not None and self._text_encoding == encoding:
            return self._text_bytes
        return self.text.encode(encoding, "surrogateescape")

    @property
    def skipped(self) -> bool:
        return bool(self.flags & 0x8000)

    @skipped.setter
    def skipped(self, value: bool):
        self.flags = self.flags | 0x8000 if value else self.flags & ~0x8000

    def replace(self, text: str | None = None, sound_resref: str | ResRef | None = None) -> None:
        """Explicitly edits and activates selected fields without resetting other metadata."""
        if text is not None:
            self.text = text
            self.text_present = bool(text)
        if sound_resref is not None:
            self.voiceover = (
                ResRef.from_bytes(sound_resref.to_bytes())
                if isinstance(sound_resref, ResRef)
                else ResRef(sound_resref)
            )
            self.sound_present = bool(self.voiceover)
        if text is not None or sound_resref is not None:
            self.skipped = False

    def copy(self) -> TLKEntry:
        """Returns an independent copy including undecoded bytes and complete metadata."""
        entry = TLKEntry(
            self.text,
            ResRef.from_bytes(self.voiceover.to_bytes()),
            flags=self.flags,
            volume_variance=self.volume_variance,
            pitch_variance=self.pitch_variance,
            sound_length_bits=self.sound_length_bits,
        )
        entry._text_bytes = self._text_bytes
        entry._text_encoding = self._text_encoding
        return entry

    @property
    def sound_length(self) -> float:
        return struct.unpack("<f", struct.pack("<I", self._sound_length_bits))[0]

    @sound_length.setter
    def sound_length(self, value: float):
        self._sound_length_bits = struct.unpack("<I", struct.pack("<f", value))[0]

    @property
    def sound_length_bits(self) -> int:
        return self._sound_length_bits

    @sound_length_bits.setter
    def sound_length_bits(self, value: int):
        self._sound_length_bits = value

    @property
    def text_present(self) -> bool:
        return bool(self.flags & 0x0001)

    @text_present.setter
    def text_present(self, value: bool):
        self.flags = self.flags | 0x0001 if value else self.flags & ~0x0001

    @property
    def sound_present(self) -> bool:
        return bool(self.flags & 0x0002)

    @sound_present.setter
    def sound_present(self, value: bool):
        self.flags = self.flags | 0x0002 if value else self.flags & ~0x0002

    @property
    def soundlength_present(self) -> bool:
        return bool(self.flags & 0x0004)

    @soundlength_present.setter
    def soundlength_present(self, value: bool):
        self.flags = self.flags | 0x0004 if value else self.flags & ~0x0004

    def __eq__(
        self,
        other: TLKEntry,
    ):
        """Compares text, the stored sound reference, and all entry metadata."""
        if self is other:
            return True
        if not isinstance(other, TLKEntry):
            return NotImplemented
        return (
            other.text == self.text
            and other.voiceover.to_bytes() == self.voiceover.to_bytes()
            and other.flags == self.flags
            and other.volume_variance == self.volume_variance
            and other.pitch_variance == self.pitch_variance
            and other.sound_length_bits == self.sound_length_bits
        )

    @property
    def text_length(self) -> int:
        """Unicode character count; use text_bytes(encoding) for the binary byte length."""
        return len(self.text)
