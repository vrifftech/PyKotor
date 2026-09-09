from __future__ import annotations

import re

from typing import TYPE_CHECKING

from pykotor.resource.formats.vis.vis_data import VIS
from pykotor.resource.type import ResourceReader, ResourceWriter, autoclose

if TYPE_CHECKING:
    from pykotor.resource.type import SOURCE_TYPES, TARGET_TYPES


def _name(value: str) -> str:
    if not value or value.startswith("#") or any(c.isspace() or ord(c) < 32 for c in value):
        raise ValueError(f"Invalid VIS room name '{value}'.")
    return value


class VISAsciiReader(ResourceReader):
    def __init__(self, source: SOURCE_TYPES, offset: int = 0, size: int = 0):
        super().__init__(source, offset, size)
        self._vis: VIS | None = None

    @autoclose
    def load(self, auto_close: bool = True) -> VIS:
        lines = [
            (number, line.split())
            for number, line in enumerate(self._reader.read_bytes(self._size).decode("ascii").splitlines(), 1)
            if line.strip() and not line.lstrip().startswith("#")
        ]
        self._vis = VIS()
        pairs: list[tuple[str, str]] = []
        index = 0
        while index < len(lines):
            number, tokens = lines[index]
            index += 1
            if len(tokens) != 2 or re.fullmatch(r"\+?[0-9]+", tokens[1]) is None:
                raise ValueError(f"Invalid VIS room/count record at line {number}.")
            observer = _name(tokens[0])
            count = int(tokens[1])
            if count > len(lines) - index:
                raise ValueError(f"Truncated VIS neighbor list at line {number}.")
            self._vis.add_room(observer)
            for record_number, record in lines[index:index + count]:
                if len(record) != 1:
                    raise ValueError(f"Invalid VIS neighbor at line {record_number}.")
                pairs.append((observer, _name(record[0])))
            index += count
        for observer, observed in pairs:
            self._vis.add_room(observed)
            self._vis.set_visible(observer, observed, visible=True)
        return self._vis


class VISAsciiWriter(ResourceWriter):
    def __init__(self, vis: VIS, target: TARGET_TYPES):
        self._vis = vis
        lines: list[str] = []
        for observer, observed in vis:
            lines.append(f"{_name(observer)} {len(observed)}\r\n")
            for room in sorted(observed):
                lines.append(f"  {_name(room)}\r\n")
        # A failed validation/encoding must not truncate the destination.
        self._data = "".join(lines).encode("ascii")
        super().__init__(target)

    @autoclose
    def write(self, auto_close: bool = True):
        self._writer.write_bytes(self._data)
