from __future__ import annotations

import re
from itertools import islice
from typing import TYPE_CHECKING

from pykotor.resource.formats.twoda.twoda_data import TwoDA
from pykotor.resource.type import ResourceReader, ResourceWriter, autoclose

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pykotor.resource.type import SOURCE_TYPES, TARGET_TYPES


def _tokens(line: str) -> Iterator[str]:
    """Tokenize one native 2DA line; backslashes have no escape meaning."""
    position = 0
    while position < len(line):
        while position < len(line) and line[position] in " \t":
            position += 1
        if position == len(line):
            return
        start = position
        if line[position] == '"':
            end = line.find('"', position + 1)
            if end < 0:
                raise ValueError("Unterminated quoted value in text 2DA.")
            yield line[position + 1:end]
            position = end + 1
        else:
            while position < len(line) and line[position] not in " \t\r\n\0":
                position += 1
            if position == start:
                raise ValueError("Invalid delimiter in text 2DA.")
            yield line[start:position]


def _token(value: str) -> str:
    """Encode one token without silently changing native parser semantics."""
    if any(character in value for character in "\r\n\0"):
        raise ValueError("A text 2DA token cannot contain a newline or null byte.")
    if not value or value.startswith('"') or any(character in value for character in " \t"):
        if '"' in value:
            raise ValueError("This text 2DA value cannot be represented with native quoting.")
        return f'"{value}"'
    return value


class TwoDATextReader(ResourceReader):
    def __init__(self, source: SOURCE_TYPES, offset: int = 0, size: int = 0):
        super().__init__(source, offset, size)

    @autoclose
    def load(self, auto_close: bool = True) -> TwoDA:
        # Latin-1 is a reversible byte mapping, not language/encoding detection.
        data = self._reader.read_bytes(self._size).decode("latin-1")
        lines = iter(re.split(r"\r\n|\r|\n", data))
        if next(lines, "").rstrip(" \t") != "2DA V2.0":
            raise ValueError("Expected a text 2DA V2.0 header.")

        twoda = TwoDA(version="V2.0")
        headers: list[str] | None = None
        has_default = False
        for line in lines:
            tokens = _tokens(line)
            values = list(tokens if headers is None else islice(tokens, len(headers) + 1))
            if not values:
                continue
            if headers is None:
                if not has_default and values[0].lower() == "default:":
                    twoda.default = values[1] if len(values) > 1 else ""
                    has_default = True
                    continue
                headers = values
                for header in headers:
                    twoda.add_column(header)
                continue

            row = twoda.add_row(values[0])
            for column, header in enumerate(headers, 1):
                if column >= len(values):
                    value = twoda.default
                else:
                    value = "" if values[column] == "****" else values[column]
                twoda.set_cell(row, header, value)

        if headers is None:
            raise ValueError("Text 2DA column headers are missing.")
        return twoda


class TwoDATextWriter(ResourceWriter):
    def __init__(self, twoda: TwoDA, target: TARGET_TYPES):
        super().__init__(target)
        self._twoda = twoda

    @autoclose
    def write(self, auto_close: bool = True):
        headers = self._twoda.get_headers()
        if not headers:
            raise ValueError("A text 2DA must have column headers.")
        lines = ["2DA V2.0", "", f"DEFAULT: {_token(self._twoda.default)}"]
        lines.append("\t".join(_token(header) for header in headers))
        for index, label in enumerate(self._twoda.get_labels()):
            values = [self._twoda.get_cell(index, header) for header in headers]
            # Omitted trailing cells use DEFAULT:, including a literal ****
            # default, which is distinct from an explicit empty (****) cell.
            while values and values[-1] == self._twoda.default:
                values.pop()
            if "****" in values:
                raise ValueError("A literal **** cell is not representable in text 2DA.")
            lines.append("\t".join([_token(str(label)), *(_token(value) if value else "****" for value in values)]))
        self._writer.write_bytes(("\n".join(lines) + "\n").encode("latin-1"))
