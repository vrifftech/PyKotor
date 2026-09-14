"""Source decoding, locations, and compilation diagnostics."""
from __future__ import annotations

import codecs
import re
from bisect import bisect_right
from dataclasses import dataclass, field

from pykotor.resource.formats.ncs.string_encoding import decode_ncs_string


def _display_text(text: str) -> str:
    """Expand tabs and escape control characters without changing the source."""
    return "".join(char if char.isprintable() else repr(char)[1:-1] for char in text.expandtabs(4))


@dataclass(frozen=True)
class SourceLocation:
    """One-based character coordinates and the unmodified source line."""

    filename: str
    line: int
    column: int
    line_text: str = ""

    def __str__(self) -> str:
        return f"{_display_text(self.filename)}:{self.line}:{self.column}"

    def excerpt(self) -> str:
        """Show a bounded line excerpt with a caret aligned after tab expansion."""
        rendered = _display_text(self.line_text)
        caret = len(_display_text(self.line_text[:self.column - 1]))
        start = max(0, caret - 60)
        end = min(len(rendered), start + 120)
        prefix = "..." if start else ""
        suffix = "..." if end < len(rendered) else ""
        return f"    {prefix}{rendered[start:end]}{suffix}\n    {' ' * (len(prefix) + caret - start)}^"


@dataclass(frozen=True)
class SourceDocument:
    """Source text with a reusable line index; LF, CRLF and CR are line breaks."""

    filename: str
    text: str
    _line_starts: tuple[int, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        starts = (0, *(match.end() for match in re.finditer(r"\r\n|\r|\n", self.text)))
        object.__setattr__(self, "_line_starts", starts)

    def location(self, offset: int) -> SourceLocation:
        """Locate a character or EOF. PLY can advance its cursor just past EOF."""
        offset = max(0, min(offset, len(self.text)))
        index = bisect_right(self._line_starts, offset) - 1
        start = self._line_starts[index]
        end = self._line_starts[index + 1] if index + 1 < len(self._line_starts) else len(self.text)
        line_text = self.text[start:end].rstrip("\r\n")
        return SourceLocation(self.filename, index + 1, offset - start + 1, line_text)


class CompileError(Exception):
    """Compilation error with optional source coordinates and an include trace."""

    def __init__(
        self,
        message: str,
        line_num: int | None = None,
        context: str | None = None,
        *,
        location: SourceLocation | None = None,
    ):
        self.message = message
        self.line_num = location.line if location is not None else line_num
        self.context = context
        self.location = location
        self.include_stack: list[SourceLocation] = []
        super().__init__(self._format())

    def _format(self) -> str:
        if self.location is not None:
            result = f"{self.location}: error: {self.message}\n{self.location.excerpt()}"
        elif self.line_num is not None:
            result = f"Line {self.line_num}: {self.message}"
        else:
            result = self.message
        if self.context:
            result += f"\n  Context: {self.context}"
        for include in self.include_stack:
            result += f"\nincluded from {include}"
        return result

    def add_include(self, location: SourceLocation | None) -> None:
        """Add the referring directive while preserving the original error type."""
        if location is None:
            return
        if self.location is None:
            self.location = location
            self.line_num = location.line
        elif location != self.location:
            self.include_stack.append(location)
        self.args = (self._format(),)


def describe_token(lexeme: str) -> str:
    """Quote the source spelling, never a transformed AST/operator object."""
    display = repr(lexeme)
    return display if len(display) <= 80 else display[:76] + "..." + display[-1]


def decode_nss_source(
    data: bytes,
    *,
    source_name: str = "<string>",
    encoding: str | None = None,
) -> str:
    """Decode source as UTF-8 with a lossless Windows-1252 fallback.

    An encoding override applies to byte inputs; Unicode inputs are unchanged.
    A UTF-8 BOM requires valid UTF-8. UTF-16 and UTF-32 BOMs are rejected.
    """
    location = SourceLocation(source_name, 1, 1)
    if encoding is not None:
        try:
            encoding = codecs.lookup(encoding).name
        except LookupError as exc:
            raise CompileError(f"Unknown source encoding {encoding!r}", location=location) from exc
        if encoding == "cp1252":
            return decode_ncs_string(data)
        if encoding not in ("utf-8", "utf-8-sig"):
            raise CompileError(
                f"Unsupported source encoding {encoding!r}; use UTF-8 or Windows-1252",
                location=location,
            )
    else:
        if data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE, codecs.BOM_UTF32_BE)):
            raise CompileError(
                "UTF-16/UTF-32 source is not supported; save as UTF-8 or Windows-1252",
                location=location,
            )
        if not data.startswith(codecs.BOM_UTF8):
            try:
                return data.decode("utf-8")
            except UnicodeDecodeError:
                return decode_ncs_string(data)

    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        prefix = exc.object[:exc.start].decode("utf-8-sig")
        location = SourceDocument(source_name, prefix).location(len(prefix))
        raise CompileError(
            f"Invalid UTF-8 source byte 0x{exc.object[exc.start]:02X}; "
            "use source_encoding='windows-1252' for a legacy byte file",
            location=location,
        ) from exc
