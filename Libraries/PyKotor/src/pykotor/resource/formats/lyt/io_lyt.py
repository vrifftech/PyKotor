from __future__ import annotations

import math
import re

from typing import TYPE_CHECKING

from pykotor.common.geometry import Vector3, Vector4
from pykotor.resource.formats.lyt.lyt_data import LYT, LYTDoorHook, LYTObstacle, LYTRoom, LYTTrack
from pykotor.resource.type import ResourceReader, ResourceWriter, autoclose

if TYPE_CHECKING:
    from pykotor.resource.type import SOURCE_TYPES, TARGET_TYPES


def _integer(value: str) -> int:
    if re.fullmatch(r"[+-]?[0-9]+", value) is None:
        raise ValueError(f"Invalid LYT integer '{value}'.")
    return int(value)


def _name(value: str) -> str:
    if not value or value.startswith("#") or any(c.isspace() or ord(c) < 32 for c in value):
        raise ValueError(f"Invalid LYT name '{value}'.")
    return value


def _number(value: str | float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("LYT coordinates and quaternions must be finite.")
    return result


class LYTAsciiReader(ResourceReader):
    def __init__(self, source: SOURCE_TYPES, offset: int = 0, size: int = 0):
        super().__init__(source, offset, size)
        self._lyt: LYT | None = None

    @autoclose
    def load(self, auto_close: bool = True) -> LYT:
        # Read only this resource's declared range; never consume a following
        # resource or discard bytes through permissive string decoding.
        lines = [
            (number, line.split())
            for number, line in enumerate(self._reader.read_bytes(self._size).decode("ascii").splitlines(), 1)
            if line.strip() and not line.lstrip().startswith("#")
        ]
        self._lyt = LYT()
        sections = {
            "roomcount": (self._lyt.rooms, LYTRoom),
            "trackcount": (self._lyt.tracks, LYTTrack),
            "obstaclecount": (self._lyt.obstacles, LYTObstacle),
            "doorhookcount": (self._lyt.doorhooks, LYTDoorHook),
        }
        seen: set[str] = set()
        started = finished = False
        index = 0
        while index < len(lines):
            number, tokens = lines[index]
            index += 1
            keyword = tokens[0].lower()
            if finished:
                raise ValueError(f"Unexpected LYT content after donelayout at line {number}.")
            if keyword == "filedependancy" and not started and not seen:
                if len(tokens) < 2:
                    raise ValueError(f"Missing LYT file dependency at line {number}.")
                continue
            if keyword == "beginlayout":
                if len(tokens) != 1 or started or seen:
                    raise ValueError(f"Invalid beginlayout at line {number}.")
                started = True
                continue
            if keyword == "donelayout":
                if len(tokens) != 1 or not (started or seen):
                    raise ValueError(f"Unexpected donelayout at line {number}.")
                finished = True
                continue
            # Exporter metadata outside the four layout tables (for example
            # othercount) is not consumed by the engine's layout loader.
            if keyword not in sections:
                continue
            if len(tokens) != 2 or keyword in seen:
                raise ValueError(f"Invalid or repeated LYT section at line {number}.")
            count = _integer(tokens[1])
            if count < 0 or count > len(lines) - index:
                raise ValueError(f"Invalid or truncated {keyword} at line {number}.")
            seen.add(keyword)
            target, record_type = sections[keyword]
            for record_number, record in lines[index:index + count]:
                fields = 10 if keyword == "doorhookcount" else 4
                if len(record) != fields:
                    raise ValueError(f"Invalid {keyword} record at line {record_number}.")
                if keyword == "doorhookcount":
                    position = Vector3(*(_number(value) for value in record[3:6]))
                    w, x, y, z = (_number(value) for value in record[6:10])
                    target.append(LYTDoorHook(
                        _name(record[0]), _name(record[1]), position,
                        Vector4(x, y, z, w), _integer(record[2]),
                    ))
                else:
                    target.append(record_type(
                        _name(record[0]), Vector3(*(_number(value) for value in record[1:4])),
                    ))
            index += count
        if not (started or seen) or (started and not finished):
            raise ValueError("Missing or incomplete LYT layout.")
        return self._lyt


class LYTAsciiWriter(ResourceWriter):
    def __init__(self, lyt: LYT, target: TARGET_TYPES):
        self._lyt = lyt
        # Validate and encode everything before opening even a direct writer's
        # destination. Public path writes additionally use atomic replacement.
        self._data = self._build()
        super().__init__(target)

    def _build(self) -> bytes:
        lines = ["beginlayout"]
        for keyword, records in (
            ("roomcount", self._lyt.rooms),
            ("trackcount", self._lyt.tracks),
            ("obstaclecount", self._lyt.obstacles),
        ):
            lines.append(f"   {keyword} {len(records)}")
            for record in records:
                x, y, z = (_number(value) for value in record.position)
                lines.append(f"      {_name(record.model)} {x} {y} {z}")
        lines.append(f"   doorhookcount {len(self._lyt.doorhooks)}")
        for hook in self._lyt.doorhooks:
            unknown = _integer(str(hook.unknown))
            x, y, z = (_number(value) for value in hook.position)
            qx, qy, qz, qw = (_number(value) for value in hook.orientation)
            lines.append(
                f"      {_name(hook.room)} {_name(hook.door)} {unknown} "
                f"{x} {y} {z} {qw} {qx} {qy} {qz}",
            )
        lines.append("donelayout")
        # CLYT's positional parser expects CRLF, even though Scene also accepts LF.
        return "\r\n".join(lines).encode("ascii")

    @autoclose
    def write(self, auto_close: bool = True):
        self._writer.write_bytes(self._data)
