from __future__ import annotations

import os
import shutil
import tempfile

from pathlib import Path
from typing import TYPE_CHECKING

from pykotor.common.stream import BinaryReader, BinaryWriter
from pykotor.resource.formats.txi.txi_document import TXI

if TYPE_CHECKING:
    from pykotor.resource.type import SOURCE_TYPES, TARGET_TYPES


def read_txi(source: SOURCE_TYPES, offset: int = 0, size: int | None = None) -> TXI:
    if isinstance(source, BinaryReader):
        position = source.position()
        try:
            source.skip(offset)
            return TXI(source.read_bytes(source.remaining() if size is None else size))
        finally:
            source.seek(position)
    with BinaryReader.from_auto(source, offset) as reader:
        return TXI(reader.read_bytes(reader.remaining() if size is None else size))


def bytes_txi(txi: TXI) -> bytes:
    return bytes(txi)


def write_txi(txi: TXI, target: TARGET_TYPES) -> None:
    data = bytes(txi)
    if not isinstance(target, (str, os.PathLike)):
        with BinaryWriter.to_auto(target) as writer:
            writer.write_bytes(data)
        return
    path = Path(target)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name.lower()}.", suffix=".tmp", delete=False) as temp:
        temporary_path = Path(temp.name)
    try:
        temporary_path.write_bytes(data)
        if path.exists():
            shutil.copymode(path, temporary_path)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
