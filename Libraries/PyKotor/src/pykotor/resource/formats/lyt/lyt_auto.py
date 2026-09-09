from __future__ import annotations

import os
import shutil
import tempfile

from pathlib import Path
from typing import TYPE_CHECKING

from pykotor.resource.formats.lyt.io_lyt import LYTAsciiReader, LYTAsciiWriter
from pykotor.resource.type import ResourceType

if TYPE_CHECKING:
    from pykotor.resource.formats.lyt.lyt_data import LYT
    from pykotor.resource.type import SOURCE_TYPES, TARGET_TYPES


def read_lyt(
    source: SOURCE_TYPES,
    offset: int = 0,
    size: int | None = None,
) -> LYT:
    """Returns an LYT instance from the source.

    The file format (LYT only) is automatically determined before parsing the data.

    Args:
    ----
        source: The source of the data.
        offset: The byte offset of the file inside the data.
        size: Number of bytes to allowed to read from the stream. If not specified, uses the whole stream.

    Raises:
    ------
        FileNotFoundError: If the file could not be found.
        IsADirectoryError: If the specified path is a directory (Unix-like systems only).
        PermissionError: If the file could not be accessed.
        ValueError: If the file was corrupted or the format could not be determined.

    Returns:
    -------
        An LYT instance.
    """
    return LYTAsciiReader(source, offset, size or 0).load()


def write_lyt(
    lyt: LYT,
    target: TARGET_TYPES,
    file_format: ResourceType = ResourceType.LYT,
):
    """Writes the LYT data to the target location with the specified format (LYT only).

    Args:
    ----
        lyt: The LYT file being written.
        target: The location to write the data to.
        file_format: The file format.

    Raises:
    ------
        IsADirectoryError: If the specified path is a directory (Unix-like systems only).
        PermissionError: If the file could not be written to the specified destination.
        ValueError: If the specified format was unsupported.
    """
    if file_format is not ResourceType.LYT:
        raise ValueError("Unsupported format specified; use LYT.")
    if not isinstance(target, (str, os.PathLike)):
        LYTAsciiWriter(lyt, target).write()
        return
    path = Path(target)
    if path.is_dir():
        error = PermissionError if os.name == "nt" else IsADirectoryError
        raise error(f"Cannot write LYT to directory '{path}'.")
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name.lower()}.", suffix=".tmp", delete=False) as temp:
        temporary_path = Path(temp.name)
    try:
        LYTAsciiWriter(lyt, temporary_path).write()
        if path.exists():
            shutil.copymode(path, temporary_path)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def bytes_lyt(
    lyt: LYT,
    file_format: ResourceType = ResourceType.LYT,
) -> bytes:
    """Returns the LYT data in the specified format (LYT only) as a bytes object.

    This is a convenience method that wraps the write_lyt() method.

    Args:
    ----
        lyt: The target LYT.
        file_format: The file format.

    Raises:
    ------
       ValueError: If the specified format was unsupported.

    Returns:
    -------
        The LYT data.
    """
    data = bytearray()
    write_lyt(lyt, data, file_format)
    return bytes(data)
