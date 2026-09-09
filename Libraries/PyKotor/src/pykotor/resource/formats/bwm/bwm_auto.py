from __future__ import annotations

import os
import shutil
import tempfile

from pathlib import Path
from typing import TYPE_CHECKING

from pykotor.resource.formats.bwm.io_bwm import BWMBinaryReader, BWMBinaryWriter
from pykotor.resource.type import ResourceType

if TYPE_CHECKING:
    from pykotor.resource.formats.bwm.bwm_data import BWM
    from pykotor.resource.type import SOURCE_TYPES, TARGET_TYPES


def read_bwm(
    source: SOURCE_TYPES,
    offset: int = 0,
    size: int | None = None,
) -> BWM:
    """Returns a BWM instance from the source.

    Args:
    ----
        source: The source of the data.
        offset: The byte offset of the file inside the data
        size: Number of bytes to allowed to read from the stream. If not specified, uses the whole stream.

    Raises:
    ------
        FileNotFoundError: If the file could not be found.
        IsADirectoryError: If the specified path is a directory (Unix-like systems only).
        PermissionError: If the file could not be accessed.
        ValueError: If the file was corrupted.

    Returns:
    -------
        A BWM instance.
    """
    return BWMBinaryReader(source, offset, size or 0).load()


def write_bwm(
    wok: BWM,
    target: TARGET_TYPES,
    file_format: ResourceType = ResourceType.WOK,
):
    """Writes the WOK data to the target location with the specified format (WOK, PWK or DWK).

    Args:
    ----
        wok: The WOK file being written.
        target: The location to write the data to.
        file_format: The file format.

    Raises:
    ------
        IsADirectoryError: If the specified path is a directory (Unix-like systems only).
        PermissionError: If the file could not be written to the specified destination.
        ValueError: If the specified format was unsupported.
    """
    if file_format not in (ResourceType.WOK, ResourceType.PWK, ResourceType.DWK):
        raise ValueError("Unsupported format specified; use WOK, PWK or DWK.")
    if not isinstance(target, (str, os.PathLike)):
        BWMBinaryWriter(wok, target).write()
        return
    path = Path(target)
    if path.is_dir():
        error = PermissionError if os.name == "nt" else IsADirectoryError
        raise error(f"Cannot write a walkmesh to directory '{path}'.")
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name.lower()}.", suffix=".tmp", delete=False) as temp:
        temporary_path = Path(temp.name)
    try:
        BWMBinaryWriter(wok, temporary_path).write()
        if path.exists():
            shutil.copymode(path, temporary_path)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def bytes_bwm(
    bwm: BWM,
    file_format: ResourceType = ResourceType.WOK,
) -> bytes:
    """Returns the BWM data in the specified format (WOK, PWK or DWK) as a bytes object.

    This is a convenience method that wraps the write_bwm() method.

    Args:
    ----
        bwm: The target BWM.
        file_format: The file format.

    Raises:
    ------
        ValueError: If the specified format was unsupported.

    Returns:
    -------
        The BWM data.
    """
    data = bytearray()
    write_bwm(bwm, data, file_format)
    return bytes(data)
