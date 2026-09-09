from __future__ import annotations
import os
from pathlib import Path
from typing import TYPE_CHECKING
from pykotor.resource.formats.mdl.io_mdl import MDLBinaryReader, MDLBinaryWriter, _read_source
from pykotor.resource.formats.mdl.io_mdl_ascii import MDLAsciiReader, MDLAsciiWriter
from pykotor.resource.type import ResourceType
if TYPE_CHECKING:
    from pykotor.resource.formats.mdl.mdl_data import MDL
    from pykotor.resource.type import SOURCE_TYPES, TARGET_TYPES

def detect_mdl(source: SOURCE_TYPES, offset: int=0) -> ResourceType:
    """Identify a binary preamble or an ASCII model declaration at the requested offset."""
    data = _read_source(source, offset)
    if len(data) >= 4 and data[:4] == bytes(4):
        return ResourceType.MDL
    for line in data.splitlines():
        line = line.strip()
        if not line or line.startswith(b'#'):
            continue
        if line.lower().startswith(b'newmodel '):
            return ResourceType.MDL_ASCII
        break
    return ResourceType.INVALID

def _companion(path: os.PathLike | str, *, must_exist: bool = True) -> Path:
    path = Path(path)
    expected = path.with_suffix('.mdx').name.lower()
    matches = [p for p in path.parent.iterdir() if p.name.lower() == expected and p.is_file()]
    if len(matches) > 1 or must_exist and not matches:
        raise FileNotFoundError(f"Expected exactly one MDX companion for '{path}'")
    return matches[0] if matches else path.with_suffix('.mdx')

def read_mdl(source: SOURCE_TYPES, offset: int=0, size: int | None=None, source_ext: SOURCE_TYPES | None=None, offset_ext: int=0, size_ext: int=0) -> MDL:
    """Read bounded MDL/MDX resources; a loose .mdl uses its matching .mdx sibling.

    Byte buffers and embedded resources require an explicit companion when their
    vertex streams are needed. An MDL-only model can still be inspected, renamed,
    transformed at the hierarchy level, or converted without inventing MDX data.
    """
    data = _read_source(source, offset, size or 0)
    if len(data) >= 4 and data[:4] == bytes(4):
        if source_ext is None and isinstance(source, (str, os.PathLike)) and (Path(source).suffix.lower() == '.mdl'):
            source_ext = _companion(source)
        return MDLBinaryReader(data, source_ext=source_ext, offset_ext=offset_ext, size_ext=size_ext).load()
    if detect_mdl(data) is ResourceType.MDL_ASCII:
        return MDLAsciiReader(data).load()
    raise ValueError('Not a binary MDL or recognized ASCII model')

def write_mdl(mdl: MDL, target: TARGET_TYPES, file_format: ResourceType=ResourceType.MDL, target_ext: TARGET_TYPES | None=None):
    """Serialize both buffers before writing; path output commits a staged MDL/MDX pair.

    A loose path without an explicit companion uses the same stem with .mdx.
    Memory callers should supply two separate buffers or use bytes_mdl_pair().
    """
    if file_format is ResourceType.MDL:
        if target_ext is None and isinstance(target, (str, os.PathLike)):
            target_ext = _companion(target, must_exist=False)
        MDLBinaryWriter(mdl, target, target_ext).write()
    elif file_format is ResourceType.MDL_ASCII:
        MDLAsciiWriter(mdl, target).write()
    else:
        raise ValueError('Unsupported model output format')

def bytes_mdl(mdl: MDL, file_format: ResourceType=ResourceType.MDL) -> bytes:
    """Return the MDL resource only, without ever aliasing it to the MDX output."""
    if file_format is ResourceType.MDL:
        return MDLBinaryWriter(mdl, None).encode()[0]
    data = bytearray()
    write_mdl(mdl, data, file_format)
    return bytes(data)

def bytes_mdl_pair(mdl: MDL) -> tuple[bytes, bytes]:
    """Return a complete independently serialized MDL/MDX pair."""
    data, extra = MDLBinaryWriter(mdl, None).encode()
    if extra is None:
        raise ValueError('The MDX companion was not supplied when the model was loaded')
    return (data, extra)
