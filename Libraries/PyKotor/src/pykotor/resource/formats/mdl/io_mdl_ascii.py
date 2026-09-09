from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pykotor.resource.formats.mdl.mdl_data import MDL
    from pykotor.resource.type import SOURCE_TYPES, TARGET_TYPES

class MDLAsciiReader:
    """ASCII models are explicitly unsupported until a complete parser is available.

    Returning an empty MDL is not parsing, and can destroy the source on save.
    Construction does not acquire a stream or modify a borrowed reader.
    """

    def __init__(self, source: SOURCE_TYPES, offset: int=0, size: int=0):
        self.source, self.offset, self.size = (source, offset, size)

    def load(self, auto_close: bool=True) -> MDL:
        raise NotImplementedError('ASCII MDL import is not implemented; use a binary MDL/MDX pair')

class MDLAsciiWriter:
    """Do not silently export a preview-only subset of a binary model."""

    def __init__(self, mdl: MDL, target: TARGET_TYPES):
        self.mdl, self.target = (mdl, target)

    def write(self, auto_close: bool=True):
        raise NotImplementedError('Lossless ASCII MDL export is not implemented; save the binary pair')
