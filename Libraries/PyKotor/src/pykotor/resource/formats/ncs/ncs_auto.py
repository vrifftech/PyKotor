from __future__ import annotations

from typing import TYPE_CHECKING

from pykotor.common.scriptdefs import KOTOR_CONSTANTS, KOTOR_FUNCTIONS, TSL_CONSTANTS, TSL_FUNCTIONS
from pykotor.common.scriptlib import KOTOR_LIBRARY, TSL_LIBRARY
from pykotor.resource.formats.ncs.compiler.classes import DEFAULT_MAX_INCLUDE_DEPTH
from pykotor.resource.formats.ncs.compiler.parser import NssParser
from pykotor.resource.formats.ncs.io_ncs import NCSBinaryReader, NCSBinaryWriter
from pykotor.resource.formats.ncs.ncs_data import NCS
from pykotor.resource.formats.ncs.optimizers import RemoveNopOptimizer, RemoveUnusedBlocksOptimizer
from pykotor.resource.type import ResourceType

if TYPE_CHECKING:
    from ply import yacc

    from pykotor.common.misc import Game
    from pykotor.resource.formats.ncs.ncs_data import NCSOptimizer
    from pykotor.resource.type import SOURCE_TYPES, TARGET_TYPES
    from utility.system.path import Path


def read_ncs(
    source: SOURCE_TYPES,
    offset: int = 0,
    size: int | None = None,
) -> NCS:
    """Read an NCS object from a file, byte buffer, or stream."""
    return NCSBinaryReader(source, offset, size or 0).load()


def write_ncs(
    ncs: NCS,
    target: TARGET_TYPES,
    file_format: ResourceType = ResourceType.NCS,
):
    """Write an NCS object to a file or writable buffer."""
    if file_format is ResourceType.NCS:
        NCSBinaryWriter(ncs, target).write()
    else:
        msg = "Unsupported format specified; use NCS."
        raise ValueError(msg)


def bytes_ncs(
    ncs: NCS,
    file_format: ResourceType = ResourceType.NCS,
) -> bytearray:
    """Serialize an NCS object to bytes."""
    data = bytearray()
    write_ncs(ncs, data, file_format)
    return data


def compile_nss(
    source: str | bytes,
    game: Game,
    optimizers: list[NCSOptimizer] | None = None,
    library_lookup: list[str | Path] | list[Path] | list[str] | str | Path | None = None,
    *,
    errorlog: yacc.NullLogger | None = None,
    debug: bool = False,
    max_include_depth: int = DEFAULT_MAX_INCLUDE_DEPTH,
    source_name: str = "<string>",
    source_encoding: str | None = None,
) -> NCS:
    """Compile NSS source using the selected game API and optimizers.

    source_name labels diagnostics. source_encoding selects UTF-8 or Windows-1252;
    None detects each file independently. The root counts toward max_include_depth.
    """
    nss_parser = NssParser(
        functions=KOTOR_FUNCTIONS if game.is_k1() else TSL_FUNCTIONS,
        constants=KOTOR_CONSTANTS if game.is_k1() else TSL_CONSTANTS,
        library=KOTOR_LIBRARY if game.is_k1() else TSL_LIBRARY,
        library_lookup=library_lookup,
        errorlog=errorlog,
        debug=debug,
        max_include_depth=max_include_depth,
        source_encoding=source_encoding,
    )

    ncs = NCS()

    block = nss_parser.parse(source, source_name=source_name, debug=debug)
    block.compile(ncs)

    optimizers = list(optimizers or [])
    if not any(isinstance(optimizer, RemoveUnusedBlocksOptimizer) for optimizer in optimizers):
        optimizers.insert(0, RemoveUnusedBlocksOptimizer())
    if not any(isinstance(optimizer, RemoveNopOptimizer) for optimizer in optimizers):
        dead_code_index = next(
            (i for i, optimizer in enumerate(optimizers) if isinstance(optimizer, RemoveUnusedBlocksOptimizer)),
            -1,
        )
        optimizers.insert(dead_code_index + 1, RemoveNopOptimizer())

    for optimizer in optimizers:
        optimizer.reset()
    ncs.optimize(optimizers)
    return ncs
