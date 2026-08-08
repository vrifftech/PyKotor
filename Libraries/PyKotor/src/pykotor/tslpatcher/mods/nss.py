"""Represents patches specific to [CompileList] logic."""

from __future__ import annotations

import os
import re
import shlex

from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

from pykotor.common.stream import BinaryReader, BinaryWriter
from pykotor.resource.formats.ncs import (
    bytes_ncs,
    compile_nss as compile_with_builtin,
)
from pykotor.resource.formats.ncs.compiler.classes import EntryPointError
from pykotor.resource.formats.ncs.compilers import ExternalNCSCompiler
from pykotor.tools.encoding import decode_bytes_with_fallbacks
from pykotor.tools.path import CaseAwarePath
from pykotor.tslpatcher.mods.template import PatcherModifications
from utility.error_handling import universal_simplify_exception
from utility.system.path import Path, PurePath, PureWindowsPath

if TYPE_CHECKING:
    from typing_extensions import Literal

    from pykotor.common.misc import Game
    from pykotor.resource.formats.ncs.ncs_data import NCS
    from pykotor.resource.type import SOURCE_TYPES
    from pykotor.tslpatcher.logger import PatchLogger
    from pykotor.tslpatcher.memory import PatcherMemory


class MutableString:
    def __init__(self, value: str):
        self.value: str = value

    def __str__(self):
        return self.value


class ModificationsNSS(PatcherModifications):
    def __init__(self, filename, replace=None, modifiers=None):
        super().__init__(filename, replace, modifiers)
        self.saveas = str(PurePath(filename).with_suffix(".ncs"))
        self.action: str = "Compile"
        self.nwnnsscomp_path: Path | None = None
        self.temp_script_folder: Path | None = None
        self.compiler_flags: str = ""
        self.skip_if_not_replace = True

    def patch_resource(
        self,
        nss_source: SOURCE_TYPES,
        memory: PatcherMemory,
        logger: PatchLogger,
        game: Game,
    ) -> bytes | Literal[True]:
        """Takes the source nss bytes and replaces instances of 2DAMEMORY# and StrRef# with the values in patcher memory. Compiles the
        source bytes and returns the ncs compiled script as a bytes object.

        Args:
        ----
            nss_source: SOURCE_TYPES: NSS source object to apply modifications to
            memory: (PatcherMemory): current memory of 2damemory and strref
            logger (PatchLogger): Logging object
            game (Game): KOTOR Game enum value

        Returns:
        -------
            bytes: Compiled NCS bytes

        Processing Logic:
        ----------------
            1. Loads NSS source bytes and decodes
            2. Replaces 2DAMEMORY# and StrRef# tokens with values from patcher memory
            3. Attempts to compile with external NWN compiler if on Windows
            4. Falls back to built-in compiler if external isn't available, fails, or not on Windows
        """
        with BinaryReader.from_auto(nss_source) as reader:
            nss_bytes: bytes = reader.read_all()
        if nss_bytes is None:
            logger.add_error("Invalid nss source provided to ModificationsNSS.apply()")
            return True

        # Replace memory tokens in the script, and save to the file.
        source = MutableString(decode_bytes_with_fallbacks(nss_bytes))
        self.apply(source, memory, logger, game)
        if self.temp_script_folder is None:
            raise RuntimeError("CompileList working directory was not prepared before compilation.")
        temp_script_file = self.temp_script_folder / PureWindowsPath(self.sourcefile).name.lower()

        BinaryWriter.dump(temp_script_file, source.value.encode(encoding="windows-1252", errors="ignore"))

        # Compile with external on windows, fall back to built-in if mac/linux or if external fails.
        is_windows = os.name == "nt"
        nwnnsscomp_exists = self.nwnnsscomp_path is not None and self.nwnnsscomp_path.safe_isfile()
        if is_windows and self.nwnnsscomp_path and nwnnsscomp_exists:
            nwnnsscompiler = ExternalNCSCompiler(self.nwnnsscomp_path)
            try:
                detected_nwnnsscomp: str = nwnnsscompiler.get_info().name
            except ValueError:
                detected_nwnnsscomp: str = "<UNKNOWN>"
            if detected_nwnnsscomp != "TSLPATCHER":
                logger.add_warning(
                    "The nwnnsscomp.exe in the tslpatchdata folder is not the expected TSLPatcher version.\n"
                    f"PyKotor has detected that the provided nwnnsscomp.exe is the '{detected_nwnnsscomp}' version.\n"
                    "PyKotor will compile regardless, but this may not yield the expected result.",
                )
            try:
                return self._compile_with_external(temp_script_file, nwnnsscompiler, logger, game)
            except Exception as e:  # pylint: disable=W0718  # noqa: BLE001
                logger.add_error(str(universal_simplify_exception(e)))

        if is_windows:
            if not self.nwnnsscomp_path or not nwnnsscomp_exists:
                logger.add_note("nwnnsscomp.exe was not found in the 'tslpatchdata' folder, using the built-in compilers...")
            else:
                logger.add_error(f"An error occurred while compiling '{self.sourcefile}' with nwnnsscomp.exe, falling back to the built-in compilers...")
        else:
            logger.add_note(f"Patching from a unix operating system, compiling '{self.sourcefile}' using the built-in compilers...")

        # Compile using built-in script compiler if external compiler fails.
        try:
            ncs: NCS = compile_with_builtin(
                source.value,
                game,
                [],  # [RemoveNopOptimizer(), RemoveMoveSPEqualsZeroOptimizer(), RemoveUnusedBlocksOptimizer()],  # TODO: ncs optimizers need testing
                library_lookup=[CaseAwarePath.pathify(self.temp_script_folder)],
            )
        except EntryPointError as e:
            logger.add_note(str(e))
            return True
        return bytes(bytes_ncs(ncs))

    def apply(
        self,
        nss_source: MutableString,
        memory: PatcherMemory,
        logger: PatchLogger,
        game: Game,
    ):
        """Applies memory patches to the source script.

        Args:
        ----
            nss_source: A mutable string. This function can't return anything in order to stay compatible with the superclass so we modify it in place.
            memory: PatcherMemory object containing StrRef and 2DAMEMORY tokens from earlier patches.
            logger: PatchLogger object for logging.
            game: Game enum representing the kotor game being patched. Not used here anymore, but is provided for backwards compatibility reasons.

        Processing Logic:
        ----------------
            - Replaces defined #2DAMEMORY# tokens with their stored values
            - Replaces defined #StrRef# tokens with their stored string references
            - Leaves undefined tokens unchanged and logs a warning
        """
        def replace_tokens(token_name: str, memory_dict: dict[int, Any]) -> None:
            search_pattern = re.compile(rf"#{token_name}([0-9]+)#")
            for token_id in sorted(memory_dict):
                token = f"#{token_name}{token_id}#"
                if token not in nss_source.value:
                    continue

                replacement_value = memory_dict[token_id]
                if isinstance(replacement_value, PureWindowsPath):
                    replacement_value = str(replacement_value)

                logger.add_verbose(f"{self.sourcefile}: Replacing '{token}' with '{replacement_value}'")
                nss_source.value = nss_source.value.replace(token, str(replacement_value))

            undefined_tokens = {
                int(match.group(1))
                for match in search_pattern.finditer(nss_source.value)
                if int(match.group(1)) not in memory_dict
            }
            for token_id in sorted(undefined_tokens):
                logger.add_warning(
                    f"{token_name}{token_id} was not defined before use in '{self.sourcefile}'; "
                    "leaving token unchanged.",
                )

        replace_tokens("2DAMEMORY", memory.memory_2da)
        replace_tokens("StrRef", memory.memory_str)

    def _compile_with_external(
        self,
        temp_script_file: Path,
        nwnnsscompiler: ExternalNCSCompiler,
        logger: PatchLogger,
        game: Game,
    ) -> bytes | Literal[True]:
        with TemporaryDirectory() as tempdir:
            tempcompiled_filepath: Path = Path(tempdir, "temp_script.ncs")
            try:
                compiler_flags = shlex.split(self.compiler_flags)
            except (TypeError, ValueError) as exc:
                logger.add_error(f"Invalid ScriptCompilerFlags value '{self.compiler_flags}': {exc}")
                compiler_flags = []
            stdout, stderr = nwnnsscompiler.compile_script(
                temp_script_file,
                tempcompiled_filepath,
                game,
                extra_args=compiler_flags,
            )
            result: bool | bytes = "File is an include file, ignored" in stdout
            if not result:
                # Return the compiled bytes
                result = BinaryReader.load_file(tempcompiled_filepath)

        # Parse the output.
        if stdout.strip():
            for line in stdout.split("\n"):
                if line.strip():
                    logger.add_verbose(line)
        if stderr.strip():
            for line in stderr.split("\n"):
                if line.strip():
                    logger.add_error(f"nwnnsscomp error: {line}")

        return result
