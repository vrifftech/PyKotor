"""Represents patches specific to [CompileList] logic."""

from __future__ import annotations

import os
import re

from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

from pykotor.common.stream import BinaryReader, BinaryWriter
from pykotor.resource.formats.ncs import (
    bytes_ncs,
    compile_nss as compile_with_builtin,
)
from pykotor.resource.formats.ncs.compiler.classes import EntryPointError
from pykotor.resource.formats.ncs.compilers import ExternalNCSCompiler
from pykotor.resource.formats.ncs.compiler.source import decode_nss_source
from pykotor.resource.formats.ncs.string_encoding import encode_ncs_string
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
        """Replace memory tokens in NSS source and return compiled NCS bytes."""
        with BinaryReader.from_auto(nss_source) as reader:
            nss_bytes: bytes = reader.read_all()
        if nss_bytes is None:
            logger.add_error("Invalid nss source provided to ModificationsNSS.apply()")
            return True

        source_text = decode_nss_source(nss_bytes, source_name=self.sourcefile)
        source = MutableString(source_text)
        self.apply(source, memory, logger, game)
        if self.temp_script_folder is None:
            raise RuntimeError("CompileList working directory was not prepared before compilation.")
        temp_script_file = self.temp_script_folder / PureWindowsPath(self.sourcefile).name.lower()

        processed_bytes = nss_bytes if source.value == source_text else encode_ncs_string(source.value)
        BinaryWriter.dump(temp_script_file, processed_bytes)

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
            except EntryPointError as exc:
                logger.add_note(str(exc))
                return True
            except Exception as e:
                logger.add_error(str(universal_simplify_exception(e)))

        if is_windows:
            if not self.nwnnsscomp_path or not nwnnsscomp_exists:
                logger.add_note("nwnnsscomp.exe was not found in the 'tslpatchdata' folder, using the built-in compilers...")
            else:
                logger.add_error(f"An error occurred while compiling '{self.sourcefile}' with nwnnsscomp.exe, falling back to the built-in compilers...")
        else:
            logger.add_note(f"Patching from a unix operating system, compiling '{self.sourcefile}' using the built-in compilers...")

        try:
            ncs: NCS = compile_with_builtin(
                source.value,
                game,
                [],
                library_lookup=[CaseAwarePath.pathify(self.temp_script_folder)],
                source_name=self.sourcefile,
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
        """Replace StrRef and 2DAMEMORY tokens in the mutable source string."""
        def replace_tokens(token_name: str, memory_dict: dict[int, Any]) -> None:
            search_pattern = re.compile(rf"#{token_name}([0-9]+)#")
            highest_token = max(memory_dict, default=0) if token_name == "2DAMEMORY" else 0
            previous_token = -1
            while True:
                token_ids = {
                    int(match.group(1))
                    for match in search_pattern.finditer(nss_source.value)
                    if match.group(1) == str(int(match.group(1)))
                    and int(match.group(1)) > previous_token
                    and (int(match.group(1)) in memory_dict or 1 <= int(match.group(1)) <= highest_token)
                }
                if not token_ids:
                    break
                token_id = min(token_ids)
                previous_token = token_id
                token = f"#{token_name}{token_id}#"
                replacement_value = memory_dict.get(token_id, "")
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

    @staticmethod
    def _split_compiler_flags(flags: str) -> list[str]:
        """Split command-line flags without treating backslashes as POSIX escapes."""
        arguments: list[str] = []
        index = 0
        while index < len(flags):
            if flags[index] in " \t":
                index += 1
                continue
            argument: list[str] = []
            quoted = False
            while index < len(flags) and (quoted or flags[index] not in " \t"):
                backslashes = 0
                while index < len(flags) and flags[index] == "\\":
                    backslashes += 1
                    index += 1
                if index < len(flags) and flags[index] == '"':
                    argument.append("\\" * (backslashes // 2))
                    if backslashes % 2:
                        argument.append('"')
                    elif quoted and index + 1 < len(flags) and flags[index + 1] == '"':
                        argument.append('"')
                        index += 1
                    else:
                        quoted = not quoted
                    index += 1
                else:
                    argument.append("\\" * backslashes)
                    if index < len(flags) and (quoted or flags[index] not in " \t"):
                        argument.append(flags[index])
                        index += 1
            if quoted:
                raise ValueError("Unclosed double quote in ScriptCompilerFlags.")
            arguments.append("".join(argument))
        return arguments

    def _compile_with_external(
        self,
        temp_script_file: Path,
        nwnnsscompiler: ExternalNCSCompiler,
        logger: PatchLogger,
        game: Game,
    ) -> bytes | Literal[True]:
        with TemporaryDirectory() as tempdir:
            tempcompiled_filepath: Path = Path(tempdir, "temp_script.ncs")
            compiler_flags = self._split_compiler_flags(self.compiler_flags)
            stdout, stderr = nwnnsscompiler.compile_script(
                temp_script_file,
                tempcompiled_filepath,
                game,
                extra_args=compiler_flags,
            )
            result: bool | bytes = "File is an include file, ignored" in stdout
            if not result:
                result = BinaryReader.load_file(tempcompiled_filepath)

        if stdout.strip():
            for line in stdout.split("\n"):
                if line.strip():
                    logger.add_verbose(line)
        if stderr.strip():
            for line in stderr.split("\n"):
                if line.strip():
                    logger.add_error(f"nwnnsscomp error: {line}")

        return result
