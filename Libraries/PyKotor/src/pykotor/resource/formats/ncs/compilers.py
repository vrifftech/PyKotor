from __future__ import annotations

import subprocess
from collections.abc import Sequence
from datetime import date
from enum import Enum
from typing import TYPE_CHECKING, NamedTuple

from pykotor.common.misc import Game
from pykotor.common.stream import BinaryReader
from pykotor.resource.formats.ncs.compiler.classes import EntryPointError
from pykotor.resource.formats.ncs.ncs_auto import compile_nss, write_ncs
from pykotor.resource.formats.ncs.ncs_data import NCSCompiler
from utility.misc import generate_hash
from utility.system.path import Path

if TYPE_CHECKING:
    import os

    from subprocess import CompletedProcess

    from pykotor.resource.formats.ncs.ncs_data import NCS, NCSOptimizer


class InbuiltNCSCompiler(NCSCompiler):
    def compile_script(  # noqa: PLR0913
        self,
        source_path: os.PathLike | str,
        output_path: os.PathLike | str,
        game: Game,
        optimizers: list[NCSOptimizer] | None = None,
        *,
        debug: bool = False,
        source_encoding: str | None = None,
    ):
        source_filepath: Path = Path.pathify(source_path)
        nss_data: bytes = BinaryReader.load_file(source_filepath)
        ncs: NCS = compile_nss(
            nss_data,
            game,
            optimizers,
            library_lookup=[source_filepath.parent],
            debug=debug,
            source_name=str(source_filepath),
            source_encoding=source_encoding,
        )
        write_ncs(ncs, output_path)


class ExternalCompilerConfig(NamedTuple):
    sha256: str
    name: str
    release_date: date
    author: str
    commandline: dict[str, list[str]]


class KnownExternalCompilers(Enum):
    TSLPATCHER = ExternalCompilerConfig(
        sha256="539EB689D2E0D3751AEED273385865278BEF6696C46BC0CAB116B40C3B2FE820",
        name="TSLPatcher",
        release_date=date(2009, 1, 1),
        author="todo",
        commandline={
            "compile": ["-c", "{source}", "-o", "{output}"],
            "decompile": ["-d", "{source}", "-o", "{output}"],
        },
    )
    KOTOR_TOOL = ExternalCompilerConfig(
        sha256="E36AA3172173B654AE20379888EDDC9CF45C62FBEB7AB05061C57B52961C824D",
        name="KOTOR Tool",
        release_date=date(2005, 1, 1),
        author="Fred Tetra",
        commandline={
            "compile": ["-c", "--outputdir", "{output_dir}", "-o", "{output_name}", "-g", "{game_value}", "{source}"],
            "decompile": ["-d", "--outputdir", "{output_dir}", "-o", "{output_name}", "-g", "{game_value}", "{source}"],
        },
    )
    V1 = ExternalCompilerConfig(
        sha256="EC3E657C18A32AD13D28DA0AA3A77911B32D9661EA83CF0D9BCE02E1C4D8499D",
        name="v1.3 first public release",
        release_date=date(2003, 12, 31),
        author="todo",
        commandline={
            "compile": ["-c", "{source}", "{output}"],
            "decompile": ["-d", "{source}", "{output}"],
        },
    )
    KOTOR_SCRIPTING_TOOL = ExternalCompilerConfig(
        sha256="B7344408A47BE8780816CF68F5A171A09640AB47AD1A905B7F87DE30A50A0A92",
        name="KOTOR Scripting Tool",
        release_date=date(2016, 5, 18),
        author="James Goad",
        commandline={
            "compile": ["-c", "--outputdir", "{output_dir}", "-o", "{output_name}", "-g", "{game_value}", "{source}"],
            "decompile": ["-d", "--outputdir", "{output_dir}", "-o", "{output_name}", "-g", "{game_value}", "{source}"],
        },
    )
    DENCS = ExternalCompilerConfig(
        sha256="539EB689D2E0D3751AEED273385865278BEF6696C46BC0CAB116B40C3B2FE820",
        name="DeNCS",
        release_date=date(2006, 5, 30),
        author="todo",
        commandline={},
    )
    XOREOS = ExternalCompilerConfig(
        sha256="todo",
        name="Xoreos Tools",
        release_date=date(1, 1, 1),
        author="todo",
        commandline={},
    )
    KNSSCOMP = ExternalCompilerConfig(
        sha256="todo",
        name="knsscomp",
        release_date=date(1, 1, 1),
        author="Nick Hugi",
        commandline={},
    )

    @classmethod
    def from_sha256(cls: type[KnownExternalCompilers], sha256: str) -> KnownExternalCompilers:
        uppercase_sha256: str = sha256.upper()
        for known_ext_compiler in cls:
            if known_ext_compiler.value.sha256 == uppercase_sha256:
                return known_ext_compiler

        msg = f"No compilers found with sha256 hash '{uppercase_sha256}'"
        raise ValueError(msg)


class NwnnsscompConfig:
    """Command-line options for a script tool invocation."""

    DEFAULT_COMMANDLINE = {
        "compile": ["-c", "{source}", "-o", "{output}"],
        "decompile": ["-d", "{source}", "-o", "{output}"],
    }

    def __init__(
        self,
        sha256_hash: str,
        sourcefile: Path,
        outputfile: Path,
        game: Game,
    ):
        self.sha256_hash: str = sha256_hash
        self.source_file: Path = sourcefile
        self.output_file: Path = outputfile
        self.output_dir: Path = outputfile.parent
        self.output_name: str = outputfile.name
        self.game: Game = game

        try:
            self.chosen_compiler: KnownExternalCompilers | None = KnownExternalCompilers.from_sha256(self.sha256_hash)
        except ValueError:
            self.chosen_compiler = None

    def get_compile_args(
        self,
        executable: str,
        extra_args: Sequence[str] = (),
    ) -> list[str]:
        args = self._format_args(self._commandline("compile"), executable)
        args[1:1] = extra_args
        return args

    def get_decompile_args(self, executable: str) -> list[str]:
        return self._format_args(self._commandline("decompile"), executable)

    def _commandline(self, operation: str) -> list[str]:
        if self.chosen_compiler is not None:
            commandline = self.chosen_compiler.value.commandline.get(operation)
            if commandline:
                return commandline
        return self.DEFAULT_COMMANDLINE[operation]

    def _format_args(self, args_list: list[str], executable: str) -> list[str]:
        formatted_args: list[str] = [
            arg.format(
                source=self.source_file,
                output=self.output_file,
                output_dir=self.output_dir,
                output_name=self.output_name,
                game_value="1" if self.game.is_k1() else "2",
            )
            for arg in args_list
        ]
        formatted_args.insert(0, executable)
        return formatted_args


class ExternalNCSCompiler(NCSCompiler):
    def __init__(self, nwnnsscomp_path: os.PathLike | str):
        self.nwnnsscomp_path: Path
        self.filehash: str
        self.change_nwnnsscomp_path(nwnnsscomp_path)

    def get_info(self) -> KnownExternalCompilers:
        return KnownExternalCompilers.from_sha256(self.filehash)

    def change_nwnnsscomp_path(self, nwnnsscomp_path: os.PathLike | str):
        self.nwnnsscomp_path = Path(nwnnsscomp_path)
        self.filehash: str = generate_hash(self.nwnnsscomp_path, hash_algo="sha256").upper()

    def config(
        self,
        source_file: os.PathLike | str,
        output_file: os.PathLike | str,
        game: Game | int,
        *,
        debug: bool = False,
    ) -> NwnnsscompConfig:
        """Resolve paths and build command-line options."""
        source_filepath = Path(source_file).absolute()
        output_filepath = Path(output_file).absolute()
        if not isinstance(game, Game):
            game = Game(game)
        return NwnnsscompConfig(self.filehash, source_filepath, output_filepath, game)

    def compile_script(
        self,
        source_file: os.PathLike | str,
        output_file: os.PathLike | str,
        game: Game | int,
        timeout: int = 60,
        *,
        extra_args: Sequence[str] = (),
        debug: bool = False,
    ) -> tuple[str, str]:
        """Compile a script and return its standard output and error text."""
        config: NwnnsscompConfig = self.config(source_file, output_file, game)

        result: CompletedProcess[str] = subprocess.run(
            args=config.get_compile_args(str(self.nwnnsscomp_path), extra_args),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            cwd=self.nwnnsscomp_path.parent,
        )

        stdout, stderr = self._get_output(result)
        if "File is an include file, ignored" in stdout:
            msg = "This file has no entry point and cannot be compiled (Most likely an include file)."
            raise EntryPointError(msg)

        if result.returncode != 0:
            diagnostics = stderr.strip() or stdout.strip() or f"return code {result.returncode}"
            raise RuntimeError(f"External script compiler failed: {diagnostics}")

        output_filepath = config.output_file
        if not output_filepath.safe_isfile():
            raise FileNotFoundError(f"External script compiler did not create '{output_filepath}'.")
        if output_filepath.stat().st_size == 0:
            raise ValueError(f"External script compiler created an empty output file at '{output_filepath}'.")

        return stdout, stderr

    def decompile_script(
        self,
        source_file: os.PathLike | str,
        output_file: os.PathLike | str,
        game: Game | int,
        timeout: int = 60,
    ) -> tuple[str, str]:
        """Disassemble a script to the requested output path."""
        config: NwnnsscompConfig = self.config(source_file, output_file, game)

        result: CompletedProcess[str] = subprocess.run(
            args=config.get_decompile_args(str(self.nwnnsscomp_path)),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            cwd=self.nwnnsscomp_path.parent,
        )
        stdout, stderr = self._get_output(result)
        if result.returncode != 0:
            diagnostics = stderr.strip() or stdout.strip() or f"return code {result.returncode}"
            raise RuntimeError(f"External script decompiler failed: {diagnostics}")

        output_filepath = config.output_file
        if not output_filepath.safe_isfile():
            raise FileNotFoundError(f"External script decompiler did not create '{output_filepath}'.")
        if output_filepath.stat().st_size == 0:
            raise ValueError(f"External script decompiler created an empty output file at '{output_filepath}'.")
        return stdout, stderr

    def _get_output(self, result: CompletedProcess[str]) -> tuple[str, str]:
        stdout: str = result.stdout
        stderr: str = (
            f"No error provided, but return code is nonzero: ({result.returncode})"
            if result.returncode != 0 and (not result.stderr or not result.stderr.strip())
            else result.stderr
        )

        if "Error:" in stdout:
            stdout_lines: list[str] = stdout.split("\n")
            error_line: str = ""
            filtered_stdout_lines: list[str] = []
            for line in stdout_lines:
                if "Error:" in line:
                    error_line += "\n" + line
                else:
                    filtered_stdout_lines.append(line)

            stdout = "\n".join(filtered_stdout_lines)

            if error_line:
                if stderr:
                    stderr += "\n" + error_line
                else:
                    stderr = error_line
        return stdout, stderr
