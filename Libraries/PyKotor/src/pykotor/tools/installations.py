"""Read-only desktop installation discovery, separate from installation validation.

Platform probes supply candidates; the shared pipeline resolves data roots and
identifies games without loading their resources. No drives are recursively
searched, no launcher is required to be running, and no global cache is kept.
"""
from __future__ import annotations

import configparser
import json
import os
import platform
import re

from collections import deque
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import TYPE_CHECKING

from pykotor.common.misc import Game

if TYPE_CHECKING:
    from collections.abc import Iterable


@dataclass(frozen=True)
class KotorInstallation:
    """A discovery result, not a guarantee that installing a mod here is safe.

    ``found_at`` can be a storefront directory or application bundle, whereas
    ``root`` contains chitin.key. Conflicting/unidentified games remain None.
    """

    found_at: Path
    root: Path
    game: Game | None
    sources: tuple[str, ...]
    evidence: tuple[str, ...]


_STEAM_APPS = {"32370": Game.K1, "208580": Game.K2}
_GOG_APPS = {"1207666283": Game.K1, "1421404581": Game.K2}
_STEAM_NAMES = ("swkotor", "Knights of the Old Republic II", "kotor2")
_MAX_METADATA_BYTES = 1024 * 1024
_MAX_LIBRARIES = 128
_MAX_ROOT_DIRECTORIES = 64
_MAX_APPLICATIONS = 512
_MAX_PREFIXES = 64
_MAX_REGISTRY_BYTES = 16 * 1024 * 1024
_MAX_SCAN_REGISTRY_BYTES = 64 * 1024 * 1024
_VDF_TOKEN = re.compile(r'\s+|//[^\r\n]*|"(?:\\.|[^"\\])*"|[{}]|[^\s{}"]+')


def _read_vdf(path: Path) -> dict:
    """Read the bounded KeyValues subset used by Steam library/app metadata.

    Handles old numeric library values and modern blocks, comments, BOMs,
    quoted/bare tokens and escaped Windows separators. Bad files are ignored.
    """
    try:
        with path.open("rb") as stream:
            raw = stream.read(_MAX_METADATA_BYTES + 1)
        if len(raw) > _MAX_METADATA_BYTES:
            return {}
        text = raw.decode("utf-8-sig")
        tokens: list[str] = []
        pos = 0
        while pos < len(text):
            match = _VDF_TOKEN.match(text, pos)
            if match is None:
                return {}
            token = match.group()
            if not token.isspace() and not token.startswith("//"):
                tokens.append(token)
            pos = match.end()
        index = 0

        def value(token: str) -> str:
            if token.startswith('"'):
                # Do not use unicode_escape: it corrupts non-ASCII paths and
                # interprets a literal Windows \\t or \\n as whitespace.
                return re.sub(r'\\([\\"])', r'\1', token[1:-1])
            return token

        def block(depth: int = 0) -> dict:
            nonlocal index
            if depth > 32:
                raise ValueError("Metadata nesting limit exceeded")
            result: dict = {}
            while index < len(tokens):
                key = tokens[index]
                index += 1
                if key == "}":
                    if depth == 0:
                        raise ValueError("Unexpected closing brace")
                    return result
                if key == "{" or index >= len(tokens):
                    raise ValueError("Missing key or value")
                token = tokens[index]
                index += 1
                if token == "}":
                    raise ValueError("Missing value")
                name = value(key).casefold()
                if name in result:
                    raise ValueError("Duplicate metadata key")
                result[name] = block(depth + 1) if token == "{" else value(token)
            if depth:
                raise ValueError("Unclosed metadata block")
            return result

        return block()
    except (OSError, UnicodeError, ValueError, RecursionError):
        return {}


class _Scan:
    """Per-scan filesystem cache using real paths, never CaseAwarePath equality."""

    def __init__(self):
        self._entries: dict[Path, tuple[Path, ...]] = {}
        self._directories: dict[str, Path | None] = {}
        self.registry_bytes_left = _MAX_SCAN_REGISTRY_BYTES

    def children(self, directory: Path) -> tuple[Path, ...]:
        if directory not in self._entries:
            try:
                self._entries[directory] = tuple(sorted(directory.iterdir(), key=lambda p: (p.name.casefold(), p.name)))
            except OSError:
                self._entries[directory] = ()
        return self._entries[directory]

    def child(self, directory: Path, name: str) -> Path | None:
        # Most paths already have the right case; avoid enumerating whole
        # parent directories unless the exact lookup fails.
        exact = directory / name
        try:
            if exact.exists():
                return exact
        except (OSError, ValueError):
            return None
        entries = self.children(directory)
        matches = [p for p in entries if p.name.casefold() == name.casefold()]
        # Do not arbitrarily choose between distinct case-sensitive entries.
        return matches[0] if len(matches) == 1 else None

    def directory(self, value: os.PathLike | str) -> Path | None:
        text = os.fspath(value)
        if text in self._directories:
            return self._directories[text]
        result = None
        try:
            path = Path(os.path.expandvars(text)).expanduser()
            if not path.is_absolute():
                return None
            if not path.is_dir():
                current = Path(path.anchor)
                for part in path.parts[1:]:
                    current = self.child(current, part)
                    if current is None:
                        break
                path = current
            if path is not None and path.is_dir():
                result = path.resolve(strict=True)
        except (OSError, RuntimeError, ValueError):
            pass
        self._directories[text] = result
        return result

    def file(self, directory: Path, name: str) -> Path | None:
        path = self.child(directory, name)
        try:
            return path if path is not None and path.is_file() else None
        except OSError:
            return None

    @staticmethod
    def identity(path: Path) -> tuple:
        try:
            info = path.stat()
            if info.st_ino:
                return ("inode", info.st_dev, info.st_ino)
        except OSError:
            pass
        return ("path", os.path.normcase(str(path)))

    def data_roots(self, candidate: Path) -> list[Path]:
        """Bounded layout walk, not a recursive scan of game assets or disks."""
        pending = deque([(candidate, 0)])
        seen: set[tuple] = set()
        roots = []
        # Native bundles plus the historical Aspyr/TransGaming wrapper layout.
        containers = {
            "steamassets", "contents", "assets", "gamedata", "resources",
            "transgaming", "c_drive", "program files", "program files (x86)",
            "lucasarts", "swkotor", "swkotor2", "kotor2", "game",
        }
        while pending and len(seen) < _MAX_ROOT_DIRECTORIES:
            directory, depth = pending.popleft()
            identity = self.identity(directory)
            if identity in seen:
                continue
            seen.add(identity)
            if self.file(directory, "chitin.key") is not None:
                roots.append(directory)
                continue
            if depth >= 9:
                continue
            for child in self.children(directory):
                name = child.name.casefold()
                if name not in containers and not name.endswith(".app"):
                    continue
                try:
                    if not child.is_symlink() and child.is_dir():
                        pending.append((child, depth + 1))
                except OSError:
                    continue
        return roots

    def identify(self, root: Path) -> tuple[set[Game], list[str]]:
        games: set[Game] = set()
        evidence = []
        for game, filenames in (
            (Game.K1, ("swkotor.exe", "swkotor", "kotor1", "swkotor.ini", "goggame-1207666283.info")),
            (Game.K2, ("swkotor2.exe", "swkotor2", "kotor2", "swkotor2.ini", "goggame-1421404581.info")),
        ):
            for name in filenames:
                if self.file(root, name) is not None:
                    games.add(game)
                    evidence.append(f"K{game.value}: {name}")
        # These are weaker layout markers. Use them only without executable,
        # INI or storefront-file evidence; a tie must not default to K1.
        if not games:
            for game, name in ((Game.K1, "streamwaves"), (Game.K2, "streamvoice")):
                child = self.child(root, name)
                if child is not None and self.directory(child) is not None:
                    games.add(game)
                    evidence.append(f"K{game.value}: {name}/")
        return games, evidence


def _steam_roots(system: str) -> list[str]:
    if system == "Windows":
        from pykotor.tools.registry import find_steam_registry_paths

        roots = find_steam_registry_paths()
        for variable, default in (("ProgramFiles(x86)", r"C:\Program Files (x86)"), ("ProgramFiles", r"C:\Program Files")):
            roots.append(str(Path(os.environ.get(variable) or default) / "Steam"))
        return roots
    if system == "Darwin":
        return ["~/Library/Application Support/Steam", "~/Library/Applications/Steam"]
    if system == "Linux":
        xdg = os.environ.get("XDG_DATA_HOME", "")
        roots = [str(Path(xdg) / "Steam")] if xdg and Path(xdg).is_absolute() else []
        return roots + [
            "~/.local/share/Steam", "~/.steam/steam", "~/.steam/root", "~/.steam/debian-installation",
            "~/.var/app/com.valvesoftware.Steam/data/Steam",
            "~/.var/app/com.valvesoftware.Steam/.local/share/Steam",
            "~/snap/steam/common/.local/share/Steam",
        ]
    return []


def _steam_candidates(scan: _Scan, system: str) -> Iterable[tuple[Path, Game | None, str]]:
    libraries: list[Path] = []
    seen: set[tuple] = set()

    def add(value: str | Path):
        if len(libraries) >= _MAX_LIBRARIES:
            return
        library = scan.directory(value)
        if library is not None and scan.identity(library) not in seen:
            seen.add(scan.identity(library))
            libraries.append(library)

    for value in _steam_roots(system):
        add(value)
    # Read both metadata locations, including the legacy numeric-string form.
    # A library may name another library; identity + count bounds prevent loops.
    for library in libraries:
        for folder in ("steamapps", "config"):
            directory = scan.child(library, folder)
            metadata = scan.file(directory, "libraryfolders.vdf") if directory is not None else None
            data = _read_vdf(metadata).get("libraryfolders", {}) if metadata else {}
            if isinstance(data, dict):
                for key, item in data.items():
                    value = item.get("path") if isinstance(item, dict) else item
                    if key.isdecimal() and isinstance(value, str) and value.strip():
                        add(value)

    for library in libraries:
        steamapps = scan.child(library, "steamapps")
        common = scan.child(steamapps, "common") if steamapps is not None else None
        if common is None:
            continue
        for appid, game in _STEAM_APPS.items():
            manifest = scan.file(steamapps, f"appmanifest_{appid}.acf")
            state = _read_vdf(manifest).get("appstate", {}) if manifest else {}
            if not isinstance(state, dict) or state.get("appid") != appid:
                continue
            name = state.get("installdir")
            # Steam supplies a single directory name, not an absolute path or
            # traversal. Symlinked game directories themselves remain supported.
            if not isinstance(name, str) or not name.strip() or name in {".", ".."} or any(c in name for c in '/\\:\x00'):
                continue
            candidate = scan.directory(common / name)
            if candidate is not None:
                yield candidate, game, f"Steam app {appid}: {manifest}"
        # Missing/stale metadata still permits known names, but a guessed name
        # is not evidence of game identity.
        for name in _STEAM_NAMES:
            candidate = scan.directory(common / name)
            if candidate is not None:
                yield candidate, None, f"Steam fallback: {library}"


def _application_candidates(scan: _Scan) -> Iterable[tuple[Path, Game | None, str]]:
    # App Store / drag-and-drop bundles, including renamed bundles and one
    # grouping folder. Do not follow directory symlinks into arbitrary trees.
    pending = deque()
    for value in ("/Applications", "~/Applications"):
        directory = scan.directory(value)
        if directory is not None:
            pending.append((directory, 0))
    seen: set[tuple] = set()
    offered = 0
    while pending and len(seen) < 64 and offered < _MAX_APPLICATIONS:
        directory, depth = pending.popleft()
        identity = scan.identity(directory)
        if identity in seen:
            continue
        seen.add(identity)
        for child in scan.children(directory):
            if offered >= _MAX_APPLICATIONS:
                break
            try:
                if child.is_symlink() or not child.is_dir():
                    continue
                offered += 1
                if child.suffix.casefold() == ".app" or scan.file(child, "chitin.key") is not None:
                    yield child, None, f"Applications: {directory}"
                elif depth == 0:
                    pending.append((child, depth + 1))
            except OSError:
                continue



def _xdg_path(variable: str, default: str) -> Path:
    value = os.environ.get(variable, "")
    return Path(value) if value and Path(value).is_absolute() else Path(default).expanduser()


def _read_json(path: Path) -> dict:
    try:
        with path.open("rb") as stream:
            raw = stream.read(_MAX_METADATA_BYTES + 1)
        if len(raw) > _MAX_METADATA_BYTES:
            return {}
        value = json.loads(raw.decode("utf-8-sig"))
        return value if isinstance(value, dict) else {}
    except (OSError, UnicodeError, ValueError, RecursionError):
        return {}


def _heroic_candidates(scan: _Scan, system: str) -> Iterable[tuple[Path, Game | None, str]]:
    """Read GOG's installed list, not a user's uninstalled/owned library."""
    if system == "Windows":
        configs = [Path(os.environ.get("APPDATA") or Path.home() / "AppData/Roaming") / "heroic"]
    elif system == "Darwin":
        configs = [Path.home() / "Library/Application Support/heroic"]
    elif system == "Linux":
        configs = [
            _xdg_path("XDG_CONFIG_HOME", "~/.config") / "heroic",
            Path.home() / ".var/app/com.heroicgameslauncher.hgl/config/heroic",
        ]
    else:
        return
    for config in configs:
        store = scan.directory(config / "gog_store")
        metadata = scan.file(store, "installed.json") if store is not None else None
        installed = _read_json(metadata).get("installed", []) if metadata else []
        if not isinstance(installed, list):
            continue
        for item in installed:
            if not isinstance(item, dict):
                continue
            game = _GOG_APPS.get(str(item.get("appName", "")))
            value = item.get("install_path")
            if game is not None and isinstance(value, str) and value.strip():
                directory = scan.directory(value)
                if directory is not None:
                    yield directory, game, f"Heroic GOG {item['appName']}: {metadata}"


def _lutris_candidates(scan: _Scan) -> Iterable[tuple[Path, Game | None, str]]:
    """Read installed records (including retail/custom games) without YAML dependencies."""
    import sqlite3
    import time

    config_home = _xdg_path("XDG_CONFIG_HOME", "~/.config")
    data_home = _xdg_path("XDG_DATA_HOME", "~/.local/share")
    roots = [
        (config_home / "lutris", data_home / "lutris"),
        (Path.home() / ".var/app/net.lutris.Lutris/config/lutris",
         Path.home() / ".var/app/net.lutris.Lutris/data/lutris"),
    ]
    databases: list[Path] = []
    for config, data in roots:
        databases.append(data / "pga.db")
        for folder in (config, data):
            parser = configparser.ConfigParser(interpolation=None)
            try:
                with (folder / "lutris.conf").open("r", encoding="utf-8-sig") as stream:
                    text = stream.read(_MAX_METADATA_BYTES + 1)
                if len(text) > _MAX_METADATA_BYTES:
                    continue
                parser.read_string(text)
                for section in parser.values():
                    value = section.get("pga_path")
                    if value:
                        candidate = Path(os.path.expandvars(value)).expanduser()
                        if candidate.is_absolute():
                            databases.append(candidate)
            except (OSError, UnicodeError, ValueError, configparser.Error):
                continue
    for database in dict.fromkeys(databases):
        connection = None
        try:
            if not database.is_file():
                continue
            connection = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True, timeout=0.1)
            connection.execute("PRAGMA query_only=ON")
            deadline = time.monotonic() + 0.5
            connection.set_progress_handler(lambda: int(time.monotonic() > deadline), 1000)
            columns = {row[1] for row in connection.execute("PRAGMA table_info(games)")}
            if not {"installed", "directory"} <= columns:
                continue
            fields = [name for name in ("directory", "executable", "service", "service_id", "name", "slug") if name in columns]
            predicates = []
            for name in ("name", "slug"):
                if name in columns:
                    predicates.extend([f"lower({name}) LIKE '%kotor%'", f"lower({name}) LIKE '%knights%old%republic%'"])
            if {"service", "service_id"} <= columns:
                predicates.append("(service='gog' AND CAST(service_id AS TEXT) IN ('1207666283','1421404581'))")
            if not predicates:
                continue
            query = f"SELECT {', '.join(fields)} FROM games WHERE installed=1 AND ({' OR '.join(predicates)}) LIMIT 128"
            # Materialize and close the read-only DB before filesystem work.
            rows = [dict(zip(fields, row)) for row in connection.execute(query)]
        except (OSError, ValueError, sqlite3.Error):
            continue
        finally:
            if connection is not None:
                connection.close()
        for row in rows:
            value = row.get("directory")
            directory = scan.directory(value) if isinstance(value, str) and value else None
            game = _GOG_APPS.get(str(row.get("service_id"))) if row.get("service") == "gog" else None
            source = f"Lutris: {database}"
            if directory is not None:
                # A prefix may contain games other than the selected record.
                # Apply product identity only to a data-root candidate, not to
                # unrelated installations found via that prefix's registry.
                yield directory, game, source
            executable = row.get("executable")
            if isinstance(executable, str) and executable and not PureWindowsPath(executable).drive:
                target = Path(executable).expanduser()
                if not target.is_absolute() and directory is not None:
                    target = directory / target
                parent = scan.directory(target.parent)
                if parent is not None:
                    yield parent, game, source


def _collection_candidates(scan: _Scan, values: Iterable[os.PathLike | str], source: str) -> Iterable[tuple[Path, Game | None, str]]:
    """Inspect immediate entries in known install collections, never entire disks."""
    seen: set[tuple] = set()
    count = 0
    for value in values:
        directory = scan.directory(value)
        if directory is None or scan.identity(directory) in seen:
            continue
        seen.add(scan.identity(directory))
        for child in scan.children(directory):
            if count >= _MAX_APPLICATIONS:
                return
            count += 1
            candidate = scan.directory(child)
            if candidate is not None:
                yield candidate, None, f"{source}: {directory}"


def _windows_collections(program_files: Iterable[Path], drive: Path) -> list[Path]:
    locations = [drive / "GOG Games", drive / "Games", drive / "Amazon Games/Library"]
    for folder in program_files:
        locations.extend((folder / "LucasArts", folder / "BioWare", folder / "GOG Games", folder / "GOG Galaxy/Games"))
    return locations


def _nonsteam_locations(scan: _Scan, system: str) -> Iterable[tuple[Path, Game | None, str]]:
    if system == "Windows":
        programs = [Path(os.environ.get(variable) or default) for variable, default in (
            ("ProgramFiles", r"C:\Program Files"), ("ProgramFiles(x86)", r"C:\Program Files (x86)"),
            ("ProgramW6432", r"C:\Program Files"),
        )]
        drive = Path((os.environ.get("SystemDrive") or "C:") + "\\")
        locations = _windows_collections(programs, drive)
    elif system in {"Linux", "Darwin"}:
        locations = [Path.home() / name for name in ("Games", "GOG Games", "Games/Heroic", "Games/GOG Games")]
    else:
        return
    yield from _collection_candidates(scan, locations, "Non-Steam install location")


def _wine_unescape(value: str) -> str:
    """Decode Wine's string/key escapes without corrupting literal Unicode."""
    def replace(match: re.Match) -> str:
        token = match.group(1)
        if token.startswith("x"):
            return chr(int(token[1:], 16))
        if token[0] in "01234567":
            return chr(int(token, 8))
        return {"n": "\n", "r": "\r", "t": "\t"}.get(token, token)
    text = re.sub(r'\\(x[0-9a-fA-F]{1,4}|[0-7]{1,3}|[\\"\[\]nrt])', replace, value)
    # Wine represents characters outside the BMP as UTF-16 surrogate pairs.
    return text.encode("utf-16-le", errors="surrogatepass").decode("utf-16-le", errors="replace")


def _wine_path(scan: _Scan, prefix: Path, value: str) -> Path | None:
    """Map an explicitly recorded Windows path through Wine's drive mappings."""
    variables = {"programfiles": r"C:\Program Files", "programfiles(x86)": r"C:\Program Files (x86)", "systemdrive": "C:"}
    value = re.sub(r"%([^%]+)%", lambda m: variables.get(m[1].casefold(), m[0]), value).strip('"')
    if not value or any(c in value for c in "\x00\r\n") or "%" in value:
        return None
    path = PureWindowsPath(value)
    if not path.is_absolute() or len(path.drive) != 2 or path.drive[1] != ":" or ".." in path.parts:
        return None
    drive = scan.directory(prefix / "dosdevices" / path.drive.lower())
    if drive is None and path.drive.casefold() == "c:":
        drive = scan.directory(prefix / "drive_c")
    return scan.directory(drive.joinpath(*path.parts[1:])) if drive is not None else None


def _wine_registry_candidates(scan: _Scan, prefix: Path) -> Iterable[tuple[Path, Game | None, str]]:
    from pykotor.tools.registry import KOTOR_REG_PATHS

    product_keys: dict[str, Game] = {}
    for game, architectures in KOTOR_REG_PATHS.items():
        for entries in architectures.values():
            for key, _ in entries:
                product_keys[key.split("\\", 1)[1].casefold().replace("\\wow6432node", "")] = game
    uninstall = "software\\microsoft\\windows\\currentversion\\uninstall\\"
    product_keys.update({uninstall + product + "_is1": game for product, game in _GOG_APPS.items()})
    product_keys[uninstall + "amazongames/star wars - knights of the old"] = Game.K1
    header = re.compile(r"^\[((?:\\.|[^\]\\])*)\]")
    setting = re.compile(r'^"((?:\\.|[^"\\])*)"=(?:str\(2\):)?"((?:\\.|[^"\\])*)"\s*$')

    for name in ("system.reg", "user.reg"):
        metadata = scan.file(prefix, name)
        if metadata is None or scan.registry_bytes_left <= 0:
            continue
        key = ""
        fields: dict[str, str] = {}

        def emit() -> Iterable[tuple[Path, Game | None, str]]:
            game = product_keys.get(key)
            display = re.sub(r"[^a-z0-9]+", " ", fields.get("displayname", "").casefold())
            if game is None and not (key.startswith(uninstall) and "knights of the old republic" in display):
                return
            for field in ("path", "internalpath", "installlocation"):
                if fields.get(field):
                    directory = _wine_path(scan, prefix, fields[field])
                    if directory is not None:
                        yield directory, game, f"Wine registry: {metadata} [{key}]"

        try:
            consumed = 0
            with metadata.open("rb") as stream:
                # Limit both total bytes and line size, even for malformed data.
                while consumed < _MAX_REGISTRY_BYTES and scan.registry_bytes_left > 0:
                    raw = stream.readline(min(65537, _MAX_REGISTRY_BYTES - consumed, scan.registry_bytes_left))
                    if not raw:
                        yield from emit()
                        break
                    consumed += len(raw)
                    scan.registry_bytes_left -= len(raw)
                    if len(raw) >= 65537:
                        break
                    line = raw.decode("utf-8", errors="replace").strip()
                    match = header.match(line)
                    if match:
                        yield from emit()
                        key = _wine_unescape(match[1]).casefold().replace("\\wow6432node", "")
                        fields = {}
                    elif key in product_keys or key.startswith(uninstall):
                        match = setting.fullmatch(line)
                        if match:
                            field = _wine_unescape(match[1]).casefold()
                            if field in {"path", "internalpath", "installlocation", "displayname"}:
                                fields[field] = _wine_unescape(match[2])
        except (OSError, ValueError):
            continue


def _wine_candidates(scan: _Scan, system: str, seeds: Iterable[tuple[Path, Game | None, str]]) -> Iterable[tuple[Path, Game | None, str]]:
    """GOG offline/retail installs in Wine, Bottles, CrossOver and launcher prefixes.

    Only known prefix containers, launcher-recorded directories and explicit
    registry paths are visited. Wine itself is never invoked.
    """
    if system not in {"Linux", "Darwin"}:
        return
    values = [str(Path.home() / ".wine")]
    if os.environ.get("WINEPREFIX"):
        values.insert(0, os.environ["WINEPREFIX"])
    if system == "Linux":
        containers = [
            _xdg_path("XDG_DATA_HOME", "~/.local/share") / "bottles/bottles",
            Path.home() / ".var/app/com.usebottles.bottles/data/bottles/bottles",
            Path.home() / ".PlayOnLinux/wineprefix", Path.home() / ".cxoffice",
        ]
    else:
        containers = [Path.home() / "Library/Application Support/CrossOver/Bottles", Path.home() / "Library/PlayOnMac/wineprefix"]
    containers.extend(Path.home() / name for name in ("Games/Heroic/Prefixes", "Games/Heroic/Prefixes/default"))
    values.extend(str(path) for path, _, _ in _collection_candidates(scan, containers, "Wine prefix collection"))
    values.extend(str(path) for path, _, _ in seeds)
    seen: set[tuple] = set()
    for value in dict.fromkeys(values):
        directory = scan.directory(value)
        if directory is None:
            continue
        for suffix in ("", "pfx", "prefix", "Contents/Resources", "Contents/Resources/wineprefix", "Contents/SharedSupport/prefix"):
            prefix = scan.directory(directory / suffix)
            if prefix is None:
                continue
            drive = scan.directory(prefix / "drive_c")
            if drive is None and scan.file(prefix, "system.reg") is None and scan.file(prefix, "user.reg") is None:
                continue
            identity = scan.identity(prefix)
            if identity in seen:
                continue
            if len(seen) >= _MAX_PREFIXES:
                return
            seen.add(identity)
            yield from _wine_registry_candidates(scan, prefix)
            if drive is not None:
                programs = [drive / name for name in ("Program Files", "Program Files (x86)")]
                yield from _collection_candidates(scan, _windows_collections(programs, drive), "Wine non-Steam location")
                # Retail installations can sit directly below Program Files.
                for parent in [drive, *programs]:
                    for name in ("SWKotOR", "SWKotOR2", "KotOR", "KotOR2"):
                        candidate = scan.directory(parent / name)
                        if candidate is not None:
                            yield candidate, None, f"Wine retail location: {prefix}"

def discover_kotor_installations() -> list[KotorInstallation]:
    """Discover desktop game-data roots without touching game contents.

    Reads Steam and Heroic GOG metadata, Windows retail/GOG registry entries,
    macOS bundles, Linux Lutris records, Wine prefixes and known locations. Unknown or
    conflicting identity is retained in this detailed API, not guessed. Each
    call is a fresh scan; UI callers should run it off-thread and cache results.
    """
    from pykotor.tools.path import get_default_paths

    system = platform.system()
    scan = _Scan()
    candidates = list(_steam_candidates(scan, system))
    candidates.extend(_heroic_candidates(scan, system))
    candidates.extend(_nonsteam_locations(scan, system))
    if system == "Linux":
        candidates.extend(_lutris_candidates(scan))
    if system == "Windows":
        from pykotor.tools.registry import find_kotor_registry_paths

        for game, value, source in find_kotor_registry_paths():
            directory = scan.directory(value)
            if directory is not None:
                candidates.append((directory, game, source))
    elif system == "Darwin":
        candidates.extend(_application_candidates(scan))
    for values in get_default_paths().get(system, {}).values():
        for value in values:
            directory = scan.directory(value)
            if directory is not None:
                candidates.append((directory, None, "Known location"))

    # Snapshot seeds before extending: prefix discovery must not consume its
    # own output or recursively expand arbitrary launcher directories.
    candidates.extend(_wine_candidates(scan, system, tuple(candidates)))

    roots_cache: dict[tuple, list[Path]] = {}
    found: dict[tuple, dict] = {}
    for candidate, hint, source in candidates:
        identity = scan.identity(candidate)
        if identity not in roots_cache:
            roots_cache[identity] = scan.data_roots(candidate)
        for root in roots_cache[identity]:
            key = scan.identity(root)
            if key not in found:
                games, evidence = scan.identify(root)
                found[key] = {
                    "found_at": candidate, "root": root, "games": games,
                    "sources": set(), "evidence": set(evidence),
                }
            entry = found[key]
            entry["sources"].add(source)
            if hint is not None:
                entry["games"].add(hint)
                entry["evidence"].add(f"K{hint.value}: {source}")

    results = []
    for entry in found.values():
        games = entry["games"]
        evidence = entry["evidence"]
        if len(games) > 1:
            evidence.add("Conflicting game identity; manual selection required")
        elif not games:
            evidence.add("Data root found, but game identity is unknown")
        results.append(KotorInstallation(
            entry["found_at"], entry["root"], next(iter(games)) if len(games) == 1 else None,
            tuple(sorted(entry["sources"])), tuple(sorted(evidence)),
        ))
    return sorted(results, key=lambda result: (result.game.value if result.game else 99, str(result.root).casefold(), str(result.root)))
