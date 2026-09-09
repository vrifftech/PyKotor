from __future__ import annotations

import os
import pathlib
import tempfile
import shutil

from datetime import datetime
from tkinter import messagebox
from typing import TYPE_CHECKING
from threading import Event

from pykotor.common.misc import Game
from pykotor.resource.formats.tlk import read_tlk, write_tlk
from pykotor.tools.misc import is_mod_file
from pykotor.tools.path import CaseAwarePath
from pykotor.tslpatcher.logger import PatchLogger
from utility.error_handling import universal_simplify_exception
from utility.system.path import Path

if TYPE_CHECKING:
    from pykotor.extract.installation import Installation


# TODO: the aspyr patch contains some required files in the override folder, hardcode them and ignore those here.
def uninstall_all_mods(installation: Installation):
    """Uninstalls all mods from the game.

    What this method really does is delete all the contents of the override folder and delete all .MOD files from
    the modules folder. Then it removes all appended TLK entries using
    the hardcoded number of entries depending on the game. There are 49,265 TLK entries in KOTOR 1, and 136,329 in TSL.
    """
    root_path = installation.path()
    override_path = installation.override_path()
    modules_path = installation.module_path()

    # Remove any TLK changes
    dialog_tlk_path = CaseAwarePath(root_path, "dialog.tlk")
    dialog_tlk = read_tlk(dialog_tlk_path)
    dialog_tlk.entries = dialog_tlk.entries[:49265] if installation.game() == Game.K1 else dialog_tlk.entries[:136329]
    # TODO: With the new Replace TLK syntax, the above TLK reinstall isn't possible anymore.
    # Here, we should write the dialog.tlk and then check it's sha1 hash compared to vanilla.
    # We could keep the vanilla TLK entries in a tlkdefs.py file, similar to our nwscript.nss defs.
    # This implementation would be required regardless in K2 anyway as this function currently isn't determining if the Aspyr patch and/or TSLRCM is installed.
    write_tlk(dialog_tlk, dialog_tlk_path)

    # Remove all override files
    for file_path in override_path.safe_iterdir():
        file_path.unlink()

    # Remove any .MOD files
    for file_path in modules_path.safe_iterdir():
        if is_mod_file(file_path.name):
            file_path.unlink()


class ModUninstaller:
    """A class that provides functionality to uninstall a selected mod using the most recent backup folder created during the last install.

    Args:
    ----
        backups_location_path (Path): The path to the location of the backup folders.
        game_path (Path): The path to the game folder.
        logger (PatchLogger | None, optional): An optional logger object. Defaults to a new PatchLogger.

    Attributes:
    ----------
        backups_location_path (Path): The path to the location of the backup folders.
        game_path (Path): The path to the game folder.
        log (PatchLogger): The logger object.

    Methods:
    -------
        is_valid_backup_folder(folder: Path, datetime_pattern="%Y-%m-%d_%H.%M.%S") -> bool:
            Check if a folder name is a valid backup folder name based on a datetime pattern.
        get_most_recent_backup(backup_folder_path: Path) -> Path | None:
            Returns the most recent valid backup folder.
        restore_backup(backup_folder: Path, existing_files: set[str], files_in_backup: list[Path]):
            Restores a game backup folder to the existing game files.
        get_backup_info() -> tuple[Path | None, set[str], list[Path], int]:
            Get information about the most recent valid backup.
        uninstall_selected_mod():
            Uninstalls the selected mod using the most recent backup folder created during the last install.
    """

    def __init__(self, backups_location_path: Path, game_path: Path, logger: PatchLogger | None = None, *, dialogs=messagebox):
        self.backups_location_path = Path.pathify(backups_location_path)
        self.game_path = Path.pathify(game_path)
        self.log = logger or PatchLogger()
        self.dialogs = dialogs

    @staticmethod
    def is_valid_backup_folder(folder: Path, datetime_pattern="%Y-%m-%d_%H.%M.%S") -> bool:
        """Check if a folder name is valid backup folder name based on datetime pattern.

        Args:
        ----
            folder: Path object of the folder to validate
            datetime_pattern: String pattern to match folder name against (default: "%Y-%m-%d_%H.%M.%S").

        Returns:
        -------
            bool: True if folder name matches datetime pattern, False otherwise

        Processing Logic:
        ----------------
            - Try to parse folder name as datetime string with given pattern
            - Return True if parsing succeeds without error
            - Return False if parsing fails with ValueError
        """
        try:
            datetime.strptime(folder.name, datetime_pattern).astimezone()
        except ValueError:
            return False
        else:
            return True

    @staticmethod
    def get_most_recent_backup(backup_folder: os.PathLike | str) -> Path | None:
        """Find the newest nonempty backup without displaying dialogs during discovery."""
        root = pathlib.Path(backup_folder)
        if not root.is_dir():
            return None
        valid = [p for p in root.iterdir() if p.is_dir() and not p.is_symlink()
                 and ModUninstaller.is_valid_backup_folder(p) and any(p.iterdir())]
        if not valid:
            return None
        return Path(max(valid, key=lambda p: datetime.strptime(p.name, "%Y-%m-%d_%H.%M.%S")))

    def restore_backup(self, backup_folder: Path, existing_files: set[str], files_in_backup: list[Path], *, should_cancel: Event | None = None):
        """Validate all paths first, retain the backup, and commit one complete file at a time."""
        root = pathlib.Path(backup_folder).resolve()
        game = pathlib.Path(self.game_path).resolve()
        copies = []
        destinations = set()
        for file in files_in_backup:
            source = pathlib.Path(file)
            if source == root / "remove these files.txt":
                continue
            source.resolve().relative_to(root)
            relative = source.relative_to(root)
            destination = pathlib.Path(CaseAwarePath.get_case_sensitive_path(game / relative))
            destination.resolve().relative_to(game)
            if not source.is_file():
                raise FileNotFoundError(source)
            copies.append((source, destination))
            destinations.add(destination.resolve())
        removals = []
        for filename in existing_files:
            target = pathlib.Path(filename)
            if not target.is_absolute():
                raise ValueError(f"Invalid removal path: {filename}")
            target.resolve().relative_to(game)
            if target.exists() and not target.is_file():
                raise ValueError(f"Removal target is not a file: {target}")
            if target.resolve() not in destinations:
                removals.append(target)
        # Restore old files before deleting newly installed ones. If any step fails,
        # the caller reports partial restoration and never offers to delete the backup.
        for source, destination in copies:
            if should_cancel is not None and should_cancel.is_set():
                raise InterruptedError("Restoration stopped between files; the backup is retained.")
            source.resolve().relative_to(root)
            destination.resolve().relative_to(game)
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".restore-", suffix=".tmp", delete=False) as stream:
                    temporary = pathlib.Path(stream.name)
                shutil.copy2(source, temporary)
                destination.resolve().relative_to(game)
                os.replace(temporary, destination)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
            self.log.add_note(f"Restored '{destination.relative_to(game)}'.")
        for target in removals:
            if should_cancel is not None and should_cancel.is_set():
                raise InterruptedError("Restoration stopped between files; the backup is retained.")
            target.resolve().relative_to(game)
            target.unlink(missing_ok=True)
            self.log.add_note(f"Removed '{target.relative_to(game)}'.")

    def get_backup_info(self) -> tuple[Path | None, set[str], list[Path], int]:
        folder = self.get_most_recent_backup(self.backups_location_path)
        if folder is None:
            self.dialogs.showerror("No backups found", f"No nonempty HoloPatcher backup exists at '{self.backups_location_path}'. No game files were changed.")
            return None, set(), [], 0
        root = pathlib.Path(folder).resolve()
        delete_list = root / "remove these files.txt"
        files_to_delete = set()
        if delete_list.is_file():
            # Reading backup metadata must not follow a symlink outside the backup.
            delete_list.resolve().relative_to(root)
            files_to_delete = {line.strip() for line in delete_list.read_text(encoding="utf-8").splitlines() if line.strip()}
        game_root = pathlib.Path(self.game_path).resolve()
        for name in files_to_delete:
            target = pathlib.Path(name)
            if not target.is_absolute():
                raise ValueError(f"Invalid removal path in backup: {name}")
            target.resolve().relative_to(game_root)
        existing = {name for name in files_to_delete if pathlib.Path(name).is_file()}
        if len(existing) != len(files_to_delete) and not self.dialogs.askyesno(
            "Backup out of date or mismatched",
            "Some files listed by this backup are absent. Restore mods in reverse installation order, and verify the selected package and game directory. Continue?",
        ):
            return None, set(), [], 0
        files = []
        folder_count = 0
        for directory, folders, names in os.walk(root, followlinks=False):
            for name in folders:
                child = pathlib.Path(directory, name)
                if child.is_symlink():
                    raise ValueError(f"Backup directory cannot be a symbolic link: {child}")
            folder_count += len(folders)
            for name in names:
                item = pathlib.Path(directory, name)
                item.resolve().relative_to(root)
                if item != delete_list:
                    files.append(Path(item))
        return folder, existing, files, folder_count

    def uninstall_selected_mod(self, *, should_cancel: Event | None = None) -> bool:
        """Return failure immediately after any restore error; retain failed backups."""
        try:
            folder, existing, files, folder_count = self.get_backup_info()
            if folder is None:
                return False
            self.log.add_note(f"Using backup '{folder}'.")
            if should_cancel is not None and should_cancel.is_set():
                return False
            if not self.dialogs.askyesno(
                "Restore this backup?",
                f"Restore {len(files)} files and remove {len(existing)} installed files in '{self.game_path}'?\n\n"
                f"Backup: {folder}\nUninstall in reverse installation order. This selects the most recent package backup, not a namespace-specific backup.",
            ):
                return False
            self.restore_backup(folder, existing, files, should_cancel=should_cancel)
        except Exception as exc:
            title, message = universal_simplify_exception(exc)
            # The error dialog must still be shown if the operation logger is unavailable.
            self.dialogs.showerror(title, f"Backup restoration did not complete. Some earlier files may already have been restored.\n\n{message}\n\nThe backup is retained. Do not delete it.")
            return False
        if should_cancel is not None and should_cancel.is_set():
            return True  # Restoration finished; cancellation still prevents backup removal.
        if self.dialogs.askyesno(
            "Uninstall completed",
            f"Successfully restored backup '{folder.name}'. Delete this restored backup?",
        ):
            try:
                shutil.rmtree(folder)
            except OSError as exc:
                self.dialogs.showwarning("Backup retained", f"Restoration succeeded, but the backup could not be deleted: {exc}")
        return True
