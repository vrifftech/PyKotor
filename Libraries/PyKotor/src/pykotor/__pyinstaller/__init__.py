"""PyInstaller hooks for both installed PyKotor and source-checkout consumers."""
from pathlib import Path


def get_hook_dirs() -> list[str]:
    """Return hooks without importing GUI packages or PyInstaller itself."""
    return [str(Path(__file__).resolve().parent)]
