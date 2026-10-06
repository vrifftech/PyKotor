"""Packaging hooks must be discoverable without importing the GUI or PyInstaller."""
import importlib.util
from pathlib import Path
import runpy
import unittest

ROOT = Path(__file__).resolve().parents[1]
HOOK_DIR = ROOT / "Libraries/PyKotor/src/pykotor/__pyinstaller"


class PyInstallerHookTests(unittest.TestCase):
    def test_hook_directory(self):
        spec = importlib.util.spec_from_file_location("pykotor_packaging_hooks", HOOK_DIR / "__init__.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(module.get_hook_dirs(), [str(HOOK_DIR)])

    def test_ply_grammar_keeps_source(self):
        hook = runpy.run_path(str(HOOK_DIR / "hook-pykotor.resource.formats.ncs.compiler.py"))
        self.assertEqual(hook["module_collection_mode"], "pyz+py")

    def test_wheel_hook_entry_point(self):
        metadata = (ROOT / "Libraries/PyKotor/pyproject.toml").read_text(encoding="utf-8")
        self.assertIn("[project.entry-points.pyinstaller40]", metadata)
        self.assertIn('hook-dirs = "pykotor.__pyinstaller:get_hook_dirs"', metadata)


if __name__ == "__main__":
    unittest.main()
