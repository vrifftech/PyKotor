# PyKotor

Shared resource libraries and the HoloPatcher backend for Knights of the
Old Republic and The Sith Lords.

Python **3.10 or newer** is required. CI targets 3.10–3.13 on Windows, Linux and
macOS for the core suite.

## Source development

Create/activate a virtual environment, then install `requirements-dev.txt`.
To work offline, provide the dependency wheels locally:

```text
python -m pip install --no-index --find-links /path/to/wheels -r requirements-dev.txt
python -m pytest tests -ra
python -m ruff check Libraries tests
```

Pytest registers the four source directories through the root `pyproject.toml`.
Ruff is the only configured formatter/linter. Its default gate checks syntax-level
errors without rewriting legacy style. `ruff format` may be used deliberately on
files being edited; a whole-repository formatting diff is not part of cleanup.
Type stubs and `typing-extensions` remain available for development.

The core CI job installs only the core/decoding test requirements. The separate
wheel job builds all four libraries and installs their actual runtime dependencies.
It does not silently skip failing tests or use a source-directory `.pth` to conceal
missing wheel contents.


## Building distributable libraries

With `build`, setuptools and wheel already installed, run from this root:

```text
python -m build --no-isolation --outdir dist Libraries/Utility
python -m build --no-isolation --outdir dist Libraries/PyKotor
python -m build --no-isolation --outdir dist Libraries/PyKotorGL
python -m build --no-isolation --outdir dist Libraries/PyKotorFont
```

The commands use each library's explicit PEP 517/621 metadata, not a custom
`setup.py` parser. No GUI/test packages are included in the core wheels, and the
core/GL/font wheels do not overlap in owned files.

For offline installation into a separate environment, include both these four
matching wheels and the necessary third-party wheels in the local wheel directory:

```text
python -m pip install --no-index --find-links /path/to/wheels PyKotor PyKotorGL PyKotorFont
python -m pip check
```

## Standalone applications

Set `PYKOTOR_ROOT` to this repository and run either frontend's `run.py --backend-info` to verify selection. Application-specific
`HOLOPATCHER_PYKOTOR_ROOT` and `HOLOCRON_PYKOTOR_ROOT` overrides take precedence.
The frontends deliberately consume the audited source checkout (and embed it at
PyInstaller build time) rather than silently choosing an installed copy.

