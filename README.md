# PyKotor

Shared resource libraries and the audited HoloPatcher backend for Knights of the
Old Republic and The Sith Lords. HoloPatcher **2.0b** and Holocron Toolset
**4.0.0b** remain separate repositories; no frontend is embedded here.

## Workspace and distributions

| Directory | Distribution | Import namespace |
|---|---|---|
| `Libraries/Utility` | `PyKotorUtility` | `utility` |
| `Libraries/PyKotor` | `PyKotor` | `pykotor` core packages |
| `Libraries/PyKotorGL` | `PyKotorGL` | `pykotor.gl` |
| `Libraries/PyKotorFont` | `PyKotorFont` | `pykotor.font` |

All four distributions retain version `1.7`; the exact audited contents are
identified by `SOURCE_SNAPSHOT.json` and `SOURCE_FILES.sha256`, not that inherited
number alone. Build components from the same snapshot. The runtime import
names and frontend backend-selection paths are unchanged.

Python **3.10 or newer** is required. CI targets 3.10–3.13 on Windows, Linux and
macOS for the core suite. That configured matrix is not a claim that every native
GUI/platform combination was executed during this cleanup.

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

### Optional local font fixtures

The retained font tests live under `Libraries/PyKotorFont/src/tests`. Supply your
own licensed fonts at the `files/roboto/Roboto-Black.ttf` and
`files/TH Sarabun New Regular/TH Sarabun New Regular.ttf` paths relative to that
file. No font binaries are included. The tests now use file-relative inputs and
temporary outputs, not a global working-directory change or Windows font path.
They are manual fixture tests, outside the root core-suite path; no skips or test
assertions were added or weakened during cleanup.

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

`PyKotor` declares its `PyKotorUtility` dependency; GL and Font declare the matching
core dependency. Do not obtain a different same-version backend from an index and
assume it has these audited changes. `secure_xml`, `encodings`, `images` and `font`
are core optional extras; GL's `accelerate` extra remains optional.

## Standalone applications

Set `PYKOTOR_ROOT` to this repository and run either frontend's `run.py --backend-info` to verify selection. Application-specific
`HOLOPATCHER_PYKOTOR_ROOT` and `HOLOCRON_PYKOTOR_ROOT` overrides take precedence.
The frontends deliberately consume the audited source checkout (and embed it at
PyInstaller build time) rather than silently choosing an installed copy.

The selected-backup restorer and all audited file-format implementations remain.
The unused destructive `uninstall_all_mods()` helper and unused Utility updater/GUI
frameworks have been retired. See `CLEANUP_NOTES.md` for exact scope.

## License

Existing license texts and author attribution are retained in the root and each
library directory. This cleanup does not change their terms.
