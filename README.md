# PyKotor

PyKotor is the shared Python implementation for reading, modifying, and writing resources used by *Star Wars: Knights of the Old Republic* and *The Sith Lords*.

This repository contains reusable libraries and their tests. End-user applications are maintained separately:

- **HoloPatcher** — TSLPatcher-compatible mod installer.
- **Holocron Toolset** — graphical resource editor.

## Repository layout

```text
Libraries/PyKotor/      Core game-resource library and HoloPatcher backend
Libraries/Utility/      Shared platform and utility code
Libraries/PyKotorGL/    Rendering support used by graphical tools
Libraries/PyKotorFont/  Bitmap-font generation support
tests/                  PyKotor library and patcher tests
```

Application-specific source, build scripts, release workflows, and GUI tests do not belong in this repository.

## Development setup

Create a virtual environment and install the repository development requirements:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
$env:PYTHONPATH = "$PWD\Libraries\PyKotor\src;$PWD\Libraries\Utility\src;$PWD\Libraries\PyKotorGL\src;$PWD\Libraries\PyKotorFont\src"
python -m pytest tests
```

Linux or macOS:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
export PYTHONPATH="$PWD/Libraries/PyKotor/src:$PWD/Libraries/Utility/src:$PWD/Libraries/PyKotorGL/src:$PWD/Libraries/PyKotorFont/src"
python -m pytest tests
```

Each distributable library keeps its own packaging metadata under `Libraries/<project>/`.

## HoloPatcher development

Use the standalone HoloPatcher repository and point it at this checkout with `HOLOPATCHER_PYKOTOR_ROOT`. The selected directory must contain both `Libraries/PyKotor/src/pykotor` and `Libraries/Utility/src/utility`.

## Holocron Toolset development

Use the standalone Holocron Toolset repository. It consumes PyKotor, Utility, PyKotorGL, and PyKotorFont from this repository during source development and executable builds.

## License

See `LICENSE` and the license files in the individual library directories.
