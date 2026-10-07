# PyKotorGL

The `pykotor.gl` namespace extension used by Holocron Toolset. It requires the
matching PyKotor library plus NumPy, PyOpenGL and PyGLM. The `accelerate` extra
is optional. Setuptools is a build tool, not a rendering/runtime dependency.

This wheel owns only `pykotor.gl` and its children. It does not contain a second
copy of the core `pykotor` packages or replace their namespace initializer.

## Build and installation

Python 3.10 or newer is required. The current CI matrix targets 3.10–3.13;
that matrix is not a claim that every platform/version has been tested here.

From this package directory, use the standard PEP 517 backend:

```text
python -m build --no-isolation
```

Build all required workspace wheels from this same source snapshot and install
them together. For offline installation, use `pip --no-index --find-links` with
the local backend wheels and the required third-party wheels. Do not substitute
an unrelated package-index build for a matching audited backend component.

See the repository root README for the full three-package build sequence. The
license text is retained unchanged in `LICENSE`.
