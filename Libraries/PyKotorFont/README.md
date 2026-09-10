# PyKotorFont

The `pykotor.font` namespace extension for bitmap-font/TXI generation. Its
runtime dependencies are the matching PyKotor distribution and Pillow. It does
not require Django, PLY directly, OpenGL or a GUI framework.

Font binaries are not included. Supply appropriately licensed fonts locally for
font generation and the retained manual font-fixture tests.

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

See the repository root README for the full four-package build sequence. The
license text is retained unchanged in `LICENSE`.
