# PyKotor

The audited core game-resource library and TSLPatcher-compatible installation
backend. This package retains the engine-verified format and preservation repairs;
it is not the unmodified upstream 1.7 source despite retaining that base version.
See the repository's `SOURCE_SNAPSHOT.json` for exact provenance.

The wheel requires the matching **PyKotorUtility** distribution and PLY for the
NSS compiler. `secure_xml`, `encodings`, `images` and `font` are optional extras.
No graphics or GUI toolkit is required by the core distribution. Individual
platform/Tk helper operations still require their host facilities.

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
