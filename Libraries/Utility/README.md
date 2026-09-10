# PyKotorUtility

The shared `utility` Python package used by the separately maintained PyKotor,
HoloPatcher and Holocron Toolset repositories. The distribution is named
**PyKotorUtility** to distinguish it from unrelated packages called Utility.
Its import name remains `utility`; no application imports need changing.

It includes the maintained path, platform-dialog, logging, error, tooltip and
RTF/RTE helpers. The unused updater/crypto and alternate GUI communication
frameworks are no longer part of this distribution. Tcl/Tk is provided by the
Python/system installation, not by a pip dependency.

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
