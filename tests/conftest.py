"""Test-wide shims for optional third-party dependencies.

The repository modules import `requests` at module import time, but this
execution environment may not have it installed. Provide a tiny stub so unit
tests can import project modules that do not actually perform HTTP calls.
"""

import sys
import types


if "requests" not in sys.modules:
    req = types.ModuleType("requests")

    class HTTPError(Exception):
        pass

    class Timeout(Exception):
        pass

    class _Exc:
        HTTPError = HTTPError
        Timeout = Timeout

    req.exceptions = _Exc()

    def _not_available(*_args, **_kwargs):  # pragma: no cover
        raise RuntimeError("requests stub called in tests")

    req.get = _not_available
    req.post = _not_available
    sys.modules["requests"] = req


if "nbformat" not in sys.modules:
    nbm = types.ModuleType("nbformat")

    class _V4:
        @staticmethod
        def new_notebook():
            return types.SimpleNamespace(cells=[])

        @staticmethod
        def new_markdown_cell(src):
            return types.SimpleNamespace(cell_type="markdown", source=src)

        @staticmethod
        def new_code_cell(src):
            return types.SimpleNamespace(cell_type="code", source=src)

    nbm.v4 = _V4()
    sys.modules["nbformat"] = nbm
