"""Test the sgnts __init__ module."""

import sys


def test_version_import_error(monkeypatch):
    """Test fallback version when _version module is not available."""
    # Remove sgnts from sys.modules to force reimport
    modules_to_remove = [m for m in sys.modules if m.startswith("sgnts")]
    for module in modules_to_remove:
        del sys.modules[module]

    # Mock importlib.import_module to raise ImportError for _version
    import importlib

    original_import = importlib.import_module

    def mock_import(name, *args, **kwargs):
        if name == "._version" and args and args[0] == "sgnts":
            raise ImportError("No module named 'sgnts._version'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", mock_import)

    # Also need to handle the 'from' import statement
    import builtins

    original_builtins_import = builtins.__import__

    def mock_builtins_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "_version" and fromlist == ("version",) and level == 1:
            raise ImportError("No module named '_version'")
        return original_builtins_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", mock_builtins_import)

    # Now import sgnts - it should fall back to "0.0.0"
    import sgnts

    assert sgnts.__version__ == "0.0.0"

    # Clean up: remove sgnts from sys.modules again
    modules_to_remove = [m for m in sys.modules if m.startswith("sgnts")]
    for module in modules_to_remove:
        del sys.modules[module]
