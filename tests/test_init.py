"""Test the sgnts __init__ module."""

import sys
import builtins


def test_version_import_error(monkeypatch):
    """Test fallback version when _version module is not available."""
    # Remove sgnts from sys.modules to force reimport
    for module in list(sys.modules):
        if module.startswith("sgnts"):
            del sys.modules[module]

    # Mock __import__ to raise ImportError for _version
    original_import = builtins.__import__

    def mock_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "_version" and level == 1:
            raise ImportError("No module named '_version'")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    # Import sgnts - it should fall back to "0.0.0"
    import sgnts

    assert sgnts.__version__ == "0.0.0"
