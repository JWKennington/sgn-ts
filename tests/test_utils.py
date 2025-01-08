import sys
from unittest import mock


def test_utils():
    # Test the default gwpy
    from sgnts.utils import gpsnow

    gpsnow()

    # Test missing gwpy
    original = sys.modules
    keys = ["gwpy", "sgnts"]
    clean = {k: v for k, v in original.items() if all(key not in k for key in keys)}
    clean.update({"gwpy": None})
    with mock.patch.dict("sys.modules", clear=True, values=clean):
        from sgnts.utils import gpsnow

        gpsnow()

    # Test missing gwpy and missing gpstime
    original = sys.modules
    keys = ["gpstime", "gwpy", "sgnts"]
    clean = {k: v for k, v in original.items() if all(key not in k for key in keys)}
    clean.update({"gpstime": None, "gwpy": None})
    with mock.patch.dict("sys.modules", clear=True, values=clean):
        from sgnts.utils import gpsnow

        gpsnow()
