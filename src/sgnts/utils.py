import time
from datetime import datetime
from typing import Callable

from sgn.base import get_sgn_logger

LOGGER = get_sgn_logger("sgn-ts")

gpsnow: Callable

try:
    from gwpy.time import to_gps

    def _gpsnow():
        return float(to_gps(datetime.utcnow()))

    gpsnow = _gpsnow

except ImportError:
    try:
        from gpstime import gpsnow as __gpsnow

        gpsnow = __gpsnow

    except ImportError:
        # accurate for "now" as of this writing
        def ___gpsnow():
            return time.time() - 315964782

        gpsnow = ___gpsnow
        LOGGER.warning(
            (
                "A GPS time function could not be imported, GPS times will not "
                "be leap second accurate.  For more accurate times install the "
                "'gwpy' or 'gpstime' package."
            )
        )
