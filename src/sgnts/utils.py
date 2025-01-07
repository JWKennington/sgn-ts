import time
from datetime import datetime

from sgn.base import get_sgn_logger

LOGGER = get_sgn_logger("sgn-ts")


try:
    from gwpy.time import to_gps

    gpsnow = lambda: float(to_gps(datetime.utcnow()))
except ImportError:
    try:
        from gpstime import gpsnow  # type: ignore
    except ImportError:
        # accurate for "now" as of this writing
        gpsnow = lambda: time.time() - 315964782  # type: ignore
        LOGGER.warning(
            (
                "A GPS time function could not be imported, GPS times will not "
                "be leap second accurate.  For more accurate times install the "
                "'gwpy' or 'gpstime' package."
            )
        )
