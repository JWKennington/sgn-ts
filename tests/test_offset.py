#!/usr/bin/env python3

from sgnts.base.offset import Offset


def test_offset():
    OLD = Offset.MAX_RATE
    Offset.set_max_rate(OLD * 2)
    Offset.set_max_rate(OLD)

    GPSNS = 122287885562500000  # corresponds to a whole number of 1/16th second buffers
    OFFSET = Offset.fromns(GPSNS)
    assert (GPSNS - Offset.tons(OFFSET)) == 0

    GPSNS = (
        1422287885562500000  # corresponds to a whole number of 1/16th second buffers
    )
    OFFSET = Offset.fromns(GPSNS)
    assert (GPSNS - Offset.tons(OFFSET)) == 0


if __name__ == "__main__":
    test_offset()
