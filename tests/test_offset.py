#!/usr/bin/env python3

from sgnts.base.offset import Offset


def test_offset():
    OLD = Offset.MAX_RATE
    Offset.set_max_rate(OLD * 2)
    Offset.set_max_rate(OLD)
