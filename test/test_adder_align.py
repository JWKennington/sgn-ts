#!/usr/bin/env python3

import numpy as np

from sgn.apps import Pipeline

from sgnts.sinks import FakeSeriesSink
from sgnts.sources import FakeSeriesSrc
from sgnts.transforms import Adder, Align, Resampler


def test_adder(capsys):

    pipeline = Pipeline()
    max_age = 1000000000000

    #
    #       ----------   H1   -------------
    #      | src1     | ---- | downsample  |
    #       ----------   SR1  -------------
    #             |              |
    #             |              |
    #             |           H1 | SR2
    #             |     ------------
    #          H1 |    | upsample   |
    #         SR1 |     ------------
    #             |        |
    #             |     H1 | SR1
    #             |        |
    #             |        |
    #             -----------
    #            |   add     |
    #             -----------
    #                   |
    #                H1 | SR1
    #             -----------
    #            |   snk1    |
    #             -----------
    #

    pipeline.insert(
        FakeSeriesSrc(
            name="src1",
            source_pad_names=("H1",),
            num_buffers=2,
            rate=2048,
            num_samples=2048,
            signal_type="sin",
        ),
        Resampler(
            name="down",
            source_pad_names=("H1",),
            sink_pad_names=("H1",),
            inrate=2048,
            outrate=512,
        ),
        Resampler(
            name="up",
            source_pad_names=("H1",),
            sink_pad_names=("H1",),
            inrate=512,
            outrate=2048,
        ),
        Adder(
            name="add",
            source_pad_names=("A",),
            sink_pad_names=("A","B"),
            max_age=max_age,
        ),
        FakeSeriesSink(
            name="snk1",
            sink_pad_names=("H1",),
        ),
        link_map={
            "down:sink:H1": "src1:src:H1",
            "up:sink:H1": "down:src:H1",
            "add:sink:A": "up:src:H1",
            "add:sink:B": "src1:src:H1",
            "snk1:sink:H1": "add:src:A",
        },
    )

    pipeline.run()

if __name__ == "__main__":
    test_adder(None)
