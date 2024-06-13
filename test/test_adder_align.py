#!/usr/bin/env python3

import numpy as np

from sgn.apps import Pipeline

from sgnts.sinks import FakeSeriesSink
from sgnts.sources import FakeSeriesSrc
from sgnts.transforms import Adder, Align, Resampler


def test_resampler(capsys):

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
    #            |   align   |
    #             -----------
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
            duration=1,
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
        Align(
            name="align",
            sink_pad_names=("A","B"),
            source_pad_names=("A","B"),
            max_age=max_age,
        ),
        Adder(
            name="add",
            source_pad_names=("H1",),
            sink_pad_names=("A", "B"),
        ),
        FakeSeriesSink(
            name="snk1",
            sink_pad_names=("H1",),
        ),
        link_map={
            "down:sink:H1": "src1:src:H1",
            "up:sink:H1": "down:src:H1",
            "align:sink:A": "up:src:H1",
            "add:sink:A": "align:src:A",
            "align:sink:B": "src1:src:H1",
            "add:sink:B": "align:src:B",
            "snk1:sink:H1": "add:src:H1",
        },
    )

    pipeline.run()
    if capsys is not None:
        captured = capsys.readouterr()
        assert (
            captured.out.strip()
            == """
buffer flow:  ('src1:src:H1' -> 'corr1:src:H1' -> 'mm1:src:H1'+'src1:src:H1' -> 'down:src:H1' -> 'corr2:src:H1' -> 'mm2:src:H1' -> 'up:src:H1') -> 'add:src:H1' -> 'snk1:sink:H1' offset 0 time 0
buffer flow:  ('src1:src:H1' -> 'corr1:src:H1' -> 'mm1:src:H1'+'src1:src:H1' -> 'down:src:H1' -> 'corr2:src:H1' -> 'mm2:src:H1' -> 'up:src:H1') -> 'add:src:H1' -> 'snk1:sink:H1' offset 15104 time 921875000
buffer flow:  ('src1:src:H1' -> 'corr1:src:H1' -> 'mm1:src:H1'+'src1:src:H1' -> 'down:src:H1' -> 'corr2:src:H1' -> 'mm2:src:H1' -> 'up:src:H1') -> 'add:src:H1' -> 'snk1:sink:H1' offset 31488 time 1921875000
""".strip()
        )


if __name__ == "__main__":
    test_resampler(None)
