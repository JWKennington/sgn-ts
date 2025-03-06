#!/usr/bin/env python3

import numpy as np
from sgn.apps import Pipeline

from sgnts.sources import FakeSeriesSource
from sgnts.sinks import NullSeriesSink
from sgnts.transforms import Adder, Correlate, Matmul, Resampler


def test_lloid():

    pipeline = Pipeline()

    #
    #       ----------   H1   -------------
    #      | src1     | ---- | downsample  |
    #       ----------   SR1  -------------
    #             |              |
    #           H1|SR1           |H1 SR2
    #          ------          -------
    #         |corr1 |        | corr2 |
    #          ------          -------
    #             |              |
    #             |              |
    #           H1|SR1           | H1 SR2
    #          ------          -------
    #         |mm1   |        | mm2   |
    #          ------          -------
    #             |              |
    #             |              |
    #             |           H1 | SR2
    #             |     ------------
    #          H1 |    | upsample   |
    #         SR1 |     ------------
    #             |        |
    #             |     H1 | SR1
    #             -----------
    #            |   add     |
    #             -----------
    #                   |
    #                H1 | SR1
    #             -----------
    #            |   snk1    |
    #             -----------
    #

    max_age = 1000000000000

    pipeline.insert(
        FakeSeriesSource(
            name="src1",
            source_pad_names=("H1",),
            rate=2048,
            signal_type="sin",
            end=8,
        ),
        Resampler(
            name="down",
            source_pad_names=("H1",),
            sink_pad_names=("H1",),
            inrate=2048,
            outrate=512,
        ),
        Correlate(
            name="corr2",
            source_pad_names=("H1",),
            sink_pad_names=("H1",),
            sample_rate=512,
            filters=np.random.rand(10, 2048),
        ),
        Matmul(
            name="mm2",
            source_pad_names=("H1",),
            sink_pad_names=("H1",),
            matrix=np.random.rand(1000, 10),
        ),
        Resampler(
            name="up",
            source_pad_names=("H1",),
            sink_pad_names=("H1",),
            inrate=512,
            outrate=2048,
        ),
        Correlate(
            name="corr1",
            source_pad_names=("H1",),
            sink_pad_names=("H1",),
            sample_rate=2048,
            filters=np.random.rand(10, 2048),
        ),
        Matmul(
            name="mm1",
            source_pad_names=("H1",),
            sink_pad_names=("H1",),
            matrix=np.random.rand(1000, 10),
        ),
        Adder(
            name="add",
            source_pad_names=("H1",),
            sink_pad_names=("frombuf", "tobuf"),
            max_age=max_age,
        ),
        NullSeriesSink(
            name="snk1",
            sink_pad_names=("H1",),
            verbose=True,
        ),
        link_map={
            "down:snk:H1": "src1:src:H1",
            "corr2:snk:H1": "down:src:H1",
            "mm2:snk:H1": "corr2:src:H1",
            "up:snk:H1": "mm2:src:H1",
            "corr1:snk:H1": "src1:src:H1",
            "mm1:snk:H1": "corr1:src:H1",
            "add:snk:frombuf": "up:src:H1",
            "add:snk:tobuf": "mm1:src:H1",
            "snk1:snk:H1": "add:src:H1",
        },
    )

    pipeline.run()


if __name__ == "__main__":
    test_lloid()
