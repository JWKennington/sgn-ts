#!/usr/bin/env python3

import numpy as np

from sgn.apps import Pipeline

from sgnts.sinks import FakeSeriesSink
from sgnts.sources import FakeSeriesSrc
from sgnts.transforms import Adder, Correlate, Matmul, Resampler


def test_lloid(capsys):

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
        Correlate(
            name="corr2",
            source_pad_names=("H1",),
            sink_pad_names=("H1",),
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
        FakeSeriesSink(
            name="snk1",
            sink_pad_names=("H1",),
        ),
        link_map={
            "down:sink:H1": "src1:src:H1",
            "corr2:sink:H1": "down:src:H1",
            "mm2:sink:H1": "corr2:src:H1",
            "up:sink:H1": "mm2:src:H1",
            "corr1:sink:H1": "src1:src:H1",
            "mm1:sink:H1": "corr1:src:H1",
            "add:sink:frombuf": "up:src:H1",
            "add:sink:tobuf": "mm1:src:H1",
            "snk1:sink:H1": "add:src:H1",
        },
    )

    pipeline.run()


if __name__ == "__main__":
    test_lloid(None)
