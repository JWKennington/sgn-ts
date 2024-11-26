#!/usr/bin/env python3

from sgn.apps import Pipeline

from sgnts.sources import FakeSeriesSource
from sgnts.sinks import NullSeriesSink
from sgnts.transforms import Align


def test_align(capsys):

    pipeline = Pipeline()

    #
    #       ----------    -------   --------
    #      | src1     |  | src2  | | src3   |
    #       ----------    -------   --------
    #              \         |      /
    #           H1  \     L1 |     / V1
    #               ----------------
    #              | sync           |
    #               ----------------
    #                 |        |    \
    #             H1  |      L1|     \ V1
    #           ---------   -------   --------
    #          | snk1    | | snk2  | |  snk3  |
    #           ---------   -------   --------

    inrate = 256
    H1_t0 = 2
    L1_t0 = 4
    V1_t0 = 6
    max_age = 100 * 1e9
    duration = 10

    pipeline.insert(
        FakeSeriesSource(
            name="src1",
            source_pad_names=("H1",),
            rate=inrate,
            t0=H1_t0,
            end=H1_t0 + duration,
        ),
        Align(
            name="trans1",
            sink_pad_names=("H1", "L1", "V1"),
            source_pad_names=("H1", "L1", "V1"),
            max_age=max_age,
        ),
        NullSeriesSink(
            name="snk1",
            sink_pad_names=("H1",),
            verbose=True,
        ),
        FakeSeriesSource(
            name="src2",
            source_pad_names=("L1",),
            rate=inrate,
            t0=L1_t0,
            end=L1_t0 + duration,
        ),
        NullSeriesSink(
            name="snk2",
            sink_pad_names=("L1",),
            verbose=True,
        ),
        FakeSeriesSource(
            name="src3",
            source_pad_names=("V1",),
            rate=inrate,
            t0=V1_t0,
            end=V1_t0 + duration,
        ),
        NullSeriesSink(
            name="snk3",
            sink_pad_names=("V1",),
            verbose=True,
        ),
        link_map={
            "trans1:snk:H1": "src1:src:H1",
            "snk1:snk:H1": "trans1:src:H1",
            "trans1:snk:L1": "src2:src:L1",
            "snk2:snk:L1": "trans1:src:L1",
            "trans1:snk:V1": "src3:src:V1",
            "snk3:snk:V1": "trans1:src:V1",
        },
    )

    pipeline.run()


if __name__ == "__main__":
    test_align(None)
