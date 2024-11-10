#!/usr/bin/env python3

from sgn.apps import Pipeline

from sgnts.sinks import FakeSeriesSink
from sgnts.sources import FakeSeriesSrc


def test_multi_sink(capsys):

    pipeline = Pipeline()

    #
    #       ----------    -------   --------
    #      | src1     |  | src2  | | src3   |
    #       ----------    -------   --------
    #              \         |      /
    #           H1  \     L1 |     / V1
    #               ----------------
    #              | sink           |
    #               ----------------

    inrate = 256

    t0 = 0
    end = 10

    pipeline.insert(
        FakeSeriesSrc(
            name="src1",
            source_pad_names=("H1",),
            rate=inrate,
            t0=t0,
            end=end,
        ),
        FakeSeriesSrc(
            name="src2",
            source_pad_names=("L1",),
            rate=inrate,
            t0=t0,
            end=end,
        ),
        FakeSeriesSrc(
            name="src3",
            source_pad_names=("V1",),
            rate=inrate,
            t0=t0,
            end=end,
        ),
        FakeSeriesSink(
            name="snk3",
            sink_pad_names=(
                "H1",
                "L1",
                "V1",
            ),
            verbose=True,
        ),
        link_map={
            "snk3:sink:H1": "src1:src:H1",
            "snk3:sink:L1": "src2:src:L1",
            "snk3:sink:V1": "src3:src:V1",
        },
    )

    pipeline.run()


if __name__ == "__main__":
    test_multi_sink(None)
