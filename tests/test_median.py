#!/usr/bin/env python3

from sgn.apps import Pipeline
from sgn.sinks import NullSink

from sgnts.sources import FakeSeriesSource
from sgnts.transforms import Median


def test_median():

    pipeline = Pipeline()

    #
    #       ----------
    #      | src1     |
    #       ----------
    #              \
    #           H1  \ SR2
    #           ------------
    #          | Median    |
    #           ------------
    #                 \
    #             H1   \ SR2
    #             ---------
    #            | snk1    |
    #             ---------

    inrate = 256

    pipeline.insert(
        FakeSeriesSource(
            name="src1",
            source_pad_names=("H1",),
            rate=inrate,
            signal_type="white",
            end=8,
        ),
        Median(
            name="trans1",
            source_pad_names=("H1",),
            sink_pad_names=("H1",),
            overlap_offsets=(512, 0),
        ),
        NullSink(
            name="snk1",
            sink_pad_names=("H1",),
            verbose=True,
        ),
        link_map={
            "trans1:snk:H1": "src1:src:H1",
            "snk1:snk:H1": "trans1:src:H1",
        },
    )

    pipeline.run()


if __name__ == "__main__":
    test_median()
