#!/usr/bin/env python3

from sgn.apps import Pipeline

from sgnts.sources import SegmentSource
from sgnts.sinks import NullSeriesSink


def test_align(capsys):

    pipeline = Pipeline()

    #
    #       -----------------
    #      | segment src1    |
    #       -----------------
    #              |
    #           ---------
    #          | snk1    |
    #           ---------

    inrate = 256
    t0 = 0.0
    end = 15.0
    segments = [(1e9, 2e9), (10e9, 11e9)]
    pipeline.insert(
        SegmentSource(
            name="src1",
            source_pad_names=("seg",),
            rate=inrate,
            t0=t0,
            end=end,
            segments=segments,
        ),
        NullSeriesSink(
            name="snk1",
            sink_pad_names=("seg",),
            verbose=True,
        ),
        link_map={
            "snk1:snk:seg": "src1:src:seg",
        },
    )

    pipeline.run()


if __name__ == "__main__":
    test_align(None)
