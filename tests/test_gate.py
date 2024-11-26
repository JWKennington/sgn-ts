#!/usr/bin/env python3

from sgn.apps import Pipeline

from sgnts.sources import FakeSeriesSource, SegmentSource
from sgnts.sinks import NullSeriesSink
from sgnts.transforms import Gate


def test_gate():

    pipeline = Pipeline()

    #
    #       ---------     ---------
    #      | segsrc  |   | datasrc |
    #       ---------     ---------
    #                \      /
    #               -----------
    #              |   gate    |
    #               -----------
    #                    |
    #                ---------
    #               | snk    |
    #                ---------

    inrate = 256
    t0 = 0.0
    end = 15.0
    segments = [(1_000_000_000, 2_250_000_000), (10_000_000_000, 12_500_000_000)]
    pipeline.insert(
        SegmentSource(
            name="segsrc",
            source_pad_names=("seg",),
            rate=inrate,
            t0=t0,
            end=end,
            segments=segments,
        ),
        FakeSeriesSource(
            name="datasrc",
            source_pad_names=("data",),
            rate=inrate,
            t0=t0,
            end=end,
        ),
        Gate(
            name="gate",
            source_pad_names=("gate",),
            sink_pad_names=("data", "control"),
            control="control",
        ),
        NullSeriesSink(
            name="snk",
            sink_pad_names=("gate",),
            verbose=True,
        ),
        link_map={
            "gate:snk:data": "datasrc:src:data",
            "gate:snk:control": "segsrc:src:seg",
            "snk:snk:gate": "gate:src:gate",
        },
    )

    pipeline.run()


if __name__ == "__main__":
    test_gate()
