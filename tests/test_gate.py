#!/usr/bin/env python3

from sgn.apps import Pipeline

from sgnts.sinks import FakeSeriesSink
from sgnts.sources import SegmentSrc, FakeSeriesSrc
from sgnts.transforms import Gate


def test_gate(capsys):

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
    num_samples = 256
    t0 = 0.0
    end = 15.0
    segments = [(1_000_000_000, 2_000_000_000), (10_000_000_000, 11_000_000_000)]
    num_buffers = 20
    pipeline.insert(
        SegmentSrc(
            name="segsrc",
            source_pad_names=("seg",),
            rate=inrate,
            num_samples=num_samples,
            t0=t0,
            end=end,
            segments=segments,
        ),
        FakeSeriesSrc(
            name="datasrc",
            source_pad_names=("data",),
            num_buffers=num_buffers,
            rate=inrate,
            num_samples=num_samples,
            t0=t0,
        ),
        Gate(
            name="gate",
            source_pad_names=("gate",),
            sink_pad_names=("data", "control"),
            control="control",
        ),
        FakeSeriesSink(
            name="snk",
            sink_pad_names=("gate",),
        ),
        link_map={
            "gate:sink:data": "datasrc:src:data",
            "gate:sink:control": "segsrc:src:seg",
            "snk:sink:gate": "gate:src:gate",
        },
    )

    pipeline.run()


if __name__ == "__main__":
    test_gate(None)
