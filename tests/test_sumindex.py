#!/usr/bin/env python3

from sgn.apps import Pipeline

from sgnts.sources import FakeSeriesSource
from sgnts.sinks import NullSeriesSink
from sgnts.transforms import SumIndex


def test_sumindex():

    pipeline = Pipeline()

    pipeline.insert(
        FakeSeriesSource(
            name="src1",
            source_pad_names=("H1",),
            end=16,
            rate=2048,
            signal_type="sin",
            sample_shape=(3, 3),
            ngap=2,
        ),
        SumIndex(
            name="sumindex",
            source_pad_names=("H1",),
            sink_pad_names=("H1",),
            sl=[slice(0, 1), slice(0, 2)],
        ),
        NullSeriesSink(name="snk1", sink_pad_names=("H1",), verbose=True),
        link_map={
            "sumindex:snk:H1": "src1:src:H1",
            "snk1:snk:H1": "sumindex:src:H1",
        },
    )

    pipeline.run()


if __name__ == "__main__":
    test_sumindex()
