#!/usr/bin/env python3
import pytest
from sgn.apps import Pipeline

from sgnts.sources import FakeSeriesSource
from sgnts.sinks import DumpSeriesSink
from sgnts.sinks import NullSeriesSink


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
        FakeSeriesSource(
            name="src1",
            source_pad_names=("H1",),
            rate=inrate,
            t0=t0,
            end=end,
        ),
        FakeSeriesSource(
            name="src2",
            source_pad_names=("L1",),
            rate=inrate,
            t0=t0,
            end=end,
            signal_type="impulse",
            impulse_position=1,
            verbose=True,
        ),
        FakeSeriesSource(
            name="src3",
            source_pad_names=("V1",),
            rate=inrate,
            t0=t0,
            end=end,
            signal_type="impulse",
            impulse_position=-1,
        ),
        NullSeriesSink(
            name="snk3",
            sink_pad_names=(
                "H1",
                "L1",
                "V1",
            ),
            verbose=True,
        ),
        link_map={
            "snk3:snk:H1": "src1:src:H1",
            "snk3:snk:L1": "src2:src:L1",
            "snk3:snk:V1": "src3:src:V1",
        },
    )

    pipeline.run()


def test_invalid_fake_series():
    pipeline = Pipeline()
    src = FakeSeriesSource(
        name="blah",
        source_pad_names=("V1",),
        rate=2048,
        t0=0,
        end=1,
        signal_type="blah",
    )
    sink = NullSeriesSink(
        name="blah2",
        sink_pad_names=("V1",),
    )
    pipeline.insert(src, sink, link_map={sink.snks["V1"]: src.srcs["V1"]})
    with pytest.raises(ValueError):
        pipeline.run()


def test_invalid_dump_series_pads():
    with pytest.raises(ValueError):
        DumpSeriesSink(
            sink_pad_names=("H1", "L1"),
        )


if __name__ == "__main__":
    test_multi_sink(None)
