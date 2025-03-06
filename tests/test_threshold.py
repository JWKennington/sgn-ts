#!/usr/bin/env python3

from sgn.apps import Pipeline

from sgnts.sources import FakeSeriesSource
from sgnts.sinks import NullSeriesSink
from sgnts.transforms import Threshold


def test_threshold(capsys):

    pipeline = Pipeline()

    #
    #         ---------
    #        | datasrc |
    #         ---------
    #             |
    #        -----------
    #       | threshold |
    #        -----------
    #             |
    #        -----------
    #       | threshold2 |
    #        -----------
    #             |
    #         --------
    #        | snk    |
    #         --------

    inrate = 256
    t0 = 0.0
    end = 5.0
    threshold = 1.5
    startwn = 10
    stopwn = 10
    pipeline.insert(
        FakeSeriesSource(
            name="datasrc",
            source_pad_names=("data",),
            rate=inrate,
            t0=t0,
            end=end,
        ),
        Threshold(
            name="threshold",
            source_pad_names=("data",),
            sink_pad_names=("data",),
            threshold=threshold,
            startwn=startwn,
            stopwn=stopwn,
        ),
        Threshold(
            name="threshold2",
            source_pad_names=("data",),
            sink_pad_names=("data",),
            threshold=threshold,
            startwn=startwn,
            stopwn=stopwn,
            invert=True,
        ),
        NullSeriesSink(
            name="snk",
            sink_pad_names=("data",),
            verbose=True,
        ),
        link_map={
            "threshold:snk:data": "datasrc:src:data",
            "threshold2:snk:data": "threshold:src:data",
            "snk:snk:data": "threshold2:src:data",
        },
    )

    pipeline.run()


if __name__ == "__main__":
    test_threshold(None)
