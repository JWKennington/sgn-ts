#!/usr/bin/env python3

from sgn.apps import Pipeline

from sgnts.sinks import FakeSeriesSink
from sgnts.sources import SegmentSrc, FakeSeriesSrc
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
    #         --------
    #        | snk    |
    #         --------

    inrate = 256
    num_samples = 256
    t0 = 0.0
    end = 15.0
    num_buffers = 20
    threshold = 0.9
    startwn = 10
    stopwn = 10
    pipeline.insert(
        FakeSeriesSrc(
            name="datasrc",
            source_pad_names=("data",),
            num_buffers=num_buffers,
            rate=inrate,
            num_samples=num_samples,
            t0=t0,
        ),
        Threshold(
            name="threshold",
            source_pad_names=("threshold",),
            sink_pad_names=("data",),
            threshold = threshold,
            startwn = startwn,
            stopwn = stopwn,
            ),
        FakeSeriesSink(
            name="snk",
            sink_pad_names=("threshold",),
        ),
        link_map={
            "threshold:sink:data": "datasrc:src:data",
            "snk:sink:threshold": "threshold:src:threshold",
        },
    )

    pipeline.run()


if __name__ == "__main__":
    test_threshold(None)
