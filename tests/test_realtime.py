#!/usr/bin/env python3

from sgn.apps import Pipeline

from sgnts.sinks import FakeSeriesSink
from sgnts.sources import RealTimeWhiteNoiseSrc


def test_realtime(capsys):

    pipeline = Pipeline()

    inrate = 256
    num_samples = 256
    t0 = 0.0
    num_buffers = 2
    pipeline.insert(
        RealTimeWhiteNoiseSrc(
            name="src",
            source_pad_names=("H1",),
            rate=inrate,
            num_samples=num_samples,
            t0=t0,
            num_buffers=num_buffers,
        ),
        FakeSeriesSink(
            name="snk",
            sink_pad_names=("H1",),
            verbose=True,
        ),
        link_map={
            "snk:sink:H1": "src:src:H1",
        },
    )

    pipeline.run()


if __name__ == "__main__":
    test_realtime(None)
