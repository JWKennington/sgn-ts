#!/usr/bin/env python3

from sgn.apps import Pipeline
import time
from sgnts.sinks import NullSeriesSink
from sgnts.sources import FakeSeriesSource
from sgnts.base import TSTransform


def test_realtime(capsys):

    class Slow(TSTransform):
        def __post_init__(self):
            super().__post_init__()
            self.slowcounter = 0

        def new(self, pad):
            if self.slowcounter == 1:
                time.sleep(2)
            self.slowcounter += 1
            return self.preparedframes[self.sink_pads[0]]

    pipeline = Pipeline()

    inrate = 256
    duration = 4
    pipeline.insert(
        FakeSeriesSource(
            name="src",
            source_pad_names=("H1",),
            rate=inrate,
            duration=duration,
            real_time=True,
        ),
        Slow(name="slow", source_pad_names=("H1",), sink_pad_names=("H1",)),
        NullSeriesSink(
            name="snk",
            sink_pad_names=("H1",),
            verbose=True,
        ),
        link_map={
            "slow:snk:H1": "src:src:H1",
            "snk:snk:H1": "slow:src:H1",
        },
    )

    pipeline.run()


if __name__ == "__main__":
    test_realtime(None)
