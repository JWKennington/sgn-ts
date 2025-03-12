#!/usr/bin/env python3

from dataclasses import dataclass
from sgn.apps import Pipeline
import time
from sgnts.base import TSResourceSource
from sgnts.base.buffer import SeriesBuffer
from sgnts.base.offset import Offset
from sgnts.sinks import NullSeriesSink
from sgnts.utils import gpsnow
import numpy
import queue
from sgn.sources import SignalEOS


#
# NOTE this mocks e.g., an arrakis server
#
@dataclass
class DataServer:
    block_duration: int = 2
    simulate_skip_data: bool = False

    description = {
        "H1:FOO": {"rate": 2048, "sample-shape": ()},
        "L1:FOO": {"rate": 2048, "sample-shape": ()},
    }

    def stream(self, channels, start=None, end=None):
        assert not (set(channels) - set(self.description))
        t0 = int(gpsnow()) - 1.0 if start is None else start
        while True:
            out = {}
            if end is not None and t0 >= end:
                return
            for channel in channels:
                sample_shape, rate = (
                    self.description[channel]["sample-shape"],
                    self.description[channel]["rate"],
                )
                shape = sample_shape + (self.block_duration * rate,)
                out[channel] = {
                    "t0": t0,
                    "data": numpy.random.randn(*shape),
                    "rate": rate,
                    "sample_shape": sample_shape,
                }
            t0 += self.block_duration
            # Simulate a data skip if requested
            if self.simulate_skip_data:
                t0 += 2
            # simulate real-time if start is None
            if start is None:
                time.sleep(max(0, t0 - gpsnow()))
            yield out


@dataclass
class FakeLiveSource(TSResourceSource):
    simulate_skip_data: bool = False
    block_duration: int = 4

    def __post_init__(self):
        self.server = DataServer(
            block_duration=self.block_duration,
            simulate_skip_data=self.simulate_skip_data,
        )
        super().__post_init__()

    def get_data(self):
        for stream in self.server.stream(self.srcs, self.start_time, self.end_time):
            for channel, block in stream.items():
                pad = self.srcs[channel]

                buf = SeriesBuffer(
                    offset=Offset.fromsec(block["t0"]),
                    data=block["data"],
                    sample_rate=block["rate"],
                )
                self.in_queue[pad].put(buf)
            try:
                self.stop_thread.get(0)
                break
            except queue.Empty:
                pass


def test_resource_source():

    pipeline = Pipeline()

    src = FakeLiveSource(
        name="src",
        source_pad_names=("H1:FOO",),
        duration=10,
        block_duration=4,
    )
    snk = NullSeriesSink(
        name="snk",
        sink_pad_names=("H1",),
        verbose=True,
    )
    pipeline.insert(
        src,
        snk,
        link_map={snk.snks["H1"]: src.srcs["H1:FOO"]},
    )

    with SignalEOS():
        pipeline.run()


if __name__ == "__main__":
    test_resource_source()
