#!/usr/bin/env python3

from dataclasses import dataclass
import time
import numpy

import pytest
from sgn.apps import Pipeline
from sgn.sources import SignalEOS
from sgnts.base import TSResourceSource
from sgnts.base.buffer import SeriesBuffer
from sgnts.base.offset import Offset
from sgnts.sinks import NullSeriesSink
from sgnts.utils import gpsnow


#
# NOTE this mocks e.g., an arrakis server
#
@dataclass
class DataServer:
    block_duration: int = 2
    simulate_skip_data: bool = False
    simulate_hang: int = 0

    description = {
        "H1:FOO": {"rate": 2048, "sample-shape": ()},
        "L1:FOO": {"rate": 2048, "sample-shape": ()},
    }

    def stream(self, channels, start=None, end=None):
        assert not (set(channels) - set(self.description))
        t0 = int(gpsnow()) - 1.0 if start is None else start
        time.sleep(self.simulate_hang)
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
    simulate_hang: int = 0

    def __post_init__(self):
        self.server = DataServer(
            block_duration=self.block_duration,
            simulate_skip_data=self.simulate_skip_data,
            simulate_hang=self.simulate_hang,
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
                yield pad, buf


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


def test_resource_fail():

    pipeline = Pipeline()

    src = FakeLiveSource(
        name="src",
        source_pad_names=("H1:BAR",),
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
        link_map={snk.snks["H1"]: src.srcs["H1:BAR"]},
    )

    with pytest.raises(RuntimeError):
        pipeline.run()


def test_resource_hang():

    pipeline = Pipeline()

    src = FakeLiveSource(
        name="src",
        source_pad_names=("H1:FOO",),
        duration=10,
        block_duration=4,
        simulate_hang=2,
        in_queue_timeout=1,
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

    with pytest.raises(ValueError):
        pipeline.run()


if __name__ == "__main__":
    test_resource_source()
