#!/usr/bin/env python3

from dataclasses import dataclass
from sgn.apps import Pipeline
import time
from sgnts.sinks import FakeSeriesSink
from sgnts.base import  TSResourceSource
from sgnts.base.buffer import SeriesBuffer
from sgnts.base.offset import Offset
from sgnts.utils import gpsnow
import numpy
import queue
from sgn.sources import SignalEOS


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
            if end is not None and t0>= end:
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
    server: object = None

    def __post_init__(self):
        super().__post_init__()

    def thread_get_data(self):
        try:

            for stream in self.server.stream(self.srcs, self.start_time, self.end):
                # if the queue is full, sleep and try again after wait seconds
                if any(q.full() for q in self.in_queue.values()):
                    time.sleep(self.blocking_wait_time)
                    continue

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

        except Exception as e:
            print(e)
            self.exception_queue.put(e)


def test_resource_source():

    pipeline = Pipeline()

    block_duration = 4

    src = FakeLiveSource(
        name="src",
        source_pad_names=("H1:FOO",),
        #start_time=0,
        duration=10,
        server=DataServer(block_duration=block_duration, simulate_skip_data=True),
        in_queue_timeout=block_duration + 2,
    )
    snk = FakeSeriesSink(
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
