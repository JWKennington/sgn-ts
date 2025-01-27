#!/usr/bin/env python3

from dataclasses import dataclass
from sgn.apps import Pipeline
import time
from sgnts.sinks import FakeSeriesSink
from sgnts.base import TSThreadedResource, TSResourceSource
from sgnts.base.buffer import SeriesBuffer
from sgnts.base.offset import Offset
from sgnts.utils import gpsnow
from typing import ClassVar
import numpy
import queue
import sys
from sgn.sources import SignalEOS

@dataclass
class DataServer:
    realtime: bool=True
    block_duration: int=2

    description =  {
                   "H1:FOO": {"rate": 2048, "sample-shape": ()},
                   "L1:FOO": {"rate": 2048, "sample-shape": ()},
                   }

    def stream(self, channels, start=None, end=None):
        assert not (set(channels) - set(self.description))
        if start is None:
            t0 = int(gpsnow()) - 1.0
        while True:
            out = {}
            for channel in channels:
                sample_shape, rate = self.description[channel]["sample-shape"], self.description[channel]["rate"]
                shape = sample_shape + (self.block_duration * rate,)
                out[channel] = {"t0": t0, "data": numpy.random.randn(*shape), "rate": rate, "sample_shape": sample_shape}
            t0 += self.block_duration
            if self.realtime:
                time.sleep(max(0, t0 - gpsnow()))
            yield out

@dataclass
class Resource(TSThreadedResource):
    def __post_init__(self):
        super().__post_init__()
        self.server = DataServer()

    def thread_get_data(self):
        try:

            for stream in self.server.stream(self.srcs):

                # if the queue is full, sleep and try again after wait seconds
                if any(q.full() for q in self.in_queue.values()):
                    time.sleep(self.blocking_wait_time)
                    continue

                for channel,block in stream.items():
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
            print (e)
            self.exception_queue.put(e)


@dataclass
class FakeLiveSource(TSResourceSource):
    pass


def test_resource_source():

    pipeline = Pipeline()

    resource = Resource(duration=10)

    src = FakeLiveSource(
        name="src",
        source_pad_names=("H1:FOO",),
        resource=resource,
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

    with SignalEOS() as control:
        pipeline.run()


if __name__ == "__main__":
    test_resource_source()
