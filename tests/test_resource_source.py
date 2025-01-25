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


@dataclass
class LiveServer(TSThreadedResource):
    def __post_init__(self):
        super().__post_init__()
        self.__end = int(gpsnow() + 10)

    def thread_get_data(self):
        try:
            # Hypothetical list of all channels and metadata that
            # this "server" can support
            # NOTE: this comes from some server query for real.
            __full_description: ClassVar[dict] = {
                "H1:FOO": {"rate": 2048, "sample-shape": ()}
            }
    
            # Make sure the user asked for channels we have
            assert not (set(self.pad_dict.values()) - set(__full_description))
            self.description = {
                pad: __full_description[name] for pad, name in self.pad_dict.items()
            }
    
    
            t0 = int(gpsnow()) - 1.0
            # Simulate a process that is pulling buffers out of a server.
            # NOTE: Replace this part with actual queries
            while True:
                for pad in self.description:
                    # Assume 1 second buffers, hence the rate in shape
                    shape = self.description[pad]["sample-shape"] + (self.description[pad]["rate"],)
                    buf = SeriesBuffer(
                        offset=Offset.fromsec(t0),
                        shape=shape,
                        sample_rate=self.description[pad]["rate"],
                    )
                    buf.set_data(numpy.random.randn(*buf.shape))
                    self.in_queue[pad].put(buf)
                t0 += 1
                time.sleep(max(0, t0 - gpsnow()))
                try:
                    self.stop_thread.get(0)
                    break
                except queue.Empty:
                    pass

        except Exception as e:
            self.exception_queue.put(e)

    @property
    def end(self):
        """The ending time of the resource"""
        return self.__end


@dataclass
class FakeLiveSource(TSResourceSource):
    pass


def test_resource_source():

    pipeline = Pipeline()

    liveserver = LiveServer()

    src = FakeLiveSource(
        name="src",
        source_pad_names=("H1:FOO",),
        resource=liveserver,
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

    pipeline.run()


if __name__ == "__main__":
    test_resource_source()
