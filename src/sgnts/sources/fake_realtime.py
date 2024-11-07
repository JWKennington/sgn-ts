import time
from dataclasses import dataclass

import numpy as np
from sgn.base import SourcePad

from sgnts.base import Offset, TSFrame, TSSource


@dataclass
class RealTimeWhiteNoiseSrc(TSSource):
    """A time-series source that generates fake data in fixed-size buffers in real-time

    Args:
        rate:
            int, the sample rate of the data
        duration:
            float, the duration for the source to continue to produce frames, in seconds
    """

    rate: int = 2048
    duration: float = float("+inf")

    def __post_init__(self):
        self.stride = Offset.SAMPLE_STRIDE_AT_MAX_RATE

        # init start time
        # FIXME: How to define t0? Currently derived from time.time()
        self.t0_offset = Offset.fromsec(time.time()) // self.stride * self.stride
        self.t0 = Offset.tosec(self.t0_offset)
        self.next_time = self.t0_offset + self.stride

        self.end = self.t0 + self.duration

        super().__post_init__()

        for pad in self.source_pads:
            self.setup_buffers_on_pad(channels=(), rate=self.rate, pad=pad)

    def new(self, pad: SourcePad) -> TSFrame:
        """New TSFrames are created on "pad" at fixed time intervals that keeps up with
        the stride specified in Offset.SAMPLE_STRIDE_AT_MAX_RATE. EOS is set if we have
        surpassed the requested duration.

        Args:
            pad:
                SourcePad, the pad for which to produce a new TSFrame

        Returns:
            TSFrame, the TSFrame with random data
        """

        # Produce buffers at every fixed interval
        now = Offset.fromsec(time.time())
        sleep = Offset.tosec(self.next_time - now)
        if sleep > 0:
            # There might be cases where sleep < 0 and we are behind? In that case
            # don't sleep
            time.sleep(sleep)
        self.next_time = self.next_time + self.stride

        metadata = {"name": "'%s'" % pad.name}

        frame = self.prepare_frame(pad, data=None, metadata=metadata)

        for buf in frame:
            buf.set_data(np.random.randn(buf.samples))

        return frame
