from dataclasses import dataclass

import numpy as np
from sgn.base import SourceElement

from ..base import OFFSET_RATE, Offset, SeriesBuffer, TSFrame


@dataclass
class FakeSeriesSrc(SourceElement):
    """
    A time-series source that generates fake data in fixed-size buffers.

    Parameters:
    -----------
    num_buffers: int
        is required and sets how many buffers will be created before setting "EOS"
    rate: int
        the sample rate of the data
    channels: tuple
        the channels of the data
    duration: float
        duration of the data buffer, in seconds
    signal_type: str
        currently supported types: (1) 'white': white noise data. (2) 'sin' or 'sine':
        sine wave data
    fsin: float
        frequency of the sine wave is signal_type = 'sin'
    t0: float
        start time of first buffer, in seconds
    """

    num_buffers: int = 0
    rate: int = 2048
    channels: tuple = ()
    duration: float = 1
    signal_type: str = "white"
    fsin: float = 5
    t0: float = 0

    def __post_init__(self):
        super().__post_init__()
        self.cnt = {p: 0 for p in self.source_pads}
        self.offset = {p: Offset.sec2offset(self.t0) for p in self.source_pads}
        self.shape = self.channels + (int(self.rate * self.duration),)

    def create_data(self, offset):
        if self.signal_type == "white":
            return np.random.rand(*self.shape)
        elif self.signal_type == "sin" or self.signal_type == "sine":
            t0 = Offset.offset2sec(offset)
            return np.sin(
                self.fsin
                * np.linspace(t0, t0 + self.duration, self.shape[-1], endpoint=False)
            )
        else:
            raise ValueError("Unknown signal type")

    def new(self, pad):
        """
        New buffers are created on "pad" with an instance specific count and a
        name derived from the pad name. "EOS" is set if we have surpassed the requested
        number of buffers.
        """
        self.cnt[pad] += 1
        noffset = int(OFFSET_RATE * self.duration)
        data = self.create_data(self.offset[pad])
        outbuf = SeriesBuffer(
            offset=self.offset[pad],
            noffset=noffset,
            offset_ref_t0=0,
            data=data,
        )

        self.offset[pad] += noffset

        return TSFrame(
            buffers=[outbuf],
            metadata={"cnt": self.cnt, "name": "'%s'" % pad.name},
            EOS=self.cnt[pad] > self.num_buffers,
        )
