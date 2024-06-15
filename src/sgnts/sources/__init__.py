from dataclasses import dataclass

import numpy as np

from ..base import Offset, SeriesBuffer, TSFrame, TSSource


@dataclass
class FakeSeriesSrc(TSSource):
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
    signal_type: str
        currently supported types: (1) 'white': white noise data. (2) 'sin' or 'sine':
        sine wave data
    fsin: float
        frequency of the sine wave is signal_type = 'sin'
    ngap: int
        Frequency of gap buffers, will generate a gap buffer every ngap buffers.
        ngap=0: do not generate gap buffers.
        ngap=-1: generates gap buffers randomly.
    random_seed: int
        set the random seed, used for signal_type = 'white'
    """

    num_buffers: int = 0
    rate: int = 2048
    channels: tuple = ()
    signal_type: str = "white"
    fsin: float = 5
    ngap: int = 0
    random_seed: int = None

    def __post_init__(self):
        super().__post_init__()
        self.cnt = {p: 0 for p in self.source_pads}
        self.shape = self.channels + (int(self.rate * self.duration),)
        if self.signal_type == "white" and self.random_seed is not None:
            np.random.seed(self.random_seed)

    def create_data(self, offset):
        if self.signal_type == "white":
            return np.random.rand(*self.shape)
        elif self.signal_type == "sin" or self.signal_type == "sine":
            t0 = Offset.tosec(offset)
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
        noffset = Offset.fromsec(self.duration)
        ngap = self.ngap
        if (ngap == -1 and np.random.rand(1) > 0.5) or (ngap > 0 and self.cnt[pad] % ngap == 0):
            data = None
        else:
            data = self.create_data(self.offset[pad])

        outbuf = SeriesBuffer(
            offset=self.offset[pad],
            noffset=noffset,
            sample_rate=self.rate,
            channels=self.channels,
            data=data,
        )

        self.offset[pad] += noffset

        return TSFrame(
            buffers=[outbuf],
            metadata={"cnt": self.cnt, "name": "'%s'" % pad.name},
            EOS=self.cnt[pad] > self.num_buffers,
        )


# @dataclass
# class SegmentSrc(TSSource):
#    """
#
#    Parameters:
#    -----------
#    rate: int
#        the sample rate of the data
#    segments: tuple
#        A tuple of segment tuples
#    """
#
#    rate: int = 2048
#    segments: tuple = None
#
#    def __post_init__(self):
#        super().__post_init__()
#        self.segments = sorted(slice(*s) for s in self.segments if s[1] < self.t0)
#
#    def new(self, pad):
#        """
#        New buffers are created on "pad" with an instance specific count and a
#        name derived from the pad name. "EOS" is set if we have surpassed the requested
#        number of buffers.
#        """
#        noffset = Offset.fromsec(self.duration)
#        outbuf = SeriesBuffer(
#            offset=self.offset[pad],
#            noffset=noffset,
#            offset_ref_t0=0,
#            data=data,
#        )
#        intersecting_segments = []
#        outslice = outbuf.slice
#        for n, s in self.segments:
#            if s > outslice:
#                break
#            if outslice & s:
#                intersecting_segments.append(s)
#        self.segments = self.segments[n:]
#        # FIXME IMPLEMENT BUFFER SPLITTING BY LIST OF SLICES
#        outbufs = outbuf.split(intersecting_segments)
#
#        self.offset[pad] += noffset
#
#        return TSFrame(
#            buffers=[outbuf],
#        )
