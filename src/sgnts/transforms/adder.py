from dataclasses import dataclass, field

from sgn.base import TransformElement

from ..base import Audioadapter, SeriesBuffer, TSFrame, TSTransform

import numpy as np


@dataclass
class Adder(TSTransform):
    """
    Add up all the buffers from all the sink pads
    """

    def __post_init__(self):
        self.inbufs = {}
        super().__post_init__()
        self.sample_rate = None
        self.channels = None

    def pull(self, pad, bufs):
        self.inbufs[pad] = bufs
        # This is a subclass of TSTransform, so the frames are already aligned
        if self.sample_rate is None and len(bufs.buffers) > 0:
            self.sample_rate = bufs[0].sample_rate
        else:
            for buf in bufs:
                assert buf.sample_rate == self.sample_rate 

        if self.channels is None and len(bufs.buffers) > 0:
            self.channels = bufs[0].channels
        else:
            for buf in bufs:
                assert buf.channels == self.channels

    def transform(self, pad):
        EOS = any(b.EOS for b in self.inbufs.values())

        #print(self.preparedframes)
        print(self.inbufs)
        frames = list(self.preparedframes.values())
        if frames[0] is None:
            return TSFrame(EOS=EOS)
        else:
            # use the first frame as basis
            out = np.concatenate([buf.filleddata for buf in frames[0]])
            # add to the first frame
            for f in frames:
                print('1',f)
            for f in frames[1:]:
                i0 = 0
                for buf in f:
                    if not buf.is_gap:
                        out[...,i0:i0+buf.size] += buf.data
                    i0 += buf.size

        return TSFrame(buffers=[SeriesBuffer(
            offset=offset,
            noffset=noffset,
            data=out,
            sample_rate=self.sample_rate,
            channels=self.channels
        )], EOS=EOS)

