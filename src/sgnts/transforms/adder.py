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

    def pull(self, pad, bufs):
        self.inbufs[pad] = bufs
        # This is a subclass of TSTransform, so the frames are already aligned
        if self.sample_rate is None and len(bufs.buffers) > 0:
            self.sample_rate = bufs[0].sample_rate
        else:
            for buf in bufs:
                assert buf.sample_rate == self.sample_rate 

    def transform(self, pad):
        EOS = any(b.EOS for b in self.inbufs.values())

        print('inbufs in adder',self.inbufs)
        frames = [b.buffers for b in self.inbufs.values()]
        if len(frames[0]) == 0:
            return TSFrame(EOS=EOS)
        else:
            # use the first frame as basis
            out = np.concatenate([buf.filleddata for buf in frames[0]])
            # add to the first frame
            for f in frames[1:]:
                i0 = 0
                for buf in f:
                    if not buf.is_gap:
                        out[...,i0:i0+buf.samples] += buf.data
                    i0 += buf.samples

            return TSFrame(buffers=[SeriesBuffer(
                offset=frames[0][0].offset,
                sample_rate=self.sample_rate,
                data=out,
            )], EOS=EOS)

