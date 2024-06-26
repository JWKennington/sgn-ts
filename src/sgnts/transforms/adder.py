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
        super().__post_init__()

    def transform(self, pad):
        frames = list(self.preparedframes.values())
        assert len(set(f.sample_rate for f in frames)) == 1
        assert len(set(f.offset for f in frames)) == 1
        assert len(set(f.end_offset for f in frames)) == 1
        assert len(set(f.shape for f in frames)) == 1

        if all(frame.is_gap for frame in frames):
            out = None
            shape = frames[0].shape
        else:
            # use the first frame as basis
            out = np.concatenate([buf.filleddata for buf in frames[0]], axis=-1)
            shape = out.shape
            # add to the first frame
            for f in frames[1:]:
                i0 = 0
                for buf in f:
                    if not buf.is_gap:
                        out[..., i0 : i0 + buf.samples] += buf.data
                    i0 += buf.samples

        return TSFrame(
            buffers=[
                SeriesBuffer(
                    offset=frames[0].offset,
                    sample_rate=frames[0].sample_rate,
                    data=out,
                    shape=shape,
                )
            ],
            EOS=frames[0].EOS,
        )
