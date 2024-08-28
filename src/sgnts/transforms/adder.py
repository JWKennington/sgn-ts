from dataclasses import dataclass, field

from ..base import Audioadapter, SeriesBuffer, TSFrame, TSTransform, ArrayOps

import numpy as np


@dataclass
class Adder(TSTransform):
    """
    Add up all the buffers from all the sink pads
    """

    lib: int = ArrayOps
    coeff_map: dict[str, float] = None
    addslices_map: dict[str, tuple[slice]] = None

    def __post_init__(self):
        super().__post_init__()

    def transform(self, pad):
        frames = list(self.preparedframes.values())
        if self.coeff_map is not None:
            keys = list(k.name.split(":")[-1] for k in self.preparedframes.keys())
        assert len(set(f.sample_rate for f in frames)) == 1
        assert len(set(f.offset for f in frames)) == 1
        assert len(set(f.end_offset for f in frames)) == 1

        if self.addslices_map is None:
            assert len(set(f.shape for f in frames)) == 1
        else:
            assert len(set(f.shape[-1] for f in frames)) == 1

        if all(frame.is_gap for frame in frames):
            out = None
            shape = frames[0].shape
        else:
            # use the first frame as basis
            if len(frames[0].buffers) == 1:
                out = frames[0].buffers[0].filleddata(self.lib.zeros_func)
            else:
                out = self.lib.cat_func(
                    [buf.filleddata(self.lib.zeros_func) for buf in frames[0]], axis=-1
                )
            if self.coeff_map is not None:
                out *= self.coeff_map[keys[0]]
            shape = out.shape
            # add to the first frame
            for i, f in enumerate(frames[1:]):
                if self.coeff_map is not None:
                    coeff = self.coeff_map[keys[i + 1]]
                else:
                    coeff = 1
                i0 = 0
                for buf in f:
                    if not buf.is_gap:
                        if self.addslices_map is None:
                            out[..., i0 : i0 + buf.samples] += buf.data * coeff
                        else:
                            slices = self.addslices_map[keys[i + 1]] + (
                                slice(i0, i0 + buf.samples),
                            )
                            out[slices] += buf.data * coeff

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
            metadata=frames[0].metadata,
        )
