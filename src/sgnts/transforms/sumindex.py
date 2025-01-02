from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
from typing import Optional

from sgn.base import SourcePad

from sgnts.base import ArrayBackend, NumpyBackend, SeriesBuffer, TSFrame, TSTransform


@dataclass
class SumIndex(TSTransform):
    """Sum array values over slices in the zero-th dimension.

    Args:
        sl:
            list[slice], the slices to sum over
        backend:
            type[ArrayBackend], the wrapper around array operations.
    """

    sl: Optional[list[slice]] = None
    backend: type[ArrayBackend] = NumpyBackend

    def __post_init__(self):
        super().__post_init__()
        assert self.sl is not None
        for sl in self.sl:
            assert isinstance(sl, slice)

    # FIXME: wraps are not playing well with mypy.  For now ignore and hope
    # that a future version of mypy will be able to handle this
    @wraps(TSTransform.new)
    def new(self, pad: SourcePad) -> TSFrame:  # type: ignore
        frame = self.preparedframes[self.sink_pads[0]]

        outbufs = []
        for buf in frame:
            if buf.is_gap:
                out = None
            else:
                data = buf.data
                data_all = []
                # NOTE mypy complains about None not being iterable but None
                # should actually be impossible at this point.
                assert self.sl is not None
                for sl in self.sl:
                    if sl.stop - sl.start == 1:
                        data_all.append((data[sl.start, :, :]))
                    else:
                        data_all.append(self.backend.sum(data[sl, :, :], axis=0))

                out = self.backend.stack(data_all)

            # NOTE mypy complains about None not being iterable but None should
            # actually be impossible at this point.
            assert self.sl is not None
            outbuf = SeriesBuffer(
                offset=buf.offset,
                sample_rate=buf.sample_rate,
                data=out,
                shape=(len(self.sl),) + buf.shape[-2:],
            )
        outbufs.append(outbuf)

        return TSFrame(buffers=outbufs, EOS=frame.EOS, metadata=frame.metadata)
