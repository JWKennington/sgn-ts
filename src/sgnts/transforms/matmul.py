from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
from typing import Optional

from sgn.base import SourcePad

from sgnts.base import (
    Array,
    ArrayBackend,
    NumpyBackend,
    SeriesBuffer,
    TSFrame,
    TSTransform,
)


@dataclass
class Matmul(TSTransform):
    """Performs matrix multiplication with provided matrix.

    Args:
        matrix:
            Optional[Array], the matrix to multiply the data with, out = matrix x data
        backend:
            type[ArrayBackend], the array backend for array operations
    """

    matrix: Optional[Array] = None
    backend: type[ArrayBackend] = NumpyBackend

    def __post_init__(self):
        super().__post_init__()
        assert (
            len(self.sink_pads) == 1 and len(self.source_pads) == 1
        ), "only one sink_pad and one source_pad is allowed"
        assert self.matrix is not None
        self.shape = self.matrix.shape

    # FIXME: wraps are not playing well with mypy.  For now ignore and hope
    # that a future version of mypy will be able to handle this
    @wraps(TSTransform.new)
    def new(self, pad: SourcePad) -> TSFrame:  # type: ignore
        outbufs = []
        # loop over the input data, only perform matmul on non-gaps
        frame = self.preparedframes[self.sink_pads[0]]
        for inbuf in frame:
            is_gap = inbuf.is_gap

            if is_gap:
                data = None
            else:
                data = self.backend.matmul(self.matrix, inbuf.data)

            outbuf = SeriesBuffer(
                offset=inbuf.offset,
                sample_rate=inbuf.sample_rate,
                data=data,
                shape=self.shape[:-1] + (inbuf.samples,),
            )
            outbufs.append(outbuf)

        return TSFrame(buffers=outbufs, EOS=frame.EOS, metadata=frame.metadata)
