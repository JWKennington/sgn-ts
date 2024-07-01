from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..base import SeriesBuffer, TSTransform, TSFrame


@dataclass
class Matmul(TSTransform):
    """
    Performs matrix multiplication with provided matrix.

    If a pad receives more then one buffer, matmul will be performed
    on the list of buffers one by one. The source pad will also output
    a list of buffers.

    Parameters:
    -----------
    matrix: Sequence[Any]
        the matrix to multiply the data with, out = matrix x data

    Assumptions:
    ------------
    - There is only one sink pad and one source pad
    """

    matrix: Sequence[Any] = None

    def __post_init__(self):
        super().__post_init__()
        assert (
            len(self.sink_pads) == 1 and len(self.source_pads) == 1
        ), "only one sink_pad and one source_pad is allowed"

    def matmul(self, a, b):
        return np.matmul(a, b)

    def transform(self, pad):
        """
        Matmul over list of buffers
        """
        outbufs = []
        # loop over the input data, only perform matmul on non-gaps
        frame = self.preparedframes[self.sink_pads[0]]
        for inbuf in frame:
            is_gap = inbuf.is_gap

            if is_gap:
                data = None
            else:
                data = self.matmul(self.matrix, inbuf.data)

            outbuf = SeriesBuffer(
                offset=inbuf.offset,
                sample_rate=inbuf.sample_rate,
                data=data,
                shape=self.matrix.shape[:-1] + (inbuf.samples,),
            )
            outbufs.append(outbuf)

        return TSFrame(buffers=outbufs, EOS=frame.EOS, metadata=frame.metadata)
