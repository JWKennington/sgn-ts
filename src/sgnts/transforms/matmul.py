from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sgn.base import SourcePad

from sgnts.base import Array, SeriesBuffer, TSFrame, TSTransform


@dataclass
class Matmul(TSTransform):
    """Performs matrix multiplication with provided matrix.

    Args:
        matrix:
            Sequence[Any], the matrix to multiply the data with, out = matrix x data
    """

    matrix: Optional[Array] = None

    def __post_init__(self):
        super().__post_init__()
        assert (
            len(self.sink_pads) == 1 and len(self.source_pads) == 1
        ), "only one sink_pad and one source_pad is allowed"

    def matmul(self, a: Array, b: Array) -> Array:
        """Matrix multiplication of two arrays.
            out = a x b

        Args:
            a:
                Array, the first array
            b:
                Array, the second array

        Returns:
            Array, the result of the matrix multiplication
        """
        return np.matmul(a, b)

    def transform(self, pad: SourcePad) -> TSFrame:
        """Matmul of a matrix with the incoming data.

        Args:
            pad:
                SourcePad, the source pad that outputs the transformed frame

        Returns:
            TSFrame, the output TSFrame
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
