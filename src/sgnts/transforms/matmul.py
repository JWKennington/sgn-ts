from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..base import SeriesBuffer, TransformElement, TSFrame


@dataclass
class Matmul(TransformElement):
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
        assert len(self.sink_pads) == 1 and len(self.source_pads) == 1, (
        "only one sink_pad and one source_pad is allowed")

    def pull(self, pad, bufs):
        """
        Assumes there is only one sink pad, if the user wants 
        to matmul multitple channels of data, 
        connect multiple matmul elements
        """
        self.inbufs = bufs

    def transform(self, pad):
        """
        Matmul over list of buffers
        """
        inbufs = self.inbufs
        outbufs = []
        # loop over the input data, only perform matmul on non-gaps
        EOS = inbufs.EOS
        metadata = {}
        metadata["cnt:%s" % inbufs.metadata["name"]] = inbufs.metadata["cnt"]
        metadata["cnt"] = inbufs.metadata["cnt"]
        metadata["name"] = "%s -> '%s'" % (
            inbufs.metadata["name"],
            pad.name,
        )
        for inbuf in inbufs:
            is_gap = inbuf.is_gap

            if is_gap:
                # produce a zeros buffer
                if inbuf.data is None:
                    data = None
                else:
                    data = np.zeros(self.matrix.shape[:-1]+(inbuf.size,))
            else:
                data = np.matmul(self.matrix, inbuf.data)

            outbuf = SeriesBuffer(
                offset=inbuf.offset,
                noffset=inbuf.noffset,
                data=data,
                offset_ref_t0=inbuf.offset_ref_t0,
                is_gap = inbuf.is_gap,
            )
            outbufs.append(outbuf)

        return TSFrame(buffers=outbufs, metadata=metadata, EOS=EOS)
