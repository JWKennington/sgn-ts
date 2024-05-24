from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..base import SeriesBuffer, TransformElement


@dataclass
class Matmul(TransformElement):
    """
    Performs matrix multiplication with provided matrix.

    Parameters:
    -----------
    matrix: Sequence[Any]
        the matrix to multiple the data with
    """

    matrix: Sequence[Any] = None

    def __post_init__(self):
        super().__post_init__()

    def pull(self, pad, bufs):
        """
        Assumes there is only one sink pad, if the user wants 
        to matmul multitple channels of data, 
        connect multiple matmul elements
        """
        self.inbufs = bufs

    def transform(self, pad):
        """
        The transform buffer just update the name to show the graph history.
        Useful for proving it works.  "EOS" is set if any input buffers are at EOS.
        """
        inbufs = self.inbufs
        EOS = inbufs[-1].EOS
        #metadata = inbufs.metadata
        # if metadata is None:
        metadata = {}
        metadata["cnt:%s" % inbufs[-1].metadata["name"]] = inbufs[-1].metadata["cnt"]
        metadata["cnt"] = inbufs[-1].metadata["cnt"]
        metadata["name"] = "%s -> '%s'" % (
            inbufs[-1].metadata["name"],
            pad.name,
        )

        # transform all the input data
        data = np.concatenate([b.data for b in inbufs],axis=-1)
        data = np.matmul(self.matrix, data)
        return [SeriesBuffer(
            offset=inbufs[0].offset,
            noffset=sum(b.noffset for b in inbufs),
            data=data,
            offset_ref_t0=inbufs[0].offset_ref_t0,
            metadata=metadata,
            EOS=EOS,
        )]
