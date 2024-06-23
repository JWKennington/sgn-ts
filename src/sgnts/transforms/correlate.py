from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy
from sgn.base import TransformElement

from ..base import Audioadapter, Offset, SeriesBuffer, TSFrame, TSTransform


@dataclass
class Correlate(TSTransform):
    """
    Correlates input data with filters

    Parameters:
    -----------
    filters: Sequence[Any]
        the filter to correlate over

    Assumptions:
    ------------
    - There is only one sink pad and one source pad
    """

    filters: Sequence[Any] = None

    def __post_init__(self):
        assert self.filters is not None
        self.overlap = (self.filters.shape[-1] - 1, 0)
        super().__post_init__()
        assert (
            len(self.sink_pads) == 1 and len(self.source_pads) == 1
        ), "only one sink_pad and one source_pad is allowed"

    def transform(self, pad):
        """
        Correlates data with filters
        """
        data = self.preparedframes[self.sink_pads[0]].buffers[0].data

        # FIXME: consider gaps
        # FIXME: Are there multi-channel correlation in numpy or scipy?
        # FIXME: consider multi-dimensional filters
        os = []
        for i in range(self.filters.shape[0]):
            os.append(scipy.signal.correlate(data, self.filters[i], mode="valid"))
        out = np.vstack(os)

        outframe = self.preparedoutframes[self.sink_pads[0]]
        outshape = self.filters.shape[:-1] + (outframe.shape[-1],)
        outframe.buffers[0].update_data(out, shape=outshape)
        return outframe
