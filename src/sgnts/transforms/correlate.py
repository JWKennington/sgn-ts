from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy
from sgn.base import TransformElement

from ..base import (
    Audioadapter,
    Offset,
    SeriesBuffer,
    TSFrame,
    TSTransform,
    AdapterConfig,
)


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
        if self.adapter_config is None:
            self.adapter_config = AdapterConfig()
        self.adapter_config.overlap = (self.filters.shape[-1] - 1, 0)
        super().__post_init__()
        assert (
            len(self.sink_pads) == 1 and len(self.source_pads) == 1
        ), "only one sink_pad and one source_pad is allowed"

    def transform(self, pad):
        """
        Correlates data with filters
        """
        outbufs = []
        outoffsets = self.preparedoutoffsets[self.sink_pads[0]]
        frames = self.preparedframes[self.sink_pads[0]]
        for i, buf in enumerate(frames):
            if buf.is_gap:
                data = None
            else:
                # FIXME: consider gaps
                # FIXME: Are there multi-channel correlation in numpy or scipy?
                # FIXME: consider multi-dimensional filters
                os = []
                for j in range(self.filters.shape[0]):
                    os.append(
                        scipy.signal.correlate(buf.data, self.filters[j], mode="valid")
                    )
                data = np.vstack(os)
            outoffset = outoffsets[i]
            outbufs.append(
                SeriesBuffer(
                    offset=outoffset["offset"],
                    sample_rate=buf.sample_rate,
                    data=data,
                    shape=self.filters.shape[:-1]
                    + (Offset.tosamples(outoffset["noffset"], buf.sample_rate),),
                )
            )
        return TSFrame(buffers=outbufs, EOS=frames.EOS)
