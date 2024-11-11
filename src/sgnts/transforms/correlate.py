from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy
from sgn.base import SourcePad

from sgnts.base import AdapterConfig, Array, Offset, SeriesBuffer, TSFrame, TSTransform


@dataclass
class Correlate(TSTransform):
    """Correlates input data with filters

    Args:
        filters:
            Array, the filter to correlate over
    """

    sample_rate: int = None
    filters: Optional[Array] = None

    def __post_init__(self):
        assert self.filters is not None
        self.shape = self.filters.shape
        if self.adapter_config is None:
            self.adapter_config = AdapterConfig()
        self.adapter_config.overlap = (
            Offset.fromsamples(self.shape[-1] - 1, self.sample_rate),
            0,
        )
        self.adapter_config.pad_zeros_startup = False
        super().__post_init__()
        assert (
            len(self.sink_pads) == 1 and len(self.source_pads) == 1
        ), "only one sink_pad and one source_pad is allowed"

    def corr(self, data: Array) -> Array:
        """Correlate an array of data with an array of filters.

        Args:
            data:
                Array, the data to correlate with the filters

        Returns:
            Array, the result of the correlation
        """
        # FIXME: try with array ops
        os = []
        shape = self.filters.shape
        self.filters = self.filters.reshape(-1, shape[-1])
        for j in range(self.filters.shape[0]):
            os.append(scipy.signal.correlate(data, self.filters[j], mode="valid"))
        return np.vstack(os).reshape(shape[:-1] + (-1,))

    def transform(self, pad: SourcePad) -> TSFrame:
        """Correlate data in the buffers in the frames with filters.

        Args:
            pad:
                SourcePad, the source pad to produce frames

        Returns:
            TSFrame, the output TSFrame
        """
        outbufs = []
        outoffsets = self.preparedoutoffsets[self.sink_pads[0]]
        frames = self.preparedframes[self.sink_pads[0]]
        for i, buf in enumerate(frames):
            assert buf.sample_rate == self.sample_rate
            if buf.is_gap:
                data = None
            else:
                # FIXME: Are there multi-channel correlation in numpy or scipy?
                # FIXME: consider multi-dimensional filters
                data = self.corr(buf.data)
            outoffset = outoffsets[i]
            outbufs.append(
                SeriesBuffer(
                    offset=outoffset["offset"],
                    sample_rate=buf.sample_rate,
                    data=data,
                    shape=self.shape[:-1]
                    + (Offset.tosamples(outoffset["noffset"], buf.sample_rate),),
                )
            )
        return TSFrame(buffers=outbufs, EOS=frames.EOS)
