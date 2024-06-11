from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy
from sgn.base import TransformElement

from ..base import Audioadapter, Offset, SeriesBuffer, TSFrame


@dataclass
class Correlate(TransformElement):
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
        self.audioadapter = Audioadapter()
        super().__post_init__()
        assert len(self.sink_pads) == 1 and len(self.source_pads) == 1, (
        "only one sink_pad and one source_pad is allowed")

    def pull(self, pad, bufs):
        """
        Assumes there is only one sink pad, if the user wants 
        to correltate multitple channels of data, 
        connect multiple correlate elements
        """
        self.inbufs = bufs
        self.nnew = 0 # len of the new data
        for buf in bufs:
            self.audioadapter.push(buf)
            self.nnew += buf.size
        offset0 = bufs[0].offset
        offset1 = bufs[-1].offset + bufs[-1].noffset

        # the offset segment we want to produce
        self.this_segment = (offset0, offset1)
        self.this_noffset = offset1 - offset0
        self.sample_rate = bufs[0].sample_rate
        self.offset_ref_t0 = bufs[0].offset_ref_t0

    def transform(self, pad):
        """
        Correlates data with filters
        """
        inbufs = self.inbufs
        EOS = inbufs.EOS
        metadata = {}
        metadata["cnt:%s" % inbufs.metadata["name"]] = inbufs.metadata["cnt"]
        metadata["cnt"] = inbufs.metadata["cnt"]
        metadata["name"] = "%s -> '%s'" % (
            inbufs.metadata["name"],
            pad.name,
        )

        nfilter_samples = self.filters.shape[-1]

        A = self.audioadapter

        shift = Offset.nsamples2offset(nfilter_samples - 1, self.sample_rate)

        start = max(0, self.this_segment[0] - shift)

        request_segment = (start, self.this_segment[1])

        data, _, _ = A.copy_samples_by_offset_segment(request_segment)
        nworkspace = self.nnew + nfilter_samples - 1
        ndata = data.shape[-1]

        if ndata < nworkspace:
            data = np.pad(data, (nworkspace - ndata, 0), "constant")

        os = []
        for i in range(self.filters.shape[0]):
            os.append(scipy.signal.correlate(data, self.filters[i], mode="valid"))
        out = np.vstack(os)

        next_offset = request_segment[1] - shift

        if next_offset > A.get_available_offset_segment()[0]:
            A.flush_samples_by_end_offset_segment(request_segment[1] - shift)

        return TSFrame(buffers=[SeriesBuffer(
            offset=self.this_segment[0],
            noffset=self.this_noffset,
            data=out,
            offset_ref_t0=self.offset_ref_t0,
        )], metadata=metadata, EOS=EOS)
