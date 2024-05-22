from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy
from sgn.base import TransformElement

from ..base import Audioadapter, Offset, SeriesBuffer


@dataclass
class Correlate(TransformElement):
    """
    A fake transform element.
    """

    filters: Sequence[Any] = None

    def __post_init__(self):
        self.inbuf = {}
        self.audioadapters = {}
        assert self.filters is not None
        super().__post_init__()

    def get_buffer(self, pad, buf):
        self.inbuf[pad] = buf
        if pad not in self.audioadapters:
            self.audioadapters[pad] = Audioadapter()
        self.audioadapters[pad].push(buf)
        self.nnew = buf.data.shape[-1]
        self.this_segment = (buf.offset, buf.offset + buf.noffset)
        self.this_noffset = buf.noffset
        self.sample_rate = buf.sample_rate
        self.offset_ref_t0 = buf.offset_ref_t0

    def transform_buffer(self, pad):
        """
        The transform buffer just update the name to show the graph history.
        Useful for proving it works.  "EOS" is set if any input buffers are at EOS.
        """
        EOS = any(b.EOS for b in self.inbuf.values())
        metadata = {}
        for b in self.inbuf.values():
            metadata["cnt:%s" % b.metadata["name"]] = b.metadata["cnt"]
            metadata["cnt"] = b.metadata["cnt"]
        metadata["name"] = "%s -> '%s'" % (
            "+".join(b.metadata["name"] for b in self.inbuf.values()),
            pad.name,
        )
        nfilter_samples = self.filters.shape[-1]

        A = self.audioadapters[self.sink_pads[0]]

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

        return SeriesBuffer(
            offset=self.this_segment[0],
            noffset=self.this_noffset,
            data=out,
            offset_ref_t0=self.offset_ref_t0,
            metadata=metadata,
            EOS=EOS,
        )
