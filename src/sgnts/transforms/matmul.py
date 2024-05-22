from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..base import SeriesBuffer, TransformElement


@dataclass
class Matmul(TransformElement):
    """
    A fake transform element.
    """

    matrix: Sequence[Any] = None

    def __post_init__(self):
        self.inbuf = {}
        super().__post_init__()

    def get_buffer(self, pad, buf):
        self.inbuf[pad] = buf

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

        b = self.inbuf[self.sink_pads[0]]
        data = b.data

        data = np.matmul(self.matrix, data)
        return SeriesBuffer(
            offset=b.offset,
            noffset=b.noffset,
            data=data,
            offset_ref_t0=b.offset_ref_t0,
            metadata=metadata,
            EOS=EOS,
        )


transforms_registry += ("Matmul",)
