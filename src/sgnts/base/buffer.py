from sgn.base import *
from .offset import *
from collections.abc import Sequence
from typing import Any

@dataclass
class SeriesBuffer(Buffer):
    """Timeseries buffer with associated metadata.

    Parameters
    ----------
    offset : int
        The number of offset samples (defined at sample rate OFFSET_RATE)
        since offset_ref_t0. Similar to "t0".
    noffset : int
        The number of offset samples (defined at sample rate OFFSET_RATE)
        in the buffer. Similar to "duration".
    offset_ref_t0 : int
        The reference time to start the offset counter, in nanoseconds.
    data : Sequence
        The timeseries data.

    """
    offset: int = None
    noffset: int = None
    offset_ref_t0: int = None
    data: Sequence[Any] = None

    @property
    def t0(self):
        return self.offset_ref_t0 + Offset.offset2ns(self.offset)

    @property
    def duration(self):
        return Offset.offset2ns(self.noffset)

    @property
    def end(self):
        return self.t0 + self.duration

    @property
    def size(self):
        return self.data.shape[-1]

    @property
    def sample_rate(self):
        return int(self.size / Offset.offset2sec(self.noffset))

