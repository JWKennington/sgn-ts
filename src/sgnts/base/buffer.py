from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any
import numpy

from sgn.base import Frame

from .offset import Offset


@dataclass
class SeriesBuffer:
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
    is_gap: bool = None

    def __post_init__(self):
        assert isinstance(self.offset, int)
        assert isinstance(self.noffset, int)
        assert isinstance(self.offset_ref_t0, int)

    def __repr__(self):
        with numpy.printoptions(threshold=3, edgeitems=1):
            return "SeriesBuffer(offset=%d, noffset=%d, offset_ref_t0=%d, size=%d, duration=%d, data=%s)" % (self.offset, self.noffset, self.offset_ref_t0, 0 if self.size is None else self.size, self.duration, self.data)

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
    def end_offset(self):
        return self.offset + self.noffset

    @property
    def size(self):
        if self.data is None:
            return None
        else:
            return self.data.shape[-1]

    @property
    def sample_rate(self):
        return int(self.size / Offset.offset2sec(self.noffset))

    def __contains__(self, item):
        if isinstance(item, int):
            return self.t0 <= item < self.end
        else:
            return False

    def __lt__(self, item):
        if isinstance(item, int):
            return self.end < item

    def __le__(self, item):
        if isinstance(item, int):
            return self.end <= item

    def __ge__(self, item):
        if isinstance(item, int):
            return self.end > item

    def split(self, ts):
        assert self.t0 <= ts < self.end
        midoffset = int(round(Offset.ns2offset(ts - self.t0)))
        midsamples = int(round(Offset.offset2nsamples(midoffset, self.sample_rate)))
        return SeriesBuffer(offset = self.offset, noffset = midoffset, offset_ref_t0 = self.offset_ref_t0, is_gap = self.is_gap, data = None if self.data is None else self.data[:midsamples,], metadata = self.metadata), SeriesBuffer(offset = self.offset + midoffset, noffset = self.noffset - midoffset, offset_ref_t0 = self.offset_ref_t0, is_gap = self.is_gap, data = None if self.data is None else self.data[midsamples:,], metadata = self.metadata)

    def pad_buffer(self, t0, is_gap = True, data = None):
        assert t0 < self.t0
        delta_offset = int(round(Offset.ns2offset(t0 - self.t0)))
        new_offset = int(round(Offset.ns2offset(t0 - self.offset_ref_t0)))
        new_noffset = int(round(Offset.ns2offset(self.t0 - t0)))
        return SeriesBuffer(offset = new_offset, noffset = new_noffset, offset_ref_t0 = self.offset_ref_t0, is_gap = is_gap, data = data)


@dataclass
class TSFrame(Frame):
    """An sgn Frame object that holds a list of buffers

    Parameters
    ----------
    buffers : list
        List of SeriesBuffers

    """

    buffers: int = None

    def __getitem__(self, item):
        return self.buffers[item]

    def __iter__(self):
        return iter(self.buffers)

    def __repr__(self):
        out = "%s ::" % self.metadata["__graph__"]
        for buf in self:
            out += "\n\t%s" % buf
        return out
