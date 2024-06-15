from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any
import numpy

from sgn.base import Frame

from .offset import Offset
from .slice_tools import TSSlice


@dataclass
class SeriesBuffer:
    """Timeseries buffer with associated metadata.

    Parameters
    ----------
    offset : int
        The number of offset samples (defined at sample rate OFFSET_RATE)
        since Offset.offset_ref_t0. Similar to "t0".
    noffset : int
        The number of offset samples (defined at sample rate OFFSET_RATE)
        in the buffer. Similar to "duration".
    sample_rate : int
        The sample rate belonging to the set of Offset.ALLOWED_RATES
    channels : tuple
        The channels in the data, can be multi-dimensional. If channels =
        (A, B), and the size of data is N, the shape of the data array is
        (A, B, N).
    data : Sequence
        The timeseries data or None. If not None, the inferred sample
        rate must equal the provided sample rate

    """

    offset: int = None
    noffset: int = None
    sample_rate: int = None
    channels: tuple = None
    data: Sequence[Any] = None

    def __post_init__(self):
        assert isinstance(self.offset, int)
        assert isinstance(self.noffset, int)
        assert isinstance(self.channels, tuple)
        assert self.sample_rate in Offset.ALLOWED_RATES
        if self.data is not None:
            assert self.__check_data()

    def __repr__(self):
        with numpy.printoptions(threshold=3, edgeitems=1):
            return (
                "SeriesBuffer(offset=%d, noffset=%d, size=%d, duration=%d, data=%s)"
                % (
                    self.offset,
                    self.noffset,
                    self.size,
                    self.duration,
                    self.data,
                )
            )

    @property
    def slice(self):
        return TSSlice(self.offset, self.end_offset)

    @property
    def t0(self):
        return Offset.offset_ref_t0 + Offset.tons(self.offset)

    @property
    def duration(self):
        return Offset.tons(self.noffset)

    @property
    def end(self):
        return self.t0 + self.duration

    @property
    def end_offset(self):
        return self.offset + self.noffset

    @property
    def size(self):
        if self.data is None:
            return int(self.sample_rate * Offset.tosec(self.noffset))
            return Offset.tosamples(self.noffset, self.sample_rate)
        else:
            return self.data.shape[-1]

    def __check_data(self):
        return (
            self.sample_rate == int(self.size / Offset.tosec(self.noffset))
        ) and (
            (self.channels == self.data.shape[:-1])
        )

    @property
    def is_gap(self):
        if self.data is None:
            return True
        else:
            return False

    @property
    def shape(self):
        return self.channels + (self.size,)

    @property
    def filleddata(self):
        if self.data is not None:
            return self.data
        else:
            return numpy.zeros(self.shape)

    def __contains__(self, item):
        if isinstance(item, int):
            return self.offset <= item < self.end_offset
        else:
            return False

    def __lt__(self, item):
        if isinstance(item, int):
            return self.end_offset < item

    def __le__(self, item):
        if isinstance(item, int):
            return self.end_offset <= item

    def __ge__(self, item):
        if isinstance(item, int):
            return self.offset >= item

    def __gt__(self, item):
        if isinstance(item, int):
            return self.offset > item

    def pad_buffer(self, off, data=None):
        assert off < self.offset
        return SeriesBuffer(
            offset=off,
            noffset=self.offset - off,
            sample_rate=self.sample_rate,
            channels=self.channels,
            data=data,
        )

    def split(self, off):
        assert self.offset <= off < self.end_offset
        midoffset = off - self.offset
        midsamples = Offset.tosamples(midoffset, self.sample_rate)
        return SeriesBuffer(
            offset=self.offset,
            noffset=midoffset,
            sample_rate=self.sample_rate,
            channels=self.channels,
            data=None if self.data is None else self.data[:midsamples,],
        ), SeriesBuffer(
            offset=self.offset + midoffset,
            noffset=self.noffset - midoffset,
            sample_rate=self.sample_rate,
            channels=self.channels,
            data=None if self.data is None else self.data[midsamples:,],
        )


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
