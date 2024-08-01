from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any
import numpy

from sgn.base import Frame

from .offset import Offset
from .slice_tools import TSSlice, TSSlices


@dataclass
class SeriesBuffer:
    """Timeseries buffer with associated metadata.

    Parameters
    ----------
    offset : int
        The number of offset samples (defined at sample rate Offset.OFFSET_RATE)
        since Offset.offset_ref_t0. Similar to "t0".
    sample_rate : int
        The sample rate belonging to the set of Offset.ALLOWED_RATES
    data : Sequence
        The timeseries data or None.
    shape : tuple
        The shape of the data regardless of gaps. Required if data is None, and
        represents the shape of the absent data.
    """

    offset: int = None
    sample_rate: int = None
    data: Sequence[Any] = None
    shape: tuple = None

    def __post_init__(self):
        assert isinstance(self.offset, int)
        assert self.sample_rate in Offset.ALLOWED_RATES
        if self.data is None:
            assert isinstance(self.shape, tuple)
        elif isinstance(self.data, int) and self.data == 1:
            assert isinstance(self.shape, tuple)
            self.data = numpy.ones(self.shape)
        elif isinstance(self.data, int) and self.data == 0:
            assert isinstance(self.shape, tuple)
            self.data = numpy.zeros(self.shape)
        elif self.shape is None:
            self.shape = self.data.shape
        else:
            assert self.shape == self.data.shape
            for t in self.shape:
                assert isinstance(t, int)

    @staticmethod
    def fromoffsetslice(offslice, sample_rate, data=None, channels=()):
        shape = channels + (
            Offset.tosamples(offslice.stop - offslice.start, sample_rate),
        )
        return SeriesBuffer(
            offset=offslice.start, sample_rate=sample_rate, data=data, shape=shape
        )

    def __repr__(self):
        with numpy.printoptions(threshold=3, edgeitems=1):
            return (
                "SeriesBuffer(offset=%d, offset_end=%d, shape=%s, sample_rate=%d, duration=%d, data=%s)"
                % (
                    self.offset,
                    self.end_offset,
                    self.shape,
                    self.sample_rate,
                    self.duration,
                    self.data,
                )
            )

    def __bool__(self):
        return self.data is not None

    @property
    def slice(self):
        return TSSlice(self.offset, self.end_offset)

    @property
    def noffset(self):
        return Offset.fromsamples(self.samples, self.sample_rate)

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
    def samples(self):
        return self.shape[-1]

    @property
    def is_gap(self):
        if self.data is None:
            return True
        else:
            return False

    def filleddata(self, zeros_func):
        if self.data is not None:
            return self.data
        else:
            return zeros_func(self.shape)

    def __contains__(self, item):
        if isinstance(item, int):
            return self.offset <= item < self.end_offset
        else:
            return False

    def __lt__(self, item):
        if isinstance(item, int):
            return self.end_offset < item
        elif isinstance(item, SeriesBuffer):
            return self.end_offset < item.end_offset

    def __le__(self, item):
        if isinstance(item, int):
            return self.end_offset <= item
        elif isinstance(item, SeriesBuffer):
            return self.end_offset <= item.end_offset

    def __ge__(self, item):
        if isinstance(item, int):
            return self.offset >= item
        elif isinstance(item, SeriesBuffer):
            return self.end_offset >= item.end_offset

    def __gt__(self, item):
        if isinstance(item, int):
            return self.offset > item
        elif isinstance(item, SeriesBuffer):
            return self.end_offset > item.end_offset

    def pad_buffer(self, off, data=None):
        assert off < self.offset
        return SeriesBuffer(
            offset=off,
            sample_rate=self.sample_rate,
            data=data,
            shape=self.shape[:-1]
            + (Offset.tosamples(self.offset - off, self.sample_rate),),
        )

    def sub_buffer(self, slc, gap=False):
        assert slc in self.slice
        startsamples, stopsamples = Offset.tosamples(
            slc.start - self.offset, self.sample_rate
        ), Offset.tosamples(slc.stop - self.offset, self.sample_rate)
        gap = gap or self.data is None
        if not gap:
            data = self.data[..., startsamples:stopsamples]
        else:
            data = None
        return SeriesBuffer(
            offset=slc.start,
            sample_rate=self.sample_rate,
            data=data,
            shape=self.shape[:-1] + (stopsamples - startsamples,),
        )

    def split(self, boundaries, contiguous=False):
        out = []
        if isinstance(boundaries, int):
            boundaries = TSSlices(self.slice.split(boundaries))
        if not isinstance(boundaries, TSSlices):
            raise NotImplementedError
        for slc in boundaries.slices:
            assert slc in self.slice
            out.append(self.sub_buffer(slc))
        if contiguous:
            gap_boundaries = boundaries.invert(self.slice)
            for slc in gap_boundaries.slices:
                out.append(self.sub_buffer(slc, gap=True))
        return sorted(out)


@dataclass
class TSFrame(Frame):
    """An sgn Frame object that holds a list of buffers

    Parameters
    ----------
    buffers : list
        List of SeriesBuffers

    """

    buffers: int = None

    def __post_init__(self):
        super().__post_init__()
        assert len(self.buffers) > 0
        self.is_gap = all([b.is_gap for b in self.buffers])

    def __getitem__(self, item):
        return self.buffers[item]

    def __iter__(self):
        return iter(self.buffers)

    def __repr__(self):
        out = "%s ::" % self.metadata["__graph__"]
        for buf in self:
            out += "\n\t%s" % buf
        return out

    @property
    def offset(self):
        return self.buffers[0].offset

    @property
    def end_offset(self):
        return self.buffers[-1].end_offset

    @property
    def slice(self):
        return TSSlice(self.offset, self.end_offset)

    @property
    def shape(self):
        return self.buffers[0].shape[:-1] + (sum(b.samples for b in self.buffers),)

    @property
    def sample_rate(self):
        return self.buffers[0].sample_rate
