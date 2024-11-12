from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

import numpy
from sgn.base import Frame

from sgnts.base.array_ops import (
    Array,
    ArrayBackend,
    NumpyArray,
    NumpyBackend,
    TorchArray,
    TorchBackend,
)
from sgnts.base.offset import Offset
from sgnts.base.slice_tools import TSSlice, TSSlices
from sgnts.base.time import Time


@dataclass
class SeriesBuffer:
    """Timeseries buffer with associated metadata.

    Args:
        offset:
            int, the offset of the buffer. See Offset class for definitions.
        sample_rate:
            int, the sample rate belonging to the set of Offset.ALLOWED_RATES
        data:
            Optional[Union[int, Array]], the timeseries data or None.
        shape:
            tuple, the shape of the data regardless of gaps. Required if data is None
            or int, and represents the shape of the absent data.
        backend:
            type[ArrayBackend], default NumpyBackend, the wrapper around array
            operations
    """

    offset: int
    sample_rate: int
    data: Optional[Union[int, Array]] = None
    shape: tuple = (-1,)
    backend: type[ArrayBackend] = NumpyBackend

    def __post_init__(self):
        assert isinstance(self.offset, int)
        if self.sample_rate not in Offset.ALLOWED_RATES:
            raise ValueError(
                "%s not in allowed rates %s" % (self.sample_rate, Offset.ALLOWED_RATES)
            )
        if self.data is None:
            assert self.shape != (-1,)
        elif isinstance(self.data, int) and self.data == 1:
            assert self.shape != (-1,)
            self.data = self.backend.ones(self.shape)
        elif isinstance(self.data, int) and self.data == 0:
            assert self.shape != (-1,)
            self.data = self.backend.zeros(self.shape)
        elif self.shape == (-1,):
            self.shape = self.data.shape
        else:
            assert self.shape == self.data.shape
            for t in self.shape:
                assert isinstance(t, int)

    @staticmethod
    def fromoffsetslice(
        offslice: TSSlice,
        sample_rate: int,
        data: Optional[Union[int, Array]] = None,
        channels: tuple[int, ...] = (),
    ) -> "SeriesBuffer":
        """Create a SeriesBuffer from a requested offset slice.

        Args:
            offslice:
                TSSlice, the offset slices the buffer spans
            sample_rate:
                int, the sample rate of the buffer
            data:
                Optional[Union[int, Array]], the data in the buffer
            channels:
                tuple[int, ...], the number of channels except the last dimension of the
                shape of the data, i.e., channels = data.shape[:-1]

        Returns:
            SeriesBuffer, the buffer that spans the requested offset slice
        """
        shape = channels + (
            Offset.tosamples(offslice.stop - offslice.start, sample_rate),
        )
        return SeriesBuffer(
            offset=offslice.start, sample_rate=sample_rate, data=data, shape=shape
        )

    def __repr__(self):
        with numpy.printoptions(threshold=3, edgeitems=1):
            return (
                "SeriesBuffer(offset=%d, offset_end=%d, shape=%s, sample_rate=%d,"
                " duration=%d, data=%s)"
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

    def set_data(self, data: Optional[Array] = None) -> None:
        """Set the data attribute to the newly provided data.

        Args:
            data:
                Optiona[Array], the new data to set to
        """
        if data is not None and self.shape != data.shape:
            raise ValueError("Data are incompatible shapes")
        # it really isn't clear to me if this should be by reference or copy...
        self.data = data

    @property
    def tarr(self) -> Array:
        """An array of time stamps for each sample of the data in the buffer, in
        seconds.

        Returns:
            Array, the time array
        """
        return (
            self.backend.arange(self.samples) / self.sample_rate
            + self.t0 / Time.SECONDS
        )

    def __eq__(self, value: object) -> bool:
        is_series_buffer = isinstance(value, SeriesBuffer)
        if not is_series_buffer:
            return False
        if not (value.shape == self.shape):
            return False
        if type(self.data) is not type(value.data):
            return False
        if isinstance(self.data, NumpyArray) and isinstance(value.data, NumpyArray):
            share_data = NumpyBackend.all(self.data == value.data)
        elif isinstance(self.data, TorchArray) and isinstance(value.data, TorchArray):
            share_data = TorchBackend.all(self.data == value.data)
        else:
            # Will need to expand this conditional if/when other data types are added
            return False
        share_offset = value.offset == self.offset
        share_sample_rate = value.sample_rate == self.sample_rate
        return share_data and share_offset and share_sample_rate

    @property
    def slice(self) -> TSSlice:
        """The offset slice that the buffer spans.

        Returns:
            TSSlices, the offset slice
        """
        return TSSlice(self.offset, self.end_offset)

    @property
    def noffset(self) -> int:
        """The number of offsets the buffer spans, which is the buffer's duration in
        terms of offsets.

        Returns:
            int, the offset duration
        """
        return Offset.fromsamples(self.samples, self.sample_rate)

    @property
    def t0(self) -> int:
        """The start time of the buffer, in integer nanoseconds.

        Returns:
            int, buffer start time
        """
        return Offset.offset_ref_t0 + Offset.tons(self.offset)

    @property
    def duration(self) -> int:
        """The duration of the buffer, in integer nanoseconds.

        Returns:
            int, the buffer duration
        """
        return Offset.tons(self.noffset)

    @property
    def end(self) -> int:
        """The end time of the buffer, in integer nanoseconds.

        Returns:
            int, buffer end time
        """
        return self.t0 + self.duration

    @property
    def end_offset(self) -> int:
        """The end offset of the buffer.

        Returns:
            int, buffer end offset
        """
        return self.offset + self.noffset

    @property
    def samples(self) -> int:
        """The number of samples the buffer carries.

        Return:
            int, the number of samples
        """
        return self.shape[-1]

    @property
    def is_gap(self) -> bool:
        """Whether the buffer is a gap. This is determined by whether the data is None.

        Returns:
            bool, whether the buffer is a gap
        """
        return self.data is None

    def filleddata(self, zeros_func) -> Array:
        """Fill the data with zeros if buffer is a gap, otherwise return the data.

        Args:
            zeros_func:
                the function to produce a zeros array

        Returns:
            Array, the filled data
        """
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

    def _insert(self, data, offset) -> None:
        """TODO workshop the name
        Adds data from a whose slice is
        fully contained within self's into self.
        Does not do safety checks."""
        insertion_index = Offset.tosamples(
            offset - self.offset, sample_rate=self.sample_rate
        )
        self.data[..., insertion_index : insertion_index + data.shape[-1]] += data

    @property
    def _backend_from_data(self):
        if isinstance(self.data, NumpyArray):
            return NumpyBackend
        elif isinstance(self.data, TorchArray):
            # FIXME: should this just throw an error?
            if self.data.device != TorchBackend.DEVICE:
                print(
                    f"Changing TorchBackend device from {TorchBackend.DEVICE} to"
                    f" {self.data.device}"
                )
                TorchBackend.set_device(self.data.device)
            if self.data.dtype != TorchBackend.DTYPE:
                print(
                    f"Changing TorchBackend dtype from {TorchBackend.DTYPE} to"
                    f" {self.data.dtype}"
                )
                TorchBackend.set_dtype(self.data.dtype)
            return TorchBackend
        else:
            return None

    def __add__(self, item: "SeriesBuffer") -> "SeriesBuffer":
        """Add two `SeriesBuffer`s, padding as necessary.

        Args:
            item:
                SeriesBuffer, The other component of the addition. Must be a
                SeriesBuffer, must have the same sample rate as self, and its data must
                be the same type (e.g. numpy array or pytorch Tensor)

        Returns:
            SeriesBuffer, The SeriesBuffer resulting from the addition
        """
        # Choose the correct backend
        # Handle polymorphism more smoothly in the future?
        # It's python so maybe this is the best option available
        if not isinstance(item, SeriesBuffer):
            raise TypeError("Both arguments must be of the SeriesBuffer type")
        # A bit convoluted, cases are:
        # - if both None then output gap
        # - if one None fill the gap and add with other's backend
        # - if neither None but disagree raise an error
        backend = self._backend_from_data
        if (backend != item._backend_from_data) and (
            item._backend_from_data is not None
        ):
            raise TypeError("Incompatible data types")
        if backend is None and item._backend_from_data is not None:
            backend = item._backend_from_data
        if self.shape[:-1] != item.shape[:-1]:
            raise ValueError("All dimensions except the padding dimension must match")
        if self.sample_rate != item.sample_rate:
            raise ValueError("Sample rates must match")
        new_buffer = self.fromoffsetslice(
            self.slice | item.slice,
            sample_rate=self.sample_rate,
            data=None,
            channels=self.shape[:-1],
        )
        if backend is None:
            return new_buffer

        new_buffer.data = new_buffer.filleddata(backend.zeros)
        self_filled_data = self.filleddata(backend.zeros)
        item_filled_data = item.filleddata(backend.zeros)

        new_buffer._insert(self_filled_data, self.offset)
        new_buffer._insert(item_filled_data, item.offset)

        return new_buffer

    def pad_buffer(
        self, off: int, data: Optional[Union[int, Array]] = None
    ) -> "SeriesBuffer":
        """Generate a buffer to pad before this buffer.

        Args:
            off:
                int, the offset to start the padding. Must be earlier than this buffer.
            data:
                Optional[Union[int, Array]], the data of the pad buffer

        Returns:
            SeriesBuffer, the pad buffer
        """
        assert off < self.offset
        return SeriesBuffer(
            offset=off,
            sample_rate=self.sample_rate,
            data=data,
            shape=self.shape[:-1]
            + (Offset.tosamples(self.offset - off, self.sample_rate),),
        )

    def sub_buffer(self, slc: TSSlice, gap: bool = False) -> "SeriesBuffer":
        """Generate a sub buffer whose offset slice is within this buffer.

        Args:
            slc:
                TSSlice, the offset slice of the sub buffer
            gap:
                bool, if True, set the sub buffer to a gap

        Returns:
            SeriesBuffer, the sub buffer
        """
        assert slc in self.slice
        startsamples, stopsamples = Offset.tosamples(
            slc.start - self.offset, self.sample_rate
        ), Offset.tosamples(slc.stop - self.offset, self.sample_rate)
        if not gap and self.data is not None and not isinstance(self.data, int):
            data = self.data[..., startsamples:stopsamples]
        else:
            data = None

        return SeriesBuffer(
            offset=slc.start,
            sample_rate=self.sample_rate,
            data=data,
            shape=self.shape[:-1] + (stopsamples - startsamples,),
        )

    def split(
        self, boundaries: Union[int, TSSlices], contiguous: bool = False
    ) -> list["SeriesBuffer"]:
        """Split the buffer according to the requested offset boundaries.

        Args:
            boundaries:
                Union[int, TSSlices], the offset boundaries to split the buffer into.
            contiguous:
                bool, if True, will generate gap buffers when there are discontinuities

        Returns:
            list[SeriesBuffer], a list of SeriesBuffers split up according to the
            offset boundaries
        """
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

    Args:
        buffers:
            list[SeriesBuffer], An iterable of SeriesBuffers
    """

    buffers: list[SeriesBuffer] = field(default_factory=list)

    def __post_init__(self):
        super().__post_init__()
        assert len(self.buffers) > 0
        self.__sanity_check(self.buffers)
        self.is_gap = all([b.is_gap for b in self.buffers])

    def __getitem__(self, item):
        return self.buffers[item]

    def __iter__(self):
        return iter(self.buffers)

    def __repr__(self):
        out = ""
        for buf in self:
            out += "\n\t%s" % buf
        return out

    def __len__(self):
        return len(self.buffers)

    def __sanity_check(self, bufs: list[SeriesBuffer]) -> None:
        """Sanity check that the buffers don't overlap nor have discontinuities.

        Args:
            bufs:
                list[SeriesBuffer], the buffers to perform the sanity check on
        """
        # FIXME: is there a smart way using TSSlics?
        if len(bufs) > 1:
            slices = [buf.slice for buf in bufs]
            off0 = slices[0].stop
            for sl in slices[1:]:
                assert off0 == sl.start
                off0 = sl.stop

    def set_buffers(self, bufs: list[SeriesBuffer]) -> None:
        """Set the buffers attribute to the bufs provided.

        Args:
            bufs:
                list[SeriesBuffers], the list of buffers to set to
        """
        self.__sanity_check(bufs)
        self.buffers = bufs

    @property
    def offset(self) -> int:
        """The offset of the TSFrame, which is the offset of the first buffer.

        Returns:
            int, the offset of the TSFrame
        """
        return self.buffers[0].offset

    @property
    def end_offset(self) -> int:
        """The end offset of the TSFrame, which is the end offset of the last buffer.

        Returns:
            int, the end offset of the TSFrame
        """
        return self.buffers[-1].end_offset

    @property
    def slice(self) -> TSSlice:
        """The offset slice of the TSFrame.

        Returns:
            TSSclie, the offset slice of the TSFrame
        """
        return TSSlice(self.offset, self.end_offset)

    @property
    def shape(self) -> tuple[int, ...]:
        """The shape of the TSFrame.

        Returns:
            tuple[int, ...], the shape of the TSFrame
        """
        return self.buffers[0].shape[:-1] + (sum(b.samples for b in self.buffers),)

    @property
    def sample_rate(self) -> int:
        """The sample rate of the TSFrame.

        Returns:
            int, the sample rate
        """
        return self.buffers[0].sample_rate

    @classmethod
    def from_buffer_kwargs(cls, **kwargs):
        """A short hand for the following:

        >>> buf = SeriesBuffer(**kwargs)
        >>> frame = TSFrame(buffers=[buf])
        """
        return cls(buffers=[SeriesBuffer(**kwargs)])

    def __next__(self):
        """
        return a new empty frame that is like the current one but advanced to the next offset, e.g.,

        >>> frame = TSFrame.from_buffer_kwargs(offset=0, sample_rate=2048, shape=(2048,))
        >>> print (frame)

                SeriesBuffer(offset=0, offset_end=16384, shape=(2048,), sample_rate=2048, duration=1000000000, data=None)
        >>> print (next(frame))
        """
        return self.from_buffer_kwargs(
            offset=self.end_offset, sample_rate=self.sample_rate, shape=self.shape
        )
