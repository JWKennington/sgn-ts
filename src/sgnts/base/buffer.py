from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Union

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
class EventBuffer:
    """Event buffer with associated metadata.

    Args:
        ts:
            int, Start time of event buffer in ns
        te:
            int, End time of event buffer in ns
        data:
            Any, Data of the event
    """

    ts: Union[int, None] = None
    te: Union[int, None] = None
    data: Any = None

    def __post_init__(self):
        if (
            not isinstance(self.ts, int)
            or not isinstance(self.te, int)
            or not (self.ts <= self.te)
        ):
            raise ValueError(
                "ts and te must be integers and ts must be <= te,"
                f"got {self.ts} and {self.te}"
            )

    def __repr__(self):
        with numpy.printoptions(threshold=3, edgeitems=1):
            return "EventBuffer(ts=%d, te=%d, data=%s)" % (
                self.ts,
                self.te,
                self.data,
            )

    def __bool__(self):
        return self.data is not None

    @property
    def slice(self):
        return TSSlice(self.ts, self.te)

    @property
    def duration(self):
        return self.te - self.ts

    @property
    def is_gap(self):
        if self.data is None:
            return True
        else:
            return False

    def __contains__(self, item):
        # FIXME should this conditional actually be open from above?
        if isinstance(item, int):
            return self.ts <= item <= self.te
        else:
            return False

    def __lt__(self, item):
        if isinstance(item, int):
            return self.te < item
        elif isinstance(item, EventBuffer):
            return self.te < item.te

    def __le__(self, item):
        if isinstance(item, int):
            return self.te <= item
        elif isinstance(item, EventBuffer):
            return self.te <= item.te

    def __ge__(self, item):
        if isinstance(item, int):
            return self.ts >= item
        elif isinstance(item, EventBuffer):
            return self.te >= item.te

    def __gt__(self, item):
        if isinstance(item, int):
            return self.ts > item
        elif isinstance(item, EventBuffer):
            return self.te > item.te


@dataclass
class EventFrame(Frame):
    """An sgn Frame object that holds a dictionary of events.

    Args:
        events:
            dict, Dictionary of EventBuffers
    """

    events: Union[dict, None] = None

    def __post_init__(self):
        super().__post_init__()
        assert self.events is not None and len(self.events) > 0

    def __getitem__(self, item):
        return self.events[item]

    def __iter__(self):
        # FIXME this will just iterate over event keys. Is that what we want?
        return iter(self.events)

    def __repr__(self):
        out = (
            f"EventFrame(EOS={self.EOS}, is_gap={self.is_gap}, "
            f"metadata={self.metadata}, events={{\n"
        )
        for evt, v in self.events.items():
            out += f"    {evt}: {v},\n"
        out += "}})"
        return out


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
            if self.shape == (-1,):
                raise ValueError("if data is None self.shape must be given")
        elif isinstance(self.data, int) and self.data == 1:
            if self.shape == (-1,):
                raise ValueError("if data is 1 self.shape must be given")
            self.data = self.backend.ones(self.shape)
        elif isinstance(self.data, int) and self.data == 0:
            if self.shape == (-1,):
                raise ValueError("if data is 0 self.shape must be given")
            self.data = self.backend.zeros(self.shape)
        elif self.shape == (-1,):
            self.shape = self.data.shape
        else:
            if self.shape != self.data.shape:
                raise ValueError(
                    "Array size mismatch: self.shape and self.data.shape "
                    "must agree,"
                    f"got {self.shape} and {self.data.shape} "
                    f"with data {self.data}"
                )

        assert isinstance(self.shape, tuple)
        assert len(self.shape) > 0

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

    def new(
        self,
        offslice: Optional[TSSlice] = None,
        data: Optional[Union[int, Array]] = None,
    ):
        """
        Return a new buffer from an existing one and optionally change the offsets.
        """
        return SeriesBuffer.fromoffsetslice(
            self.slice if offslice is None else offslice,
            self.sample_rate,
            data,
            self.shape[:-1],
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

    @property
    def properties(self):
        return {
            "offset": self.offset,
            "end_offset": self.end_offset,
            "t0": self.t0,
            "end": self.end,
            "shape": self.shape,
            "sample_shape": self.sample_shape,
            "sample_rate": self.sample_rate,
        }

    def __bool__(self):
        return self.data is not None

    def __len__(self):
        return 0 if self.data is None else len(self.data)

    def set_data(self, data: Optional[Array] = None) -> None:
        """Set the data attribute to the newly provided data.

        Args:
            data:
                Optiona[Array], the new data to set to
        """
        if isinstance(data, int) and data == 1:
            self.data = self.backend.ones(self.shape)
        elif isinstance(data, int) and data == 0:
            self.data = self.backend.zeros(self.shape)
        elif data is not None and self.shape != data.shape:
            raise ValueError("Data are incompatible shapes")
        else:
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

    def __eq__(self, value: Union[SeriesBuffer, Any]) -> bool:
        # FIXME this is a bit convoluted.  In order for some of these tests to
        # be triggered strange manipulation of objects would have to occur.
        # Consider making the SeriesBuffer properties read only where possible.
        is_series_buffer = isinstance(value, SeriesBuffer)
        if not is_series_buffer:
            return False
        if not (value.shape == self.shape):
            return False
        # FIXME is this the right check? Or do we want to check dtype? Under
        # what circumstances will this check fail?
        if type(self.data) is not type(value.data):
            return False
        if isinstance(self.data, NumpyArray) and isinstance(value.data, NumpyArray):
            share_data = NumpyBackend.all(self.data == value.data)
        elif isinstance(self.data, TorchArray) and isinstance(value.data, TorchArray):
            share_data = TorchBackend.all(self.data == value.data)
        elif self.data is None and value.data is None:
            share_data = True
        else:
            # Will need to expand this conditional if/when other data types are added
            raise ValueError("invalid data object")
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
        assert len(self.shape) > 0
        return self.shape[-1]

    @property
    def sample_shape(self) -> tuple:
        """return the sample shape"""
        return self.shape[:-1]

    @property
    def is_gap(self) -> bool:
        """Whether the buffer is a gap. This is determined by whether the data is None.

        Returns:
            bool, whether the buffer is a gap
        """
        return self.data is None

    def filleddata(self, zeros_func=None) -> Array:
        """Fill the data with zeros if buffer is a gap, otherwise return the data.

        Args:
            zeros_func:
                the function to produce a zeros array

        Returns:
            Array, the filled data
        """
        if zeros_func is None:
            zeros_func = self.backend.zeros

        if self.data is not None:
            return self.data
        else:
            return zeros_func(self.shape)

    def __contains__(self, item):
        # FIXME, is this what we want?
        if isinstance(item, int):
            # The end offset is not actually in the buffer hence the second "<" vs "<="
            return self.offset <= item < self.end_offset
        elif isinstance(item, SeriesBuffer):
            return (self.offset <= item.offset) and (item.end_offset <= self.end_offset)
        else:
            return False

    def __lt__(self, item):
        assert isinstance(item, SeriesBuffer)
        return self.end_offset < item.end_offset

    def __le__(self, item):
        assert isinstance(item, SeriesBuffer)
        return self.end_offset <= item.end_offset

    def __ge__(self, item):
        assert isinstance(item, SeriesBuffer)
        return self.end_offset >= item.end_offset

    def __gt__(self, item):
        assert isinstance(item, SeriesBuffer)
        return self.end_offset > item.end_offset

    def _insert(self, data: Array, offset) -> None:
        """TODO workshop the name
        Adds data from a whose slice is
        fully contained within self's into self.
        Does not do safety checks."""
        insertion_index = Offset.tosamples(
            offset - self.offset, sample_rate=self.sample_rate
        )
        # FIXME: this is a thorny issue because of how generous we are with the type
        # of data and the type of Array.  Fixing this will involve being
        # stricter about types and more careful throughout the array_ops
        # module.
        self.data[
            ..., insertion_index : insertion_index + data.shape[-1]
        ] += data  # type: ignore

    @property
    def _backend_from_data(self):
        if isinstance(self.data, NumpyArray):
            return NumpyBackend
        elif isinstance(self.data, TorchArray):
            if (
                self.data.device != TorchBackend.DEVICE
                or self.data.dtype != TorchBackend.DTYPE
            ):
                raise ValueError("TorchArray and data backends are incompatable")
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
        if (
            (backend != item._backend_from_data)
            and (item._backend_from_data is not None)
            and (backend is not None)
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

    def __getitem__(self, item):
        return self.buffers[item]

    def __iter__(self):
        return iter(self.buffers)

    def __repr__(self):
        out = (
            f"TSFrame(EOS={self.EOS}, is_gap={self.is_gap}, "
            f"metadata={self.metadata}, buffers=[\n"
        )
        for buf in self:
            out += f"    {buf},\n"
        out += "])"
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
        self.is_gap = all([b.is_gap for b in self.buffers])

        # Check all backends are the same
        backends = {buf.backend for buf in bufs}
        assert (
            len(backends) == 1
        ), f"All buffers must have the same backend, got {backends}"

        # TODO check that all sample rates are the same

    def set_buffers(self, bufs: list[SeriesBuffer]) -> None:
        """Set the buffers attribute to the bufs provided.

        Args:
            bufs:
                list[SeriesBuffers], the list of buffers to set to
        """
        self.buffers = bufs
        self.__sanity_check(bufs)

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
    def t0(self) -> float:
        """The t0 of the TSFrame, which is the t0 of the first buffer.

        Returns:
            float, the t0 of the TSFrame
        """
        return self.buffers[0].t0

    @property
    def end(self) -> float:
        """The end of the TSFrame, which is the end time of the last buffer.

        Returns:
            float, the end time of the TSFrame
        """
        return self.buffers[-1].end

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
    def sample_shape(self) -> tuple:
        """return the sample shape"""
        return self.buffers[0].sample_shape

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

    @property
    def backend(self) -> type[ArrayBackend]:
        """The backend of the buffers.

        Returns:
            type[ArrayBackend], the backend of the buffers
        """
        return self.buffers[0].backend

    def heartbeat(self, EOS=False):
        frame = TSFrame.from_buffer_kwargs(
            offset=self.offset,
            sample_rate=self.sample_rate,
            shape=self.sample_shape + (0,),
            data=None,
        )
        frame.EOS = EOS
        return frame

    def __next__(self):
        """
        return a new empty frame that is like the current one but advanced to
        the next offset, e.g.,

        >>> frame = TSFrame.from_buffer_kwargs(offset=0,
                        sample_rate=2048, shape=(2048,))
        >>> print (frame)

                SeriesBuffer(offset=0, offset_end=16384, shape=(2048,),
                             sample_rate=2048, duration=1000000000, data=None)
        >>> print (next(frame))
        """
        return self.from_buffer_kwargs(
            offset=self.end_offset, sample_rate=self.sample_rate, shape=self.shape
        )

    def __contains__(self, other):
        return other.slice in self.slice

    def intersect(self, other):
        """
        Intersect self with another frame and return up to three
        frames, the frame before, the intersecting frame and the frame after.  For
        example, given two frames A and B:

        A:
                SeriesBuffer(offset=0, offset_end=4096, shape=(32,),
                             sample_rate=128, duration=250000000, data=None)
                SeriesBuffer(offset=4096, offset_end=20480, shape=(128,),
                             sample_rate=128, duration=1000000000, data=None)
        B:
                SeriesBuffer(offset=2048, offset_end=10240, shape=(64,),
                             sample_rate=128, duration=500000000, data=None)
                SeriesBuffer(offset=10240, offset_end=174080, shape=(1280,),
                             sample_rate=128, duration=10000000000, data=None)

        B.intersect(A):

                before Frame:
                SeriesBuffer(offset=0, offset_end=2048, shape=(16,),
                             sample_rate=128, duration=125000000, data=None)

                intersecting Frame:
                SeriesBuffer(offset=2048, offset_end=4096, shape=(16,),
                             sample_rate=128, duration=125000000, data=None)
                SeriesBuffer(offset=4096, offset_end=20480, shape=(128,),
                             sample_rate=128, duration=1000000000, data=None)

                after Frame: None

        A.intersect(B):

                before Frame: None

                intersecting Frame:
                SeriesBuffer(offset=2048, offset_end=10240, shape=(64,),
                             sample_rate=128, duration=500000000, data=None)
                SeriesBuffer(offset=10240, offset_end=20480, shape=(80,),
                             sample_rate=128, duration=625000000, data=None)

                after Frame:
                SeriesBuffer(offset=20480, offset_end=174080, shape=(1200,),
                             sample_rate=128, duration=9375000000, data=None)
        """
        bbuf = []
        inbuf = []
        abuf = []
        for buf in other.buffers:
            if buf.end_offset <= self.offset:
                bbuf.append(buf)
            elif buf.offset >= self.end_offset:
                abuf.append(buf)
            elif buf in self:
                inbuf.append(buf)
            else:
                outside_slices = TSSlices(self.slice - buf.slice).search(buf.slice)
                outside_bufs = buf.split(outside_slices)
                for obuf in outside_bufs:
                    assert (obuf.end_offset <= self.offset) or (
                        obuf.offset >= self.end_offset
                    )
                    if obuf.end_offset <= self.offset:
                        bbuf.append(obuf)
                    else:
                        abuf.append(obuf)
                inbuf.extend(buf.split(TSSlices([self.slice & buf.slice])))
        return (
            None if not bbuf else TSFrame(buffers=bbuf),
            None if not inbuf else TSFrame(buffers=inbuf),
            None if not abuf else TSFrame(buffers=abuf),
        )

    def filleddata(self) -> "TSFrame":
        """Combine the buffers of the frame into a single buffer,
        analogous to itertools.chain.

        Returns:
            TSFrame, the frame with a single buffer
        """
        arrays = [buf.filleddata() for buf in self.buffers]
        data = self.backend.cat(arrays, axis=-1)
        return TSFrame.from_buffer_kwargs(
            offset=self.offset,
            sample_rate=self.sample_rate,
            shape=self.shape,
            data=data,
        )
