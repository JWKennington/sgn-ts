from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import total_ordering
from typing import Any, Iterable, Literal, Optional, Protocol, Union, runtime_checkable

import numpy
from sgn.frames import DataSpec, Frame

from sgnts.base.array_ops import (
    Array,
    ArrayBackend,
    NumpyArray,
    NumpyBackend,
    TorchArray,
    TorchBackend,
)
from sgnts.base.offset import Offset, TimeUnits
from sgnts.base.slice_tools import TSSlice, TSSlices
from sgnts.base.time import Time


@runtime_checkable
class TimeLike(Protocol):
    offset: int

    @property
    def time(self) -> int:
        """The reference time, in integer nanoseconds.

        Returns:
            int, buffer time
        """
        return Offset.offset_ref_t0 + Offset.tons(self.offset)

    @property
    def t0(self) -> int:
        """The start (reference) time, in integer nanoseconds.

        Returns:
            int, buffer start time
        """
        return self.time

    @property
    def start(self) -> int:
        """The start (reference) time, in integer nanoseconds.

        Returns:
            int, buffer start time
        """
        return self.time


@total_ordering
@runtime_checkable
class TimeSpanLike(TimeLike, Protocol):
    noffset: int

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TimeSpanLike):
            return NotImplemented
        return self.end_offset == other.end_offset

    def __lt__(self, other: TimeSpanLike) -> bool:
        return self.end_offset < other.end_offset

    @property
    def end_offset(self) -> int:
        """The end offset.

        Returns:
            int, end offset
        """
        return self.offset + self.noffset

    @property
    def slice(self) -> TSSlice:
        """The offset slice that the item spans.

        Returns:
            TSSlices, the offset slice
        """
        return TSSlice(self.offset, self.end_offset)

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


@dataclass
class Event(TimeLike):
    """Event with metadata.

    Args:
        offset:
            int, the offset of the buffer. See Offset class for definitions.
        data:
            Any, data of the event
    """

    offset: int
    data: Any = None

    @classmethod
    def from_time(cls, time: int, data: Any = None) -> Event:
        """Create an Event from a reference time (in nanoseconds)."""
        offset = Offset.fromns(time) - Offset.offset_ref_t0
        return cls(offset=offset, data=data)


@dataclass(eq=False)
class EventBuffer(TimeSpanLike):
    """Event buffer with associated metadata.

    Args:
        offset:
            int, the offset of the buffer. See Offset class for definitions.
        noffset:
            int, the number of offsets the buffer spans, or the duration.
        data:
            Sequence[Any], event data covering the span in question.
    """

    offset: int
    noffset: int
    data: Sequence[Any] = field(default_factory=list)

    def __post_init__(self):
        if not isinstance(self.offset, int) or not isinstance(self.noffset, int):
            msg = "offset and noffset must be integers"
            raise ValueError(msg)

    @classmethod
    def from_span(
        cls, start: int, end: int, data: Sequence[Any] | None = None
    ) -> EventBuffer:
        """Create an EventBuffer from start/end times (in nanoseconds)."""
        if not isinstance(start, int) or not isinstance(end, int) or not (start <= end):
            raise ValueError(
                "start and end must be integers and start must be <= end,"
                f"got {start} and {end}"
            )
        offset = Offset.fromns(start)
        noffset = Offset.fromns(end) - offset
        if data is None:
            data = []
        return cls(offset=offset, noffset=noffset, data=data)

    def __iter__(self):
        return iter(self.data)

    def __getitem__(self, idx: int) -> Any:
        return self.data[idx]

    @property
    def events(self) -> Sequence[Any]:
        """The event data."""
        return self.data

    def __repr__(self):
        with numpy.printoptions(threshold=3, edgeitems=1):
            return "EventBuffer(offset=%d, end_offset=%d, data=%s)" % (
                self.offset,
                self.end_offset,
                self.data,
            )

    def __bool__(self):
        return bool(self.data)

    @property
    def is_gap(self):
        return not self.data

    def __contains__(self, item):
        # FIXME, is this what we want?
        if isinstance(item, int):
            # The end offset is not actually in the buffer hence the second "<" vs "<="
            return self.offset <= item < self.end_offset
        elif isinstance(item, EventBuffer):
            return (self.offset <= item.offset) and (item.end_offset <= self.end_offset)
        else:
            return False


class TimeSpanFrame(Frame, TimeSpanLike):
    """Base class for frames with time span semantics.

    TimeSpanFrame combines Frame's data-carrying capabilities with
    TimeSpanLike's temporal semantics (start/end offsets).

    All TimeSpanFrame subclasses must be iterable.
    """

    @abstractmethod
    def __iter__(self):
        """Iterate over the frame's data elements."""
        ...


@dataclass(eq=False)
class EventFrame(TimeSpanFrame):
    """An sgn Frame object that holds a list of EventBuffers.

    EventFrame can be created with data (offset/noffset computed from buffers)
    or empty with explicit offset/noffset for incremental population.

    Args:
        data: list[EventBuffer], EventBuffers to hold
        offset: int, explicit offset when creating empty frame
        noffset: int, explicit noffset (duration) when creating empty frame
    """

    data: list[EventBuffer] = field(default_factory=list)
    offset: int = 0
    noffset: int = 0

    def __post_init__(self):
        super().__post_init__()
        # If data exists, compute offset/noffset from data
        if self.data:
            # Ensure user didn't try to manually set offset/noffset
            if self.offset != 0 or self.noffset != 0:
                raise ValueError(
                    "Cannot specify offset/noffset when providing data - "
                    "they are computed from data"
                )
            # Compute from data
            self.offset = self.data[0].offset
            self.noffset = self.data[-1].end_offset - self.offset

            # Validate computed values
            if (
                not isinstance(self.start, int)
                or not isinstance(self.end, int)
                or not (self.start <= self.end)
            ):
                raise ValueError(
                    "start and end must be integers and start must be <= end,"
                    f"got {self.start} and {self.end}"
                )

    def __iter__(self):
        return iter(self.data)

    def __getitem__(self, idx: int) -> EventBuffer:
        return self.data[idx]

    def __contains__(self, other):
        return other.slice in self.slice

    @property
    def events(self):
        """The list of Events."""
        return [event for buffer in self.data for event in buffer.data]

    def __repr__(self):
        out = (
            f"EventFrame(EOS={self.EOS}, is_gap={self.is_gap}, "
            f"metadata={self.metadata}, buffers={{\n"
        )
        for buf in self.data:
            out += f"    {buf},\n"
        out += "}})"
        return out

    def append(self, item: EventBuffer) -> None:
        """Append EventBuffer with validation.

        Validates that buffer falls within frame bounds (offset to offset+noffset)
        and is contiguous with previous buffers.

        Args:
            item: EventBuffer to append

        Raises:
            AssertionError if validation fails
        """
        frame_end_offset = self.offset + self.noffset

        # Check buffer falls within bounds
        assert (
            self.offset <= item.offset
        ), f"Buffer offset {item.offset} starts before frame offset {self.offset}"
        assert item.end_offset <= frame_end_offset, (
            f"Buffer end_offset {item.end_offset} extends beyond frame "
            f"end_offset {frame_end_offset}"
        )

        # Check contiguity with previous buffer
        if self.data:
            assert item.offset == self.data[-1].end_offset, (
                f"Buffer offset {item.offset} is not contiguous with "
                f"previous buffer end {self.data[-1].end_offset}"
            )

        self.data.append(item)

    def validate_span(self) -> None:
        """Validate that data fully spans the offset/noffset range.

        Checks that:
        - First buffer starts at frame offset
        - Last buffer ends at frame offset+noffset (the frame's end_offset)
        - All buffers are contiguous

        Raises:
            AssertionError if validation fails
        """
        if self.data:
            frame_end_offset = self.offset + self.noffset

            assert self.data[0].offset == self.offset, (
                f"First buffer offset {self.data[0].offset} != "
                f"frame offset {self.offset}"
            )
            assert self.data[-1].end_offset == frame_end_offset, (
                f"Last buffer end_offset {self.data[-1].end_offset} != "
                f"frame end_offset {frame_end_offset}"
            )
            # Check all buffers are contiguous
            for i in range(1, len(self.data)):
                assert self.data[i].offset == self.data[i - 1].end_offset, (
                    f"Gap between buffer {i-1} (end={self.data[i-1].end_offset}) "
                    f"and buffer {i} (start={self.data[i].offset})"
                )


@dataclass(frozen=True)
class SeriesDataSpec(DataSpec):
    """Data specification for timeseries.

    Args:
        sample_rate:
            int, the sample rate associated with the data.
        data_type:
            Any, the data type associated with the data.
    """

    sample_rate: int
    data_type: Any


@dataclass(eq=False)
class SeriesBuffer(TimeSpanLike):
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
        assert len(self.shape) > 0, f"Buffer shape cannot be empty, got {self.shape}"

        for t in self.shape:
            assert isinstance(t, int)

        # set the data specification
        self.spec = SeriesDataSpec(
            sample_rate=self.sample_rate, data_type=self.backend.DTYPE
        )

    def __and__(self, other):
        sl = self.slice & other.slice
        if sl:
            return self.sub_buffer(sl)
        else:
            return None

    def isfinite(self):
        return self.slice.isfinite()

    def copy(
        self,
        offset: int | None = None,
        sample_rate: int | None = None,
        data: int | Array | None = None,
        is_gap: bool | None = None,
        shape: tuple | None = None,
        backend: type[ArrayBackend] | None = None,
    ) -> SeriesBuffer:
        """Returns a copy of the TSFrame with requested modifications.

        Any attributes not being changed will inherit from the original
        TSFrame.

        Args:
            offset:
                int, optional, the offset of the buffer. See Offset class for
                definitions.
            sample_rate:
                int, optional, the sample rate belonging to the set of
                Offset.ALLOWED_RATES
            data:
                int | Array, optional, the timeseries data.
            is_gap:
                bool, optional, set the buffer as a gap (or non-gap).
            shape:
                tuple, optional, the shape of the data regardless of gaps.
                Required if data is None or int, and represents the shape of
                the absent data.
            backend:
                type[ArrayBackend], optional, the wrapper around array
                operations
        """
        offset = self.offset if offset is None else offset
        sample_rate = self.sample_rate if sample_rate is None else sample_rate
        shape = self.shape if shape is None else shape
        backend = self.backend if backend is None else backend

        # using data=None as a case to decide whether to modify the buffer's
        # data with user-specified data needs extra care due to data=None also
        # being used to define the presence of a gap, so instead we enumerate
        # the possible cases based on whether is_gap is set and what the value
        # is if it is set.
        # NOTE: this can be simplified but is written as such to be explicit
        if is_gap is None:  # inherit buffer's gap status
            buf_data = self.data if data is None else data
        elif is_gap:  # change to gap
            buf_data = None
        else:  # change to non-gap
            buf_data = data

        return SeriesBuffer(
            offset=offset, sample_rate=sample_rate, data=buf_data, shape=shape
        )

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
        assert (
            offslice.units == TimeUnits.OFFSETS
        ), f"offset slice must be in offsets, got {offslice.units}"
        shape = channels + (
            Offset.tosamples(int(offslice.stop - offslice.start), sample_rate),
        )
        return SeriesBuffer(
            offset=int(offslice.start), sample_rate=sample_rate, data=data, shape=shape
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
                Optional[Array], the new data to set to
        """
        if isinstance(data, int) and data == 1:
            self.data = self.backend.ones(self.shape)
        elif isinstance(data, int) and data == 0:
            self.data = self.backend.zeros(self.shape)
        elif isinstance(data, (int, float, complex)):
            # Handle any numeric value by creating an array filled with that value
            self.data = self.backend.full(self.shape, data)
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
    def noffset(self) -> int:
        """The number of offsets spanned by this buffer.

        Returns:
            int, the offset duration
        """
        return Offset.fromsamples(self.samples, self.sample_rate)

    @noffset.setter
    def noffset(self, other: int) -> None:
        msg = "cannot set noffset on a SeriesBuffer"
        raise AttributeError(msg)

    @property
    def samples(self) -> int:
        """The number of samples the buffer carries.

        Return:
            int, the number of samples
        """
        assert len(self.shape) > 0, f"Buffer shape cannot be empty, got {self.shape}"
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

    def __add__(self, item: SeriesBuffer) -> SeriesBuffer:
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
        assert (
            off < self.offset
        ), f"Requested offset {off} must be before buffer offset {self.offset}"
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
        assert (
            slc in self.slice
        ), f"Requested slice {slc} not contained in buffer slice {self.slice}"
        startsamples, stopsamples = Offset.tosamples(
            int(slc.start - self.offset), self.sample_rate
        ), Offset.tosamples(int(slc.stop - self.offset), self.sample_rate)
        if not gap and self.data is not None and not isinstance(self.data, int):
            data = self.data[..., startsamples:stopsamples]
        else:
            data = None

        return SeriesBuffer(
            offset=int(slc.start),
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
            assert (
                slc in self.slice
            ), f"Slice {slc} must be within buffer bounds {self.slice}"
            out.append(self.sub_buffer(slc))
        if contiguous:
            gap_boundaries = boundaries.invert(self.slice)
            for slc in gap_boundaries.slices:
                out.append(self.sub_buffer(slc, gap=True))
        return sorted(out)

    def plot(
        self,
        ax=None,
        label: Optional[str] = None,
        channel: Optional[Union[int, tuple]] = None,
        gap_color: str = "red",
        gap_alpha: float = 0.3,
        show_gaps: bool = True,
        time_unit: Literal["s", "ms", "ns", "gps"] = "gps",
        **kwargs,
    ):
        """Plot the buffer's time-series data.

        Requires matplotlib. Install with: pip install sgn-ts[plot]

        Args:
            ax:
                matplotlib Axes, optional. If None, creates a new figure and axes.
            label:
                str, optional. Legend label for this buffer's data line.
            channel:
                int or tuple[int, ...], optional. For multi-dimensional data,
                specifies which channel(s) to plot. If None and data is
                multi-dimensional, plots all channels.
            gap_color:
                str, color for gap region shading. Default 'red'.
            gap_alpha:
                float, alpha transparency for gap region shading. Default 0.3.
            show_gaps:
                bool, whether to show gap indicators. Default True.
            time_unit:
                str, time unit for x-axis: 's' (seconds since start), 'ms',
                'ns', or 'gps' (absolute GPS time). Default 'gps'.
            **kwargs:
                Additional keyword arguments passed to ax.plot().

        Returns:
            tuple: (fig, ax) matplotlib figure and axes objects
        """
        from sgnts.plotting import plot_buffer

        return plot_buffer(
            self,
            ax=ax,
            label=label,
            channel=channel,
            gap_color=gap_color,
            gap_alpha=gap_alpha,
            show_gaps=show_gaps,
            time_unit=time_unit,
            **kwargs,
        )


def ensure_nonempty(func):
    """Decorator to ensure TSFrame has buffers before accessing properties/methods.

    Raises ValueError with a helpful message if the frame is empty.
    """

    def wrapper(self, *args, **kwargs):
        if len(self.buffers) == 0:
            raise ValueError(
                f"TSFrame.{func.__name__} cannot be used when there are no buffers "
                f"in the frame. Use TSFrame.fill() to populate the frame."
            )
        return func(self, *args, **kwargs)

    return wrapper


@dataclass(eq=False, kw_only=True)
class TSCollectFrame(TimeSpanFrame):
    """A collector for incrementally building a TSFrame with validation.

    TSCollectFrame provides atomic all-or-nothing buffer collection:
    - Buffers are collected in a temporary list
    - Validation occurs on close()
    - Only commits to parent TSFrame if all validations pass
    - Can be used as a context manager for automatic cleanup

    Args:
        parent_frame: TSFrame, the frame to populate

    Usage:
        # Context manager (automatic close)
        frame = TSFrame(offset=0, noffset=1000)
        with frame.fill() as collector:
            collector.append(buf1)
            collector.append(buf2)
        # frame now has buffers

        # Manual (explicit control)
        frame = TSFrame(offset=0, noffset=1000)
        collector = frame.fill()
        collector.append(buf1)
        collector.close()
    """

    parent_frame: TSFrame
    _buffers: list[SeriesBuffer] = field(default_factory=list, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    def __post_init__(self):
        super().__post_init__()
        # Inherit offset/noffset from parent
        self.offset = self.parent_frame.offset
        self.noffset = self.parent_frame.noffset
        self.EOS = self.parent_frame.EOS
        self.metadata = self.parent_frame.metadata

    def __iter__(self):
        """Iterate over collected buffers."""
        return iter(self._buffers)

    def __len__(self) -> int:
        """Return number of collected buffers."""
        return len(self._buffers)

    def __enter__(self) -> TSCollectFrame:
        """Enter context manager."""
        if len(self.parent_frame.buffers) > 0:
            raise ValueError(
                "Cannot use fill() on a TSFrame that already has buffers. "
                "TSCollectFrame can only populate empty frames."
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager - close if no exception occurred."""
        if exc_type is None:
            self.close()
        return False

    def append(self, item: SeriesBuffer) -> None:
        """Append SeriesBuffer to temporary collection.

        Validates that buffer falls within frame bounds and is contiguous
        with previous buffers. Does not commit to parent frame until close().

        Args:
            item: SeriesBuffer to append
        """
        if self._closed:
            raise ValueError("Cannot append to closed TSCollectFrame")

        frame_end_offset = self.offset + self.noffset

        # Check buffer falls within bounds
        assert (
            self.offset <= item.offset
        ), f"Buffer offset {item.offset} starts before frame offset {self.offset}"
        assert item.end_offset <= frame_end_offset, (
            f"Buffer end_offset {item.end_offset} extends beyond frame "
            f"end_offset {frame_end_offset}"
        )

        # Check contiguity with previous buffer
        if self._buffers:
            assert item.offset == self._buffers[-1].end_offset, (
                f"Buffer offset {item.offset} is not contiguous with "
                f"previous buffer end {self._buffers[-1].end_offset}"
            )
        else:
            # First buffer must start at frame offset
            assert item.offset == self.offset, (
                f"First buffer offset {item.offset} must match "
                f"frame offset {self.offset}"
            )

        self._buffers.append(item)

    def extend(self, items: Iterable[SeriesBuffer]) -> None:
        """Extend with multiple SeriesBuffers, validating each.

        Args:
            items: Iterable of SeriesBuffers to append
        """
        for item in items:
            self.append(item)

    def __iadd__(self, item: SeriesBuffer) -> TSCollectFrame:
        """Support += operator for appending."""
        self.append(item)
        return self

    def validate_span(self) -> None:
        """Validate that buffers fully span the offset/noffset range.

        Checks that:
        - First buffer starts at frame offset
        - Last buffer ends at frame offset+noffset (the frame's end_offset)
        """
        if not self._buffers:
            raise ValueError("Cannot validate empty collector - no buffers added")

        frame_end_offset = self.offset + self.noffset

        assert self._buffers[0].offset == self.offset, (
            f"First buffer offset {self._buffers[0].offset} != "
            f"frame offset {self.offset}"
        )
        assert self._buffers[-1].end_offset == frame_end_offset, (
            f"Last buffer end_offset {self._buffers[-1].end_offset} != "
            f"frame end_offset {frame_end_offset}"
        )

    def close(self) -> None:
        """Validate and commit buffers to parent TSFrame.

        This validates that buffers span the frame's offset/noffset range,
        then atomically commits them to the parent frame using set_buffers(),
        which performs additional validation (contiguity, consistent specs, etc.).

        After close(), this TSCollectFrame cannot be used again.
        """
        if self._closed:
            raise ValueError("TSCollectFrame already closed")

        # Validate that buffers span the frame's range
        self.validate_span()

        # Atomically commit to parent frame
        # set_buffers() handles contiguity, backend, and spec validation
        self.parent_frame.set_buffers(self._buffers)

        # Mark as closed
        self._closed = True


@dataclass(eq=False)
class TSFrame(TimeSpanFrame):
    """An sgn Frame object that holds a list of buffers

    TSFrame can be created with data (offset/noffset computed from buffers)
    or empty with explicit offset/noffset for incremental population.

    Args:
        buffers: list[SeriesBuffer], SeriesBuffers to hold
        offset: int, explicit offset when creating empty frame
        noffset: int, explicit noffset (duration) when creating empty frame
    """

    buffers: list[SeriesBuffer] = field(default_factory=list)
    offset: int = 0
    noffset: int = 0

    def __post_init__(self):
        super().__post_init__()

        # If buffers exist, compute offset/noffset from buffers
        if self.buffers:
            # Ensure user didn't try to manually set offset/noffset
            if self.offset != 0 or self.noffset != 0:
                raise ValueError(
                    "Cannot specify offset/noffset when providing buffers - "
                    "they are computed from buffers"
                )

            # Compute from buffers
            self.offset = self.buffers[0].offset
            self.noffset = self.buffers[-1].end_offset - self.offset

            # Validate and update buffer-dependent attributes
            self.validate_buffers()
            self.update_buffer_attrs()
            self.spec = self.buffers[0].spec
        else:
            # Empty frame - offset/noffset are used as-is
            # Set default attributes for empty frame
            self.is_gap = False
            self.spec = None

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

    def fill(self) -> TSCollectFrame:
        """Create a TSCollectFrame for atomically populating this frame.

        Returns a TSCollectFrame that can be used to incrementally add buffers
        with validation. The buffers are only committed to this frame when
        close() is called (or automatically via context manager).

        Returns:
            TSCollectFrame: A collector for building this frame

        Usage:
            # Context manager (recommended - automatic close)
            frame = TSFrame(offset=0, noffset=1000)
            with frame.fill() as collector:
                collector.append(buf1)
                collector.append(buf2)
            # frame now has buffers

            # Manual (explicit control)
            frame = TSFrame(offset=0, noffset=1000)
            collector = frame.fill()
            collector.append(buf1)
            collector.close()  # commits to frame
        """
        return TSCollectFrame(parent_frame=self)

    def validate_span(self) -> None:
        """Validate that buffers fully span the offset/noffset range.

        Checks that:
        - First buffer starts at frame offset
        - Last buffer ends at frame offset+noffset (the frame's end_offset)
        - All buffers are contiguous
        """
        if self.buffers:
            frame_end_offset = self.offset + self.noffset

            assert self.buffers[0].offset == self.offset, (
                f"First buffer offset {self.buffers[0].offset} != "
                f"frame offset {self.offset}"
            )
            assert self.buffers[-1].end_offset == frame_end_offset, (
                f"Last buffer end_offset {self.buffers[-1].end_offset} != "
                f"frame end_offset {frame_end_offset}"
            )
            # validate_buffers checks contiguity
            self.validate_buffers()

    def validate_buffers(self) -> None:
        """Sanity check that the buffers don't overlap nor have discontinuities.

        Args:
            bufs:
                list[SeriesBuffer], the buffers to perform the sanity check on
        """
        # FIXME: is there a smart way using TSSlics?

        if len(self.buffers) > 1:
            slices = [buf.slice for buf in self.buffers]
            off0 = slices[0].stop
            for sl in slices[1:]:
                assert off0 == sl.start, (
                    f"Buffer offset {off0} must match slice start {sl.start} "
                    f"for contiguous buffers"
                )
                off0 = sl.stop

        # Check all backends are the same
        backends = {buf.backend for buf in self.buffers}
        assert (
            len(backends) == 1
        ), f"All buffers must have the same backend, got {backends}"

        # check that data specifications are all the same
        data_specs = {buf.spec for buf in self.buffers}
        assert (
            len(data_specs) == 1
        ), f"All buffers must have the same data specifications, got {data_specs}"

    def update_buffer_attrs(self):
        """Helper method for updating buffer dependent attributes.

        This is useful since buffers are mutable, and there are cases where we modify
        the buffer contents after the TSFrame has been created, e.g., when preparing a
        return frame in a "new" method.
        """
        self.is_gap = all([b.is_gap for b in self.buffers])

    def set_buffers(self, bufs: list[SeriesBuffer]) -> None:
        """Set the buffers attribute to the bufs provided.

        Args:
            bufs:
                list[SeriesBuffers], the list of buffers to set to
        """
        self.buffers = bufs
        self.validate_buffers()
        self.update_buffer_attrs()

    @property
    @ensure_nonempty
    def shape(self) -> tuple[int, ...]:
        """The shape of the TSFrame.

        Returns:
            tuple[int, ...], the shape of the TSFrame
        """
        return self.buffers[0].shape[:-1] + (sum(b.samples for b in self.buffers),)

    @property
    @ensure_nonempty
    def samples(self) -> int:
        """The number of samples in the Frame.

        Return:
            int, the number of samples
        """
        return sum(buf.samples for buf in self.buffers)

    @property
    @ensure_nonempty
    def sample_shape(self) -> tuple:
        """return the sample shape"""
        return self.buffers[0].sample_shape

    @property
    @ensure_nonempty
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
    @ensure_nonempty
    def backend(self) -> type[ArrayBackend]:
        """The backend of the buffers.

        Returns:
            type[ArrayBackend], the backend of the buffers
        """
        return self.buffers[0].backend

    @ensure_nonempty
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

    @ensure_nonempty
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
                    ), (
                        f"Buffer overlap detected - output buffer "
                        f"[{obuf.offset}, {obuf.end_offset}] must not overlap "
                        f"with frame range [{self.offset}, {self.end_offset}]"
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

    @property
    @ensure_nonempty
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

    @ensure_nonempty
    def filleddata(self, zeros_func=None) -> Array:
        """Combined buffer data for the entire frame with zeros filled
        in for buffer gaps.

        Basically SeriesBuffer.filleddata() for the entire frame.

        Args:
            zeros_func:
                the function to produce a zeros array

        Returns:
            Array, the filled data

        """
        arrays = [buf.filleddata(zeros_func) for buf in self.buffers]
        return self.backend.cat(arrays, axis=-1)

    @ensure_nonempty
    def search(self, buf):
        out = []
        for b in self:
            intersects = b & buf
            if intersects is not None and intersects.isfinite():
                out.append(intersects)
        return out

    @ensure_nonempty
    def align(self, tsslices) -> "TSFrame":
        "Align buffers according to the TSSlices provided"
        assert (
            self.slice == tsslices.slice
        ), "The boundaries provided are not aligned with the frame boundaries"
        bufs = []
        for aligned_buf in [self[0].new(offslice) for offslice in tsslices]:
            searched_bufs = self.search(aligned_buf)
            # promote any gaps
            if any([b.is_gap for b in searched_bufs]):
                bufs.append(aligned_buf)
                continue
            # otherwise add the data from the found sub buffers
            for sb in searched_bufs:
                aligned_buf = aligned_buf + sb
            bufs.append(aligned_buf)
        return TSFrame(buffers=bufs)

    def plot(
        self,
        ax=None,
        label: Optional[str] = None,
        channel: Optional[Union[int, tuple]] = None,
        gap_color: str = "red",
        gap_alpha: float = 0.3,
        show_gaps: bool = True,
        time_unit: Literal["s", "ms", "ns", "gps"] = "gps",
        multichannel: Literal["overlay", "subplots"] = "overlay",
        **kwargs,
    ):
        """Plot the frame's time-series data.

        Requires matplotlib. Install with: pip install sgn-ts[plot]

        Args:
            ax:
                matplotlib Axes, optional. If None, creates a new figure and axes.
                Ignored if multichannel='subplots' and channel is None.
            label:
                str, optional. Legend label for this frame's data line.
            channel:
                int or tuple[int, ...], optional. For multi-dimensional data,
                specifies which channel(s) to plot. If None and data is
                multi-dimensional, behavior depends on multichannel parameter.
            gap_color:
                str, color for gap region shading. Default 'red'.
            gap_alpha:
                float, alpha transparency for gap region shading. Default 0.3.
            show_gaps:
                bool, whether to show gap indicators. Default True.
            time_unit:
                str, time unit for x-axis: 's' (seconds since start), 'ms',
                'ns', or 'gps' (absolute GPS time). Default 'gps'.
            multichannel:
                str, how to handle multi-channel data when channel is None:
                'overlay' plots all channels on the same axes,
                'subplots' creates a subplot for each channel. Default 'overlay'.
            **kwargs:
                Additional keyword arguments passed to ax.plot().

        Returns:
            tuple: (fig, ax) matplotlib figure and axes objects. If
                   multichannel='subplots', ax is an array of axes.
        """
        from sgnts.plotting import plot_frame

        return plot_frame(
            self,
            ax=ax,
            label=label,
            channel=channel,
            gap_color=gap_color,
            gap_alpha=gap_alpha,
            show_gaps=show_gaps,
            time_unit=time_unit,
            multichannel=multichannel,
            **kwargs,
        )
