"""The audioadapter stores buffers of data into a deque."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Union

from sgnts.base.array_ops import Array, ArrayBackend, NumpyBackend
from sgnts.base.buffer import SeriesBuffer
from sgnts.base.offset import Offset
from sgnts.base.slice_tools import TSSlice


@dataclass
class AdapterConfig:
    """Config to hold parameters used for the audioadapter in _TSTransSink.

    Args:
        enable:
            bool | None, controls whether adapter processing is enabled.
            - None (default): Auto-detect based on configuration
              - Disabled if all config values remain at defaults
              - Enabled if any non-default config is provided
              - Enabled if any configuration method is called
            - True: Force enable adapter processing
            - False: Force disable adapter processing
            Default: None
        overlap:
            tuple[int, int], the overlap before and after the data segment to process,
            in offsets
        stride:
            int, the stride to produce, in offsets
        pad_zeros_startup:
            bool, when overlap is provided, whether to pad zeros in front of the
            first buffer, or wait until there is enough data.
        skip_gaps:
            bool, produce a whole gap buffer if there are any gaps in the copied data
            segment
        backend:
            type[ArrayBackend], the ArrayBackend wrapper
        align_to:
            int or None, alignment boundary in offsets
            When set, output offsets will be aligned to multiples of this value.
            For example:
            - Offset.fromsec(1) aligns to integer seconds
            - Offset.fromsamples(1024, rate) aligns to 1024-sample boundaries
            Default: None (no alignment)
        align_buffers:
            bool, when True, aligns buffer slices to the minimum sampling rate
            across all pads. This expands gaps and shrinks data slices to ensure
            all buffers align to integer sample boundaries at the lowest rate.
            Default: False
        offset_shift:
            int, offset shift to apply to output buffers, in offsets This is
            used for transforms that introduce latency or phase shifts. The
            output offset will be shifted by this amount: offset +
            offset_shift. Positive values shift forward in time, negative
            values shift backward. For example, a filter with latency=2 samples
            at rate=1 Hz would use offset_shift=-Offset.fromsamples(2, 1) to
            shift output backward by 2 samples.
            Default: 0 (no shift)
    """

    enable: bool | None = None
    overlap: tuple[int, int] = (0, 0)
    stride: int = 0
    pad_zeros_startup: bool = False
    skip_gaps: bool = False
    backend: type[ArrayBackend] = NumpyBackend
    align_to: int | None = None
    align_buffers: bool = False
    offset_shift: int = 0

    @property
    def is_enabled(self) -> bool:
        """Check if adapter should be enabled.

        Returns:
            True if adapter is explicitly enabled or has non-default configuration.
            False if adapter is explicitly disabled or has all default values.
        """
        if self.enable is False:
            return False
        if self.enable is True:
            return True
        # enable is None - auto-detect based on configuration
        return (
            self.overlap != (0, 0)
            or self.stride != 0
            or self.pad_zeros_startup
            or self.skip_gaps
            or self.backend != NumpyBackend
            or self.align_to is not None
            or self.align_buffers
            or self.offset_shift != 0
        )

    def alignment(
        self,
        overlap: Optional[tuple[int, int]] = None,
        stride: Optional[int] = None,
        align_to: Optional[int] = None,
        align_buffers: Optional[bool] = None,
        shift: Optional[int] = None,
    ) -> AdapterConfig:
        """Configure alignment and buffering parameters.

        Enables the adapter when called.

        Args:
            overlap: tuple[int, int], the overlap before and after the data segment
            stride: int, the stride to produce, in offsets
            align_to: int, alignment boundary in offsets
            align_buffers: bool, align buffer slices to minimum sampling rate
            shift: int, offset shift to apply to output buffers

        Returns:
            AdapterConfig, self for method chaining
        """
        if self.enable is None:
            self.enable = True

        if overlap is not None:
            self.overlap = overlap
        if stride is not None:
            self.stride = stride
        if align_to is not None:
            self.align_to = align_to
        if align_buffers is not None:
            self.align_buffers = align_buffers
        if shift is not None:
            self.offset_shift = shift
        return self

    def on_gap(self, skip: Optional[bool] = None) -> AdapterConfig:
        """Configure gap handling.

        Enables the adapter when called.

        Args:
            skip: bool, produce a whole gap buffer if there are any gaps

        Returns:
            AdapterConfig, self for method chaining
        """
        if self.enable is None:
            self.enable = True

        if skip is not None:
            self.skip_gaps = skip
        return self

    def on_startup(self, pad_zeros: Optional[bool] = None) -> AdapterConfig:
        """Configure startup behavior.

        Enables the adapter when called.

        Args:
            pad_zeros: bool, whether to pad zeros in front of the first buffer

        Returns:
            AdapterConfig, self for method chaining
        """
        if self.enable is None:
            self.enable = True

        if pad_zeros is not None:
            self.pad_zeros_startup = pad_zeros
        return self

    def valid_buffer(self, buf, data: Optional[Union[int, Array]] = 0):
        """
        Return a new buffer corresponding to the non overlapping part of a
        buffer "buf" as defined by this classes overlap properties As a special case,
        if the buffer is shape zero (a heartbeat buffer) a new heartbeat buffer is
        returned with the offsets shifted by overlap[0].
        Otherwise, in order for the buffer to be valid it must be what is expected
        based on the adapter's overlap and stride etc.
        """

        if buf.shape == (0,):
            new_slice = TSSlice(
                buf.slice[0] + self.overlap[0], buf.slice[0] + self.overlap[0]
            )
            return buf.new(new_slice, data=None)
        else:
            expected_shape = (
                Offset.tosamples(self.overlap[0], buf.sample_rate)
                + Offset.tosamples(self.overlap[1], buf.sample_rate)
                + Offset.sample_stride(buf.sample_rate),
            )
            assert buf.shape == expected_shape
            new_slice = TSSlice(
                buf.slice[0] + self.overlap[0], buf.slice[1] - self.overlap[1]
            )
            return buf.new(new_slice, data)


class Audioadapter:
    """The audioadapter stores buffers of data into a deque, and will track the copying
    and flushing of data from the adapter.

    Args:
        backend:
            type[ArrayBackend], the wrapper around array operations
    """

    def __init__(self, backend: type[ArrayBackend] = NumpyBackend):
        self.buffers: Deque[SeriesBuffer] = deque()
        self.size: int = 0
        self.gap_size: int = 0
        self.nongap_size: int = 0
        self.sample_rate: int = -1
        self.pre_cat_data: Optional[SeriesBuffer] = None
        self.backend: type[ArrayBackend] = backend

    def __len__(self) -> int:
        return len(self.buffers)

    @property
    def offset(self) -> int:
        """The offset of the first buffer in the audioadapter.

        Returns:
            int, the offset
        """
        if len(self) == 0:
            raise ValueError("Audioadapter not populated")
        return self.buffers[0].offset

    @property
    def end_offset(self) -> int:
        """The end offset of the last buffer in the audioadapter.

        Returns:
            int, the end offset
        """
        if len(self) == 0:
            raise ValueError("Audioadapter not populated")
        return self.buffers[-1].end_offset

    @property
    def slice(self) -> tuple[int, int]:
        """The offset slice of the audioadapter.

        Returns:
            tuple[int, int], the offset slice
        """
        return (self.offset, self.end_offset)

    @property
    def is_gap(self) -> bool:
        """True if all buffers are gaps.

        Returns:
            bool, if True, the whole audioadapter is a gap. If False, there are nongap
            buffers in the audioadapter
        """
        return self.nongap_size == 0

    def concatenate_data(
        self, offset_segment: Optional[tuple[int, int]] = None
    ) -> None:
        """Concatenate all the data and gaps info in the buffers, and save as attribute.

        Args:
            offset_segment:
                Optional[tuple[int, int]], only concatenate data within this offset
                segment
        """
        if self.size > 0:
            if offset_segment is not None:
                bufs = self.get_sliced_buffers(offset_segment)
            else:
                bufs = self.buffers
                offset_segment = self.slice

            cat_all = self.backend.cat(
                [b.filleddata(self.backend.zeros) for b in bufs], axis=-1
            )

            self.pre_cat_data = SeriesBuffer(
                offset=offset_segment[0],
                data=cat_all,
                sample_rate=bufs[0].sample_rate,
                shape=bufs[0].shape[:-1]
                + (
                    Offset.tosamples(
                        offset_segment[1] - offset_segment[0], bufs[0].sample_rate
                    ),
                ),
            )

    def push(self, buf: SeriesBuffer) -> None:
        """Push buffer into the deque, update metadata.

        Args:
            buf:
                SeriesBuffer, the buffer to append to the deque of the audioadapter.
        """
        if buf.noffset == 0 and len(self) > 0:
            # if there are no buffers and the very first buffer we receive
            # is a zero length buffer, still push it into the adapter
            return

        # Check if the start time is as expected
        # FIXME should we support discontinuities?
        if self.sample_rate != -1 and buf.offset != self.end_offset:
            raise ValueError(
                f"Got an unexpected buffer offset: {buf.offset=}"
                f" instead of {self.end_offset=} {buf=}"
            )

        if self.sample_rate == -1:
            self.sample_rate = buf.sample_rate
        elif buf.sample_rate != self.sample_rate:
            # buffers in the audioadapter must be the same sample rate
            raise ValueError(
                f"Inconsistent sample rate, buffer sample rate: {buf.sample_rate}"
                f" audioadpater sample rate: {self.sample_rate}"
            )

        # Store gap information
        nsamples = buf.samples
        self.size += nsamples
        if buf.is_gap is True:
            self.gap_size += nsamples
        else:
            self.nongap_size += nsamples

        if len(self) > 0 and self.buffers[-1].duration == 0:
            if buf.duration > 0:
                # Replace heartbeat buffers
                self.buffers[-1] = buf
        else:
            self.buffers.append(buf)

        self.pre_cat_data = None  # reset

    def get_sliced_buffers(
        self, offset_segment: tuple[int, int], pad_start: bool = False
    ) -> Deque[SeriesBuffer]:
        """Return buffers that lie within the offset_segment, slice up buffers if
        neeeded.

        Args:
            offset_segment:
                tuple[int, int], the offset segment to get buffers from
            pad_start:
                bool, default False, if True and if offset segment is earlier than the
                available buffers, will make front-pad the buffers with a gap buffer

        Returns:
            Deque[SeriesBuffer], the sliced buffers within the offset segment
        """
        start = offset_segment[0]
        end = offset_segment[1]

        if end > self.end_offset:
            raise ValueError(
                f"Requested end offset {end} outside of available end offset"
                f" {self.end_offset}"
            )

        if pad_start is False and start < self.offset:
            raise ValueError(
                f"Requested offset {start} outside of available offset {self.offset}"
            )

        bufs = deque(
            b for b in self.buffers if b.offset <= end and b.end_offset >= start
        )

        if pad_start is True and start < bufs[0].offset:
            # pad buffers in front
            buf = bufs[0].pad_buffer(off=start)
            bufs.appendleft(buf)

        # check buffers at each end
        if bufs[0].offset < start:
            bufs[0] = bufs[0].sub_buffer(TSSlice(start, bufs[0].end_offset))
        if bufs[-1].end_offset > end:
            bufs[-1] = bufs[-1].sub_buffer(TSSlice(bufs[-1].offset, end))

        return bufs

    def copy_samples(self, nsamples: int, start_sample: int = 0) -> Array:
        """Copy nsamples from the start_sample of the deque.

        Args:
            nsamples:
                int, the number of samples to copy out of the audioadapter
            start_sample:
                int, start the copying from this sample point

        Returns:
            Array, the array of copied samples
        """
        start_offset = Offset.fromsamples(start_sample, self.sample_rate) + self.offset
        end_offset = Offset.fromsamples(nsamples, self.sample_rate) + start_offset

        return self.copy_samples_by_offset_segment((start_offset, end_offset))

    def copy_samples_by_offset_segment(
        self, offset_segment: tuple[int, int], pad_start: bool = False
    ) -> Array:
        """Copy samples within the offset segment.

        Args:
            offset_segment:
                tuple[int, int], the offset segment
            pad_start:
                bool, default False, pad zeros in front if offset_segment[0] is earlier
                than the available segment

        Returns:
            Array, the array of copied samples
        """
        if self.pre_cat_data is None:
            avail_seg = self.slice
        else:
            avail_seg = (
                self.pre_cat_data.offset,
                self.pre_cat_data.end_offset,
            )

        assert offset_segment[1] <= avail_seg[1], (
            f"rate: {self.sample_rate} requested end segment outside of"
            f" available segment, requested: {offset_segment}, available: {avail_seg}"
        )

        if pad_start is False:
            assert offset_segment[0] >= avail_seg[0], (
                "requested start segment outside of available segment,"
                f" requested: {offset_segment}, available: {avail_seg}"
            )

        segment_has_gaps, segment_has_nongaps = self.segment_gaps_info(
            offset_segment, pad_start
        )
        # check gaps before copying
        if self.is_gap or not segment_has_nongaps:
            # no nongaps
            out = None
        else:
            if self.pre_cat_data is None:
                bufs = self.get_sliced_buffers(offset_segment, pad_start=pad_start)
                if len(bufs) == 1:
                    out = bufs[0].data
                else:
                    out = self.backend.cat(
                        [b.filleddata(self.backend.zeros) for b in bufs], axis=-1
                    )
            else:
                # find start sample
                ni = Offset.tosamples(
                    offset_segment[0] - self.pre_cat_data.offset, self.sample_rate
                )
                nsamples = Offset.tosamples(
                    offset_segment[1] - offset_segment[0], self.sample_rate
                )
                # FIXME: this is a thorny issue because of how generous we are
                # with the type of data and the type of Array.  Fixing this will
                # involve being stricter about types and more careful throughout
                # the array_ops module.
                out = self.pre_cat_data.data[..., ni : ni + nsamples]  # type: ignore

        return out

    def flush_samples(self, nsamples: int) -> None:
        """Flush nsamples from the head of the deque.

        Args:
            nsamples:
                int, the number of samples to flush from the head of the audioadapter
        """
        self.flush_samples_by_end_offset(
            self.offset + Offset.fromsamples(nsamples, self.sample_rate)
        )

    def flush_samples_by_end_offset(self, end_offset: int) -> None:
        """Flush nsamples from the head of the deque up to the end of the offset.

        Args:
            end_offset:
                int, the end offset
        """
        avail = self.slice
        if end_offset < avail[0] or end_offset > avail[1]:
            raise ValueError(
                f"offset segment outside of available segment" f" {end_offset} {avail}"
            )

        while self.size > 0:
            b = self.buffers[0]
            if b.end_offset < end_offset:
                # pop out old buffers
                self.buffers.popleft()
                if b.is_gap:
                    self.gap_size -= b.samples
                else:
                    self.nongap_size -= b.samples
                self.size -= b.samples
            elif b.end_offset == end_offset:
                if len(self) > 1:
                    self.buffers.popleft()
                else:
                    # if b.end_offset == end_offset, have a zero-length buffer in the
                    # adapter to record metadata
                    self.buffers[0] = b.sub_buffer(
                        slc=TSSlice(end_offset, end_offset), gap=True
                    )
                if b.is_gap:
                    self.gap_size -= b.samples
                else:
                    self.nongap_size -= b.samples
                self.size -= b.samples
                break
            else:
                if b.offset < end_offset:
                    # if the end_offset lies within a buffer, split the buffer
                    l, r = b.split(end_offset)
                    self.buffers[0] = r
                    if l.is_gap:
                        self.gap_size -= l.samples
                    else:
                        self.nongap_size -= l.samples
                    self.size -= l.samples
                break

        self.pre_cat_data = None

    def buffers_gaps_info(
        self, offset_segment: tuple[int, int], pad_start: bool = False
    ) -> list[bool]:
        """Return a list of booleans that flag buffers based on whether they are gaps.
        True: is_gap, False: is_nongap

        Args:
            offset_segment:
                tuple[int, int], the offset segment to get gaps info from
            pad_start:
                bool, default False, pad zeros in front if offset_segment[0] is earlier
                than the available segment

        Returns:
            list[bool], a list of booleans that flags whether buffers are gaps
        """
        return [
            b.is_gap
            for b in self.get_sliced_buffers(offset_segment, pad_start=pad_start)
        ]

    def samples_gaps_info(self, offset_segment: tuple[int, int]) -> Array:
        """Return an array of booleans that flag samples based on whether they are gaps.
        True: is_gap, False: is_nongap

        Args:
            offset_segment:
                tuple[int, int], the offset segment to get gaps info from

        Returns:
            Array, an array of booleans that flags whether buffers are gaps
        """
        return self.backend.cat(
            [
                self.backend.full((b.samples,), b.is_gap)
                for b in self.get_sliced_buffers(offset_segment)
            ],
            axis=-1,
        )

    def segment_gaps_info(
        self, offset_segment: tuple[int, int], pad_start: bool = False
    ) -> tuple[bool, bool]:
        """Identify whether there are gaps or nongaps in the requested offset_segment.

        Args:
            offset_segment:
                tuple[int, int], the offset segment to check for gaps and nongaps
            pad_start:
                bool, default False, pad zeros in front if offset_segment[0] is earlier
                than the available segment

        Returns:
            tuple[bool, bool], the tuple representing gaps information in the form
            (has_gaps, has_nongaps)
        """
        if self.is_gap:
            return True, False
        else:
            gaps = self.buffers_gaps_info(offset_segment, pad_start=pad_start)
            return True in gaps, False in gaps
