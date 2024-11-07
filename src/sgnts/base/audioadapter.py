"""
The audioadapter stores buffers of data into a deque
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Optional

from sgnts.base.array_ops import Array, ArrayOps
from sgnts.base.buffer import SeriesBuffer
from sgnts.base.offset import Offset
from sgnts.base.slice_tools import TSSlice


class Audioadapter:
    """
    The audioadapter stores buffers of data into a deque, and will
    track the copying and flushing of data from the adapter
    """

    def __init__(self, lib: type[ArrayOps] = ArrayOps):
        self.buffers: Deque[SeriesBuffer] = deque()
        self.size = 0
        self.gap_size = 0
        self.nongap_size = 0
        self.sample_rate = -1
        self.data_all: Optional[Array] = None
        self.data_all_offset_seg: Optional[tuple[int, int]] = None
        self.lib = lib

    def __len__(self) -> int:
        return len(self.buffers)

    @property
    def offset(self) -> int:
        if len(self) == 0:
            raise ValueError("Audioadapter not populated")
        return self.buffers[0].offset

    @property
    def end_offset(self) -> int:
        if len(self) == 0:
            raise ValueError("Audioadapter not populated")
        return self.offset + Offset.fromsamples(self.size, self.sample_rate)

    def concatenate_data(
        self, offset_segment: Optional[tuple[int, int]] = None
    ) -> None:
        """
        Concatenate all the data and gaps info in the buffers, and save as attribute
        """
        if self.size > 0:
            if offset_segment is not None:
                bufs = self.get_sliced_buffers(offset_segment)
                self.data_all_offset_seg = offset_segment
            else:
                bufs = self.buffers
                self.data_all_offset_seg = self.get_available_offset_segment()

            self.data_all = self.lib.cat_func(
                [b.filleddata(self.lib.zeros_func) for b in bufs], axis=-1
            )

    def push(self, buf: SeriesBuffer) -> None:
        """
        Push buffer into the deque
        """
        if buf.noffset == 0 and len(self) > 0:
            # if there are no buffers and the very first buffer we receive
            # is a zero length buffer, still push it into the adapter
            return

        if self.sample_rate == -1:
            self.sample_rate = buf.sample_rate
        elif buf.sample_rate != self.sample_rate:
            # buffers in the audioadapter must be the same sample rate
            raise ValueError(
                f"Inconsistent sample rate, buffer sample rate: {buf.sample_rate}"
                f" audioadpater sample rate: {self.sample_rate}"
            )

        # Check if the start time is as expected
        # FIXME should we support discontinuities?
        if len(self) > 0 and buf.offset != self.end_offset:
            raise ValueError(
                f"Got an unexpected buffer offset: {buf.offset=}"
                f" instead of {self.end_offset=} {buf=}"
            )

        # Store gap information
        nsamples = buf.samples
        self.size += nsamples
        is_gap = buf.is_gap
        if is_gap is True:
            self.gap_size += nsamples
        elif is_gap is False:
            self.nongap_size += nsamples
        else:
            raise ValueError(f"Unknown is_gap value {is_gap=} {type(is_gap)=}")

        self.buffers.append(buf)
        self.data_all = None  # reset the data array
        self.data_all_offset_seg = None

    def get_available_offset_segment(self) -> tuple[int, int]:
        """
        Return the full segment of all the available samples in the adapter
        """
        if self.offset is None:
            raise ValueError("Audioadapter not populated")
        return (self.offset, self.end_offset)

    def get_sliced_buffers(
        self, offset_segment: tuple[int, int], pad_start: bool = False
    ) -> Deque[SeriesBuffer]:
        """
        Return buffers that lie within the offset_segment, slice up buffers if neeeded
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
                "Requested offset {start} outside of available offset {self.offset}"
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
        """
        Copy nsamples from the start_sample of the deque
        """
        start_offset = Offset.fromsamples(start_sample, self.sample_rate) + self.offset
        end_offset = Offset.fromsamples(nsamples, self.sample_rate) + self.offset

        return self.copy_samples_by_offset_segment((start_offset, end_offset))

    def copy_samples_by_offset_segment(
        self, offset_segment: tuple[int, int], pad_zeros: bool = False
    ) -> Array:
        """
        Copy samples within the offset segment

        Parameters
        ----------
        offset_segment: tuple[int, int]
            the offset segment
        pad_zeros: bool = False
            pad zeros in front if offset_segment[0] is earlier
            than the available segment
        """
        if self.data_all_offset_seg is None:
            avail_seg = (self.offset, self.end_offset)
        else:
            avail_seg = self.data_all_offset_seg

        assert offset_segment[1] <= avail_seg[1], (
            f"rate: {self.sample_rate} requested end segment outside of"
            f"available segment, requested: {offset_segment}, available: {avail_seg}"
        )

        if pad_zeros is False:
            assert offset_segment[0] >= avail_seg[0], (
                "requested start segment outside of available segment,"
                f"requested: {offset_segment}, available: {avail_seg}"
            )

        # find start sample
        if self.data_all_offset_seg is not None:
            offset = self.data_all_offset_seg[0]
        else:
            offset = self.offset

        ni = Offset.tosamples(offset_segment[0] - offset, self.sample_rate)
        nsamples = Offset.tosamples(
            offset_segment[1] - offset_segment[0], self.sample_rate
        )

        pad_samples = 0
        if ni < 0 and pad_zeros is True:
            pad_samples = -ni
            ni = 0
            nsamples -= pad_samples

        segment_has_gaps, segment_has_nongaps = self.segment_gaps_info(offset_segment)
        # check gaps before copying
        if self.nongap_size == 0 or not segment_has_nongaps:
            # no nongaps
            out = None
        else:
            if self.data_all is None:
                bufs = self.get_sliced_buffers(offset_segment)
                if len(bufs) == 1:
                    out = bufs[0].data
                else:
                    out = self.lib.cat_func(
                        [b.filleddata(self.lib.zeros_func) for b in bufs], axis=-1
                    )
            else:
                out = self.data_all[..., ni : ni + nsamples]

        # FIXME: the checks on out are to avoid mypy errors
        if pad_samples > 0 and out is not None and not isinstance(out, int):
            out = self.lib.pad_func(out, (pad_samples, 0))

        return out

    def flush_samples(self, nsamples: int) -> None:
        """
        Flush nsamples from the head of the deque
        """
        self.flush_samples_by_end_offset_segment(
            self.offset + Offset.fromsamples(nsamples, self.sample_rate)
        )

    def flush_samples_by_end_offset_segment(self, end_offset_segment: int) -> None:
        """
        Flush nsamples from the head of the deque up to the end of the offset segment
        """
        avail = self.get_available_offset_segment()
        if end_offset_segment < avail[0]:
            return

        if end_offset_segment > avail[1]:
            raise ValueError(
                f"offset segment outside of available segment"
                f" {end_offset_segment} {avail}"
            )

        while self.size > 0:
            b = self.buffers[0]
            if b.end_offset <= end_offset_segment:
                # pop out old buffers
                self.buffers.popleft()
                if b.is_gap:
                    self.gap_size -= b.samples
                else:
                    self.nongap_size -= b.samples
                self.size -= b.samples
            else:
                if b.offset < end_offset_segment:
                    # if the end_offset_segment lies within a buffer, split the buffer
                    l, r = b.split(end_offset_segment)
                    self.buffers[0] = r
                    if l.is_gap:
                        self.gap_size -= l.samples
                    else:
                        self.nongap_size -= l.samples
                    self.size -= l.samples
                break

        self.data_all = None
        self.data_all_offset_seg = None

    def buffers_gaps_info(self, offset_segment: tuple[int, int]) -> list[bool]:
        """
        Return a list of booleans that flag buffers based on whether they are gaps
        True: is_gap, False: is_nongap
        """
        return [b.is_gap for b in self.get_sliced_buffers(offset_segment)]

    def samples_gaps_info(self, offset_segment: tuple[int, int]) -> Array:
        """
        Return an array of booleans that flag samples based on whether they are gaps
        True: is_gap, False: is_nongap
        """
        return self.lib.cat_func(
            [
                self.lib.full_func((b.samples,), b.is_gap)
                for b in self.get_sliced_buffers(offset_segment)
            ],
            axis=-1,
        )

    def segment_gaps_info(self, offset_segment: tuple[int, int]) -> tuple[bool, bool]:
        """
        Identify whether there are gaps or nongaps in the requested offset_segment
        """
        gaps = self.buffers_gaps_info(offset_segment)
        has_gaps = True in gaps
        has_nongaps = False in gaps
        return has_gaps, has_nongaps

    def is_gap(self) -> bool:
        """
        True if all buffers are gaps
        """
        if self.nongap_size == 0:
            return True
        else:
            return False
