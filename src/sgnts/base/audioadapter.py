"""The audioadapter stores buffers of data into a deque."""

from __future__ import annotations

from collections import deque
from typing import Deque, Optional

from sgnts.base.array_ops import Array, ArrayBackend, NumpyBackend
from sgnts.base.buffer import SeriesBuffer
from sgnts.base.offset import Offset
from sgnts.base.slice_tools import TSSlice


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
