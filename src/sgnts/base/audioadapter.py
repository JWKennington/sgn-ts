"""
The audioadapter stores buffers of data into a deque
"""

from collections import deque

import numpy as np
from numpy import pad

from .buffer import SeriesBuffer
from .offset import Offset
from .time import Time
from .array_ops import ArrayOps


class Audioadapter:
    """
    The audioadapter stores buffers of data into a deque, and will
    track the copying and flushing of data from the adapter
    """

    def __init__(self, lib=ArrayOps):
        self.buffers = deque()
        self.size = 0
        self.gap_size = 0
        self.nongap_size = 0
        self.skip = 0
        self.offset = None
        self.next_offset = None
        self.sample_rate = None
        self.data_all = None
        self.lib = lib

    @property
    def end_offset(self):
        if self.offset is not None:
            return self.offset + Offset.fromsamples(self.size, self.sample_rate)
        else:
            return None

    def concatenate_data(self):
        """
        Concatenate all the data and gaps info in the buffers, and save as attribute
        """
        if self.size > 0:
            self.data_all = self.lib.cat_func([b.filleddata(self.lib.zeros_func) for b in self.buffers], axis=-1)


    def push(self, buf):
        """
        Push buffer into the deque
        """
        tb = type(buf)
        assert (
            tb is SeriesBuffer
        ), f"Buffers should be of type SeriesBuffer, instead got {tb}"

        if buf.noffset == 0 and len(self.buffers) > 0:
            # if there are no buffers and the very first buffer we receive
            # is a zero lenth buffer, still push it into the adapter
            return

        sample_rate = buf.sample_rate
        if self.sample_rate is None:
            self.sample_rate = sample_rate
        else:
            # buffers in the audioadapter must be the same sample rate
            assert sample_rate == self.sample_rate, f"{sample_rate} {self.sample_rate}"

        # Check if the start time is as expected
        # FIXME should we support discontinuities?
        next_offset = self.next_offset
        if next_offset is not None and buf.offset != next_offset:
            raise ValueError(
                f"got an unexpected buffer offset: {buf.offset=}"
                f" instead of {next_offset=} sample rate: {buf.sample_rate=}"
            )
        self.next_offset = buf.end_offset

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
        self.data_all = None # reset the data array

        if self.offset is None or len(self.buffers) == 1:
            self.offset = buf.offset

    def get_available_offset_segment(self):
        """
        Return the full segment of all the available samples in the adapter
        """
        if self.offset is None:
            return (0, 0)
        else:
            return (self.offset, self.end_offset)

    def samples_remaining(self, buf, start_sample=None):
        """
        The remaining samples in the deque yet to be processed
        """
        n = buf.samples
        if start_sample is not None:
            assert start_sample <= n
            return n - start_sample
        else:
            assert self.skip <= n
            return n - self.skip

    def copy_samples(self, nsamples, start_sample=0):
        """
        Copy nsamples from the start_sample of the deque
        """
        assert nsamples > 0, f"{nsamples=} {self.sample_rate=}"
        assert nsamples == int(nsamples), f"{nsamples=} must be an integer"

        i0 = self.skip + start_sample

        # check gaps before copying
        copy_data = False
        if self.nongap_size == 0:
            # no nongaps
            out = None
        else:
            if self.data_all is None:
                out = self.lib.cat_func([b.filleddata(self.lib.zeros_func) for b in self.buffers], axis=-1)[
                    ..., i0 : i0 + nsamples
                ]
            else:
                out = self.data_all[..., i0 : i0 + nsamples]

        return out

    def copy_samples_by_offset_segment(self, offset_segment, pad_zeros=False):
        """
        Copy samples within the offset segment

        Arguments:
        ----------
        offset_segment: tuple[int, int]
            the offset segment
        pad_zeros: bool = False
            pad zeros in front if offset_segment[0] is earlier
            than the available segment
        """
        avail_seg = self.get_available_offset_segment()

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
        ni = Offset.tosamples(offset_segment[0] - self.offset, self.sample_rate)
        assert ni == int(ni), "start sample point number is not an integer"
        ni = int(ni)

        nsamples = Offset.tosamples(
            offset_segment[1] - offset_segment[0], self.sample_rate
        )
        assert nsamples == int(nsamples), (
            f"nsamples is not an integer, nsamples: {nsamples}, "
            f"segment: {offset_segment}"
        )
        nsamples = int(nsamples)

        pad_samples = 0
        if ni < 0 and pad_zeros is True:
            pad_samples = -ni
            ni = 0
            nsamples -= pad_samples

        out = self.copy_samples(nsamples, start_sample=ni)
        if pad_samples > 0 and out is not None:
            out = self.lib.pad_func(out, (pad_samples,0))

        return out

    def flush_samples(self, nsamples: int):
        """
        Flush nsamples from the head of the deque
        """
        if nsamples <= 0:
            return

        assert nsamples <= self.size, f"{nsamples} {self.size}"

        nsamples = int(nsamples)

        while nsamples:
            buf = self.buffers[0]
            n = self.samples_remaining(buf)
            is_gap = buf.is_gap
            if nsamples < n:
                self.skip += nsamples
                self.size -= nsamples
                if is_gap is True:
                    self.gap_size -= nsamples
                else:
                    self.nongap_size -= nsamples
                break
            self.skip = 0
            self.size -= n
            if is_gap is True:
                self.gap_size -= n
            else:
                self.nongap_size -= n
            nsamples -= n
            self.buffers.popleft()

        if len(self.buffers) > 0:
            buf0 = self.buffers[0]
            self.offset = buf0.offset + Offset.fromsamples(self.skip, self.sample_rate)

        self.data_all = None

    def flush_samples_by_end_offset_segment(self, end_offset_segment):
        """
        Flush nsamples from the head of the deque up to the end of the offset segment
        """
        avail = self.get_available_offset_segment()
        assert avail[0] <= end_offset_segment <= avail[1], (
            f"offset segment outside of available segment"
            f"{end_offset_segment} {avail}"
        )

        nsamples = Offset.tosamples(end_offset_segment - self.offset, self.sample_rate)
        assert nsamples == int(nsamples), "number of samples is not an integer"
        nsamples = int(nsamples)

        self.flush_samples(nsamples)

    def clear(self):
        """
        Clear out the deque and reset metadata
        """
        self.__init__()

    def get_gaps_info(self, nsamples, start_sample=0):
        """
        Return a list of booleans that flag samples based on whether they are gaps
        True: is_gap, False: is_nongap
        """
        out = self.lib.cat_func(
            [self.lib.full_func((b.samples,), b.is_gap) for b in self.buffers], axis=-1
        )
        i0 = self.skip + start_sample
        out = out[..., i0 : i0 + nsamples]
        return out

    def is_gap(self):
        """
        True if all buffers are gaps
        """
        if self.nongap_size == 0:
            return True
        else:
            return False
