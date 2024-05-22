"""
The audioadapter stores buffers of data into a deque
"""

from collections import deque
import numpy as np
from numpy import pad

from .time import Time
from .buffer import *


class Audioadapter:
    """
    The audioadapter stores buffers of data into a deque, and will
    track the copying and flushing of data from the adapter
    """

    def __init__(self):
        self.buffers = deque()
        self.is_gaps = deque()
        self.size = 0
        self.gap_size = 0
        self.nongap_size = 0
        self.skip = 0
        self.starttime = None
        self.offset = None
        self.next_offset = None
        self.sample_rate = None
        self.next_starttime = None
        self.cat_data = None
        self.cat_gaps = None
        self.channels = None
        # self.device = None
        # self.dtype = None
        self.SECONDS = Time.SECONDS
        self.zero = None

    @property
    def endtime(self):
        if self.starttime is not None:
            return self.starttime + self.size / self.sample_rate * Time.SECONDS
        else:
            return None

    @property
    def end_offset(self):
        if self.offset is not None:
            return int(self.offset + self.size * OFFSET_RATE / self.sample_rate)
        else:
            return None

    def pad_func(self, data, pad_samples):
        npad = [(0, 0)] * data.ndim
        npad[-1] = (pad_samples, 0)
        return np.pad(data, npad, "constant")

    def cat(self, xs, axis):
        return np.concatenate(xs, axis=axis)

    def zero_func(self):
        return np.zeros(1)

    def concatenate_data(self):
        """
        Concatenate all the data and gaps info in the buffers, and save as attribute
        """
        if self.size > 0:
            # self.cat_data = torch.cat([b.data for b in self.buffers], dim=-1)
            self.cat_data = self.cat([b.data for b in self.buffers], axis=-1)
            if self.cat_gaps is None and self.gap_size > 0 and self.nongap_size > 0:
                # mixture of gaps and nongaps
                self.cat_gaps = np.concatenate(self.is_gaps)

    def push(self, buf):
        """
        Push buffer into the deque
        """
        if buf.duration == 0:
            return

        tb = type(buf)
        if tb is SeriesBuffer:
            sample_rate = buf.sample_rate
        else:
            raise ValueError(
                f"Buffers should be of type SeriesBuffer, instead got {tb}"
            )

        if self.sample_rate is not None:
            # buffers in the audioadapter must be the same sample rate
            assert sample_rate == self.sample_rate, f"{sample_rate} {self.sample_rate}"

        # Check if the start time is as expected
        # FIXME should we support discontinuities?
        next_starttime = self.next_starttime
        next_offset = self.next_offset
        if next_offset is not None and buf.offset != next_offset:
            raise ValueError(
                f"got an unexpected buffer offset: {buf.offset=} \
                        instead of {next_offset=}"
            )
        if next_starttime is not None and buf.t0 - next_starttime > 1:
            raise ValueError(
                f"got an unexpected buffer timestamp: {buf.t0=} \
                        instead of {next_starttime=}"
            )
        self.next_starttime = buf.t0 + buf.duration
        self.next_offset = buf.offset + buf.noffset

        nsamples = buf.size
        self.size += nsamples
        data = buf.data
        # if self.device is None:
        #    self.device = data.device

        # if self.dtype is None:
        #    self.dtype = data.dtype

        if self.channels is None:
            self.channels = data.shape[:-1]

        if self.zero is None:
            # self.zero = torch.zeros(1, device=data.device, dtype=data.dtype)
            self.zero = self.zero_func()

        is_gap = buf.is_gap
        self.is_gaps.append(np.broadcast_to(is_gap, nsamples))
        if is_gap is True:
            self.gap_size += nsamples
        elif is_gap is False:
            self.nongap_size += nsamples
        else:
            raise ValueError(f"Unknown is_gap value {is_gap=} {type(is_gap)=}")

        self.buffers.append(buf)

        if self.starttime is None or len(self.buffers) == 1:
            self.starttime = buf.t0
            self.sample_rate = sample_rate
            self.offset = buf.offset

    def get_available_offset_segment(self):
        """
        Return the full segment of all the available samples in the adapter
        """
        if self.offset is None:
            return (0, 0)
        else:
            return (self.offset, self.end_offset)

    def get_available_segment(self):
        """
        Return the full segment of all the available samples in the adapter
        """
        return (self.starttime, self.endtime)

    def samples_remaining(self, buf, start_sample=None):
        """
        The remaining samples in the deque yet to be processed
        """
        n = buf.size
        if start_sample is not None:
            assert start_sample <= n
            return n - start_sample
        else:
            assert self.skip <= n
            return n - self.skip

    def copy_samples(self, nsamples, start_sample=0):
        """
        Copy nsamples from the head of the deque to dst
        """
        assert nsamples > 0, f"{nsamples=} {self.sample_rate=}"
        assert nsamples == int(nsamples), f"{nsamples=} must be an integer"

        copied_gap = False
        copied_nongap = False

        i0 = self.skip + start_sample

        # check gaps
        copy_data = False
        if self.gap_size == 0:
            # no gaps in buffer
            copied_nongap = True
            copied_gap = False
            copy_data = True
        elif self.nongap_size == 0:
            # no nongaps
            copied_nongap = False
            copied_gap = True
        else:
            # some gaps, some nongaps
            gaps = self.get_gaps_info(nsamples, start_sample)
            copied_gap = gaps.any().item()
            copied_nongap = ((~gaps).any()).item()
            copy_data = True

        # copy data
        if copy_data is True:
            if self.cat_data is None:
                # out = torch.cat([b.data for b in self.buffers], dim=-1)[
                #    ..., i0 : i0 + nsamples
                # ]
                out = self.cat([b.data for b in self.buffers], axis=-1)[
                    ..., i0 : i0 + nsamples
                ]
            else:
                out = self.cat_data[..., i0 : i0 + nsamples]
        else:
            if self.channels is None:
                out = self.zero.expand(nsamples)
            else:
                out = self.zero.expand(self.channels + (nsamples,))

        return out, copied_gap, copied_nongap

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

        copied_gap = False
        copied_nongap = False

        # find start sample
        ni = int((offset_segment[0] - self.offset) / (OFFSET_RATE / self.sample_rate))
        assert ni == int(ni), "start sample point number is not an integer"
        ni = int(ni)

        nsamples = int(
            (offset_segment[1] - offset_segment[0]) / (OFFSET_RATE / self.sample_rate)
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

        out, copied_gap, copied_nongap = self.copy_samples(nsamples, start_sample=ni)
        if pad_samples > 0:
            out = self.pad_func(out, pad_samples)

        return out, copied_gap, copied_nongap

    def copy_samples_by_segment(self, segment, pad_zeros=False):
        """
        Copy samples within the segment to dst

        Arguments:
        ----------
        segment: segments.segment
            the segment
        pad_zeros: bool = False
            pad zeros in front if segment[0] is earlier than the available segment
        """
        avail_seg = self.get_available_segment()

        assert segment[1] <= avail_seg[1], (
            f"rate: {self.sample_rate} requested end segment outside of"
            f"available segment, requested: {segment}, available: {avail_seg}"
        )

        if pad_zeros is False:
            assert segment[0] >= avail_seg[0], (
                "requested start segment outside of available segment,"
                f"requested: {segment}, available: {avail_seg}"
            )

        copied_gap = False
        copied_nongap = False

        # find start sample
        ni = (segment[0] - self.starttime) / 1e9 * self.sample_rate
        assert ni == int(ni), "start sample point number is not an integer"
        ni = int(ni)

        nsamples = (segment[1] - segment[0]) / 1e9 * self.sample_rate
        assert nsamples == int(
            nsamples
        ), f"nsamples is not an integer, nsamples: {nsamples}, segment: {segment}"
        nsamples = int(nsamples)

        pad_samples = 0
        if ni < 0 and pad_zeros is True:
            pad_samples = -ni
            ni = 0
            nsamples -= pad_samples

        out, copied_gap, copied_nongap = self.copy_samples(nsamples, start_sample=ni)
        if pad_samples > 0:
            out = pad(out, (pad_samples, 0), "constant")

        return out, copied_gap, copied_nongap

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
            self.is_gaps.popleft()

        if len(self.buffers) > 0:
            buf0 = self.buffers[0]
            skip_duration = (self.skip / self.sample_rate) * Time.SECONDS
            self.starttime = buf0.t0 + skip_duration
            self.offset = buf0.offset + int(self.skip * OFFSET_RATE / self.sample_rate)

        self.cat_data = None
        self.cat_gaps = None

    def flush_samples_by_end_offset_segment(self, end_offset_segment):
        """
        Flush nsamples from the head of the deque up to the end of the offset segment
        """
        avail = self.get_available_offset_segment()
        assert avail[0] <= end_offset_segment <= avail[1], (
            f"offset segment outside of available segment"
            f"{end_offset_segment} {avail}"
        )

        nsamples = (end_offset_segment - self.offset) / (OFFSET_RATE / self.sample_rate)
        assert nsamples == int(nsamples), "number of samples is not an integer"
        nsamples = int(nsamples)

        self.flush_samples(nsamples)

    def flush_samples_by_end_segment(self, segment):
        """
        Flush nsamples from the head of the deque up to the end of the segment
        """
        assert (
            segment in self.get_available_segment()
        ), "segment outside of available segment"

        nsamples = (segment[1] - self.starttime) * self.sample_rate / Time.SECONDS
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
        if self.cat_gaps is None:
            out = np.concatenate(self.is_gaps)
        else:
            out = self.cat_gaps
        i0 = self.skip + start_sample
        out = out[..., i0 : i0 + nsamples]
        return out

    def is_gap(self):
        """
        True if all buffers are gaps
        """
        if self.nongap_size == 0:
            return True
        elif self.gap_size == 0:
            return False
        else:
            gaps = np.concatenate(self.is_gaps)
            return gaps.all().item()
