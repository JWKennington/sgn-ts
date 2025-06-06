from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

from sgnts.base import Offset, SeriesBuffer, TSFrame, TSTransform


# This method is faster than np.median() for odd array lengths
def get_new_median(arr, old_med):
    N_med = len(arr)
    assert N_med % 2, "get_new_median requires odd array length"
    num_greater = (arr > old_med).sum()
    if num_greater > N_med // 2:
        new_med = min(x for x in arr if x > old_med)
    else:
        num_less = (arr < old_med).sum()
        if num_less > N_med // 2:
            new_med = max(x for x in arr if x < old_med)
        else:
            new_med = old_med
    return new_med


@dataclass
class MedianMean(TSTransform):
    """Computes a running median over the previous and subsequent
    "median_overlap_samples". If "mean_overlap_samples" is set to something other than
    the default of (0, 0), the running median is followed by a running mean. The output
    will be at the same sample rate as the input.

    Args:
        median_overlap_samples:
            tuple[int, int], how many previous and subsequent samples over which to
            take the running median
        mean_overlap_samples:
            tuple[int, int], how many previous and subsequent samples over which to
            take the running mean following the running median
        default_value:
            float | complex, the default value, used to initialize the median array and
            to fill gaps, unless default-to-median is set to True
        default_to_median:
            bool, whether to fill gaps with the default value or the current median
        reject_zeros:
            bool, whether or not to replace zeros with either the default value or
            current median
    """

    median_overlap_samples: tuple[int, int] = (2048, 0)
    mean_overlap_samples: tuple[int, int] = (0, 0)
    default_value: Optional[float, complex] = 0.0
    default_to_median: bool = True
    reject_zeros: bool = True

    def __post_init__(self):
        super().__post_init__()
        # This element is written to assume one channel, one source pad and one sink pad
        assert len(self.source_pads) == len(self.sink_pads) == 1

        # Initialize the arrays
        assert self.median_array_len > 0 and self.mean_array_len > 0
        if self.default_value is not None:
            self.median_array = np.tile(self.default_value, self.median_array_len)
            self.mean_array = np.tile(self.default_value, self.mean_array_len)
            self.samples_in_median = self.median_array_len
            self.samples_in_mean = self.mean_array_len
            self.current_median = self.default_value
        else:
            self.median_array = np.zeros(self.median_array_len, dtype=float)
            self.mean_array = np.zeros(self.mean_array_len, dtype=float)
            self.samples_in_median = self.samples_in_mean = 0
            self.current_median = 0.0 + 0.0j

        # We won't know these until the first data arrives
        self.median_array_index = self.mean_array_index = 0
        self.array_index_set = False
        self.real = None

    @property
    def median_array_len(self):
        """Get the length of the running median array"""
        return 1 + sum(self.median_overlap_samples)

    @property
    def mean_array_len(self):
        """Get the length of the running mean array"""
        return 1 + sum(self.mean_overlap_samples)

    @property
    def latency(self):
        """Get the latency of the element in samples"""
        return self.median_overlap_samples[1] + self.mean_overlap_samples[1]

    def update_median(self, new_sample):
        # Update the number of samples in the array
        if self.samples_in_median < self.median_array_len:
            self.samples_in_median += 1
        # Update our location in the array
        self.median_array_index = (self.median_array_index + 1) % self.median_array_len

        self.median_array[self.median_array_index] = new_sample
        if self.samples_in_median < self.median_array_len:
            # Then we are taking the median of a subset of this array
            median_array_subset = np.roll(
                self.median_array, self.median_array_len - self.median_array_index - 1
            )[-self.samples_in_median :]
            if self.samples_in_median % 2:
                if self.real:
                    self.current_median = get_new_median(
                        median_array_subset, self.current_median
                    )
                else:
                    self.current_median = get_new_median(
                        median_array_subset.real, self.current_median.real
                    ) + 1j * get_new_median(
                        median_array_subset.imag, self.current_median.imag
                    )
            else:
                if self.real:
                    self.current_median = np.median(median_array_subset)
                else:
                    self.current_median = np.median(
                        median_array_subset.real
                    ) + 1j * np.median(median_array_subset.imag)
        else:
            if self.samples_in_median % 2:
                if self.real:
                    self.current_median = get_new_median(
                        self.median_array, self.current_median
                    )
                else:
                    self.current_median = get_new_median(
                        self.median_array.real, self.current_median.real
                    ) + 1j * get_new_median(
                        self.median_array.imag, self.current_median.imag
                    )
            else:
                if self.real:
                    self.current_median = np.median(self.median_array)
                else:
                    self.current_median = np.median(
                        self.median_array.real
                    ) + 1j * np.median(self.median_array.imag)

    def compute_mean(self):
        new_sample = self.current_median
        # Update the number of samples in the array
        if self.samples_in_mean < self.mean_array_len:
            self.samples_in_mean += 1
        # Update our location in the arrays
        self.mean_array_index = (self.mean_array_index + 1) % self.mean_array_len

        self.mean_array[self.mean_array_index] = new_sample

        if self.samples_in_mean < self.mean_array_len:
            # Then we are taking the mean of a subset of this array
            mean_array_subset = np.roll(
                self.mean_array, self.mean_array_len - self.mean_array_index - 1
            )[-self.samples_in_mean :]
            return np.mean(mean_array_subset)
        else:
            return np.mean(self.mean_array)

    def internal(self):
        super().internal()
        frame = self.preparedframes[self.sink_pads[0]]
        self.outbufs = []
        for inbuf in frame:
            if inbuf.is_gap:
                # We need to fill in gaps with either the most recent median or the
                # default value
                samples_to_fill = Offset.tosamples(
                    inbuf.end_offset - inbuf.offset, inbuf.sample_rate
                )
                if samples_to_fill <= 0:
                    outdata = None
                else:
                    if self.real:
                        outdata = np.empty(samples_to_fill, dtype=np.float64)
                    else:
                        outdata = np.empty(samples_to_fill, dtype=np.complex128)
                    for idx in range(samples_to_fill):
                        # Take care of the real part first.  If the input is real, this
                        # is all we need to do.
                        if self.default_to_median:
                            new_sample = self.current_median
                        else:
                            new_sample = self.default_value
                        # Update the current median
                        self.update_median(new_sample)
                        # Compute the mean to set the next output value
                        outdata[idx] = self.compute_mean()
            else:
                if self.real is None:
                    self.real = not isinstance(
                        inbuf.data[0], (complex, np.complex128, np.clongdouble)
                    )
                    # Now, check whether the median and mean arrays are the right type
                    if self.real and isinstance(
                        self.median_array[0], (complex, np.complex128, np.clongdouble)
                    ):
                        # FIXME: Should this warning be something other than a print
                        # statement?
                        msg = (
                            "WARNING: MedianMean: data is real; discarding imaginary "
                            "part of default value, median array, and mean array"
                        )
                        print(msg)
                        if self.default_value is not None:
                            self.default_value = self.default_value.real
                        self.median_array = self.median_array.real
                        self.mean_array = self.mean_array.real
                    elif not self.real and isinstance(
                        self.median_array[0], (float, np.float64, np.longdouble)
                    ):
                        self.median_array = self.median_array.astype(complex)
                        self.mean_array = self.mean_array.astype(complex)
                if self.real:
                    outdata = np.empty(len(inbuf.data), dtype=np.float64)
                else:
                    outdata = np.empty(len(inbuf.data), dtype=np.complex128)
                if self.array_index_set:
                    # Check that the array index is still aligned with the offset
                    assert (
                        self.median_array_index
                        == Offset.tosamples(inbuf.offset, inbuf.sample_rate)
                        % self.median_array_len
                    )
                    assert (
                        self.mean_array_index
                        == Offset.tosamples(inbuf.offset, inbuf.sample_rate)
                        % self.mean_array_len
                    )
                else:
                    # Make sure the order of the arrays is independent of start time.
                    self.median_array_index = (
                        Offset.tosamples(inbuf.offset, inbuf.sample_rate)
                        % self.median_array_len
                    )
                    self.mean_array_index = (
                        Offset.tosamples(inbuf.offset, inbuf.sample_rate)
                        % self.mean_array_len
                    )
                    self.array_index_set = True

                for idx in range(len(inbuf.data)):
                    # Take care of the real part first.  If the input is real, this is
                    # all we need to do.
                    new_sample = inbuf.data[idx]
                    if (
                        np.isinf(new_sample)
                        or np.isnan(new_sample)
                        or (new_sample == 0 and self.reject_zeros)
                    ):
                        if self.default_to_median:
                            new_sample = self.current_median
                        else:
                            new_sample = self.default_value
                    # Update the current median
                    self.update_median(new_sample)
                    # Compute the mean to set the next output value
                    outdata[idx] = self.compute_mean()

            outbuf = SeriesBuffer(
                offset=inbuf.offset
                - Offset.fromsamples(self.latency, inbuf.sample_rate),
                sample_rate=inbuf.sample_rate,
                data=outdata,
                shape=inbuf.shape,
            )
            self.outbufs.append(outbuf)
        self.eos = frame.EOS
        self.metadata = frame.metadata

    def new(self, pad):
        return TSFrame(buffers=self.outbufs, EOS=self.eos, metadata=self.metadata)
