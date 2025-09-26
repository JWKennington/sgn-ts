from __future__ import annotations

from dataclasses import dataclass
from typing import Union

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
class Average(TSTransform):
    """Computes a running average (mean or median) over the previous and subsequent
    "avg_overlap_samples". The output will be at the same sample rate as the input.

    Args:
        avg_overlap_samples:
            tuple[int, int], how many previous and subsequent samples over which to
            take the running mean or median
        default_value:
            float | complex, the default value, used to initialize the average array and
            to fill gaps, unless default_to_avg is set to True
        method:
            str, which method to use, either 'mean' or 'median'
        default_to_avg:
            bool, whether to fill gaps with the default value or the current average
        initialize_array:
            bool, whether to fill the average array with the default value at startup
        reject_zeros:
            bool, whether or not to replace zeros with either the default value or
            current average
    """

    avg_overlap_samples: tuple[int, int] = (2048, 0)
    default_value: Union[float, complex] = 0.0
    method: str = "mean"
    default_to_avg: bool = True
    initialize_array: bool = True
    reject_zeros: bool = True

    def __post_init__(self):
        super().__post_init__()
        # This element is written to assume one channel, one source pad and one sink pad
        assert (
            len(self.source_pads) == len(self.sink_pads) == 1
        ), "Average transform requires exactly one source pad and one sink pad"

        # We only know these two methods
        assert self.method in [
            "mean",
            "median",
        ], f"Method must be 'mean' or 'median', got '{self.method}'"

        # Initialize the arrays
        assert (
            self.avg_array_len > 0
        ), f"avg_array_len must be positive, got {self.avg_array_len}"
        self.current_avg = self.default_value
        self.avg_array = np.tile(complex(self.default_value), self.avg_array_len)
        if self.initialize_array:
            self.valid_samples = self.avg_array_len
        else:
            self.valid_samples = 0

        # We won't know these until the first data arrives
        self.avg_array_index = 0
        self.array_index_set = False
        self.real = None

    @property
    def avg_array_len(self):
        """Get the length of the running average array"""
        return 1 + sum(self.avg_overlap_samples)

    def update_mean(self, new_sample, sample_is_valid):
        # Update the number of samples in the array
        if sample_is_valid and self.valid_samples < self.avg_array_len:
            self.valid_samples += 1
        # Update our location in the arrays
        self.avg_array_index = (self.avg_array_index + 1) % self.avg_array_len

        self.avg_array[self.avg_array_index] = new_sample

        if self.valid_samples < self.avg_array_len:
            # Then we are taking the mean of a subset of this array
            avg_array_subset = np.roll(
                self.avg_array, self.avg_array_len - self.avg_array_index - 1
            )[-self.valid_samples :]
            self.current_avg = np.mean(avg_array_subset)
        else:
            self.current_avg = np.mean(self.avg_array)

    def update_median(self, new_sample, sample_is_valid):
        # Update the number of samples in the array
        if sample_is_valid and self.valid_samples < self.avg_array_len:
            self.valid_samples += 1
        # Update our location in the array
        self.avg_array_index = (self.avg_array_index + 1) % self.avg_array_len

        self.avg_array[self.avg_array_index] = new_sample
        if self.valid_samples < self.avg_array_len:
            # Then we are taking the median of a subset of this array
            avg_array_subset = np.roll(
                self.avg_array, self.avg_array_len - self.avg_array_index - 1
            )[-self.valid_samples :]
            if self.valid_samples % 2:
                if self.real:
                    self.current_avg = get_new_median(
                        avg_array_subset, self.current_avg
                    )
                else:
                    self.current_avg = get_new_median(
                        avg_array_subset.real, self.current_avg.real
                    ) + 1j * get_new_median(
                        avg_array_subset.imag, self.current_avg.imag
                    )
            else:
                if self.real:
                    self.current_avg = np.median(avg_array_subset)
                else:
                    self.current_avg = np.median(
                        avg_array_subset.real
                    ) + 1j * np.median(avg_array_subset.imag)
        else:
            if self.valid_samples % 2:
                if self.real:
                    self.current_avg = get_new_median(self.avg_array, self.current_avg)
                else:
                    self.current_avg = get_new_median(
                        self.avg_array.real, self.current_avg.real
                    ) + 1j * get_new_median(self.avg_array.imag, self.current_avg.imag)
            else:
                if self.real:
                    self.current_avg = np.median(self.avg_array)
                else:
                    self.current_avg = np.median(self.avg_array.real) + 1j * np.median(
                        self.avg_array.imag
                    )

    def internal(self):
        super().internal()
        frame = self.preparedframes[self.sink_pads[0]]
        self.outbufs = []
        for inbuf in frame:
            if inbuf.is_gap:
                # We need to fill in gaps with either the most recent average or the
                # default value
                samples_to_fill = Offset.tosamples(
                    inbuf.end_offset - inbuf.offset, inbuf.sample_rate
                )
                assert (
                    samples_to_fill >= 0
                ), f"samples_to_fill cannot be negative, got {samples_to_fill}"
                if samples_to_fill == 0:
                    outdata = None
                else:
                    if self.real:
                        outdata = np.empty(samples_to_fill, dtype=np.float64)
                    else:
                        outdata = np.empty(samples_to_fill, dtype=np.complex128)
                    for idx in range(samples_to_fill):
                        if self.default_to_avg:
                            new_sample = self.current_avg
                        else:
                            new_sample = self.default_value
                        # Update the current average
                        if self.method == "mean":
                            self.update_mean(new_sample, False)
                        else:
                            self.update_median(new_sample, False)
                        outdata[idx] = self.current_avg
            else:
                if self.real is None:
                    self.real = not isinstance(
                        inbuf.data[0], (complex, np.complex128, np.clongdouble)
                    )
                    # Now, check whether the average array is the right type
                    if self.real and isinstance(
                        self.avg_array[0], (complex, np.complex128, np.clongdouble)
                    ):
                        # FIXME: Should this warning be something other than a print
                        # statement?
                        msg = (
                            "WARNING: Average: data is real; discarding imaginary "
                            "part of default value and average array"
                        )
                        print(msg)
                        self.default_value = self.default_value.real
                        self.current_avg = self.current_avg.real
                        self.avg_array = self.avg_array.real
                    elif not self.real and not isinstance(
                        self.avg_array[0], (complex, np.complex128, np.clongdouble)
                    ):
                        self.avg_array = self.avg_array.astype(complex)
                if self.real:
                    outdata = np.empty(len(inbuf.data), dtype=np.float64)
                else:
                    outdata = np.empty(len(inbuf.data), dtype=np.complex128)
                if self.array_index_set:
                    # Check that the array index is still aligned with the offset
                    expected_index = (
                        Offset.tosamples(inbuf.offset, inbuf.sample_rate)
                        % self.avg_array_len
                    )
                    assert self.avg_array_index == expected_index, (
                        f"Array index {self.avg_array_index} misaligned with "
                        f"offset-derived index {expected_index}"
                    )
                else:
                    # Make sure the order of the arrays is independent of start time.
                    self.avg_array_index = (
                        Offset.tosamples(inbuf.offset, inbuf.sample_rate)
                        % self.avg_array_len
                    )
                    self.array_index_set = True

                for idx in range(len(inbuf.data)):
                    new_sample = inbuf.data[idx]
                    if (
                        np.isinf(new_sample)
                        or np.isnan(new_sample)
                        or (new_sample == 0 and self.reject_zeros)
                    ):
                        if self.default_to_avg:
                            new_sample = self.current_avg
                        else:
                            new_sample = self.default_value
                        # Update the current average
                        if self.method == "mean":
                            self.update_mean(new_sample, False)
                        else:
                            self.update_median(new_sample, False)
                    else:
                        if self.method == "mean":
                            self.update_mean(new_sample, True)
                        else:
                            self.update_median(new_sample, True)
                    outdata[idx] = self.current_avg

            outbuf = SeriesBuffer(
                offset=inbuf.offset
                - Offset.fromsamples(self.avg_overlap_samples[1], inbuf.sample_rate),
                sample_rate=inbuf.sample_rate,
                data=outdata,
                shape=inbuf.shape,
            )
            self.outbufs.append(outbuf)
        self.eos = frame.EOS
        self.metadata = frame.metadata

    def new(self, pad):
        return TSFrame(buffers=self.outbufs, EOS=self.eos, metadata=self.metadata)
