from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

from sgnts.base import Offset, SeriesBuffer, TSFrame, TSTransform


@dataclass
class Mean(TSTransform):
    """Computes a running mean over the previous and subsequent "mean_overlap_samples".
    The output will be at the same sample rate as the input.

    Args:
        mean_overlap_samples:
            tuple[int, int], how many previous and subsequent samples over which to
            take the running mean
        default_value:
            float | complex, the default value, used to initialize the mean array and
            to fill gaps, unless default-to-mean is set to True
        default_to_mean:
            bool, whether to fill gaps with the default value or the current mean
        reject_zeros:
            bool, whether or not to replace zeros with either the default value or
            current mean
    """

    mean_overlap_samples: tuple[int, int] = (0, 0)
    default_value: Optional[Union[float, complex]] = 0.0
    default_to_mean: bool = True
    reject_zeros: bool = True

    def __post_init__(self):
        super().__post_init__()
        # This element is written to assume one channel, one source pad and one sink pad
        assert len(self.source_pads) == len(self.sink_pads) == 1

        # Initialize the arrays
        assert self.mean_array_len > 0
        if self.default_value is not None:
            self.mean_array = np.tile(self.default_value, self.mean_array_len)
            self.samples_in_mean = self.mean_array_len
            self.current_mean = self.default_value
        else:
            self.mean_array = np.zeros(self.mean_array_len, dtype=float)
            self.samples_in_mean = 0
            self.current_mean = 0.0

        # We won't know these until the first data arrives
        self.mean_array_index = 0
        self.array_index_set = False
        self.real = None

    @property
    def mean_array_len(self):
        """Get the length of the running mean array"""
        return 1 + sum(self.mean_overlap_samples)

    def update_mean(self, new_sample):
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
            self.current_mean = np.mean(mean_array_subset)
        else:
            self.current_mean = np.mean(self.mean_array)

    def internal(self):
        super().internal()
        frame = self.preparedframes[self.sink_pads[0]]
        self.outbufs = []
        for inbuf in frame:
            if inbuf.is_gap:
                # We need to fill in gaps with either the most recent mean or the
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
                        if self.default_to_mean:
                            new_sample = self.current_mean
                        else:
                            new_sample = self.default_value
                        # Update the current mean
                        self.update_mean(new_sample)
                        outdata[idx] = self.current_mean
            else:
                if self.real is None:
                    self.real = not isinstance(
                        inbuf.data[0], (complex, np.complex128, np.clongdouble)
                    )
                    # Now, check whether the mean array is the right type
                    if self.real and isinstance(
                        self.mean_array[0], (complex, np.complex128, np.clongdouble)
                    ):
                        # FIXME: Should this warning be something other than a print
                        # statement?
                        msg = (
                            "WARNING: Mean: data is real; discarding imaginary part of "
                            "default value and mean array"
                        )
                        print(msg)
                        if self.default_value is not None:
                            self.default_value = self.default_value.real
                        self.mean_array = self.mean_array.real
                    elif not self.real and not isinstance(
                        self.mean_array[0], (complex, np.complex128, np.clongdouble)
                    ):
                        self.mean_array = self.mean_array.astype(complex)
                if self.real:
                    outdata = np.empty(len(inbuf.data), dtype=np.float64)
                else:
                    outdata = np.empty(len(inbuf.data), dtype=np.complex128)
                if self.array_index_set:
                    # Check that the array index is still aligned with the offset
                    assert (
                        self.mean_array_index
                        == Offset.tosamples(inbuf.offset, inbuf.sample_rate)
                        % self.mean_array_len
                    )
                else:
                    # Make sure the order of the arrays is independent of start time.
                    self.mean_array_index = (
                        Offset.tosamples(inbuf.offset, inbuf.sample_rate)
                        % self.mean_array_len
                    )
                    self.array_index_set = True

                for idx in range(len(inbuf.data)):
                    new_sample = inbuf.data[idx]
                    if (
                        np.isinf(new_sample)
                        or np.isnan(new_sample)
                        or (new_sample == 0 and self.reject_zeros)
                    ):
                        if self.default_to_mean:
                            new_sample = self.current_mean
                        else:
                            new_sample = self.default_value
                    # Update the current mean
                    self.update_mean(new_sample)
                    outdata[idx] = self.current_mean

            outbuf = SeriesBuffer(
                offset=inbuf.offset
                - Offset.fromsamples(self.mean_overlap_samples[1], inbuf.sample_rate),
                sample_rate=inbuf.sample_rate,
                data=outdata,
                shape=inbuf.shape,
            )
            self.outbufs.append(outbuf)
        self.eos = frame.EOS
        self.metadata = frame.metadata

    def new(self, pad):
        return TSFrame(buffers=self.outbufs, EOS=self.eos, metadata=self.metadata)
