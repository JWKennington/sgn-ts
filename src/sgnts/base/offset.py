from __future__ import annotations

from enum import Enum
from typing import Optional, Union

import numpy

from sgnts.base.time import Time


class TimeUnits(str, Enum):
    """Enumeration of available time units for TSSlices."""

    OFFSETS = "offsets"
    SECONDS = "seconds"
    NANOSECONDS = "nanoseconds"
    SAMPLES = "samples"


class Offset:
    """A class for bookkeeping of sample points in the SGN-TS package.

    MAX_RATE:
      the maximum sample rate the pipeline will use. Should be a power of 2.

    ALLOWED_RATES:
      will vary from 1 to MAX_RATE by powers of 2.

    SAMPLE_STRIDE_AT_MAX_RATE:
      is the average stride that src pads should acheive per Frame in order to ensure
      that the pipeline src elements are roughly synchronous. Otherwise queues blow up
      and the pipelines get behind until they crash.

    offset_ref_t0:
      reference time to count offsets, in nanoseconds

    offset:
      Offsets count the number of samples at the MAX_RATE since the reference time
      offset_ref_t0, and are used for bookkeeping. Since offsets exactly track samples
      at the MAX_RATE, any data in ALLOWED_RATES will have sample points that lie
      exactly on an offset point. We then use offsets to synchronize between data at
      different sample rates, and to convert number of samples between different sample
      rate ratios. An offset can also be viewed as a time unit that equals 1/MAX_RATE
      seconds.

      Example:
      --------
      Suppose a pipeline has data of sample rates 16 Hz, 8 Hz, 4 Hz. If we set
      MAX_RATE = 16, offsets will track the sample points at 16 Hz. The other two data
      streams will also have sample points lying exactly on offset points, but with a
      fixed gap step.


      offsets          |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |
                       0   1   2   3   4   5   6   7   8   9   10  11  12  13  14  15

      sample rate 16   x   x   x   x   x   x   x   x   x   x   x   x   x   x   x   x
      sample rate 8    x       x       x       x       x       x       x       x
      sample rate 4    x               x               x               x

    Assumptions:
      all the sample rates in the buffers are powers of 2.

    Using offsets as a clock that is a power of 2 gives better resolution between
    sample points than using seconds/nanoseconds, because it exactly tracks samples.

    Example: If the MAX_RATE = 16384, for a buffer of data at a sample rate of 2048,
    the time difference between two nearby samples can be represented as integer offsets
    16384/2048 = 8 offsets. However, if we want to use integer nanoseconds, the time
    difference will be 1/2048 * 1e9 = 488281.25 nanoseconds, which cannot be represented
    by an integer.

    The MAX_RATE can be changed to a number that is a power of 2 and larger than the
    highest sample rate of the pipeline. As long as all sample points lie exactly on
    offset points, the bookkeeping will still work.
    """

    offset_ref_t0 = 0
    MAX_RATE = 16384
    ALLOWED_RATES = set(2**x for x in range(1 + int(numpy.log2(MAX_RATE))))
    SAMPLE_STRIDE_AT_MAX_RATE = 16384

    @classmethod
    def set_max_rate(cls, max_rate):
        cls.MAX_RATE = max_rate
        cls.ALLOWED_RATES = set(2**x for x in range(1 + int(numpy.log2(cls.MAX_RATE))))

    @classmethod
    def convert(
        cls,
        value: Union[int, float],
        from_unit: TimeUnits,
        to_unit: TimeUnits,
        from_sample_rate: Optional[int] = None,
        to_sample_rate: Optional[int] = None,
    ) -> Union[int, float]:
        """Convert a value from one time unit to another.

        Args:
            value: The value to convert.
            from_unit: The unit of the input value.
            to_unit: The unit of the output value.
            from_sample_rate: Required if from_unit is SAMPLES. Optional if
                              from_unit is NANOSECONDS (used for alignment).
            to_sample_rate: Required if to_unit is SAMPLES.

        Returns:
            The converted value in to_unit.
        """
        # 1. Optimize identity conversion
        if from_unit == to_unit:
            # Handle SAMPLES -> SAMPLES resampling
            if from_unit == TimeUnits.SAMPLES:
                if from_sample_rate == to_sample_rate:
                    return value
                # If rates differ, fall through to full conversion logic
            else:
                return value

        # 2. Convert Source -> OFFSETS (Base Unit)
        if from_unit == TimeUnits.SECONDS:
            offset_val = cls.fromsec(value)
        else:
            assert isinstance(value, int), (
                "Value must be an integer when " "converting from non-seconds units"
            )
            if from_unit == TimeUnits.OFFSETS:
                offset_val = value
            elif from_unit == TimeUnits.NANOSECONDS:
                # fromns accepts a sample_rate for alignment purposes
                offset_val = cls.fromns(value, sample_rate=from_sample_rate)
            elif from_unit == TimeUnits.SAMPLES:
                if from_sample_rate is None:
                    raise ValueError(
                        "from_sample_rate required when converting from SAMPLES"
                    )
                offset_val = cls.fromsamples(value, from_sample_rate)
            else:
                raise ValueError(f"Unknown from_unit: {from_unit}")

        # 3. Convert OFFSETS -> Target
        if to_unit == TimeUnits.OFFSETS:
            return offset_val
        elif to_unit == TimeUnits.SECONDS:
            return cls.tosec(offset_val)
        elif to_unit == TimeUnits.NANOSECONDS:
            return cls.tons(offset_val)
        elif to_unit == TimeUnits.SAMPLES:
            if to_sample_rate is None:
                raise ValueError("to_sample_rate required when converting to SAMPLES")
            return cls.tosamples(offset_val, to_sample_rate)
        else:
            raise ValueError(f"Unknown to_unit: {to_unit}")

    @staticmethod
    def sample_stride(rate: int) -> int:
        """Given Offset.SAMPLE_STRIDE_AT_MAX_RATE, derive the sample stride at the
        requested sample rate.

        Args:
            rate:
                int, the sample rate to calculate the sample stride

        Returns:
           int, the number of samples in the stride at the requested sample rate
        """
        return Offset.tosamples(Offset.SAMPLE_STRIDE_AT_MAX_RATE, rate)

    @staticmethod
    def tosec(offset: int) -> float:
        """Convert offsets to seconds.

        Args:
            offset:
                int, the offset to convert to seconds

        Returns:
            float, the time corresponding to the offset, in seconds
        """
        return offset / Offset.MAX_RATE

    @staticmethod
    def tons(offset: int) -> int:
        """Convert offsets to integer nanoseconds.

        Args:
            offset:
                int, the offset to convert to nanoseconds

        Returns:
            int, the time corresponding to the offset, in nanoseconds

        NOTE - for very large offsets, this switches to integer arithmetic to
        preserve precision. A downside is that the result will be truncated
        rather than rounded, leading to a slight bias at the 1 ns scale in
        some case. This is unlikely to cause any actual problems.  As a
        reminder, all serious bookkeeping should be done with offsets not
        timestamps.
        """
        if offset * Offset.MAX_RATE > 1e17:
            return offset * Time.SECONDS // Offset.MAX_RATE
        else:
            return round(offset / Offset.MAX_RATE * Time.SECONDS)

    @staticmethod
    def fromsec(seconds: float) -> int:
        """Convert seconds to offsets.

        Args:
            seconds:
                float, the time to convert to offsets, in seconds

        Returns:
            int, the offset corresponding to the time
        """
        return round(seconds * Offset.MAX_RATE)

    @staticmethod
    def fromns(nanoseconds: int, sample_rate: Optional[int] = None) -> int:
        """Convert nanoseconds to offsets.

        Args:
            nanoseconds:
                int, the time to convert to offsets, in nanoseconds
            sample_rate:
                int, optional sample rate to align the offset to. If provided,
                the offset will be rounded to the nearest sample boundary for
                this rate.

        Returns:
            int, the offset corresponding to the time
        """
        if sample_rate is None:
            # Standard behavior - convert directly
            return round(int(nanoseconds) / int(Time.SECONDS) * Offset.MAX_RATE)
        else:
            # Align to sample boundary
            assert (
                sample_rate in Offset.ALLOWED_RATES
            ), f"Invalid sample rate: {sample_rate}"

            # Convert nanoseconds to samples at the given rate
            samples = round(nanoseconds * sample_rate / Time.SECONDS)

            # Convert samples back to offset - this ensures alignment
            return Offset.fromsamples(int(samples), sample_rate)

    @staticmethod
    def tosamples(offset: int, sample_rate: int) -> int:
        """Convert offsets to number of sample points.

        Args:
            offset:
                int, the offset to convert to number of samples. The offset must map to
                integer number of sample points.
            sample_rate:
                int, the sample rate at which to calculate the number of samples

        Returns:
            int, the number of samples corresponding to the offset at the given sample
            rate
        """
        assert (
            sample_rate in Offset.ALLOWED_RATES
        ), f"Sample rate {sample_rate} not in ALLOWED_RATES: {Offset.ALLOWED_RATES}"
        assert not offset % (Offset.MAX_RATE // sample_rate), (
            "Offset does not map to"
            f" integer sample points. Offset: {offset}, sample rate: {sample_rate}"
        )
        return offset // (Offset.MAX_RATE // sample_rate)

    @staticmethod
    def fromsamples(samples: int, sample_rate: int) -> int:
        """Convert number of sample points to offsets.

        Args:
            samples:
                int, the number of samples to convert to offsets
            sample_rate:
                int, the sample rate at which to calculate the offset

        Returns:
            int, the offset corresponding to the number of sample points at the given
            sample rate
        """
        assert (
            sample_rate in Offset.ALLOWED_RATES
        ), f"Sample rate {sample_rate} not in ALLOWED_RATES: {Offset.ALLOWED_RATES}"
        return samples * Offset.MAX_RATE // sample_rate
