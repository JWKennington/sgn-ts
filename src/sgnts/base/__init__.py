from sgn.base import *
from collections.abc import Sequence
from enum import Enum
from typing import Any

class Time(int, Enum):
    SECONDS = 1_000_000_000
    MILLISECONDS = 1_000_000
    MICROSECONDS = 1_000
    NANOSECONDS = 1

#
# OFFSET_RATE: the number of offsets in one second.
# The OFFSET_RATE serves as a global clock that is a power of 2.
# 
# Assumptions: (1) all the sample rates in the buffers are 
# powers of 2. (2) the OFFSET_RATE is at least as large as
# the highest sample rate, so that OFFSET_RATE/sample_rate 
# is an integer, i.e., the offset difference between two nearby
# samples is an integer.
# 
# The OFFSET_RATE is used for bookkeeping. Using a clock that is
# a power of 2 gives better resolution between sample points than
# using seconds/nanoseconds.
# 
# Example: If the OFFSET_RATE = 16384, for a buffer of data at a 
# sample rate of 2048, the time difference between two nearby 
# samples is 16384/2048 = 8 offsets. However, if we want to use 
# integer nanoseconds, the time difference will be 488281.25 nanoseconds, 
# which is not enough resolution.
#
# The OFFSET_RATE can be changed to another number, but it needs 
# to be a power of 2, and at least as large as the highest sample
# rate the buffers will carry.
# 
OFFSET_RATE = 16384 

class Offset:
    @staticmethod
    def offset2sec(offset):
        return offset / OFFSET_RATE

    @staticmethod
    def offset2ns(offset):
        return int(offset / OFFSET_RATE * Time.SECONDS)

    @staticmethod
    def sec2offset(seconds):
        return seconds * OFFSET_RATE

    @staticmethod
    def ns2offset(nanoseconds):
        return nanoseconds / Time.SECONDS * OFFSET_RATE

    @staticmethod
    def offset2nsamples(offset, sample_rate):
        return int(offset / OFFSET_RATE * sample_rate)

    @staticmethod
    def nsamples2offset(nsamples, sample_rate):
        return int(nsamples / sample_rate * OFFSET_RATE)

@dataclass
class SeriesBuffer(Buffer):
    """Timeseries buffer with associated metadata.

    Parameters
    ----------
    offset : int
        The number of offset samples (defined at sample rate OFFSET_RATE)
        since offset_ref_t0. Similar to "t0".
    noffset : int
        The number of offset samples (defined at sample rate OFFSET_RATE)
        in the buffer. Similar to "duration".
    offset_ref_t0 : int
        The reference time to start the offset counter, in nanoseconds.
    data : Sequence
        The timeseries data.

    """
    offset: int = None
    noffset: int = None
    offset_ref_t0: int = None
    data: Sequence[Any] = None

    @property
    def t0(self):
        return self.offset_ref_t0 + Offset.offset2ns(self.offset)

    @property
    def duration(self):
        return Offset.offset2ns(self.noffset)

    @property
    def end(self):
        return self.t0 + self.duration

    @property
    def size(self):
        return self.data.shape[-1]

    @property
    def sample_rate(self):
        return int(self.size / Offset.offset2sec(self.noffset))

