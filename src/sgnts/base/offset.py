from .time import Time
import numpy


class Offset:
    """
    OFFSET_RATE: the number of offsets in one second.
    The OFFSET_RATE serves as a global clock that is a power of 2.

    offset: An offset is a time unit that equals 1/OFFSET_RATE seconds.

    Assumptions: (1) all the sample rates in the buffers are
    powers of 2. (2) the OFFSET_RATE is at least as large as
    the highest sample rate, so that OFFSET_RATE/sample_rate
    is an integer, i.e., the offset difference between two nearby
    samples is an integer.

    The OFFSET_RATE is used for bookkeeping. Using a clock that is
    a power of 2 gives better resolution between sample points than
    using seconds/nanoseconds.

    Example: If the OFFSET_RATE = 16384, for a buffer of data at a
    sample rate of 2048, the time difference between two nearby
    samples is 16384/2048 = 8 offsets. However, if we want to use
    integer nanoseconds, the time difference will be 488281.25 nanoseconds,
    which cannot be represented by an integer.

    The OFFSET_RATE can be changed to another number, but it needs
    to be a power of 2, and at least as large as the highest sample
    rate the buffers will carry.

    The ALLOWED_RATES will vary from 1 to OFFSET_RATE by powers of 2.

    The TARGET_OFFSET_STRIDE is the average stride that src pads should acheive
    per Frame in order to ensure that the pipeline src elements are roughly
    synchronous. Otherwise queues blow up and the pipelines get behind until they
    crash.
    """

    offset_ref_t0 = 0 # in nanoseconds
    OFFSET_RATE = 16384
    ALLOWED_RATES = set(2**x for x in range(1 + int(numpy.log2(OFFSET_RATE))))
    TARGET_OFFSET_STRIDE = 4096

    @staticmethod
    def tosec(offset: int) -> float:
        return offset / Offset.OFFSET_RATE

    @staticmethod
    def tons(offset: int) -> int:
        return round(offset / Offset.OFFSET_RATE * Time.SECONDS)

    @staticmethod
    def fromsec(seconds: float) -> int:
        return round(seconds * Offset.OFFSET_RATE)

    @staticmethod
    def fromns(nanoseconds: int) -> int:
        return round(nanoseconds / Time.SECONDS * Offset.OFFSET_RATE)

    @staticmethod
    def tosamples(offset: int, sample_rate: int) -> int:
        assert sample_rate in Offset.ALLOWED_RATES
        assert not offset % (Offset.OFFSET_RATE // sample_rate)
        return offset // (Offset.OFFSET_RATE // sample_rate)

    @staticmethod
    def fromsamples(samples: int, sample_rate: int) -> int:
        assert sample_rate in Offset.ALLOWED_RATES
        return samples * Offset.OFFSET_RATE // sample_rate

    @staticmethod
    def stridesamples(sample_rate: int) -> int:
        return Offset.tosamples(Offset.TARGET_OFFSET_STRIDE, sample_rate)
