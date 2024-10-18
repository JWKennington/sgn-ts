from .time import Time
import numpy


class Offset:
    """
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

    @staticmethod
    def sample_stride(rate: int):
        return Offset.tosamples(Offset.SAMPLE_STRIDE_AT_MAX_RATE, rate)

    @staticmethod
    def tosec(offset: int) -> float:
        return offset / Offset.MAX_RATE

    @staticmethod
    def tons(offset: int) -> int:
        return round(offset / Offset.MAX_RATE * Time.SECONDS)

    @staticmethod
    def fromsec(seconds: float) -> int:
        return round(seconds * Offset.MAX_RATE)

    @staticmethod
    def fromns(nanoseconds: int) -> int:
        return round(nanoseconds / Time.SECONDS * Offset.MAX_RATE)

    @staticmethod
    def tosamples(offset: int, sample_rate: int) -> int:
        assert sample_rate in Offset.ALLOWED_RATES
        assert not offset % (Offset.MAX_RATE // sample_rate)
        return offset // (Offset.MAX_RATE // sample_rate)

    @staticmethod
    def fromsamples(samples: int, sample_rate: int) -> int:
        assert sample_rate in Offset.ALLOWED_RATES
        return samples * Offset.MAX_RATE // sample_rate
