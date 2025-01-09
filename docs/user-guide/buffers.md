# SeriesBuffer

The most important new class in `sgnts` is the [TSFrame][sgnts.base.buffer.TSFrame]
which holds a list of [SeriesBuffer][sgnts.base.buffer.SeriesBuffer] objects.

## Introduction

The below example is a good starting point for understanding the key concepts
of `sgnts` buffers. There is plenty to unpack here, so lets go step by step.

   ```python
   import numpy
   
   numpy.random.seed(1)
   from sgnts.base.buffer import SeriesBuffer
   
   # Initialize a buffer sampled at 2048 Hz with one second of random numbers starting at offset 0
   buf = SeriesBuffer(offset=0, sample_rate=2048, data=numpy.random.randn(2048))
   
   # If we print the initialized buffer, it displays most of the important
   # propterties of the buffer e.g.,
   #
   #   - offset: 0 as specified in initialization
   #   - offset_end: derived from the data shape, sample rate and the max sample
   #     rate supported by the application (which is 16384 by default), thus, this
   #     one second of data has an offset equal to one second of samples at the max
   #     rate.
   #   - shape: derived from the input data
   #   - sample_rate: 2048 as specified in initialization
   #   - duration: 1000000000 the inferred duration of the buffer in nanoseconds (this could have rounding error!)
   #   - data: a compact form of the input data array
   repr_buf = "SeriesBuffer(offset=0, offset_end=16384, shape=(2048,), sample_rate=2048, duration=1000000000, data=[1.62434536 ... 1.20809946])"
   
   # Verify that you get the expected output - These docs are actually tested,
   # please forgive the pedantry
   assert repr(buf) == repr_buf
   ```


Offsets
:  The term `offset` is globally meaningful throughout the application and acts as a
   precise surrogate for time, i.e., an absolute "time" reference for any element
   within an sgnts application that should not suffer from any rounding error.
   Technically offsets are defined as a cumulative number of samples passed
   defined at the maximum sample rate allowed by the application.  This will be
   explained more below.

Sample Rate
:  `sample_rate` is the number of samples per second that a stretch of data
   contains. It is used to convert to actual time with nanosecond precision. In
   order to make certain gaurantees about precision in sgnts, we currently only
   support power of 2 sample rates from 1 Hz to a maximum which defaults to 16384
   Hz.  



Data
:  `data` is generally a numpy array that can be interpreted as (possibly
   multidimensional) time series data. 

## Allowed sample rates

The max sample rate and allowed rates are defined [here](https://git.ligo.org/greg/sgn-ts/-/blob/main/src/sgnts/base/offset.py?ref_type=heads#L63).

## Why offsets instead of time?

We assume that times will have finite precision and be represented as integer
nanoseconds (this allows us to capture gps times, for example).  Many buffers
cannot be represented by integer nanosecond durations, e.g., 

   ```python
   import numpy
   
   from sgnts.base.buffer import SeriesBuffer
   
   # Initialize a buffer sampled at 2048 Hz with one __sample__ (the value 1.) starting at offset 0
   buf = SeriesBuffer(offset=0, sample_rate=2048, data=numpy.array([1.0]))
   
   # If you print the buffer, you will see that it displays a duration of 488281.
   # But this is not precise! In fact the duration is 1 / 2048 = 488281.25, which
   # cannot be represented as integer nanoseconds. This is why we use integer
   # offsets for bookeeping.
   repr_buf = "SeriesBuffer(offset=0, offset_end=8, shape=(1,), sample_rate=2048, duration=488281, data=[1.])"
   
   # Verify that you get the expected output - These docs are actually tested,
   # please forgive the pedantry
   assert repr(buf) == repr_buf
   ```

## More on the relationship between offsets, samples and time

Offsets are the primary time book-keeping mechanism defined as the hypothetical number of samples since a reference time (default = 0) at the maximum supported sample rate.  It is common to encounter and need samples and timestamps.  Below are additional details about these concepts.

- time is represented as integer nanoseconds
- offsets are the number of samples at the (possibly hypothetical) highest
  sample rate. These are global properties within an instance of an
application. Think of them as a precise clock.
- samples are the number of samples at the current SeriesBuffer sample rate.
  These are almost always local quantities used for indexing within a buffer

Revisiting the above

   ```python
   import numpy
   
   numpy.random.seed(1)
   from sgnts.base.buffer import SeriesBuffer
   
   assert (
       repr(SeriesBuffer(offset=0, sample_rate=2048, data=numpy.random.randn(2048)))
       == "SeriesBuffer(offset=0, offset_end=16384, shape=(2048,), sample_rate=2048, duration=1000000000, data=[1.62434536 ... 1.20809946])"
   )
   ```

we see the following.  The user specified data as a 2048 sample long set of increasing numbers.  Since the sample_rate is also 2048
seconds, this is interpreted as 1 second of time series data. When printing the
buffer you can see `duration=1000000000` which is equal to 1e9 nanoseconds
(time is stored as integer nanoseconds).  You can see `offset_end=16384` which
indicates the number of samples that would be in this data if it where at the
maximum sample rate.  That is what an offset defines -- a sample count assuming
max sample rate.  It is critical for accurate internal bookkeeping.  You also
see `shape=(2048,)` which indicates single channel time series.  Try the
following for an example of multichannel audio:

   ```python
   import numpy
   
   numpy.random.seed(1)
   from sgnts.base.buffer import SeriesBuffer
   
   assert (
       repr(SeriesBuffer(offset=0, sample_rate=2048, data=numpy.random.randn(2, 2048)))
       == "SeriesBuffer(offset=0, offset_end=16384, shape=(2, 2048), sample_rate=2048, duration=1000000000, data=[[ 1.62434536 ...  1.20809946]\n [-1.82921963 ...  0.79494725]])"
   )
   ```

Note what happens to the offset if you change the sample rate (and in this case
also the data size)

   ```python
   import numpy
   
   numpy.random.seed(1)
   from sgnts.base.buffer import SeriesBuffer
   
   assert (
       repr(SeriesBuffer(offset=0, sample_rate=1024, data=numpy.random.randn(2, 2048)))
       == "SeriesBuffer(offset=0, offset_end=32768, shape=(2, 2048), sample_rate=1024, duration=2000000000, data=[[ 1.62434536 ...  1.20809946]\n [-1.82921963 ...  0.79494725]])"
   )
   ```

**The offset stays the same.** Remember that the offset is the sample count at the
theoretical maximum sample rate which is defined in offset.py.  

Only power of two sample rates are allowed at present to ensure that bookeeping
remains simple and accurate. 

   ```python
   import numpy
   from sgnts.base.buffer import SeriesBuffer
   
   # Try initializing a buffer with a non power-of-two sample rate, which is not allowed
   error = None
   try:
       buf = SeriesBuffer(offset=0, sample_rate=1000, data=numpy.random.randn(2, 1000))
   except ValueError as e:
       error = e
       pass
   
   # Verify that you get a helpful error message saying that the sample rate is not among the allowed rates
   assert (
       repr(error)
       == "ValueError('1000 not in allowed rates {32, 1, 2, 64, 4, 128, 256, 512, 8, 1024, 2048, 4096, 8192, 16, 16384}')"
   )
   ```

## Changing the global maximum sample rate

It is possible to increase the maximum sample rate globally in an application (though it must still be a power of 2) by modifying the [Offset][sgnts.base.offset.Offset] class

   ```python
   import numpy
   
   numpy.random.seed(1)
   from sgnts.base.buffer import SeriesBuffer
   from sgnts.base.offset import Offset
   
   # Increase the maximum sample rate to 262144 - NOTE: This is an application
   # level change affecting everything.
   Offset.set_max_rate(262144)
   
   # Initialize a buffer sampled at 32768 Hz with one second of random numbers
   # starting at offset 0
   buf = SeriesBuffer(offset=0, sample_rate=32768, data=numpy.random.randn(32768))
   
   # If we print the initialized buffer, it displays an offset equal to the max
   # rate that we specified since this is a one second buffer
   repr_buf = "SeriesBuffer(offset=0, offset_end=262144, shape=(32768,), sample_rate=32768, duration=1000000000, data=[1.62434536 ... 0.33230468])"
   
   
   # Verify that you get the expected output - These docs are actually tested,
   # please forgive the pedantry
   assert repr(buf) == repr_buf
   Offset.set_max_rate(16384)
   ```

## Advanced SeriesBuffer techniques

### Instantiating a SeriesBuffer

**From __init__()**:
:  A SeriesBuffer requires an offset and a sample rate.  
   Additionally it must have either data defined or provide a shape for the data.
   Below are some different instantiations
   ```python
   from sgnts.base.buffer import SeriesBuffer
   import numpy
   
   # A Gap buffer (data is None - NOTE we had to specify a shape though)
   assert (
       repr(SeriesBuffer(offset=0, sample_rate=128, shape=(64,), data=None))
       == "SeriesBuffer(offset=0, offset_end=8192, shape=(64,), sample_rate=128, duration=500000000, data=None)"
   )
   
   # A buffer of zeros
   assert (
       repr(SeriesBuffer(offset=0, sample_rate=128, shape=(64,), data=0))
       == "SeriesBuffer(offset=0, offset_end=8192, shape=(64,), sample_rate=128, duration=500000000, data=[0. ... 0.])"
   )
   
   # A buffer of ones
   assert (
       repr(SeriesBuffer(offset=0, sample_rate=128, shape=(64,), data=1))
       == "SeriesBuffer(offset=0, offset_end=8192, shape=(64,), sample_rate=128, duration=500000000, data=[1. ... 1.])"
   )
   
   # A buffer of existing data - NOTE do not give shape when data is provided
   assert (
       repr(
           SeriesBuffer(offset=0, sample_rate=128, data=numpy.arange(64, dtype="float64"))
       )
       == "SeriesBuffer(offset=0, offset_end=8192, shape=(64,), sample_rate=128, duration=500000000, data=[ 0. ... 63.])"
   )
   ```

**From offset slice**:
:  There is a shortcut for creating a one dimensional buffer directly from a
   TSSlice object representing offsets and a sample rate
   ```python
   from sgnts.base.buffer import SeriesBuffer
   from sgnts.base.slice_tools import TSSlice
   
   assert (
       repr(SeriesBuffer.fromoffsetslice(TSSlice(0, 8192), sample_rate=128))
       == "SeriesBuffer(offset=0, offset_end=8192, shape=(64,), sample_rate=128, duration=500000000, data=None)"
   )
   ```

**From an existing buffer**:
:  There is a shortcut to make a new similar (but empty) buffer from an existing one
   ```python
   from sgnts.base.buffer import SeriesBuffer
   
   buf = SeriesBuffer(offset=0, sample_rate=128, shape=(64,), data=1)
   assert (
       repr(buf)
       == "SeriesBuffer(offset=0, offset_end=8192, shape=(64,), sample_rate=128, duration=500000000, data=[1. ... 1.])"
   )
   assert (
       repr(buf.new())
       == "SeriesBuffer(offset=0, offset_end=8192, shape=(64,), sample_rate=128, duration=500000000, data=None)"
   )
   ```

### Buffer operators

#### Truth value of a buffer
:  A buffer is logically True only if its data is not None
   ```python
   from sgnts.base.buffer import SeriesBuffer
   
   assert bool(SeriesBuffer(offset=0, sample_rate=128, shape=(64,), data=1))
   assert not bool(SeriesBuffer(offset=0, sample_rate=128, shape=(64,), data=None))
   ```

#### Length of a buffer
:  A buffer has a length equal to the length of its data unless its data is
   None in which case the length is 0, but a buffer also has a shape property
   which must provide a valid size for data even if the data doesn't exist.
   ```python
   from sgnts.base.buffer import SeriesBuffer
   
   assert len(SeriesBuffer(offset=0, sample_rate=128, shape=(64,), data=1)) == 64
   assert len(SeriesBuffer(offset=0, sample_rate=128, shape=(64,), data=None)) == 0
   ```

#### Equality
:  Two buffers are equal if their offsets, shape, sample rate, and data are equal. 
   ```python
   from sgnts.base.buffer import SeriesBuffer
   
   assert SeriesBuffer(offset=0, sample_rate=128, shape=(32,), data=None) == SeriesBuffer(
       offset=0, sample_rate=128, shape=(32,), data=None
   )
   
   assert not SeriesBuffer(
       offset=0, sample_rate=128, shape=(32,), data=None
   ) == SeriesBuffer(offset=0, sample_rate=128, shape=(32,), data=1)
   
   assert not SeriesBuffer(
       offset=0, sample_rate=128, shape=(32,), data=None
   ) == SeriesBuffer(offset=0, sample_rate=128, shape=(33,), data=None)
   
   assert not SeriesBuffer(
       offset=0, sample_rate=128, shape=(32,), data=None
   ) == SeriesBuffer(offset=0, sample_rate=256, shape=(32,), data=None)
   
   assert not SeriesBuffer(
       offset=0, sample_rate=128, shape=(32,), data=None
   ) == SeriesBuffer(offset=16, sample_rate=128, shape=(32,), data=None)
   ```

#### Contains
:  An integer is considered to be "in" a buffer if it is a valid offset for the
   buffer e.g., `buf.offset <= item < buf.end_offset`. A SeriesBuffer is
   considered to be in a buffer if its offset span is within the buffer.
   ```python
   from sgnts.base.buffer import SeriesBuffer
   
   buf1 = SeriesBuffer(offset=0, sample_rate=128, shape=(32,), data=None)
   buf2 = SeriesBuffer(offset=0, sample_rate=128, shape=(64,), data=None)
   assert 1 in buf1
   assert 16384 not in buf1
   assert buf1 in buf2
   assert not buf2 in buf1
   ```
#### Inequality
:  Buffer comparisons e.g., ">", "<", ">=", and "<=" are implemented based on the buffers end offets
   ```python
   from sgnts.base.buffer import SeriesBuffer
   
   # Has end offset 4096
   buf1 = SeriesBuffer(offset=0, sample_rate=128, shape=(32,), data=None)
   # Has end offset 8192
   buf2 = SeriesBuffer(offset=0, sample_rate=128, shape=(64,), data=None)
   
   assert not (buf1 > buf2)
   assert buf2 > buf1
   assert not (buf1 > buf1)
   assert buf1 >= buf1
   
   assert buf1 < buf2
   assert not (buf2 < buf1)
   assert not (buf1 < buf1)
   assert buf1 <= buf1
   ```

#### Addition
:  Series buffers can be added together and padded as necessary. If a gap buffer is added to a nongap buffer, it will be treated as zeros.
   ```python
   from sgnts.base.buffer import SeriesBuffer
   
   buf1 = SeriesBuffer(offset=0, sample_rate=128, shape=(2,), data=1)
   buf2 = SeriesBuffer(offset=0, sample_rate=128, shape=(3,), data=0)
   buf3 = SeriesBuffer(offset=0, sample_rate=128, shape=(3,), data=None)
   
   # The addition of any two distinct buffers above will result in this:
   expected = "SeriesBuffer(offset=0, offset_end=384, shape=(3,), sample_rate=128, duration=23437500, data=[1. 1. 0.])"
   
   assert repr(buf1 + buf2) == expected
   assert repr(buf2 + buf1) == expected
   assert repr(buf1 + buf3) == expected
   assert repr(buf3 + buf1) == expected
   
   # The addition of buf1 with itself will result in this:
   expected = "SeriesBuffer(offset=0, offset_end=256, shape=(2,), sample_rate=128, duration=15625000, data=[2. 2.])"
   
   assert repr(buf1 + buf1) == expected
   ```
### Buffer methods

#### pad_buffer()
:  Series buffers can be asked to produce a pad buffer on the left by providing an offset earlier than the start of the buffer
   ```python
   from sgnts.base.buffer import SeriesBuffer
   
   buf = SeriesBuffer(offset=16384, sample_rate=128, shape=(2,), data=1)
   
   # produce an empty (gap, data=None) pad buffer one sample point before buf
   pad_buffer = buf.pad_buffer(16256)
   assert (
       repr(pad_buffer)
       == "SeriesBuffer(offset=16256, offset_end=16384, shape=(1,), sample_rate=128, duration=7812500, data=None)"
   )
   
   # If you actually want to elongate the original buffer you can do this (inplace is not implemented yet)
   padded_buffer = pad_buffer + buf
   assert (
       repr(padded_buffer)
       == "SeriesBuffer(offset=16256, offset_end=16640, shape=(3,), sample_rate=128, duration=23437500, data=[0. 1. 1.])"
   )
   ```

#### sub_buffer()
:  You can extract a sub buffer from a buffer by giving a valid offset TSSlice
   ```python
   from sgnts.base.buffer import SeriesBuffer
   from sgnts.base.slice_tools import TSSlice
   
   buf = SeriesBuffer(offset=16256, sample_rate=128, shape=(3,), data=1)
   subbuf = buf.sub_buffer(TSSlice(16384, 16384 + 256))
   
   assert (
       repr(buf)
       == "SeriesBuffer(offset=16256, offset_end=16640, shape=(3,), sample_rate=128, duration=23437500, data=[1. 1. 1.])"
   )
   assert (
       repr(subbuf)
       == "SeriesBuffer(offset=16384, offset_end=16640, shape=(2,), sample_rate=128, duration=15625000, data=[1. 1.])"
   )
   ```

#### split()
:  split() effectively calls sub-buffer recursively, e.g,
   ```python
   from sgnts.base.buffer import SeriesBuffer
   from sgnts.base.slice_tools import TSSlice, TSSlices
   
   buf = SeriesBuffer(offset=0, sample_rate=128, shape=(6,), data=1)
   
   # extract two sub buffers corresponding to sample points (0 and 1) and (3 and 4)
   slices = TSSlices([TSSlice(0, 256), TSSlice(384, 640)])
   split_bufs = buf.split(slices)
   assert len(split_bufs) == 2
   assert (
       repr(split_bufs[0])
       == "SeriesBuffer(offset=0, offset_end=256, shape=(2,), sample_rate=128, duration=15625000, data=[1. 1.])"
   )
   assert (
       repr(split_bufs[1])
       == "SeriesBuffer(offset=384, offset_end=640, shape=(2,), sample_rate=128, duration=15625000, data=[1. 1.])"
   )
   ```

## More details

Additional methods and properties are documented in the [API docs](/api/base/buffer/)
