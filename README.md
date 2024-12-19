<!-- index.rst content start -->

# SGN-TS (SGN TimeSeries)

SGN-TS is set of extensions to the core library `sgn`, that includes functionality
specific to TimeSeries analysis. This page is for documenatation of the `sgnts` package, but there is a family of
libraries that extend the functionality of SGN in other ways, including:

- [`sgn`](https://docs.ligo.org/greg/sgn/): Base library for SGN
- [`sgn-ligo`](https://git.ligo.org/greg/sgn-try): LIGO-specific utilities for SGN

## Installation

To install SGN-TS, simply run:

```bash
pip install sgn-ts
```

More SGN-TS-specific documentation coming soon.

## Developer's guide

Before reading this guide you should carefully read and understand the [SGN
developers guide](https://greg.docs.ligo.org/sgn/#developer-s-guide).   

The core motivation with SGN TS (sgnts) is to build Time Series (TS) handling
into SGN.  This is appropriate for e.g., signal processing applications.  Of
course nothing is stopping you from doing any of these things with just SGN,
but you will likely have to deal with some of the conceptual and technical
hurdles that this library solves.  That being said, there are many limitations
of sgnts and you should understand those carefully in the context of your
project.  We are open to making changes that reach a wider audience, so please
let us know your thoughts. 

**New concepts over SGN:**

- Data are now rigidly defined to be uniformly sampled time series.  There is
  an expectation that elements will deal with data in a synchronous way.
- Synchronization means that the continuity equation must be satisfied.  Data
  cannot be produced at a higher rate in one source element than another,
  otherwise synchronous operations will be impossible without data "piling up"
  somewhere.
- Time stamp bookeeping accuracy is important. The library aims to keep single
  sample point timing accuracies even for applications that are designed to run
  uninterupped for years.  This requires a bit of rigidity in bookeeping, but we
  try to hide as much as possible from the causual developer and user.

### Buffers and Frames

The most important new class in sgnts is the
[TSFrame](https://git.ligo.org/greg/sgn-ts/-/blob/main/src/sgnts/base/buffer.py?ref_type=heads#L415)
which holds a list of
[SeriesBuffers](https://git.ligo.org/greg/sgn-ts/-/blob/main/src/sgnts/base/buffer.py?ref_type=heads#L17)

Here we can get some familiarity with both of these objects and along the way,
other classes and concepts relevant for sgnts.

```python notest
>>> import numpy
>>> from sgnts.base.buffer import SeriesBuffer
>>> buf = SeriesBuffer(offset=0, sample_rate=2048, data=numpy.random.randn(2048))
>>> print (buf)
SeriesBuffer(offset=0, offset_end=16384, shape=(2048,), sample_rate=2048, duration=1000000000, data=[0.56649291 ... 1.39569688])
```

There is plenty to unpack here, so lets go step by step.  

**offset:**

`offset` is globally meaningful throughout the application and acts as a
precise surrogate for time, i.e., an absolute "time" reference for any element
within an sgnts application that should not suffer from any rounding error.
Technically offsets are defined as a cumulative number of samples passed
defined at the maximum sample rate allowed by the application.  This will be
explained more below.

**sample_rate:**

`sample_rate` is the number of samples per second that a stretch of data
contains. It is used to convert to actual time with nanosecond precision. In
order to make certain gaurantees about precision in sgnts, we currently only
support power of 2 sample rates from 1 Hz to a maximum which defaults to 16384
Hz.  The max sample rate and allowed rates are defined
[here](https://git.ligo.org/greg/sgn-ts/-/blob/main/src/sgnts/base/offset.py?ref_type=heads#L63).

**data:**

`data` is generally a numpy array that can be interpreted as (possibly
multidimensional) time series data. 


Now revisiting the above

```python notest
>>> buf = SeriesBuffer(offset=0, sample_rate=2048, data=numpy.random.randn(2048))
>>> print (buf)
SeriesBuffer(offset=0, offset_end=16384, shape=(2048,), sample_rate=2048, duration=1000000000, data=[0.56649291 ... 1.39569688])
```

we see the following.  The user specified data as a 2048 sample long set of
random gaussian distributed numbers.  Since the sample_rate is also 2048
seconds, this is interpreted as 1 second of time series data. When printing the
buffer you can see `duration=1000000000` which is equal to 1e9 nanoseconds
(time is stored as integer nanoseconds).  You can see `offset_end=16384` which
indicates the number of samples that would be in this data if it where at the
maximum sample rate.  That is what an offset defines -- a sample count assuming
max sample rate.  It is critical for accurate internal bookkeeping.  You also
see `shape=(2048,)` which indicates single channel time series.  Try the
following for an example of multichannel audio:

```python notest
>>> buf = SeriesBuffer(offset=0, sample_rate=2048, data=numpy.random.randn(2,2048))
>>> print (buf)
SeriesBuffer(offset=0, offset_end=16384, shape=(2, 2048), sample_rate=2048, duration=1000000000, data=[[ 0.01684876 ... -1.6963346 ]
 [-0.55875476 ...  0.58967178]])
```

Note what happens to the offset if you change the sample rate (and in this case
also the data size)

```python notest
>>> buf = SeriesBuffer(offset=0, sample_rate=1024, data=numpy.random.randn(2,1024))
>>> buf
SeriesBuffer(offset=0, offset_end=16384, shape=(2, 1024), sample_rate=1024, duration=1000000000, data=[[-0.13116052 ...  1.2223811 ]
 [-0.98786954 ... -0.56760618]])
```

**It stays the same.** Remember that the offset is the sample count at the
theoretical maximum sample rate which is defined in offset.py.  

Only power of two sample rates are allowed at present to ensure that bookeeping
remains simple and accurate. 

```python notest
>>> buf = SeriesBuffer(offset=0, sample_rate=1000, data=numpy.random.randn(2,1000))
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "<string>", line 7, in __init__
  File "/Users/crh184/Library/Python/3.9/lib/python/site-packages/sgnts/base/buffer.py", line 38, in __post_init__
    raise ValueError("%s not in allowed rates %s" % (self.sample_rate, Offset.ALLOWED_RATES))
ValueError: 1000 not in allowed rates {32, 1, 2, 64, 4, 128, 256, 512, 8, 1024, 2048, 4096, 8192, 16, 16384}
```

It is possible to increase the maximum sample rate globally in an application

```python notest
>>> import numpy
>>> from sgnts.base.buffer import SeriesBuffer
>>> from sgnts.base.offset import Offset
>>> Offset.set_max_rate(262144)
>>> buf = SeriesBuffer(offset=0, sample_rate=32768, data=numpy.random.randn(32768))
>>> print (buf)
SeriesBuffer(offset=0, offset_end=262144, shape=(32768,), sample_rate=32768, duration=1000000000, data=[-0.08916502 ...  0.89236118])
```

Buffers are not the primary data type passed around between element in sgnts.  Rather, it is a `TSFrame`.  TSFrames hold lists of buffers

```
>>> import numpy
>>> from sgnts.base.buffer import SeriesBuffer, TSFrame
>>> 
>>> # An example of just one buffer
>>> buf1 = SeriesBuffer(offset=0, sample_rate=2048, data=numpy.random.randn(2048))
>>> frame = TSFrame(buffers=[buf1])
>>> print (frame)

	SeriesBuffer(offset=0, offset_end=16384, shape=(2048,), sample_rate=2048, duration=1000000000, data=[-0.04094335 ... -1.49758223])
>>> 
>>> # An example of two contiguous buffers
>>> buf1 = SeriesBuffer(offset=0, sample_rate=2048, data=numpy.random.randn(2048))
>>> buf2 = SeriesBuffer(offset=16384, sample_rate=2048, data=numpy.random.randn(2048))
>>> frame = TSFrame(buffers=[buf1, buf2])
>>> print (frame)

	SeriesBuffer(offset=0, offset_end=16384, shape=(2048,), sample_rate=2048, duration=1000000000, data=[-1.56771352 ... -0.20928693])
	SeriesBuffer(offset=16384, offset_end=32768, shape=(2048,), sample_rate=2048, duration=1000000000, data=[-1.00442217 ... -0.75684022])
>>> 
>>> # An example of two non contiguous buffers. NOTE THIS SHOULDN'T WORK!!
>>> buf1 = SeriesBuffer(offset=0, sample_rate=2048, data=numpy.random.randn(2048))
>>> buf2 = SeriesBuffer(offset=12345, sample_rate=2048, data=numpy.random.randn(2048))
>>> frame = TSFrame(buffers=[buf1, buf2])
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "<string>", line 8, in __init__
  File "/Users/crh184/Library/Python/3.9/lib/python/site-packages/sgnts/base/buffer.py", line 455, in __post_init__
    self.__sanity_check(self.buffers)
  File "/Users/crh184/Library/Python/3.9/lib/python/site-packages/sgnts/base/buffer.py", line 485, in __sanity_check
    assert off0 == sl.start
AssertionError
```

Note in the above that TSFrames only support contiguous buffers

TSFrames offer some additional methods to describe their contents, e.g.,

```python notest
>>> buf1 = SeriesBuffer(offset=0, sample_rate=2048, data=numpy.random.randn(2048))
>>> buf2 = SeriesBuffer(offset=16384, sample_rate=2048, data=numpy.random.randn(2048))
>>> frame = TSFrame(buffers=[buf1, buf2])
>>> 
>>> # Get the offset of the first buffer
>>> print (frame.offset)
0
>>> 
>>> # Get the offset end of the last buffer
>>> print (frame.end_offset)
32768
>>> 
>>> # Get the sample rate
>>> print (frame.sample_rate)
2048
>>> 
>>> # Iterate over the buffers
>>> for buf in frame:
...     print (buf)
... 
SeriesBuffer(offset=0, offset_end=16384, shape=(2048,), sample_rate=2048, duration=1000000000, data=[0.01658589 ... 0.76543937])
SeriesBuffer(offset=16384, offset_end=32768, shape=(2048,), sample_rate=2048, duration=1000000000, data=[0.76470737 ... 0.89438121])
```

TSFrames must be initialized with at least one buffer because metadata are
derived from the buffer(s).  If you want to have an empty frame, you still have
to set one buffer with the correct metadata, e.g., 

```python notest
>>> # empty buffer
>>> buf = SeriesBuffer(offset=0, sample_rate=2048, shape=(2048,), data=None)
>>> frame = TSFrame(buffers=[buf])
```

#### Advanced TSFrame techniques

There are shortcuts for producing a new empty TSFrame that might be useful if your goal is to just spit out some similar empty frames to fill in, e.g.,

```python notest
>>> frame = TSFrame.from_buffer_kwargs(offset=0, sample_rate=2048, shape=(2048,))
>>> print (frame)

	SeriesBuffer(offset=0, offset_end=16384, shape=(2048,), sample_rate=2048, duration=1000000000, data=None)
>>> print (next(frame))

	SeriesBuffer(offset=16384, offset_end=32768, shape=(2048,), sample_rate=2048, duration=1000000000, data=None)
```

### Writing a new source element


