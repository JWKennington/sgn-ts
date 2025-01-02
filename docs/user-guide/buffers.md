# SeriesBuffers and TSFrames

The most important new class in `sgnts` is the [TSFrame][sgnts.base.buffer.TSFrame]
which holds a list of [SeriesBuffers][sgnts.base.buffer.SeriesBuffer].

Here we can get some familiarity with both of these objects and along the way,
other classes and concepts relevant for sgnts.

## Key Concepts

The below example is a good starting point for understanding the key concepts
of `sgnts` buffers. There is plenty to unpack here, so lets go step by step.

```python
import numpy
from sgnts.base.buffer import SeriesBuffer
buf = SeriesBuffer(offset=0, sample_rate=2048, data=numpy.random.randn(2048))
print(buf)
```
```
[Out] SeriesBuffer(offset=0, offset_end=16384, shape=(2048,), sample_rate=2048, duration=1000000000, data=[0.56649291 ... 1.39569688])
```



### Offsets

The term `offset` is globally meaningful throughout the application and acts as a
precise surrogate for time, i.e., an absolute "time" reference for any element
within an sgnts application that should not suffer from any rounding error.
Technically offsets are defined as a cumulative number of samples passed
defined at the maximum sample rate allowed by the application.  This will be
explained more below.

### Sample Rate

`sample_rate` is the number of samples per second that a stretch of data
contains. It is used to convert to actual time with nanosecond precision. In
order to make certain gaurantees about precision in sgnts, we currently only
support power of 2 sample rates from 1 Hz to a maximum which defaults to 16384
Hz.  The max sample rate and allowed rates are defined
[here](https://git.ligo.org/greg/sgn-ts/-/blob/main/src/sgnts/base/offset.py?ref_type=heads#L63).


### Data

`data` is generally a numpy array that can be interpreted as (possibly
multidimensional) time series data. 


## Detailed Example

Now revisiting the above

```python
import numpy
from sgnts.base.buffer import SeriesBuffer

buf = SeriesBuffer(offset=0, sample_rate=2048, data=numpy.random.randn(2048))
print (buf)
```
```
[Out] SeriesBuffer(offset=0, offset_end=16384, shape=(2048,), sample_rate=2048, duration=1000000000, data=[0.56649291 ... 1.39569688])
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

```python
import numpy
from sgnts.base.buffer import SeriesBuffer

buf = SeriesBuffer(offset=0, sample_rate=2048, data=numpy.random.randn(2,2048))
print (buf)
```
```
[Out] SeriesBuffer(offset=0, offset_end=16384, shape=(2, 2048), sample_rate=2048, duration=1000000000, data=[[ 0.01684876 ... -1.6963346 ]
 [-0.55875476 ...  0.58967178]])
```

Note what happens to the offset if you change the sample rate (and in this case
also the data size)

```python
import numpy
from sgnts.base.buffer import SeriesBuffer    

buf = SeriesBuffer(offset=0, sample_rate=1024, data=numpy.random.randn(2,1024))
buf
```
```
[Out] SeriesBuffer(offset=0, offset_end=16384, shape=(2, 1024), sample_rate=1024, duration=1000000000, data=[[-0.13116052 ...  1.2223811 ]
 [-0.98786954 ... -0.56760618]])
```

**It stays the same.** Remember that the offset is the sample count at the
theoretical maximum sample rate which is defined in offset.py.  

Only power of two sample rates are allowed at present to ensure that bookeeping
remains simple and accurate. 

```{.python notest}
import numpy
from sgnts.base.buffer import SeriesBuffer

buf = SeriesBuffer(offset=0, sample_rate=1000, data=numpy.random.randn(2,1000))
```
```
[Out] Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "<string>", line 7, in __init__
  File "/Users/crh184/Library/Python/3.9/lib/python/site-packages/sgnts/base/buffer.py", line 38, in __post_init__
    raise ValueError("%s not in allowed rates %s" % (self.sample_rate, Offset.ALLOWED_RATES))
ValueError: 1000 not in allowed rates {32, 1, 2, 64, 4, 128, 256, 512, 8, 1024, 2048, 4096, 8192, 16, 16384}
```

It is possible to increase the maximum sample rate globally in an application

```{.python notest}
import numpy
from sgnts.base.buffer import SeriesBuffer
from sgnts.base.offset import Offset

Offset.set_max_rate(262144)
buf = SeriesBuffer(offset=0, sample_rate=32768, data=numpy.random.randn(32768))
print (buf)
```
```
[Out] SeriesBuffer(offset=0, offset_end=262144, shape=(32768,), sample_rate=32768, duration=1000000000, data=[-0.08916502 ...  0.89236118])
```

Buffers are not the primary data type passed around between element in sgnts.  Rather, it is a `TSFrame`.  TSFrames hold lists of buffers

```python
import numpy
from sgnts.base.buffer import SeriesBuffer, TSFrame

# An example of just one buffer
buf1 = SeriesBuffer(offset=0, sample_rate=2048, data=numpy.random.randn(2048))
frame = TSFrame(buffers=[buf1])
print(frame)
```
```
[Out] SeriesBuffer(offset=0, offset_end=16384, shape=(2048,), sample_rate=2048, duration=1000000000, data=[-0.04094335 ... -1.49758223])
```
	
```python
import numpy
from sgnts.base.buffer import SeriesBuffer, TSFrame

# An example of two contiguous buffers
buf1 = SeriesBuffer(offset=0, sample_rate=2048, data=numpy.random.randn(2048))
buf2 = SeriesBuffer(offset=16384, sample_rate=2048, data=numpy.random.randn(2048))
frame = TSFrame(buffers=[buf1, buf2])
print (frame)
```
```
[Out] SeriesBuffer(offset=0, offset_end=16384, shape=(2048,), sample_rate=2048, duration=1000000000, data=[-1.56771352 ... -0.20928693])
	  SeriesBuffer(offset=16384, offset_end=32768, shape=(2048,), sample_rate=2048, duration=1000000000, data=[-1.00442217 ... -0.75684022])
```

```{.python notest}
import numpy
from sgnts.base.buffer import SeriesBuffer, TSFrame

# An example of two non contiguous buffers. NOTE THIS SHOULDN'T WORK!!
buf1 = SeriesBuffer(offset=0, sample_rate=2048, data=numpy.random.randn(2048))
buf2 = SeriesBuffer(offset=12345, sample_rate=2048, data=numpy.random.randn(2048))
frame = TSFrame(buffers=[buf1, buf2])
```
```
[Out] Traceback (most recent call last):
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

```python
import numpy
from sgnts.base.buffer import SeriesBuffer, TSFrame

buf1 = SeriesBuffer(offset=0, sample_rate=2048, data=numpy.random.randn(2048))
buf2 = SeriesBuffer(offset=16384, sample_rate=2048, data=numpy.random.randn(2048))
frame = TSFrame(buffers=[buf1, buf2])

# Get the offset of the first buffer, the end offset of the last buffer, and the sample rate
print(frame.offset, frame.end_offset, frame.sample_rate)
```
```
[Out] 0 32768 2048
```

```python
import numpy
from sgnts.base.buffer import SeriesBuffer, TSFrame

buf1 = SeriesBuffer(offset=0, sample_rate=2048, data=numpy.random.randn(2048))
buf2 = SeriesBuffer(offset=16384, sample_rate=2048, data=numpy.random.randn(2048))
frame = TSFrame(buffers=[buf1, buf2])

# Iterate over the buffers
for buf in frame:
    print (buf)
```
```
[Out] SeriesBuffer(offset=0, offset_end=16384, shape=(2048,), sample_rate=2048, duration=1000000000, data=[0.01658589 ... 0.76543937])
      SeriesBuffer(offset=16384, offset_end=32768, shape=(2048,), sample_rate=2048, duration=1000000000, data=[0.76470737 ... 0.89438121])
```

TSFrames must be initialized with at least one buffer because metadata are
derived from the buffer(s).  If you want to have an empty frame, you still have
to set one buffer with the correct metadata, e.g., 

```python
from sgnts.base.buffer import SeriesBuffer, TSFrame

# empty buffer
buf = SeriesBuffer(offset=0, sample_rate=2048, shape=(2048,), data=None)
frame = TSFrame(buffers=[buf])
```


This section will cover some advanced techniques for working with `sgnts` and `SeriesBuffer` objects.

## Advanced TSFrame techniques

There are shortcuts for producing a new empty TSFrame that might be useful if your goal is to
just spit out some similar empty frames to fill in, e.g.,

```python
from sgnts.base.buffer import TSFrame

frame = TSFrame.from_buffer_kwargs(offset=0, sample_rate=2048, shape=(2048,))
print (frame)
```
```
[Out] SeriesBuffer(offset=0, offset_end=16384, shape=(2048,), sample_rate=2048, duration=1000000000, data=None)
```

```python
from sgnts.base.buffer import TSFrame

frame = TSFrame.from_buffer_kwargs(offset=0, sample_rate=2048, shape=(2048,))
print (next(frame))
```
```
[Out] SeriesBuffer(offset=16384, offset_end=32768, shape=(2048,), sample_rate=2048, duration=1000000000, data=None)
```

