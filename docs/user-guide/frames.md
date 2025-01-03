# TSFrames

The most important new class in `sgnts` is the [TSFrame][sgnts.base.buffer.TSFrame]
which holds a list of [SeriesBuffers][sgnts.base.buffer.SeriesBuffer].


## TSFrames -- a container for SeriesBuffers

Buffers are passed between element in sgnts in `TSFrame` objects.  TSFrames hold lists of buffers

Simple TSFrame with one non-gap buffer:

   ```python
   import numpy
   
   numpy.random.seed(1)
   from sgnts.base.buffer import SeriesBuffer, TSFrame
   
   # An example of just one buffer in a TSFrame
   buf = SeriesBuffer(offset=0, sample_rate=2048, data=numpy.random.randn(2048))
   frame = TSFrame(buffers=[buf])
   
   # If you print the TSFrame it displays additional metadata that is derived from the buffer, e.g.,
   #
   # - EOS (end of stream): By default this is False like in sgn, but it can be used to indicate now new data is coming
   # - is_gap: will be set if **all** the input buffers to the frame are gap (i.e., data = None)
   # - metadata: This is an arbitrary dictionary of metadata about the Frame, it is not handled consistently anywhere within the framework. It is recommended to **not** use it since we might deprecate it.
   # - buffers: the list of input buffers
   repr_frame = """TSFrame(EOS=False, is_gap=False, metadata={}, buffers=[
       SeriesBuffer(offset=0, offset_end=16384, shape=(2048,), sample_rate=2048, duration=1000000000, data=[1.62434536 ... 1.20809946]),
   ])"""
   
   assert repr(frame) == repr_frame
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

