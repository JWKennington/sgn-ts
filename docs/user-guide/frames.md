# TSFrames

The [TSFrame][sgnts.base.buffer.TSFrame]
class holds a list of [SeriesBuffers][sgnts.base.buffer.SeriesBuffer].


## Introduction

Buffers are passed between element in sgnts in `TSFrame` objects.  TSFrames hold lists of buffers.  Below is a simple TSFrame with one non-gap buffer:
  
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

An example with two contiguous buffers:
	
   ```python
   import numpy
   numpy.random.seed(1)
   
   from sgnts.base.buffer import SeriesBuffer, TSFrame
   
   # An example of two contiguous buffers
   buf1 = SeriesBuffer(offset=0, sample_rate=2048, data=numpy.random.randn(2048))
   buf2 = SeriesBuffer(offset=16384, sample_rate=2048, data=numpy.random.randn(2048))
   frame = TSFrame(buffers=[buf1, buf2])
   
   # The frame will contain both series buffers in this example
   repr_frame="""TSFrame(EOS=False, is_gap=False, metadata={}, buffers=[
       SeriesBuffer(offset=0, offset_end=16384, shape=(2048,), sample_rate=2048, duration=1000000000, data=[1.62434536 ... 1.20809946]),
       SeriesBuffer(offset=16384, offset_end=32768, shape=(2048,), sample_rate=2048, duration=1000000000, data=[-1.82921963 ...  0.79494725]),
   ])"""
   
   assert repr(frame) == repr_frame
   ```

If you try to create a TSFrame without contiguous buffers, you get an error

```python
import numpy

numpy.random.seed(1)
from sgnts.base.buffer import SeriesBuffer, TSFrame

# An example of two non contiguous buffers. NOTE THIS SHOULDN'T WORK!!
buf1 = SeriesBuffer(offset=0, sample_rate=2048, data=numpy.random.randn(2048))
buf2 = SeriesBuffer(offset=12345, sample_rate=2048, data=numpy.random.randn(2048))
err = None

try:
    frame = TSFrame(buffers=[buf1, buf2])
except AssertionError as e:
    err = e

assert err is not None
```

## Properties

TSFrames have additional properties that are all derived from their input buffers.

TSFrames must be initialized with at least one buffer because metadata are
derived from the buffer(s).  
:  If you want to have an empty frame, you still have to provide a gap buffer, e.g.,
   ```python
   from sgnts.base.buffer import SeriesBuffer, TSFrame
   
   # empty buffer
   buf = SeriesBuffer(offset=0, sample_rate=2048, shape=(2048,), data=None)
   frame = TSFrame(buffers=[buf])
   repr_frame = """TSFrame(EOS=False, is_gap=True, metadata={}, buffers=[
       SeriesBuffer(offset=0, offset_end=16384, shape=(2048,), sample_rate=2048, duration=1000000000, data=None),
   ])"""
   assert repr_frame == repr(frame)
   ```


### Offset
:  The offset of a TSFrame is the offset of its first buffer
   ```python
   import numpy
   
   numpy.random.seed(1)
   
   from sgnts.base.buffer import SeriesBuffer, TSFrame
   
   buf1 = SeriesBuffer(offset=0, sample_rate=2048, data=numpy.random.randn(2048))
   buf2 = SeriesBuffer(offset=16384, sample_rate=2048, data=numpy.random.randn(2048))
   frame = TSFrame(buffers=[buf1, buf2])
   assert frame.offset == 0
   ```

### Offset end
:  The end offset of a TSFrame is the end offset of its last buffer
   ```python
   import numpy
   
   numpy.random.seed(1)
   
   from sgnts.base.buffer import SeriesBuffer, TSFrame
   
   buf1 = SeriesBuffer(offset=0, sample_rate=2048, data=numpy.random.randn(2048))
   buf2 = SeriesBuffer(offset=16384, sample_rate=2048, data=numpy.random.randn(2048))
   frame = TSFrame(buffers=[buf1, buf2])
   assert frame.end_offset == 32768
   ```

## Buffer operations

### Iteration, indexing and length
:  You can iterate over the buffers in a frame or index them. The length of a frame is the number of buffers.
   ```python
   import numpy
   
   numpy.random.seed(1)
   from sgnts.base.buffer import SeriesBuffer, TSFrame
   
   buf0 = SeriesBuffer(offset=0, sample_rate=2048, data=numpy.random.randn(2048))
   buf1 = SeriesBuffer(offset=16384, sample_rate=2048, data=numpy.random.randn(2048))
   
   frame = TSFrame(buffers=[buf0, buf1])
   
   # Iterate over the buffers
   for buf in frame:
       assert buf in (buf0, buf1)
   
   assert buf0 == frame[0]
   assert buf1 == frame[1]
   assert len(frame) == 2
   ```

## Other TSFrame initialization techniques

### Initializing a TSFrame from buffer kwargs
:  If your goal is to just produce a frame with a single buffer all in one step you can do
   ```python
   from sgnts.base.buffer import TSFrame
   
   frame = TSFrame.from_buffer_kwargs(offset=0, sample_rate=2048, shape=(2048,))
   repr_frame = """TSFrame(EOS=False, is_gap=True, metadata={}, buffers=[
       SeriesBuffer(offset=0, offset_end=16384, shape=(2048,), sample_rate=2048, duration=1000000000, data=None),
   ])"""
   
   assert repr(frame) == repr_frame
   ```

### Getting the "next" frame
:  It is common to want to produce a sequence of frames with the same
   properties, e.g., a single buffer of the same shape incremented to the next
  offset span
  ```python
  from sgnts.base.buffer import TSFrame
  
  frame = TSFrame.from_buffer_kwargs(offset=0, sample_rate=2048, shape=(2048,))
  next_frame = next(frame)
  repr_next_frame = """TSFrame(EOS=False, is_gap=True, metadata={}, buffers=[
      SeriesBuffer(offset=16384, offset_end=32768, shape=(2048,), sample_rate=2048, duration=1000000000, data=None),
  ])"""
  
  assert repr(next_frame) == repr_next_frame
  ```

## More details

Additional methods and properties are documented in the [API docs](/api/base/buffer/)
