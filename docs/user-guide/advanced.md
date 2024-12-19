# Advanced Techniques

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
