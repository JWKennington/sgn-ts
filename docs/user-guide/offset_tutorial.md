# Working with Offsets

## Introduction

The `Offset` class is a fundamental component in SGN-TS that handles the precise representation of time in digital signal processing pipelines. Instead of using floating-point seconds or large integer nanoseconds, SGN-TS uses integer "offsets" to represent time in a way that aligns perfectly with sample points at various sampling rates.

This tutorial will guide you through:

1. Understanding why offsets are needed
2. How the offset system works
3. Converting between offsets, time, and samples
4. Working with different sample rates in a unified way
5. Best practices for using offsets in your pipelines

## Why Offsets?

### The Challenge of Time Representation

In digital signal processing, we often work with data sampled at specific time intervals. For example, audio data might be sampled at 44.1 kHz, meaning we have 44,100 samples per second. When we need to synchronize multiple data streams with different sample rates, we face several challenges:

1. **Floating-point precision issues**: Representing time as floating-point seconds can lead to rounding errors
2. **Integer nanosecond overflow**: Using integer nanoseconds requires very large numbers that can overflow
3. **Alignment problems**: Sample points at different rates may not align perfectly with each other

### The Offset Solution

The `Offset` class solves these problems by:

1. Using integer arithmetic to avoid floating-point precision issues
2. Working with smaller integers than nanoseconds to avoid overflow
3. Ensuring perfect alignment of sample points across different rates

## How Offsets Work

### The Concept

An offset is an integer that counts the number of samples at the maximum allowed sample rate (`MAX_RATE`) since a reference time. Here's the key insight:

> If all sample rates are powers of 2 and the maximum rate is also a power of 2, then sample points at any allowed rate will fall exactly on offset boundaries.

### Key Parameters

The `Offset` class defines several important class variables:

```python
# Example of key Offset parameters (not tested by mkdocs)
"""
# Maximum sample rate the system will use (must be a power of 2)
Offset.MAX_RATE = 16384  # 16 kHz by default

# All allowed sample rates (powers of 2 up to MAX_RATE)
Offset.ALLOWED_RATES = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384}

# Default stride length for source elements (in samples at MAX_RATE)
Offset.SAMPLE_STRIDE_AT_MAX_RATE = 16384

# Reference time from which offsets are counted (in nanoseconds)
Offset.offset_ref_t0 = 0
"""
```

### Visual Example

Imagine we have data at three different sample rates: 16 Hz, 8 Hz, and 4 Hz. With `MAX_RATE = 16`, the offsets align perfectly with all sample points:

```
offsets          |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |
                 0   1   2   3   4   5   6   7   8   9   10  11  12  13  14  15

sample rate 16   x   x   x   x   x   x   x   x   x   x   x   x   x   x   x   x
sample rate 8    x       x       x       x       x       x       x       x
sample rate 4    x               x               x               x
```

Notice how:
- Every sample at rate 16 Hz corresponds to a single offset
- Every sample at rate 8 Hz corresponds to every other offset
- Every sample at rate 4 Hz corresponds to every fourth offset

This alignment is what makes offsets powerful for synchronizing data across different rates.

## Converting Between Representations

The `Offset` class provides several methods for converting between different time representations:

### Time to Offsets

```python
# Example of converting time to offsets (not tested by mkdocs)
"""
from sgnts.base import Offset

# Convert seconds to offsets
one_second_offset = Offset.fromsec(1.0)  # 16384 at MAX_RATE=16384

# Convert nanoseconds to offsets
billion_ns_offset = Offset.fromns(1_000_000_000)  # 16384 at MAX_RATE=16384

# Convert nanoseconds to offsets with automatic rounding to sample boundaries
# This is useful when you need the offset to align exactly with a sample point
offset_aligned = Offset.fromns(1_000_000_500, sample_rate=1024)  # Rounds to nearest sample at 1024 Hz

# Convert samples to offsets
samples_at_8khz = 8000  # 8000 samples at 8 kHz
offset_value = Offset.fromsamples(samples_at_8khz, 8192)  # 16000
"""
```

### Offsets to Time

```python
# Example of converting offsets to time (not tested by mkdocs)
"""
from sgnts.base import Offset

# Let's take an offset value
offset_value = 32768  # 2 seconds at MAX_RATE=16384

# Convert offset to seconds
seconds = Offset.tosec(offset_value)  # 2.0

# Convert offset to nanoseconds
nanoseconds = Offset.tons(offset_value)  # 2_000_000_000

# Convert offset to samples at a specific rate
samples_at_8khz = Offset.tosamples(offset_value, 8192)  # 16000
"""
```

### Sample Stride Calculation

The `sample_stride` method helps calculate how many samples correspond to a standard stride at a given rate:

```python
# Example of calculating sample stride (not tested by mkdocs)
"""
from sgnts.base import Offset

# Calculate number of samples in a standard stride at 8 kHz
stride_samples = Offset.sample_stride(8192)  # 8192 samples

# This represents the same time duration as SAMPLE_STRIDE_AT_MAX_RATE
# samples at MAX_RATE
"""
```

## Working with Different Sample Rates

### Setting the Maximum Rate

Before using offsets in your application, you should set the maximum rate to be at least as high as the highest sample rate you'll encounter:

```python
# Example of setting maximum rate (not tested by mkdocs)
"""
from sgnts.base import Offset

# If your highest sample rate is 44.1 kHz, set MAX_RATE to the next power of 2
Offset.set_max_rate(65536)  # 64 kHz > 44.1 kHz

# Now the ALLOWED_RATES will update automatically to include all powers of 2 up to 65536
"""
```

### Converting Between Sample Rates

Offsets make it easy to convert between different sample rates:

```python
# Example of converting between sample rates (not tested by mkdocs)
"""
from sgnts.base import Offset

# Convert 1000 samples at 8 kHz to equivalent samples at 16 kHz
samples_8khz = 1000
offset_value = Offset.fromsamples(samples_8khz, 8192)
samples_16khz = Offset.tosamples(offset_value, 16384)  # 2000

# This works because the offset represents an absolute time duration that
# can be expressed in different sample rates
"""
```

## Best Practices

### When to Use Offsets

Offsets should be used whenever you need to:

1. Precisely synchronize data at different sample rates
2. Convert between time representations without precision loss
3. Handle very long time series (where floating-point precision becomes an issue)

### Ensuring Valid Conversions

When using `tosamples`, make sure the offset corresponds to a whole number of samples:

```python
# Example of handling valid/invalid conversions (not tested by mkdocs)
"""
from sgnts.base import Offset

# This will work fine - 16384 offsets is exactly 4096 samples at 4096 Hz
samples = Offset.tosamples(16384, 4096)  # 4096

# This would raise an assertion error - 10000 offsets doesn't correspond
# to a whole number of samples at 4096 Hz
try:
    samples = Offset.tosamples(10000, 4096)
except AssertionError:
    print("Invalid conversion - offset doesn't align with sample points")
    
# To avoid errors, ensure offsets are multiples of (MAX_RATE / sample_rate)
# For sample_rate=4096 and MAX_RATE=16384, offsets should be multiples of 4
valid_offset = 10000 // 4 * 4  # Round to nearest valid offset
samples = Offset.tosamples(valid_offset, 4096)  # Now it works
"""
```

### Automatic Rounding to Sample Boundaries

When converting from nanoseconds to offsets, you can use the optional `sample_rate` parameter to automatically round to the nearest sample boundary:

```python
# Example of automatic rounding (not tested by mkdocs)
"""
from sgnts.base import Offset

# Without sample_rate parameter - exact conversion
raw_offset = Offset.fromns(1_000_000_123)  # May not align with sample boundaries

# With sample_rate parameter - rounds to nearest sample boundary
aligned_offset = Offset.fromns(1_000_000_123, sample_rate=1024)

# This is particularly useful when working with time segments that need to
# align exactly with sample points, such as in SegmentSource

# Example: Creating aligned time segments
segment_start_ns = 1_500_000_000 + 123  # 1.5 seconds + 123 nanoseconds
segment_end_ns = 2_000_000_000 - 456    # 2.0 seconds - 456 nanoseconds

# These will be rounded to the nearest sample boundaries at 256 Hz
start_offset = Offset.fromns(segment_start_ns, sample_rate=256)
end_offset = Offset.fromns(segment_end_ns, sample_rate=256)

# Now these offsets are guaranteed to map to integer sample points
start_samples = Offset.tosamples(start_offset, 256)  # Integer result
end_samples = Offset.tosamples(end_offset, 256)      # Integer result
"""
```

### Performance Considerations

For very large offset values, the `tons` method switches to integer arithmetic to preserve precision:

```python
# Example of handling large offset values (not tested by mkdocs)
"""
from sgnts.base import Offset, Time

# For small offsets, standard floating-point calculation is used
small_offset = 1000
ns_small = Offset.tons(small_offset)  # Uses floating-point

# For very large offsets, integer division is used to avoid precision loss
large_offset = 10**18 // Offset.MAX_RATE
ns_large = Offset.tons(large_offset)  # Uses integer division
"""
```

## Common Use Cases

### Aligning Data Streams

One of the most common uses of offsets is aligning data from different sources:

```python
# Example of aligning data streams (not tested by mkdocs)
"""
from sgnts.base import Offset, SeriesBuffer
import numpy as np

# Create buffers at different sample rates but aligned in time
start_offset = Offset.fromsec(10.0)  # Start at 10 seconds

# 1 second of data at 8 kHz
buffer_8khz = SeriesBuffer(
    offset=start_offset,
    sample_rate=8192,
    data=np.random.randn(8192)  # 8192 samples = 1 second at 8 kHz
)

# The same 1 second of data at 4 kHz
buffer_4khz = SeriesBuffer(
    offset=start_offset,
    sample_rate=4096,
    data=np.random.randn(4096)  # 4096 samples = 1 second at 4 kHz
)

# Both buffers start at the same offset and represent the same time duration
assert buffer_8khz.offset == buffer_4khz.offset
assert buffer_8khz.end_offset == buffer_4khz.end_offset
"""
```

### Calculating Time Differences

Offsets make it easy to calculate precise time differences:

```python
# Example of calculating time differences (not tested by mkdocs)
"""
from sgnts.base import Offset

# Calculate time difference between two events
event1_offset = Offset.fromsec(10.25)
event2_offset = Offset.fromsec(12.75)

# Difference in offsets
diff_offset = event2_offset - event1_offset

# Convert to time units
diff_seconds = Offset.tosec(diff_offset)  # 2.5 seconds
diff_samples_at_8khz = Offset.tosamples(diff_offset, 8192)  # 20480 samples
"""
```

## Conclusion

The `Offset` class provides a powerful solution for precise time representation in signal processing pipelines. By using integer offsets that align perfectly with sample points, SGN-TS ensures accurate synchronization across different sample rates without the precision issues of floating-point arithmetic.

Key takeaways:
- Offsets count samples at the maximum sample rate from a reference time
- All allowed sample rates must be powers of 2
- Offsets ensure perfect alignment of sample points across different rates
- Use the conversion methods to move between offsets, time, and samples

By understanding and using offsets correctly, you can build robust time-series processing pipelines that handle synchronization challenges with ease.

## Next Steps

- Learn about [SeriesBuffers and TSFrames](buffers.md) which use offsets for time tracking
- Explore the [Core Components](base_tutorial.md) to see how offsets are used in pipelines
- Study the [AudioAdapter](audioadapter_tutorial.md) to understand buffer alignment