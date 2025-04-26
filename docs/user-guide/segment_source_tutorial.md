# SegmentSource

The `SegmentSource` produces non-gap buffers for specified time segments and gap buffers elsewhere, allowing for the simulation of windowed data.

## Overview

`SegmentSource` is useful when you want to:
- Create data streams with predefined time windows
- Simulate data that's only available during specific time periods
- Test how downstream components handle gaps in data

## Basic Usage

```python
# Basic usage of SegmentSource (not tested by mkdocs)
"""
from sgnts.sources import SegmentSource
from sgnts.base import Time

# Define time segments in nanoseconds (start, end)
segments = (
    (1000000000, 2000000000),  # 1s to 2s
    (3000000000, 4000000000),  # 3s to 4s
)

# Create a segment source
segment_source = SegmentSource(
    rate=2048,      # Sample rate
    segments=segments,
    t0=0,           # Start time
    end=5           # End time (in seconds)
)

# Pull frames
frame = segment_source.pull()

# Buffers within the specified segments will contain data
# Buffers outside the segments will be gap buffers
for buf in frame:
    if buf.is_gap:
        print(f"Gap buffer at {buf.t0 / Time.SECONDS}s")
    else:
        print(f"Data buffer at {buf.t0 / Time.SECONDS}s")
"""
```

## Time Slices

The `SegmentSource` internally uses `TSSlice` and `TSSlices` to represent and manage the time segments:

```python
# Working with segments (not tested by mkdocs)
"""
from sgnts.sources import SegmentSource
from sgnts.base import TSSlice, TSSlices, Offset

# Define overlapping segments in nanoseconds
raw_segments = (
    (1000000000, 3000000000),  # 1s to 3s
    (2000000000, 4000000000),  # 2s to 4s
)

# Create a segment source
source = SegmentSource(
    rate=2048,
    segments=raw_segments,
    t0=0,
    end=5
)

# SegmentSource automatically converts these to TSSlice objects and
# simplifies overlapping segments using TSSlices.simplify()
print(source.segment_slices)  # This will show a simplified representation
"""
```

## Splitting Buffers

The `SegmentSource` splits buffers at segment boundaries, ensuring that each buffer is either entirely within a segment or entirely outside:

```python
# Understanding buffer splitting (not tested by mkdocs)
"""
from sgnts.sources import SegmentSource

# Define non-overlapping segments
segments = (
    (1000000000, 2000000000),  # 1s to 2s
    (3000000000, 4000000000),  # 3s to 4s
)

source = SegmentSource(
    rate=2048,
    segments=segments,
    t0=0,
    end=5
)

# Pull a frame
frame = source.pull()

# The frame will contain multiple buffers, split at segment boundaries
print(f"Number of buffers in frame: {len(frame)}")

# Analyze each buffer
for i, buf in enumerate(frame):
    status = "Data" if not buf.is_gap else "Gap"
    print(f"Buffer {i}: {status} at offset {buf.offset}")
"""
```

## Integration with Processing Pipeline

```python
# Integration example (not tested by mkdocs)
"""
from sgnts.sources import SegmentSource
from sgnts.transforms import AmplifyTransform
from sgnts.sinks import DumpSeriesSink

# Define segments
segments = (
    (1000000000, 2000000000),  # 1s to 2s
    (3000000000, 4000000000),  # 3s to 4s
)

# Create a pipeline
source = SegmentSource(rate=2048, segments=segments, t0=0, end=5)
amplify = AmplifyTransform(factor=2.0)
sink = DumpSeriesSink(fname="segment_output.txt")

# Connect the elements
source.add_dest(amplify)
amplify.add_dest(sink)

# Run the pipeline
for _ in range(10):
    source.process()
    amplify.process()
    sink.process()

# The output file will contain data only for the specified segments
# with gaps elsewhere
"""
```

## Best Practices

When using `SegmentSource`:

1. **Define non-overlapping segments** when possible for clarity, though the source will handle overlapping segments correctly by simplifying them

2. **Be aware of segment boundaries** - each buffer will be either entirely within or entirely outside a segment, which may result in many small buffers at segment boundaries

3. **Check for gaps** in downstream processing - ensure that components receiving data from a `SegmentSource` properly handle gap buffers

4. **Consider memory usage** - when segments are highly fragmented, many buffers may be created, which could impact performance

5. **Use precise time units** - segments are specified in nanoseconds, so be careful with unit conversions