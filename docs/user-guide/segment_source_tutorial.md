# SegmentSource Tutorial

The `SegmentSource` produces non-gap buffers for specified time segments and gap buffers elsewhere, allowing for the simulation of windowed data. The enhanced version now supports custom values for each segment, making it more flexible for creating test signals.

## Overview

`SegmentSource` is useful when you want to:
- Create data streams with predefined time windows
- Simulate data that's only available during specific time periods
- Test how downstream components handle gaps in data
- Generate test signals with different values for different time windows

## Basic Usage

The basic usage of SegmentSource involves defining time segments where data should be present:

```python
from sgnts.sources import SegmentSource

# Define time segments in nanoseconds (start, end)
segments = (
    (1000000000, 2000000000),  # 1s to 2s
    (3000000000, 4000000000),  # 3s to 4s
)

# Create a segment source
segment_source = SegmentSource(
    name="example_source",
    source_pad_names=("data",),
    rate=2048,      # Sample rate
    segments=segments,
    t0=0,           # Start time
    end=5           # End time (in seconds)
)

# Get the source pad
pad = segment_source.srcs["data"]

# Pull a frame
frame = segment_source.new(pad)

# Buffers within the specified segments will contain data
# Buffers outside the segments will be gap buffers
for buf in frame:
    if buf.is_gap:
        print(f"Gap buffer at offset {buf.offset}")
    else:
        print(f"Data buffer at offset {buf.offset} with shape {buf.shape}")
```

## Using Custom Values

The enhanced `SegmentSource` now supports a `values` parameter to specify different values for each segment:

```python
from sgnts.sources import SegmentSource

# Define segments and their corresponding values
segments = (
    (1000000000, 2000000000),  # 1s to 2s: value=10
    (3000000000, 4000000000),  # 3s to 4s: value=20
    (5000000000, 6000000000),  # 5s to 6s: value=30
)

values = (10, 20, 30)  # One value per segment

# Create source with custom values
source = SegmentSource(
    name="custom_value_source",
    source_pad_names=("data",),
    rate=2048,
    segments=segments,
    values=values,  # New parameter!
    t0=0,
    end=7
)

# Get a frame to see the values
pad = source.srcs["data"]
frame = source.new(pad)

# Check the values in non-gap buffers
for buf in frame:
    if not buf.is_gap and buf.data is not None:
        print(f"Buffer at offset {buf.offset} has value: {buf.data[0]}")
```

## Example Files

We provide several example files demonstrating different aspects of SegmentSource. You can find these in the `examples/` directory:

### [segment_source_basic.py](https://git.ligo.org/greg/sgn-ts/-/blob/main/examples/segment_source_basic.py)

This example shows how to create segments with different values and visualize them:
- Creates segments with custom values (10, 20, 30)
- Saves output using DumpSeriesSink
- Loads and visualizes the results with matplotlib
- Shows gap vs. data regions

### [segment_source_edge_cases.py](https://git.ligo.org/greg/sgn-ts/-/blob/main/examples/segment_source_edge_cases.py)

This comprehensive example demonstrates edge cases, validation, and special behaviors:
- Adjacent segments (touching but not overlapping)
- Segments partially outside the time range (automatically clipped)
- Special value handling (0, 1, and arbitrary integers)
- Error cases and validation
- Gap-only output scenarios

### Running the Examples

To run any of the examples:

```bash
# From the sgn-ts root directory
python examples/segment_source_basic.py
python examples/segment_source_edge_cases.py
```

Each example will:
- Create output files (`.txt` format from DumpSeriesSink)
- Display interactive matplotlib plots (if available)
- Print diagnostic information to the console

## Important Behaviors

### Segment Filtering

Segments that partially overlap with the source's time range (`t0` to `end`) are automatically included and clipped to fit within the bounds:

```python
from sgnts.sources import SegmentSource

source = SegmentSource(
    name="filtered_source",
    source_pad_names=("data",),
    rate=256,
    t0=0.0,
    end=5.0,  # 5 seconds
    segments=(
        (1e9, 2e9),      # 1-2s: INCLUDED (fully within)
        (4e9, 6e9),      # 4-6s: INCLUDED and clipped to 4-5s
        (7e9, 8e9),      # 7-8s: EXCLUDED (fully outside)
    ),
    values=(10, 20, 30),
)

# Check which segments were kept
print(f"Number of segments kept: {len(source.segment_data)}")
for seg_slice, orig_idx in source.segment_data:
    print(f"Segment {orig_idx}: {seg_slice.start/1e9:.1f}s to {seg_slice.stop/1e9:.1f}s")
# Output shows first two segments, with the second one clipped to 5s
```

### Special Values

The values 0 and 1 have special meaning:
- `0` creates arrays filled with zeros
- `1` creates arrays filled with ones
- Any other integer creates arrays filled with that value

```python
values = (0, 1, 42)  # zeros, ones, and array of 42s
```

### Segment Validation

The enhanced version validates that segments don't overlap:

```python
# This will raise an AssertionError
overlapping = (
    (1_000_000_000, 3_000_000_000),
    (2_000_000_000, 4_000_000_000),  # Overlaps!
)

# Adjacent segments are OK
adjacent = (
    (1_000_000_000, 2_000_000_000),
    (2_000_000_000, 3_000_000_000),  # Touches but doesn't overlap
)
```

## Best Practices

1. **Define non-overlapping segments** - The enhanced version validates that segments don't overlap and will raise an error if they do

2. **Match values to segments** - When using the `values` parameter, ensure it has the same length as `segments`

3. **Be aware of segment boundaries** - Each buffer will be either entirely within or entirely outside a segment, which may result in many small buffers at segment boundaries

4. **Check for gaps** in downstream processing - Ensure that components receiving data from a `SegmentSource` properly handle gap buffers

5. **Use precise time units** - Segments are specified in nanoseconds, so be careful with unit conversions

6. **Special values** - Remember that values 0 and 1 have special meaning (create arrays of zeros and ones respectively)

7. **Validate segment coverage** - When debugging, always check `source.segment_data` to see which segments were actually kept:
   ```python
   from sgnts.sources import SegmentSource
   
   # Create a source with segments that might be filtered
   source = SegmentSource(
       name="debug_source",
       source_pad_names=("data",),
       rate=256,
       t0=1.0,  # Start at 1 second
       end=4.0,  # End at 4 seconds
       segments=(
           (0, 2e9),        # 0-2s: partially in range
           (2e9, 3e9),      # 2-3s: fully in range
           (3.5e9, 5e9),    # 3.5-5s: partially in range
           (6e9, 7e9),      # 6-7s: fully outside
       ),
       values=(10, 20, 30, 40)
   )
   
   # Check which segments were kept
   print(f"Segments kept: {len(source.segment_data)} out of 4")
   for seg_slice, orig_idx in source.segment_data:
       print(f"  Original segment {orig_idx}: {seg_slice.start/1e9:.1f}s to {seg_slice.stop/1e9:.1f}s")
   ```

## Known Behaviors and Limitations

### Value=0 Creates Zero Arrays, Not Gaps

When `values` contains 0, it creates buffers filled with zeros, not gap buffers. This is an important distinction:

- **Gap buffers**: `data=None` (indicate missing data)
- **Zero-valued buffers**: `data=array([0, 0, 0, ...])` (indicate actual signal level of 0)

Both appear as 0 in the output file, but they are fundamentally different in meaning.

### DumpSeriesSink Output Format

`DumpSeriesSink` only writes non-gap buffers to the output file. This means:

- The output file contains only time periods with data
- Gaps are not represented in the file
- Time values may not be continuous

Example output file:
```
1.0000  10.0   # Data from first segment
1.0039  10.0
...
3.0000  20.0   # Data from second segment (gap from 2-3s not shown)
3.0039  20.0
```

If you need to preserve gap information, consider using a different sink or post-processing the output.

## Summary

The enhanced SegmentSource with custom values provides a powerful way to:
- Create test signals with known characteristics
- Simulate different signal conditions
- Generate data for pipeline testing
- Create reproducible test scenarios

The ability to specify different values for each segment makes it much more flexible than the original implementation that only supported binary (gap/no-gap) outputs.