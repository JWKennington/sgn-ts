# Working with Time Slices

## Introduction

The `slice_tools` module in SGN-TS provides robust utilities for working with time intervals, which are essential for time-series processing. The module centers around two main classes:

1. `TSSlice`: Represents a single time interval with start and stop points
2. `TSSlices`: Manages collections of `TSSlice` objects with operations for searching and manipulating time intervals

This tutorial will guide you through:

1. Creating and manipulating time slices
2. Performing operations like intersection, union, and difference
3. Working with collections of time slices
4. Using time slices for indexing arrays
5. Common patterns and best practices

## Understanding TSSlice

A `TSSlice` represents a half-open interval `[start, stop)` where start is inclusive and stop is exclusive. This matches Python's standard slicing behavior and makes it convenient for indexing arrays.

### Creating Time Slices

You can create time slices with explicit start and stop values:

```python
# Example of creating time slices (not tested by mkdocs)
"""
from sgnts.base.slice_tools import TSSlice

# Create a slice from offset 100 to 200
slice1 = TSSlice(100, 200)

# Create a slice representing the entire possible time range
full_slice = TSSlice()  # Uses default TIME_MIN and TIME_MAX values

# Create a null slice (represents no time interval)
null_slice = TSSlice(None, None)
"""
```

### Basic Properties and Operations

`TSSlice` provides several basic properties and operations:

```python
# Example of basic TSSlice operations (not tested by mkdocs)
"""
from sgnts.base.slice_tools import TSSlice

# Create a slice
slice1 = TSSlice(100, 200)

# Access the start and stop values
start = slice1.start  # 100
stop = slice1.stop    # 200

# Access via indexing (compatible with tuple unpacking)
start, stop = slice1  # 100, 200

# Convert to Python's built-in slice object (for indexing)
py_slice = slice1.slice  # slice(100, 200, 1)

# Check if a slice is valid (not null)
is_valid = bool(slice1)  # True
is_valid = bool(TSSlice(None, None))  # False

# Check if a slice has a non-zero duration
has_duration = slice1.isfinite()  # True
"""
```

## Operations on Time Slices

`TSSlice` supports various operations that make it easy to work with time intervals.

### Intersection (AND)

Find the overlap between two time slices:

```python
# Example of intersection operations (not tested by mkdocs)
"""
from sgnts.base.slice_tools import TSSlice

# Create two overlapping slices
slice1 = TSSlice(100, 300)
slice2 = TSSlice(200, 400)

# Find their intersection
overlap = slice1 & slice2  # TSSlice(200, 300)

# Non-overlapping slices result in a null slice
slice3 = TSSlice(500, 600)
no_overlap = slice1 & slice3  # TSSlice(None, None)
"""
```

### Union (OR)

Find the slice that spans both slices:

```python
# Example of union operations (not tested by mkdocs)
"""
from sgnts.base.slice_tools import TSSlice

# Create two slices
slice1 = TSSlice(100, 300)
slice2 = TSSlice(200, 400)

# Find their union (the span covering both)
span = slice1 | slice2  # TSSlice(100, 400)

# Works with disjoint slices too
slice3 = TSSlice(500, 600)
span = slice1 | slice3  # TSSlice(100, 600)
"""
```

### Addition (Merge or Keep Separate)

Add slices, merging them if they overlap:

```python
# Example of addition operations (not tested by mkdocs)
"""
from sgnts.base.slice_tools import TSSlice

# Create two overlapping slices
slice1 = TSSlice(100, 300)
slice2 = TSSlice(200, 400)

# Add them together - returns a list with the merged result
result = slice1 + slice2  # [TSSlice(100, 400)]

# Non-overlapping slices remain separate
slice3 = TSSlice(500, 600)
result = slice1 + slice3  # [TSSlice(100, 300), TSSlice(500, 600)]
"""
```

### Difference (Subtraction)

Find the parts of slices that don't overlap:

```python
# Example of subtraction operations (not tested by mkdocs)
"""
from sgnts.base.slice_tools import TSSlice

# Create two overlapping slices
slice1 = TSSlice(100, 300)
slice2 = TSSlice(200, 400)

# Find what's in their union but not in their intersection
diff = slice1 - slice2  # [TSSlice(100, 200), TSSlice(300, 400)]

# Non-overlapping slices return an empty list
slice3 = TSSlice(500, 600)
diff = slice1 - slice3  # []
"""
```

### Comparison Operations

Compare slices to determine their relative positions:

```python
# Example of comparison operations (not tested by mkdocs)
"""
from sgnts.base.slice_tools import TSSlice

# Create some slices
slice1 = TSSlice(100, 200)
slice2 = TSSlice(150, 250)
slice3 = TSSlice(300, 400)

# Check if one slice is completely after another
is_after = slice3 > slice1  # True - slice3 starts after slice1 ends

# Check if one slice is completely before another
is_before = slice1 < slice3  # True - slice1 ends before slice3 starts

# Partially overlapping slices aren't strictly greater or less
is_after = slice2 > slice1  # False - slice2 starts in the middle of slice1
is_before = slice1 < slice2  # False - slice1 ends in the middle of slice2

# Check if one slice contains another
slice4 = TSSlice(120, 180)
contains = slice1.__contains__(slice4)  # True - slice4 is within slice1
"""
```

### Splitting Slices

Split a slice at a specific point:

```python
# Example of splitting slices (not tested by mkdocs)
"""
from sgnts.base.slice_tools import TSSlice

# Create a slice
slice1 = TSSlice(100, 300)

# Split it at offset 200
left, right = slice1.split(200)  # [TSSlice(100, 200), TSSlice(200, 300)]
"""
```

## Working with TSSlices

The `TSSlices` class manages collections of `TSSlice` objects and provides operations for searching and manipulating them.

### Creating Collections of Slices

```python
# Example of creating TSSlices collections (not tested by mkdocs)
"""
from sgnts.base.slice_tools import TSSlice, TSSlices

# Create a collection of slices
slices = TSSlices([
    TSSlice(100, 200),
    TSSlice(300, 400),
    TSSlice(500, 600)
])

# Slices are automatically sorted by start time
unsorted_slices = TSSlices([
    TSSlice(500, 600),
    TSSlice(100, 200),
    TSSlice(300, 400)
])
# Result is the same as 'slices' above
"""
```

### Simplifying Collections (Merging Overlaps)

When working with multiple overlapping slices, you can simplify the collection:

```python
# Example of simplifying overlapping slices (not tested by mkdocs)
"""
from sgnts.base.slice_tools import TSSlice, TSSlices

# Create overlapping slices
slices = TSSlices([
    TSSlice(100, 300),
    TSSlice(200, 400),
    TSSlice(500, 700),
    TSSlice(600, 800)
])

# Simplify by merging overlapping slices
simplified = slices.simplify()  # TSSlices([TSSlice(100, 400), TSSlice(500, 800)])
"""
```

### Finding Intersections

Find the time interval common to all slices:

```python
# Example of finding intersections (not tested by mkdocs)
"""
from sgnts.base.slice_tools import TSSlice, TSSlices

# Create slices with a common overlap
slices = TSSlices([
    TSSlice(100, 400),
    TSSlice(200, 500),
    TSSlice(300, 600)
])

# Find the intersection of all slices
common = slices.intersection()  # TSSlice(300, 400)

# No common intersection returns a null slice
disjoint_slices = TSSlices([
    TSSlice(100, 200),
    TSSlice(300, 400),
    TSSlice(500, 600)
])
common = disjoint_slices.intersection()  # TSSlice(None, None)
"""
```

### Searching for Overlaps

Find slices that overlap with a given time range:

```python
# Example of searching for overlaps (not tested by mkdocs)
"""
from sgnts.base.slice_tools import TSSlice, TSSlices

# Create a collection of slices
slices = TSSlices([
    TSSlice(100, 200),
    TSSlice(300, 400),
    TSSlice(500, 600)
])

# Find slices that overlap with a search range
search_range = TSSlice(150, 350)

# Get overlapping slices without modifying them
overlaps = slices.search(search_range, align=False)
# TSSlices([TSSlice(100, 200), TSSlice(300, 400)])

# Get overlapping slices but truncate them to the search range
aligned_overlaps = slices.search(search_range, align=True)
# TSSlices([TSSlice(150, 200), TSSlice(300, 350)])
"""
```

### Inverting Slices

Find the gaps between slices within a boundary:

```python
# Example of inverting slices (not tested by mkdocs)
"""
from sgnts.base.slice_tools import TSSlice, TSSlices

# Create a collection of slices
slices = TSSlices([
    TSSlice(100, 200),
    TSSlice(300, 400),
    TSSlice(500, 600)
])

# Find the gaps within a boundary
boundary = TSSlice(0, 700)
gaps = slices.invert(boundary)
# TSSlices([TSSlice(0, 100), TSSlice(200, 300), TSSlice(400, 500), TSSlice(600, 700)])
"""
```

## Practical Applications

### Indexing NumPy Arrays

`TSSlice` can be used to index NumPy arrays:

```python
# Example of indexing with TSSlice (not tested by mkdocs)
"""
import numpy as np
from sgnts.base.slice_tools import TSSlice

# Create a sample array
data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

# Use a TSSlice to extract a portion of the array
slice1 = TSSlice(2, 5)
subset = data[slice1.slice]  # array([2, 3, 4])
"""
```

### Processing Time Windows

Use `TSSlice` to manage processing windows in streaming data:

```python
# Example of processing time windows (not tested by mkdocs)
"""
from sgnts.base.slice_tools import TSSlice, TSSlices
from sgnts.base import Offset
import numpy as np

# Create a processing window that advances in 0.5 second increments
window_duration = Offset.fromsec(1.0)  # 1 second window
stride = Offset.fromsec(0.5)           # 0.5 second stride

# Process data in overlapping windows
current_offset = Offset.fromsec(0.0)
end_offset = Offset.fromsec(5.0)

windows = []
while current_offset < end_offset:
    window = TSSlice(current_offset, current_offset + window_duration)
    windows.append(window)
    current_offset += stride

# Result: 9 overlapping windows covering 0 to 5 seconds
# with 50% overlap between consecutive windows
"""
```

### Gap Detection and Handling

Use `TSSlices` to identify and handle gaps in time-series data:

```python
# Example of gap detection (not tested by mkdocs)
"""
from sgnts.base.slice_tools import TSSlice, TSSlices
from sgnts.base import Offset

# Time ranges where we have data
data_ranges = TSSlices([
    TSSlice(Offset.fromsec(0.0), Offset.fromsec(2.0)),   # 0-2 seconds
    TSSlice(Offset.fromsec(3.0), Offset.fromsec(4.0)),   # 3-4 seconds
    TSSlice(Offset.fromsec(6.0), Offset.fromsec(8.0))    # 6-8 seconds
])

# Find gaps in our data within a 10-second window
full_range = TSSlice(Offset.fromsec(0.0), Offset.fromsec(10.0))
gaps = data_ranges.invert(full_range)

# gaps now contains:
# - Gap from 2.0 to 3.0 seconds
# - Gap from 4.0 to 6.0 seconds
# - Gap from 8.0 to 10.0 seconds

# We can handle these gaps differently (e.g., interpolate, fill with zeros, etc.)
for gap in gaps.slices:
    print(f"Gap from {Offset.tosec(gap.start)} to {Offset.tosec(gap.stop)} seconds")
"""
```

## Best Practices

### Handling Null Slices

Be careful when working with null slices (represented as `TSSlice(None, None)`):

```python
# Example of handling null slices (not tested by mkdocs)
"""
from sgnts.base.slice_tools import TSSlice

# Create a null slice
null_slice = TSSlice(None, None)

# Check if a slice is null before using it
if not null_slice:
    print("This is a null slice")

# Operations with null slices often result in null slices
slice1 = TSSlice(100, 200)
result = slice1 & null_slice  # TSSlice(None, None)
result = slice1 | null_slice  # TSSlice(None, None)
"""
```

### Boundary Handling

Be aware of boundary behaviors when using slices:

```python
# Example of boundary handling (not tested by mkdocs)
"""
from sgnts.base.slice_tools import TSSlice, TSSlices

# Create slices at boundaries
slice1 = TSSlice(100, 200)
slice2 = TSSlice(200, 300)  # Starts exactly where slice1 ends

# These slices are considered adjacent but not overlapping
overlap = slice1 & slice2  # TSSlice(None, None) - no overlap

# However, they will be merged when simplified
slices = TSSlices([slice1, slice2])
simplified = slices.simplify()  # TSSlices([TSSlice(100, 300)])
"""
```

### Performance Considerations

For large collections of slices, consider these performance tips:

```python
# Example of performance optimizations (not tested by mkdocs)
"""
from sgnts.base.slice_tools import TSSlice, TSSlices

# When working with many slices, simplify them first to reduce the number
many_slices = TSSlices([
    # ... many potentially overlapping slices ...
])
simplified = many_slices.simplify()

# Use the search method for efficient lookup
# This uses bisect for O(log n) performance
relevant_slices = simplified.search(TSSlice(start, end), align=True)

# For time-critical code, use the Python slice object directly
py_slice = relevant_slices.slices[0].slice
data_subset = large_array[py_slice]  # Fast indexing
"""
```

## Conclusion

The `slice_tools` module provides powerful abstractions for working with time intervals in SGN-TS. By understanding `TSSlice` and `TSSlices`, you can efficiently manage, search, and manipulate time ranges in your time-series processing pipelines.

Key takeaways:
- `TSSlice` represents a single time interval with start and stop points
- `TSSlices` manages collections of time slices with operations for searching and manipulation
- Operations like intersection, union, and difference help with complex time interval logic
- These tools integrate well with NumPy and other array-based data processing

By mastering these tools, you can build robust time-series applications that handle complex temporal relationships with ease.

## Next Steps

- Learn about [Offsets](offset_tutorial.md) for precise time representation
- Explore [SeriesBuffers and TSFrames](buffers.md) which use time slices for data handling
- Study the [Core Components](base_tutorial.md) to see how time slices fit into the pipeline architecture