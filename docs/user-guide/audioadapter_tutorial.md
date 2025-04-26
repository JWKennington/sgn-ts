# Working with AudioAdapter

## Introduction

The `Audioadapter` in SGN-TS is a powerful component for managing time-series data buffers. It stores buffer objects in a deque (double-ended queue) and provides methods to manipulate, access, and process these buffers efficiently. This tutorial will guide you through:

1. Understanding what the `Audioadapter` is and when to use it
2. Creating and managing buffers with `Audioadapter`
3. Working with data gaps
4. Copying and manipulating data
5. Common usage patterns and best practices

## Understanding AudioAdapter

The `Audioadapter` acts as a container for time-series data divided into `SeriesBuffer` objects. It maintains metadata about the buffers it contains, tracks gaps in the data, and provides methods to manipulate data across buffer boundaries.

### Key Features

- Store multiple buffers with consistent sample rates
- Keep track of gaps and non-gaps in the data
- Copy data across buffer boundaries
- Flush data to manage memory usage
- Concatenate data from multiple buffers

### When to Use AudioAdapter

Use `Audioadapter` when you need to:

- Handle streaming time-series data that arrives in chunks
- Manage data with potential gaps
- Process data incrementally without loading everything into memory
- Work with sliding windows of time-series data

## Creating and Managing Buffers

### Initializing an AudioAdapter

To create an `Audioadapter`, you simply initialize it with an optional array backend:

```python
# Example of AudioAdapter initialization (not tested by mkdocs)
"""
from sgnts.base import Audioadapter, SeriesBuffer
from sgnts.base.numpy_backend import NumpyBackend
import numpy as np

# Create an AudioAdapter with default numpy backend
adapter = Audioadapter()

# Or specify a backend explicitly
adapter = Audioadapter(backend=NumpyBackend)
"""
```

### Adding Buffers to the Adapter

You can add buffers to the adapter using the `push` method:

```python
# Example of adding buffers to AudioAdapter (not tested by mkdocs)
"""
import numpy as np
from sgnts.base import Audioadapter, SeriesBuffer, Offset

# Create an adapter
adapter = Audioadapter()

# Create and add a buffer with data
buffer1 = SeriesBuffer(
    offset=0,               # Starting offset
    sample_rate=2048,       # Samples per second
    data=np.random.rand(2048)  # 1 second of random data
)
adapter.push(buffer1)

# Add a second buffer that starts where the first one ends
buffer2 = SeriesBuffer(
    offset=buffer1.end_offset,  # Start at end of first buffer
    sample_rate=2048,
    data=np.random.rand(2048)
)
adapter.push(buffer2)
"""
```

### AudioAdapter Properties

The `Audioadapter` provides several properties to access information about the contained buffers:

```python
# Example of accessing AudioAdapter properties (not tested by mkdocs)
"""
# Continue from previous example...

# Get the offset of the first buffer
start_offset = adapter.offset  # 0

# Get the end offset of the last buffer
end_offset = adapter.end_offset  # Offset.fromsec(2) or similar

# Get the full range as a tuple
time_slice = adapter.slice  # (0, Offset.fromsec(2))

# Check if all buffers are gaps
is_all_gaps = adapter.is_gap  # False

# Get the total number of samples
total_samples = adapter.size  # 4096

# Get the count of non-gap samples
nongap_samples = adapter.nongap_size  # 4096

# Get the count of gap samples
gap_samples = adapter.gap_size  # 0
"""
```

## Working with Data Gaps

Time-series data often contains gaps. The `Audioadapter` helps you manage these gaps efficiently.

### Creating Buffers with Gaps

You can create buffers that represent gaps in the data by setting `data=None` and providing a shape:

```python
# Example of creating buffers with gaps (not tested by mkdocs)
"""
# Create a buffer that represents a gap (missing data)
gap_buffer = SeriesBuffer(
    offset=Offset.fromsec(2),  # Starting at 2 seconds
    sample_rate=2048,
    data=None,                 # No data = gap
    shape=(2048,)              # Shape of the gap (1 second)
)
adapter.push(gap_buffer)

# Now the adapter contains two data buffers and one gap buffer
"""
```

### Checking for Gaps

You can check whether specific segments contain gaps:

```python
# Example of checking for gaps (not tested by mkdocs)
"""
# Check if all buffers in the adapter are gaps
all_gaps = adapter.is_gap  # False, because we have data buffers

# Check which buffers in a specific time segment are gaps
segment = (Offset.fromsec(1.5), Offset.fromsec(3))
gap_info = adapter.buffers_gaps_info(segment)  # [False, True]

# Get detailed gap information for a segment
has_gaps, has_nongaps = adapter.segment_gaps_info(segment)
"""
```

## Copying and Manipulating Data

### Copying Data Across Buffer Boundaries

One of the most powerful features of `Audioadapter` is the ability to copy data across buffer boundaries:

```python
# Example of copying data across buffer boundaries (not tested by mkdocs)
"""
# Copy the first 10 samples
first_10 = adapter.copy_samples(10)

# Copy samples starting from a specific sample
samples_5_to_15 = adapter.copy_samples(10, start_sample=5)

# Copy samples by offset segment
start_offset = Offset.fromsec(0.5)  # 0.5 seconds in
end_offset = Offset.fromsec(1.5)    # 1.5 seconds in
segment_data = adapter.copy_samples_by_offset_segment((start_offset, end_offset))
"""
```

### Concatenating Data

You can concatenate all buffers in the adapter to create a single continuous buffer:

```python
# Example of concatenating data (not tested by mkdocs)
"""
# Concatenate all data in the adapter
adapter.concatenate_data()

# Now you can access the concatenated data
concat_data = adapter.pre_cat_data

# Or concatenate just a segment
segment = (Offset.fromsec(0.5), Offset.fromsec(1.5))
adapter.concatenate_data(segment)
segment_data = adapter.pre_cat_data
"""
```

### Flushing Data

To manage memory usage with streaming data, you can flush (remove) processed data from the adapter:

```python
# Example of flushing data (not tested by mkdocs)
"""
# Flush the first 1024 samples
adapter.flush_samples(1024)

# Or flush up to a specific offset
adapter.flush_samples_by_end_offset(Offset.fromsec(1.5))
"""
```

## Advanced Usage Patterns

### Processing Streaming Data

A common pattern is to process streaming data in blocks, adding new data to the adapter and flushing processed data:

```python
# Example of processing streaming data (not tested by mkdocs)
"""
# Process function that works on 1-second blocks with 0.5-second overlap
def process_streaming_data(adapter, new_buffer):
    # Add new buffer to the adapter
    adapter.push(new_buffer)
    
    # Define the processing window
    if adapter.size >= 2048:  # At least 1 second of data
        # Process the last second of data
        process_window = (adapter.end_offset - Offset.fromsec(1), adapter.end_offset)
        process_data = adapter.copy_samples_by_offset_segment(process_window)
        
        # Process the data...
        result = some_processing_function(process_data)
        
        # Flush everything except the last 0.5 seconds (for overlap)
        flush_point = adapter.end_offset - Offset.fromsec(0.5)
        adapter.flush_samples_by_end_offset(flush_point)
        
        return result
    
    return None
"""
```

### Working with Sliced Buffers

You can get a slice of buffers within a specific time range:

```python
# Example of working with sliced buffers (not tested by mkdocs)
"""
# Get buffers within a specific offset segment
segment = (Offset.fromsec(1), Offset.fromsec(2.5))
sliced_buffers = adapter.get_sliced_buffers(segment)

# Work with the sliced buffers individually
for buffer in sliced_buffers:
    # Do something with each buffer
    print(buffer.offset, buffer.end_offset, buffer.is_gap)
"""
```

## Best Practices

### Memory Management

- Regularly `flush_samples` to avoid memory buildup when processing streaming data
- Use `concatenate_data` only when needed, as it creates a copy of the data
- Consider gap size when working with large datasets with many gaps

### Performance Considerations

- The `Audioadapter` is optimized for sequential append-and-process patterns
- For random access to arbitrary time segments, consider alternative data structures
- When copying data, be aware that crossing buffer boundaries may involve data copying

### Error Handling

Common errors to handle:

- ValueError when trying to access properties of an empty adapter
- ValueError when trying to push buffers with inconsistent sample rates
- ValueError when trying to push buffers with discontinuous offsets
- Value/AssertionError when trying to access data outside the available range

## Conclusion

The `Audioadapter` is a powerful tool for managing time-series data in SGN-TS. It provides a flexible and efficient way to handle streaming data, manage gaps, and process data across buffer boundaries. By understanding its capabilities and usage patterns, you can simplify your time-series data processing pipelines.

## Next Steps

- Learn about [SeriesBuffer](buffers.md) to understand the container objects used by `Audioadapter`
- Explore [Offset](offset.md) to master time-based indexing
- Check out the various [transforms](../api/transforms/) that can be applied to data managed by `Audioadapter`