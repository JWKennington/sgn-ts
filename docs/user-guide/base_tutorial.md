# Core Components: Understanding the Base Module

## Introduction

The `base.py` module is the foundation of SGN-TS, providing the core architecture for building time-series processing pipelines. It defines base classes for sources, transforms, and sinks - the three fundamental components in a signal processing graph.

This tutorial will guide you through:

1. Understanding the pipeline architecture in SGN-TS
2. Working with different types of components (sources, transforms, sinks)
3. Data flow and alignment mechanisms
4. Creating custom components
5. Advanced usage patterns

## Pipeline Architecture

SGN-TS uses a directed graph architecture for signal processing, where data flows from sources through transforms to sinks. The core components are:

1. **Sources** (`TSSource`, `TSResourceSource`): Generate or retrieve time-series data
2. **Transforms** (`TSTransform`): Process incoming data and produce output data
3. **Sinks** (`TSSink`): Consume data, typically for visualization, storage, or other outputs

These components are connected through **pads**:
- **Source pads**: Output connections that deliver data to downstream components
- **Sink pads**: Input connections that receive data from upstream components

### Example Pipeline Structure

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   TSSource   │     │  TSTransform │     │    TSSink    │
│              │     │              │     │              │
│  ┌────────┐  │     │ ┌────────┐   │     │ ┌────────┐   │
│  │SourcePad├──────►│ │SinkPad │   │     │ │SinkPad │   │
│  └────────┘  │     │ └────────┘   │     │ └────────┘   │
│              │     │              │     │              │
│              │     │  ┌────────┐  │     │              │
│              │     │  │SourcePad├──────►│              │
│              │     │  └────────┘  │     │              │
└──────────────┘     └──────────────┘     └──────────────┘
```

## Key Concepts

### Offsets and Time

SGN-TS uses integer offsets to represent time. This avoids floating-point precision issues when dealing with time-series data. Key classes for handling time include:

- `Offset`: Converts between seconds, samples, and offset integers
- `Time`: Defines time constants (e.g., `Time.SECONDS`, `Time.MILLISECONDS`)
- `TSSlice`: Represents a time segment with start and end offsets

### Buffers and Frames

Data moves through the pipeline in structured containers:

- `SeriesBuffer`: Holds a chunk of data with associated metadata (offset, sample rate, etc.)
- `TSFrame`: A collection of `SeriesBuffer` objects that represents a complete frame of data

### Alignment and Adapters

Components need to handle data from multiple inputs that might arrive at different times or rates:

- `_TSTransSink`: A mixin class that provides alignment capabilities
- `AdapterConfig`: Configures how data is aligned, padded, and processed
- `Audioadapter`: Manages buffers and handles gaps in the data

## Working with Sources

Sources are components that generate time-series data. SGN-TS provides two main source types:

### TSSource

`TSSource` is a basic source that generates data in fixed-size buffers at regular intervals:

```python
# Example of TSSource usage (not tested by mkdocs)
"""
from sgnts.base import TSSource, Offset
import numpy as np

# Create a source that generates data from t=0 to t=10 seconds
source = TSSource(
    t0=0,               # Start time
    duration=10,        # Duration in seconds
    source_pad_names=["output"]  # Name the output pad
)

# Configure the source to produce samples at 1000 Hz
source.set_pad_buffer_params(
    pad=source.srcs["output"],
    sample_shape=(),    # Scalar samples (no channels)
    rate=1000           # 1000 samples per second
)

# Source will now produce frames with buffers containing the specified data
# when used in a pipeline
"""
```

### TSResourceSource

`TSResourceSource` is more flexible and driven by external data sources:

```python
# Example of TSResourceSource usage (not tested by mkdocs)
"""
from sgnts.base import TSResourceSource, SeriesBuffer, Offset
import numpy as np

class MyDataSource(TSResourceSource):
    def __init__(self, **kwargs):
        super().__init__(source_pad_names=["output"], **kwargs)
        # Initialize your data source here
        
    def get_data(self):
        # This method must be implemented to yield data
        # It runs in a separate thread
        while not self.stop_event.is_set():
            # Get data from some external source
            data = np.random.rand(1024)  # Example: random data
            
            # Create a buffer and yield it with the corresponding pad
            buf = SeriesBuffer(
                offset=Offset.fromsec(time.time()),
                data=data,
                sample_rate=1000
            )
            yield self.srcs["output"], buf
"""
```

### Key Properties of Sources

- `start_offset`: The earliest offset in the source
- `end_offset`: The latest offset in the source
- `t0`: The start time in seconds
- `duration`: The duration of the source in seconds

## Working with Transforms

Transforms process incoming data and produce output data. The `TSTransform` class handles:

1. Receiving data on sink pads
2. Aligning data from multiple inputs
3. Processing the aligned data
4. Sending the processed data to output pads

### Creating a Custom Transform

```python
# Example of creating a custom transform (not tested by mkdocs)
"""
from sgnts.base import TSTransform, TSFrame, AdapterConfig
import numpy as np

class AmplifyTransform(TSTransform):
    def __init__(self, gain=2.0, **kwargs):
        super().__init__(
            sink_pad_names=["input"],
            source_pad_names=["output"],
            # Configure how input data is aligned and processed
            adapter_config=AdapterConfig(
                overlap=(0, 0),  # No overlap needed
                stride=0,        # Process all available data
                skip_gaps=True   # Skip processing if there are gaps
            ),
            **kwargs
        )
        self.gain = gain
        
    def new(self, pad):
        # This method is called to produce output for a source pad
        # Get the aligned input frame
        input_frame = self.preparedframes[self.snks["input"]]
        
        # Process each buffer in the frame
        output_buffers = []
        for buf in input_frame:
            if buf.is_gap:
                # Pass through gaps unchanged
                output_buffers.append(buf)
            else:
                # Apply gain to the data
                processed_data = buf.data * self.gain
                
                # Create a new buffer with the processed data
                output_buffer = buf.new(buf.slice, processed_data)
                output_buffers.append(output_buffer)
        
        # Create and return the output frame
        return TSFrame(
            buffers=output_buffers,
            EOS=input_frame.EOS,
            metadata=input_frame.metadata
        )
"""
```

### Handling Alignment

The `AdapterConfig` class allows you to configure how input data is aligned and processed:

```python
# Example of AdapterConfig for different scenarios (not tested by mkdocs)
"""
from sgnts.base import AdapterConfig, NumpyBackend

# Basic configuration with no special handling
basic_config = AdapterConfig()

# Configuration for a filter with overlap (padding)
filter_config = AdapterConfig(
    overlap=(16, 16),       # Pad 16 samples before and after
    stride=1024,            # Process in blocks of 1024 samples
    pad_zeros_startup=True, # Pad zeros at the start
    skip_gaps=True,         # Skip processing if there are gaps
    backend=NumpyBackend    # Use NumPy for array operations
)

# Configuration for overlapping windows with 50% overlap
window_config = AdapterConfig(
    overlap=(0, 512),       # Overlap of 512 samples
    stride=512,             # Stride of 512 samples (50% overlap)
    pad_zeros_startup=False # Don't pad zeros at startup
)
"""
```

## Working with Sinks

Sinks are components that consume data, typically for output or storage. The `TSSink` class provides:

1. Alignment of input data (like transforms)
2. Final processing or handling of aligned data

### Creating a Custom Sink

```python
# Example of creating a custom sink (not tested by mkdocs)
"""
from sgnts.base import TSSink, AdapterConfig
import numpy as np

class PrintingSink(TSSink):
    def __init__(self, **kwargs):
        super().__init__(
            sink_pad_names=["input"],
            adapter_config=AdapterConfig(
                stride=1024,  # Process in blocks of 1024 samples
            ),
            **kwargs
        )
        
    def internal(self):
        # Call the parent's internal method to align data
        super().internal()
        
        # Get the aligned input frame
        input_frame = self.preparedframes[self.snks["input"]]
        
        # Process each buffer in the frame
        for buf in input_frame:
            if not buf.is_gap and buf.samples > 0:
                # Print some statistics about the data
                print(f"Offset: {buf.offset}, Samples: {buf.samples}")
                if buf.data is not None:
                    print(f"  Mean: {np.mean(buf.data)}")
                    print(f"  Min: {np.min(buf.data)}")
                    print(f"  Max: {np.max(buf.data)}")
"""
```

## Advanced Topics

### Data Gaps

Time-series data often contains gaps. SGN-TS provides mechanisms to handle these gaps:

```python
# Example of handling gaps (not tested by mkdocs)
"""
# Creating a buffer with a gap
gap_buffer = SeriesBuffer(
    offset=Offset.fromsec(1.0),
    sample_rate=1000,
    data=None,            # None indicates a gap
    shape=(1000,)         # Still need to specify the shape
)

# Checking if a buffer is a gap
is_gap = gap_buffer.is_gap  # True

# Configuring a transform to skip processing when gaps are present
transform = MyTransform(
    adapter_config=AdapterConfig(
        skip_gaps=True  # Skip processing if there are gaps
    )
)
"""
```

### Real-time Processing

For real-time applications, you can use `TSResourceSource` with a thread that continuously pulls data:

```python
# Example of real-time processing (not tested by mkdocs)
"""
class RealTimeSource(TSResourceSource):
    def __init__(self, device_name, **kwargs):
        super().__init__(
            start_time=None,  # None means start at current time
            source_pad_names=["output"],
            **kwargs
        )
        self.device_name = device_name
        
    def get_data(self):
        # Set up the data acquisition device
        device = open_data_device(self.device_name)
        
        # Continue until stopped
        while not self.stop_event.is_set():
            # Read a chunk of data from the device
            timestamp, data = device.read_chunk()
            
            # Create a buffer and yield it
            buf = SeriesBuffer(
                offset=Offset.fromns(timestamp),
                data=data,
                sample_rate=device.sample_rate
            )
            yield self.srcs["output"], buf
"""
```

### Working with Multiple Inputs

Transforms can have multiple inputs that need to be aligned:

```python
# Example of a transform with multiple inputs (not tested by mkdocs)
"""
class AddTransform(TSTransform):
    def __init__(self, **kwargs):
        super().__init__(
            sink_pad_names=["input1", "input2"],
            source_pad_names=["output"],
            **kwargs
        )
        
    def new(self, pad):
        # Get the aligned input frames
        input1 = self.preparedframes[self.snks["input1"]]
        input2 = self.preparedframes[self.snks["input2"]]
        
        # Process buffers from both inputs
        output_buffers = []
        for i, (buf1, buf2) in enumerate(zip(input1, input2)):
            if buf1.is_gap or buf2.is_gap:
                # If either input has a gap, output a gap
                output_buffers.append(buf1.clone_as_gap())
            else:
                # Add the data from both inputs
                result_data = buf1.data + buf2.data
                output_buffers.append(buf1.new(buf1.slice, result_data))
        
        # Create and return the output frame
        return TSFrame(
            buffers=output_buffers,
            EOS=input1.EOS or input2.EOS
        )
"""
```

## Best Practices

### Memory Management

- Use stride parameters to control memory usage in long-running pipelines
- Flush processed data regularly in resource sources
- Be mindful of buffer sizes and avoid accumulating large amounts of data

### Performance Optimization

- Use appropriate backends (NumPy, Torch) for your data processing needs
- Consider using fixed-stride processing to enable efficient batching
- Configure overlap parameters carefully to minimize redundant computations

### Error Handling

- Check for gaps in data and handle them appropriately
- Use timeout mechanisms to detect stalled pipelines
- Implement proper error handling in threaded components

## Conclusion

The `base.py` module provides the foundation for building complex time-series processing pipelines in SGN-TS. By understanding the core concepts of sources, transforms, and sinks, along with the alignment mechanisms, you can create efficient and flexible pipelines for a wide range of applications.

## Next Steps

- Explore [AudioAdapter](audioadapter_tutorial.md) for more advanced buffer management
- Learn about [Array Backends](array_backend_tutorial.md) for different computational frameworks
- Dive into specific [transforms](../api/transforms/) for common signal processing operations