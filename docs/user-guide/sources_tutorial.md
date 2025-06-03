# Sources in SGN-TS

Sources are components that generate or read time-series data and introduce it into the processing pipeline. This tutorial covers the available source components in SGN-TS and how to use them effectively.

## Overview

Source components in SGN-TS inherit from the `TSSource` base class and implement the `new` method to generate `TSFrame` objects. These frames contain time-series data in the form of `SeriesBuffer` objects.

The SGN-TS library provides several source components:

- `FakeSeriesSource`: Generates synthetic time-series data for testing and development
- `SegmentSource`: Produces non-gap buffers for specified time segments and gap buffers elsewhere

## FakeSeriesSource

The `FakeSeriesSource` generates synthetic time-series data in fixed-size buffers. It's useful for:

- Testing and debugging signal processing pipelines
- Generating controlled test signals (white noise, sine waves, impulses, etc.)
- Simulating real-time data streams

### Basic Usage

```python
# Basic usage of FakeSeriesSource (not tested by mkdocs)
"""
from sgnts.sources import FakeSeriesSource

# Create a white noise source at 2048 Hz
source = FakeSeriesSource(
    rate=2048,           # Sample rate
    sample_shape=(),     # Shape of each sample (empty tuple for 1D data)
    signal_type="white", # Type of signal to generate
    source_pad_names=("pad",),
)

# Get a frame from the source
frame = source.new(source.srcs["pad"])

# Access the data from the frame
for buf in frame:
    data = buf.data  # NumPy array containing the generated data
    print(f"Buffer shape: {data.shape}")
    print(f"Time range: {buf.t0} to {buf.t0 + buf.duration}")
"""
```

### Signal Types

`FakeSeriesSource` supports several types of signals:

```python
# Different signal types in FakeSeriesSource (not tested by mkdocs)
"""
from sgnts.sources import FakeSeriesSource

# White noise signal
white_noise = FakeSeriesSource(
    rate=2048,
    signal_type="white",
    random_seed=42  # For reproducibility
)

# Sine wave at 5 Hz
sine_wave = FakeSeriesSource(
    rate=2048,
    signal_type="sine",
    fsin=5.0  # Frequency in Hz
)

# Impulse signal (one at a specific position, zero elsewhere)
impulse = FakeSeriesSource(
    rate=2048,
    signal_type="impulse",
    impulse_position=1024  # Position of the impulse
)

# Constant signal
constant = FakeSeriesSource(
    rate=2048,
    signal_type="const",
    const=1.5  # Constant value
)
"""
```

### Real-time Simulation

`FakeSeriesSource` can simulate real-time data generation:

```python
# Real-time simulation with FakeSeriesSource (not tested by mkdocs)
"""
from sgnts.sources import FakeSeriesSource

# Create a real-time source
realtime_source = FakeSeriesSource(
    rate=2048,
    signal_type="sine",
    fsin=10.0,
    real_time=True  # Enable real-time mode
)

# When pulling frames, they will be emitted at the appropriate wall-clock time
frame = realtime_source.pull()
"""
```

### Multi-dimensional Data

You can generate multi-dimensional data by specifying the `sample_shape`:

```python
# Multi-dimensional data with FakeSeriesSource (not tested by mkdocs)
"""
from sgnts.sources import FakeSeriesSource

# Create a source for 2x4 matrix at each time point
matrix_source = FakeSeriesSource(
    rate=2048,
    sample_shape=(2, 4),  # Each sample will be a 2x4 matrix
    signal_type="white"
)

# Pull a frame
frame = matrix_source.pull()
for buf in frame:
    # Data will have shape (2, 4, N) where N is the number of time samples
    print(f"Data shape: {buf.data.shape}")
"""
```

## SegmentSource

The `SegmentSource` produces non-gap buffers for specified time segments and gap buffers elsewhere. The enhanced version now supports custom values for each segment. This is useful when you want to:

- Simulate data that's only available during specific time periods
- Create windowed segments of data for processing
- Generate test signals with different values for different time windows

### Basic Usage

```python
# Basic usage of SegmentSource (not tested by mkdocs)
"""
from sgnts.sources import SegmentSource

# Define time segments in nanoseconds (start, end)
segments = (
    (1000000000, 2000000000),  # 1s to 2s
    (3000000000, 4000000000),  # 3s to 4s
)

# Create a segment source
segment_source = SegmentSource(
    rate=2048,
    segments=segments,
    values=(10, 20),  # Optional: custom values for each segment
    t0=0,             # Start time
    end=5             # End time (in seconds)
)

# Note: SegmentSource automatically rounds segment times to the nearest
# sample boundary to ensure proper alignment with the sample rate

# Pull frames
frame = segment_source.pull()

# Buffers within the specified segments will contain data
# Buffers outside the segments will be gap buffers
for buf in frame:
    if buf.is_gap:
        print(f"Gap buffer at {buf.t0}")
    else:
        print(f"Data buffer at {buf.t0}")
"""
```

## Integrating Sources with Transforms and Sinks

Sources can be connected to transforms and sinks to create a complete processing pipeline:

```python
# Complete pipeline example (not tested by mkdocs)
"""
from sgnts.sources import FakeSeriesSource
from sgnts.transforms import AmplifyTransform
from sgnts.sinks import DumpSeriesSink

# Create a source
source = FakeSeriesSource(
    rate=2048,
    signal_type="sine",
    fsin=10.0
)

# Create a transform
amplifier = AmplifyTransform(
    gain=2.0  # Double the amplitude
)

# Create a sink
sink = DumpSeriesSink(
    fname="output.txt"  # Output file
)

# Connect the elements
source.add_dest(amplifier)
amplifier.add_dest(sink)

# Run the pipeline for 10 iterations
for _ in range(10):
    source.process()
    amplifier.process()
    sink.process()
"""
```

## Best Practices

When working with sources:

1. **Choose the appropriate source** for your needs - use `FakeSeriesSource` for testing and `SegmentSource` for windowed data with custom values

2. **Configure sample rate appropriately** - ensure your source's rate matches or is compatible with downstream components

3. **Consider real-time constraints** - when using `real_time=True`, be aware that processing must complete within the frame duration to avoid falling behind

4. **Mind memory usage** - when generating large amounts of data, consider buffer sizes and processing efficiency

5. **Use random seeds** for reproducibility when needed during testing
