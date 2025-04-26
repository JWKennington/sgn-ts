# FakeSeriesSource

The `FakeSeriesSource` generates synthetic time-series data in fixed-size buffers, making it ideal for testing and development purposes.

## Overview

`FakeSeriesSource` is a versatile source component that can generate various types of test signals:
- White noise
- Sine waves
- Impulse signals
- Constant values

It can also operate in real-time mode, simulating data that arrives at the actual sample rate.

## Basic Usage

```python
# Basic usage of FakeSeriesSource (not tested by mkdocs)
"""
from sgnts.sources import FakeSeriesSource

# Create a white noise source at 2048 Hz
source = FakeSeriesSource(
    rate=2048,           # Sample rate
    sample_shape=(),     # Shape of each sample (empty tuple for 1D data)
    signal_type="white", # Type of signal to generate
)

# Get a frame from the source
frame = source.pull()

# Access the data from the frame
for buf in frame:
    data = buf.data  # NumPy array containing the generated data
    print(f"Buffer shape: {data.shape}")
    print(f"Time range: {buf.t0} to {buf.t0 + buf.duration}")
"""
```

## Signal Types

### White Noise

```python
# White noise example (not tested by mkdocs)
"""
from sgnts.sources import FakeSeriesSource

white_noise = FakeSeriesSource(
    rate=2048,
    signal_type="white",
    random_seed=42  # For reproducibility
)
"""
```

### Sine Wave

```python
# Sine wave example (not tested by mkdocs)
"""
from sgnts.sources import FakeSeriesSource

sine_wave = FakeSeriesSource(
    rate=2048,
    signal_type="sine",
    fsin=5.0  # Frequency in Hz
)
"""
```

### Impulse Signal

```python
# Impulse signal example (not tested by mkdocs)
"""
from sgnts.sources import FakeSeriesSource

impulse = FakeSeriesSource(
    rate=2048,
    signal_type="impulse",
    impulse_position=1024  # Position of the impulse
)
"""
```

### Constant Signal

```python
# Constant signal example (not tested by mkdocs)
"""
from sgnts.sources import FakeSeriesSource

constant = FakeSeriesSource(
    rate=2048,
    signal_type="const",
    const=1.5  # Constant value
)
"""
```

## Real-time Simulation

`FakeSeriesSource` can simulate real-time data generation by delaying the emission of frames to match the wall clock:

```python
# Real-time simulation example (not tested by mkdocs)
"""
from sgnts.sources import FakeSeriesSource

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

## Multi-dimensional Data

You can generate multi-dimensional data by specifying the `sample_shape`:

```python
# Multi-dimensional data example (not tested by mkdocs)
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

## Gap Generation

`FakeSeriesSource` can generate gap buffers at specified intervals:

```python
# Gap generation example (not tested by mkdocs)
"""
from sgnts.sources import FakeSeriesSource

# Generate gap buffers every 3 frames
source_with_gaps = FakeSeriesSource(
    rate=2048,
    signal_type="white",
    ngap=3  # Generate a gap buffer every 3 buffers
)

# Generate random gaps
random_gap_source = FakeSeriesSource(
    rate=2048,
    signal_type="white",
    ngap=-1  # Generate gap buffers randomly
)
"""
```

## Best Practices

When using `FakeSeriesSource`:

1. **Use random seeds** for reproducible test data when needed
2. **Match sample rates** with downstream components
3. **Consider buffer size** to ensure it meets the needs of your pipeline
4. **Be aware of real-time constraints** - when using `real_time=True`, ensure that processing completes within the frame duration
5. **Use metadata** from frames to track signal properties, especially for impulse signals where the exact position is recorded in the metadata