# FakeSeriesSource Tutorial

The `FakeSeriesSource` generates synthetic time-series data in fixed-size buffers, making it ideal for testing and development purposes.

## Overview

`FakeSeriesSource` is a versatile source component that can generate various types of test signals:

- **White noise** - Random Gaussian noise
- **Sine waves** - Periodic sinusoidal signals
- **Impulse signals** - Single-sample pulses
- **Constant values** - Flat signal at a specified level

It can also operate in real-time mode, simulating data that arrives at the actual sample rate.

## Quick Start

Here's a minimal example to get started:

```python
from sgnts.sources import FakeSeriesSource

# Create a source that generates white noise at 2048 Hz
source = FakeSeriesSource(
    name="my_source",
    source_pad_names=("data",),
    rate=2048,
    signal_type="white",
)

# Get the source pad and pull a frame
pad = source.srcs["data"]
frame = source.new(pad)

# Inspect the generated data
for buf in frame:
    print(f"Buffer shape: {buf.shape}")
    print(f"Sample rate: {buf.sample_rate} Hz")
    print(f"Duration: {buf.duration / 1e9:.3f} seconds")
    if buf.data is not None:
        print(f"Data range: [{buf.data.min():.3f}, {buf.data.max():.3f}]")
```

## Signal Types

### White Noise

White noise generates random samples from a standard normal distribution (mean=0, std=1):

```python
from sgnts.sources import FakeSeriesSource

source = FakeSeriesSource(
    name="noise_source",
    source_pad_names=("data",),
    rate=2048,
    signal_type="white",
    random_seed=42,  # For reproducible results
)

pad = source.srcs["data"]
frame = source.new(pad)

for buf in frame:
    if buf.data is not None:
        print(f"Mean: {buf.data.mean():.4f}")  # Close to 0
        print(f"Std: {buf.data.std():.4f}")    # Close to 1
```

### Sine Wave

Generate a sinusoidal signal at a specified frequency:

```python
from sgnts.sources import FakeSeriesSource

source = FakeSeriesSource(
    name="sine_source",
    source_pad_names=("data",),
    rate=2048,
    signal_type="sine",
    fsin=10.0,  # 10 Hz sine wave
)

pad = source.srcs["data"]
frame = source.new(pad)

for buf in frame:
    if buf.data is not None:
        # Sine wave oscillates between -1 and 1
        print(f"Min: {buf.data.min():.4f}")
        print(f"Max: {buf.data.max():.4f}")
```

### Impulse Signal

Create a signal with a single non-zero sample (useful for testing filters):

```python
from sgnts.sources import FakeSeriesSource

source = FakeSeriesSource(
    name="impulse_source",
    source_pad_names=("data",),
    rate=2048,
    signal_type="impulse",
    impulse_position=100,  # Impulse at sample 100
    end=1.0,  # Run for 1 second
)

pad = source.srcs["data"]
frame = source.new(pad)

for buf in frame:
    if buf.data is not None:
        nonzero = (buf.data != 0).sum()
        print(f"Non-zero samples: {nonzero}")
```

### Constant Signal

Generate a flat signal at a specified value:

```python
from sgnts.sources import FakeSeriesSource

source = FakeSeriesSource(
    name="const_source",
    source_pad_names=("data",),
    rate=2048,
    signal_type="const",
    const=2.5,  # All samples will be 2.5
)

pad = source.srcs["data"]
frame = source.new(pad)

for buf in frame:
    if buf.data is not None:
        print(f"All values equal to 2.5: {(buf.data == 2.5).all()}")
```

## Multi-Pad Sources

You can configure different signals for different output pads using the `signals` parameter:

```python
from sgnts.sources import FakeSeriesSource

source = FakeSeriesSource(
    name="multi_source",
    source_pad_names=("channel_a", "channel_b"),
    signals={
        "channel_a": {
            "signal_type": "sine",
            "rate": 2048,
            "fsin": 5.0,
        },
        "channel_b": {
            "signal_type": "white",
            "rate": 1024,
        },
    },
)

# Each pad has its own signal configuration
for pad_name in ["channel_a", "channel_b"]:
    pad = source.srcs[pad_name]
    frame = source.new(pad)
    for buf in frame:
        print(f"{pad_name}: rate={buf.sample_rate}, shape={buf.shape}")
```

## Multi-dimensional Data

Generate data with multiple dimensions per sample by specifying `sample_shape`:

```python
from sgnts.sources import FakeSeriesSource

# Generate 3-channel data (e.g., accelerometer with X, Y, Z axes)
source = FakeSeriesSource(
    name="3d_source",
    source_pad_names=("accel",),
    rate=256,
    sample_shape=(3,),  # 3 values per time sample
    signal_type="white",
)

pad = source.srcs["accel"]
frame = source.new(pad)

for buf in frame:
    if buf.data is not None:
        # Shape is (3, N) where N is number of time samples
        print(f"Data shape: {buf.data.shape}")
        print(f"Sample shape: {buf.sample_shape}")
```

For matrix data:

```python
from sgnts.sources import FakeSeriesSource

# Generate 2x4 matrix at each time point
source = FakeSeriesSource(
    name="matrix_source",
    source_pad_names=("matrix",),
    rate=128,
    sample_shape=(2, 4),
    signal_type="white",
)

pad = source.srcs["matrix"]
frame = source.new(pad)

for buf in frame:
    if buf.data is not None:
        # Shape is (2, 4, N) where N is number of time samples
        print(f"Data shape: {buf.data.shape}")
```

## Gap Generation

`FakeSeriesSource` can generate gap buffers (buffers with `data=None`) to simulate missing data:

```python
from sgnts.sources import FakeSeriesSource

# Generate a gap every 3 buffers
source = FakeSeriesSource(
    name="gappy_source",
    source_pad_names=("data",),
    rate=2048,
    signal_type="white",
    ngap=3,  # Gap every 3rd buffer
    end=5.0,
)

pad = source.srcs["data"]
gap_count = 0
data_count = 0

for _ in range(10):
    frame = source.new(pad)
    for buf in frame:
        if buf.is_gap:
            gap_count += 1
        else:
            data_count += 1
    if frame.EOS:
        break

print(f"Data buffers: {data_count}, Gap buffers: {gap_count}")
```

For random gaps, use `ngap=-1`:

```python
from sgnts.sources import FakeSeriesSource

source = FakeSeriesSource(
    name="random_gap_source",
    source_pad_names=("data",),
    rate=2048,
    signal_type="white",
    ngap=-1,  # Random gaps (~50% probability)
    random_seed=42,
)
```

## Finite Duration Sources

Specify `end` to create a source that produces an End-of-Stream (EOS) signal after a certain time:

```python
from sgnts.sources import FakeSeriesSource

source = FakeSeriesSource(
    name="finite_source",
    source_pad_names=("data",),
    rate=2048,
    signal_type="sine",
    fsin=10.0,
    t0=0,      # Start at time 0
    end=2.0,   # End after 2 seconds
)

pad = source.srcs["data"]
frame_count = 0

while True:
    frame = source.new(pad)
    frame_count += 1
    if frame.EOS:
        print(f"EOS reached after {frame_count} frames")
        break
```

## Complete Pipeline Example

Here's a complete example showing `FakeSeriesSource` in a pipeline with a transform and sink:

```python
from sgn import Pipeline
from sgnts.sources import FakeSeriesSource
from sgnts.transforms import Amplify
from sgnts.sinks import TSFrameCollectSink

# Create source
source = FakeSeriesSource(
    name="source",
    source_pad_names=("H1",),
    rate=256,
    signal_type="sine",
    fsin=5.0,
    t0=0,
    end=1.0,  # 1 second of data
)

# Create amplifier transform
amp = Amplify(
    name="amplifier",
    sink_pad_names=("H1",),
    source_pad_names=("H1",),
    factor=2.0,
)

# Create sink to collect results
sink = TSFrameCollectSink(
    name="sink",
    sink_pad_names=("H1",),
)

# Build and run pipeline
pipeline = Pipeline()
pipeline.connect(source, amp)
pipeline.connect(amp, sink)
pipeline.run()

# Access the collected data
out_frame = sink.out_frames()["H1"]
print(f"Collected {out_frame.samples} samples")
print(f"Duration: {out_frame.duration / 1e9:.1f} seconds")
print(f"Data range: [{out_frame.filleddata().min():.2f}, {out_frame.filleddata().max():.2f}]")
```

## Real-time Mode

Enable real-time mode to simulate data arriving at the actual sample rate. In this mode, the source's `internal()` method blocks until the wall-clock time matches the frame's end time. This blocking occurs during pipeline execution:

```{.python notest}
import time
from sgn import Pipeline
from sgnts.sources import FakeSeriesSource
from sgnts.sinks import TSFrameCollectSink
from sgnts.utils import gpsnow

# Get current GPS time for t0 and set end 3 seconds later
t0 = int(gpsnow())

# Create a real-time source
source = FakeSeriesSource(
    name="realtime_source",
    source_pad_names=("data",),
    rate=2048,
    signal_type="sine",
    fsin=10.0,
    real_time=True,  # Enable real-time mode
    t0=t0,
    end=t0 + 5,      # Run for 5 seconds
)

sink = TSFrameCollectSink(
    name="sink",
    sink_pad_names=("data",),
)

pipeline = Pipeline()
pipeline.connect(source, sink)

# The pipeline will take ~5 seconds to run (real-time pacing)
start = time.time()
pipeline.run()
elapsed = time.time() - start

print(f"Pipeline ran for {elapsed:.2f}s (expected ~5s for real-time mode)")
```

Example output:

```text
Pipeline ran for 5.01s (expected ~5s for real-time mode)
```

!!! note
    Real-time mode is useful for testing pipelines that need to process data at realistic rates. The blocking happens in the source's `internal()` method during pipeline execution, ensuring frames are released at the appropriate wall-clock time.

## Configuration Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | required | Element name |
| `source_pad_names` | tuple | required | Names of output pads |
| `rate` | int | 2048 | Sample rate in Hz |
| `sample_shape` | tuple | () | Shape of each sample (empty for 1D) |
| `signal_type` | str | "white" | Signal type: "white", "sine", "impulse", "const" |
| `signals` | dict | None | Per-pad signal configuration |
| `fsin` | float | 5.0 | Sine wave frequency (Hz) |
| `impulse_position` | int | -1 | Impulse position (-1 for random) |
| `const` | float | 1.0 | Constant signal value |
| `ngap` | int | 0 | Gap frequency (0=none, -1=random, N=every Nth buffer) |
| `random_seed` | int | None | Random seed for reproducibility |
| `t0` | float | 0 | Start time (GPS seconds, or current time if `real_time=True`) |
| `end` | float | None | End time (GPS seconds, None for infinite) |
| `real_time` | bool | False | Enable real-time mode |

## Best Practices

1. **Use random seeds** for reproducible test results:
   ```{.python notest}
   source = FakeSeriesSource(..., random_seed=42)
   ```

2. **Match sample rates** with downstream components to avoid resampling issues

3. **Set `end` for finite tests** to ensure pipelines terminate cleanly

4. **Use meaningful names** to make debugging easier:
   ```{.python notest}
   source = FakeSeriesSource(name="h1_test_signal", ...)
   ```

5. **Check for gaps** when using `ngap` - downstream components should handle gap buffers appropriately

## See Also

- [SegmentSource Tutorial](segment_source_tutorial.md) - For creating data with specific time segments
- [Building Pipelines](pipeline_tutorial.md) - Complete guide to pipeline construction
- [API Reference](../api/sources/fake_series.md) - Full API documentation
