# Plotting Tutorial

SGN-TS provides built-in plotting support for visualizing time-series data in `SeriesBuffer` and `TSFrame` objects. This tutorial covers how to use the plotting functionality effectively.

## Installation

Plotting requires matplotlib as an optional dependency. Install it with:

```bash
pip install sgn-ts[plot]
```

## Quick Start

The simplest way to plot a TSFrame is to call its `plot()` method:

```{.python notest}
import matplotlib.pyplot as plt
from sgnts.sources import FakeSeriesSource

# Create a source and get a frame
source = FakeSeriesSource(
    name="source",
    source_pad_names=("data",),
    rate=256,
    signal_type="sine",
    fsin=5.0,
    t0=0,
    end=2.0,
)

pad = source.srcs["data"]
frame = source.new(pad)

# Plot the frame
fig, ax = frame.plot(label="5 Hz Sine", time_unit="s")
ax.legend()
ax.set_title("Simple Plot Example")
plt.show()
```

## Plotting with Gaps

One of the key features of the plotting system is automatic visualization of gap buffers. Gaps are shown as shaded regions:

```{.python notest}
import matplotlib.pyplot as plt
from sgn import Pipeline
from sgnts.sources import FakeSeriesSource
from sgnts.sinks import TSFrameCollectSink

# Create source with gaps every 3rd buffer
source = FakeSeriesSource(
    name="source",
    source_pad_names=("data",),
    rate=256,
    signal_type="sine",
    fsin=2.0,
    ngap=3,
    t0=0,
    end=5.0,
)

sink = TSFrameCollectSink(name="sink", sink_pad_names=("data",))

pipeline = Pipeline()
pipeline.connect(source, sink)
pipeline.run()

# Get the combined frame
frame = sink.out_frames()["data"]

# Plot with gap visualization
fig, ax = frame.plot(
    label="Sine with gaps",
    color="blue",
    gap_color="red",      # Color for gap shading
    gap_alpha=0.3,        # Transparency of gap shading
    show_gaps=True,       # Enable gap visualization (default)
    time_unit="s",
)
ax.legend()
ax.set_title("Data with Gap Visualization")
plt.show()
```

You can customize gap appearance:

```{.python notest}
# Different gap styling
fig, ax = frame.plot(
    gap_color="gray",
    gap_alpha=0.5,
)

# Disable gap visualization
fig, ax = frame.plot(show_gaps=False)
```

## Time Units

By default, the x-axis shows absolute GPS time. You can change this with the `time_unit` parameter:

```{.python notest}
# Relative time in seconds (from start of frame)
fig, ax = frame.plot(time_unit="s")

# Milliseconds
fig, ax = frame.plot(time_unit="ms")

# Nanoseconds
fig, ax = frame.plot(time_unit="ns")

# GPS time (default)
fig, ax = frame.plot(time_unit="gps")
```

## Plotting Multiple Frames

### Using the Same Axes

To plot multiple frames on the same axes, pass the `ax` parameter to subsequent calls:

```{.python notest}
import matplotlib.pyplot as plt
from sgnts.sources import FakeSeriesSource

# Create two sources with different frequencies
source1 = FakeSeriesSource(
    name="source1",
    source_pad_names=("data",),
    rate=256,
    signal_type="sine",
    fsin=2.0,
    t0=0,
    end=2.0,
)

source2 = FakeSeriesSource(
    name="source2",
    source_pad_names=("data",),
    rate=256,
    signal_type="sine",
    fsin=5.0,
    t0=0,
    end=2.0,
)

frame1 = source1.new(source1.srcs["data"])
frame2 = source2.new(source2.srcs["data"])

# Plot both on same axes
fig, ax = frame1.plot(label="2 Hz", color="blue", time_unit="s")
frame2.plot(ax=ax, label="5 Hz", color="red", time_unit="s")
ax.legend()
ax.set_title("Two Signals Overlaid")
plt.show()
```

### Using plot_frames()

For convenience, you can use the `plot_frames()` function:

```{.python notest}
import matplotlib.pyplot as plt
from sgnts.sources import FakeSeriesSource
from sgnts.plotting import plot_frames

source1 = FakeSeriesSource(
    name="source1",
    source_pad_names=("data",),
    rate=256,
    signal_type="sine",
    fsin=2.0,
    t0=0,
    end=2.0,
)

source2 = FakeSeriesSource(
    name="source2",
    source_pad_names=("data",),
    rate=256,
    signal_type="sine",
    fsin=5.0,
    t0=0,
    end=2.0,
)

frame1 = source1.new(source1.srcs["data"])
frame2 = source2.new(source2.srcs["data"])

fig, ax = plot_frames(
    [frame1, frame2],
    labels=["2 Hz", "5 Hz"],
    time_unit="s",
)
ax.legend()
plt.show()
```

## Multi-Channel Data

For data with multiple channels (e.g., `sample_shape=(3,)`), you have two display options:

### Overlay (Default)

All channels are plotted on the same axes:

```{.python notest}
import matplotlib.pyplot as plt
from sgnts.sources import FakeSeriesSource

source = FakeSeriesSource(
    name="source",
    source_pad_names=("data",),
    rate=256,
    sample_shape=(3,),  # 3 channels
    signal_type="white",
    t0=0,
    end=1.0,
)

frame = source.new(source.srcs["data"])

# Overlay all channels
fig, ax = frame.plot(multichannel="overlay", time_unit="s")
ax.legend()
ax.set_title("3 Channels Overlaid")
plt.show()
```

### Subplots

Each channel gets its own subplot:

```{.python notest}
import matplotlib.pyplot as plt
from sgnts.sources import FakeSeriesSource

source = FakeSeriesSource(
    name="source",
    source_pad_names=("data",),
    rate=256,
    sample_shape=(3,),  # 3 channels
    signal_type="white",
    t0=0,
    end=1.0,
)

frame = source.new(source.srcs["data"])

# Separate subplot for each channel
fig, axes = frame.plot(multichannel="subplots", time_unit="s")
fig.suptitle("3 Channels in Subplots")
plt.show()
```

### Plotting a Specific Channel

To plot only one channel:

```{.python notest}
import matplotlib.pyplot as plt
from sgnts.sources import FakeSeriesSource

source = FakeSeriesSource(
    name="source",
    source_pad_names=("data",),
    rate=256,
    sample_shape=(3,),  # 3 channels
    signal_type="white",
    t0=0,
    end=1.0,
)

frame = source.new(source.srcs["data"])

# Plot only channel 1
fig, ax = frame.plot(channel=1, time_unit="s")
ax.set_title("Channel 1 Only")
plt.show()
```

## Plotting Individual Buffers

You can also plot individual `SeriesBuffer` objects:

```{.python notest}
import matplotlib.pyplot as plt
import numpy as np
from sgnts.base.buffer import SeriesBuffer

# Create a buffer directly
buf = SeriesBuffer(
    offset=0,
    sample_rate=256,
    data=np.sin(2 * np.pi * 5 * np.arange(256) / 256),
    shape=(256,),
)

# Plot the buffer
fig, ax = buf.plot(label="Single buffer", time_unit="s")
ax.legend()
plt.show()
```

## Customizing Plots

All standard matplotlib `plot()` keyword arguments are passed through:

```{.python notest}
import matplotlib.pyplot as plt
from sgnts.sources import FakeSeriesSource

source = FakeSeriesSource(
    name="source",
    source_pad_names=("data",),
    rate=256,
    signal_type="sine",
    fsin=5.0,
    t0=0,
    end=2.0,
)

frame = source.new(source.srcs["data"])

fig, ax = frame.plot(
    label="Custom style",
    color="green",
    linestyle="--",
    linewidth=2,
    alpha=0.8,
    marker="o",
    markersize=2,
    time_unit="s",
)
ax.legend()
ax.set_ylabel("Amplitude")
ax.grid(True)
plt.show()
```

## Complete Pipeline Example

Here's a complete example showing a pipeline with plotting:

```{.python notest}
import matplotlib.pyplot as plt
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
    fsin=3.0,
    ngap=4,  # Add some gaps
    t0=0,
    end=5.0,
)

# Create amplifier
amp = Amplify(
    name="amp",
    sink_pad_names=("H1",),
    source_pad_names=("H1",),
    factor=2.0,
)

# Create sink
sink = TSFrameCollectSink(
    name="sink",
    sink_pad_names=("H1",),
)

# Build and run pipeline
pipeline = Pipeline()
pipeline.connect(source, amp)
pipeline.connect(amp, sink)
pipeline.run()

# Get the output frame
output_frame = sink.out_frames()["H1"]

# Create a nice plot
fig, ax = output_frame.plot(
    label="Amplified signal",
    color="blue",
    gap_color="red",
    gap_alpha=0.2,
    time_unit="s",
)

ax.set_ylabel("Amplitude")
ax.set_title("Pipeline Output with Gaps")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## See Also

- [Plotting API Reference](../api/plotting.md) - Full API documentation
- [FakeSeriesSource Tutorial](fake_series_source_tutorial.md) - Creating test data
- [Building Pipelines](pipeline_tutorial.md) - Complete pipeline guide
- [Buffers](buffers.md) - Understanding SeriesBuffer
- [Frames](frames.md) - Understanding TSFrame
