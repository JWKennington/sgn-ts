# TSPlotSink Tutorial

`TSPlotSink` is a convenient sink that collects frames during pipeline execution and provides built-in plotting methods for visualizing the results. It extends `TSFrameCollectSink` with plotting capabilities.

## Installation

Plotting requires matplotlib as an optional dependency:

```bash
pip install sgn-ts[plot]
```

## Quick Start

The simplest way to visualize pipeline output:

```{.python notest}
import matplotlib.pyplot as plt
from sgn import Pipeline
from sgnts.sources import FakeSeriesSource
from sgnts.sinks import TSPlotSink

# Create source and plot sink
source = FakeSeriesSource(
    name="source",
    source_pad_names=("data",),
    rate=256,
    signal_type="sine",
    fsin=5.0,
    t0=0,
    end=2.0,
)

sink = TSPlotSink(name="output", sink_pad_names=("data",))

# Build and run pipeline
pipeline = Pipeline()
pipeline.connect(source, sink)
pipeline.run()

# Plot the collected data
fig, ax = sink.plot(time_unit="s")
ax.set_ylabel("Amplitude")
ax.legend()
plt.show()
```

## Multiple Pads

`TSPlotSink` handles multiple input pads, making it easy to compare signals.

### Overlay Mode (Default)

All pads are plotted on the same axes:

```{.python notest}
import matplotlib.pyplot as plt
from sgn import Pipeline
from sgnts.sources import FakeSeriesSource
from sgnts.sinks import TSPlotSink

# Source with multiple output pads, each with different frequency
source = FakeSeriesSource(
    name="source",
    source_pad_names=("H1", "L1", "V1"),
    signals={
        "H1": {"signal_type": "sine", "rate": 256, "fsin": 2.0},
        "L1": {"signal_type": "sine", "rate": 256, "fsin": 3.0},
        "V1": {"signal_type": "sine", "rate": 256, "fsin": 5.0},
    },
    t0=0,
    end=2.0,
)

sink = TSPlotSink(name="Detector Comparison", sink_pad_names=("H1", "L1", "V1"))

pipeline = Pipeline()
pipeline.connect(source, sink)
pipeline.run()

# All three signals overlaid on same axes
# Labels default to pad names, title defaults to element name
fig, ax = sink.plot(time_unit="s")
ax.legend()
ax.set_ylabel("Strain")
plt.show()
```

### Subplots Mode

Each pad gets its own subplot, stacked vertically with a shared x-axis:

```{.python notest}
import matplotlib.pyplot as plt
from sgn import Pipeline
from sgnts.sources import FakeSeriesSource
from sgnts.sinks import TSPlotSink

# Different signal types for each pad
source = FakeSeriesSource(
    name="source",
    source_pad_names=("H1", "L1", "V1"),
    signals={
        "H1": {"signal_type": "sine", "rate": 256, "fsin": 3.0},
        "L1": {"signal_type": "white", "rate": 256},
        "V1": {"signal_type": "const", "rate": 256, "const": 0.5},
    },
    t0=0,
    end=2.0,
)

sink = TSPlotSink(name="Detector Comparison", sink_pad_names=("H1", "L1", "V1"))

pipeline = Pipeline()
pipeline.connect(source, sink)
pipeline.run()

# Each pad in its own subplot
fig, axes = sink.plot(layout="subplots", time_unit="s")
plt.tight_layout()
plt.show()
```

## Smart Defaults

`TSPlotSink` uses metadata to provide sensible defaults:

| Attribute | Default Value |
|-----------|---------------|
| Labels | Pad names (e.g., "H1", "L1") |
| Title | Element name (for multi-pad plots) |
| Figure size | `(10, 4)` for overlay, `(10, 2.5 * n_pads)` for subplots |

## Custom Labels

Override the default labels with a dictionary:

```{.python notest}
import matplotlib.pyplot as plt
from sgn import Pipeline
from sgnts.sources import FakeSeriesSource
from sgnts.sinks import TSPlotSink

source = FakeSeriesSource(
    name="source",
    source_pad_names=("H1", "L1"),
    rate=256,
    signal_type="sine",
    fsin=5.0,
    t0=0,
    end=2.0,
)

sink = TSPlotSink(name="plot", sink_pad_names=("H1", "L1"))

pipeline = Pipeline()
pipeline.connect(source, sink)
pipeline.run()

fig, ax = sink.plot(
    labels={"H1": "Hanford", "L1": "Livingston"},
    title="LIGO Detectors",
    time_unit="s",
)
ax.legend()
plt.show()
```

## Plotting a Subset of Pads

Plot only specific pads using the `pads` parameter:

```{.python notest}
import matplotlib.pyplot as plt
from sgn import Pipeline
from sgnts.sources import FakeSeriesSource
from sgnts.sinks import TSPlotSink

source = FakeSeriesSource(
    name="source",
    source_pad_names=("raw", "filtered", "residual"),
    rate=256,
    signal_type="sine",
    fsin=5.0,
    t0=0,
    end=2.0,
)

sink = TSPlotSink(name="plot", sink_pad_names=("raw", "filtered", "residual"))

pipeline = Pipeline()
pipeline.connect(source, sink)
pipeline.run()

# Only plot two of the three pads
fig, ax = sink.plot(pads=["raw", "filtered"], time_unit="s")
ax.legend()
plt.show()
```

## Gap Visualization

Gaps in the data are automatically shown as shaded regions:

```{.python notest}
import matplotlib.pyplot as plt
from sgn import Pipeline
from sgnts.sources import FakeSeriesSource
from sgnts.sinks import TSPlotSink

# Source that generates gaps every 3rd buffer
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

sink = TSPlotSink(name="plot", sink_pad_names=("data",))

pipeline = Pipeline()
pipeline.connect(source, sink)
pipeline.run()

# Gaps shown as red shaded regions
fig, ax = sink.plot(
    time_unit="s",
    show_gaps=True,
    gap_color="red",
    gap_alpha=0.3,
)
ax.set_title("Signal with Gaps")
plt.show()
```

To hide gaps:

```{.python notest}
fig, ax = sink.plot(show_gaps=False)
```

## Plotting on Existing Axes

Add pipeline output to an existing figure:

```{.python notest}
import matplotlib.pyplot as plt
from sgn import Pipeline
from sgnts.sources import FakeSeriesSource
from sgnts.sinks import TSPlotSink

source = FakeSeriesSource(
    name="source",
    source_pad_names=("data",),
    rate=256,
    signal_type="sine",
    fsin=5.0,
    t0=0,
    end=2.0,
)

sink = TSPlotSink(name="plot", sink_pad_names=("data",))

pipeline = Pipeline()
pipeline.connect(source, sink)
pipeline.run()

# Create figure with custom setup
fig, ax = plt.subplots(figsize=(12, 5))
ax.axhline(0, color="gray", linestyle="--", label="Reference")

# Add pipeline data to existing axes
sink.plot(ax=ax, time_unit="s")
ax.legend()
ax.set_title("Custom Figure")
plt.show()
```

## Time Units

Control the x-axis time representation:

```{.python notest}
# Relative time in seconds
fig, ax = sink.plot(time_unit="s")

# Milliseconds
fig, ax = sink.plot(time_unit="ms")

# Nanoseconds
fig, ax = sink.plot(time_unit="ns")

# GPS time (default)
fig, ax = sink.plot(time_unit="gps")
```

## Custom Figure Size

Override the automatic figure size:

```{.python notest}
# Custom size for overlay
fig, ax = sink.plot(figsize=(14, 6))

# Custom size for subplots
fig, axes = sink.plot(layout="subplots", figsize=(12, 10))
```

## Complete Pipeline Example

Here's a complete example showing a processing pipeline with visualization:

```{.python notest}
import matplotlib.pyplot as plt
from sgn import Pipeline
from sgnts.sources import FakeSeriesSource
from sgnts.transforms import Amplify
from sgnts.sinks import TSPlotSink

# Create source with gaps
source = FakeSeriesSource(
    name="source",
    source_pad_names=("H1",),
    rate=256,
    signal_type="sine",
    fsin=3.0,
    ngap=4,
    t0=0,
    end=5.0,
)

# Amplify the signal
amp = Amplify(
    name="amp",
    sink_pad_names=("H1",),
    source_pad_names=("H1",),
    factor=2.0,
)

# Collect and plot
sink = TSPlotSink(name="Amplified Output", sink_pad_names=("H1",))

# Build pipeline
pipeline = Pipeline()
pipeline.connect(source, amp)
pipeline.connect(amp, sink)
pipeline.run()

# Visualize results
fig, ax = sink.plot(
    time_unit="s",
    gap_color="red",
    gap_alpha=0.2,
)

ax.set_ylabel("Amplitude")
ax.set_title("Pipeline Output with Gaps")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## Accessing Raw Frames

`TSPlotSink` extends `TSFrameCollectSink`, so you can still access the raw frames for custom processing:

```{.python notest}
# Get collected frames as dict[pad_name, TSFrame]
frames = sink.out_frames()

# Access specific pad's frame
h1_frame = frames["H1"]
print(f"Samples: {h1_frame.samples}")
print(f"Duration: {h1_frame.duration / 1e9:.1f} seconds")

# Get concatenated data (gaps filled with zeros)
data = h1_frame.filleddata()
```

## API Reference

See the [TSPlotSink API Reference](../api/sinks/plot.md) for complete documentation of all parameters and methods.

## See Also

- [Plotting Tutorial](plotting_tutorial.md) - Plotting TSFrame and SeriesBuffer directly
- [TSFrameCollectSink Tutorial](collect_sink_tutorial.md) - Base collection functionality
- [Building Pipelines](pipeline_tutorial.md) - Complete pipeline guide
