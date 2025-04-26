# Gate Transform

The `Gate` transform selectively passes or blocks data based on a control signal, allowing for conditional processing within a pipeline.

## Overview

The `Gate` transform:
- Takes input from two sink pads: one for data and one for control
- Uses the control signal to determine when to pass or block the data
- Works based on whether the control buffers are gaps or not
- Preserves the time alignment of data
- Can be used to implement conditional processing logic

## Basic Usage

```python
# Basic usage of Gate (not tested by mkdocs)
"""
from sgnts.transforms import Gate
from sgnts.sources import FakeSeriesSource, SegmentSource

# Create a data source with continuous signal
data_source = FakeSeriesSource(
    rate=2048,
    signal_type="sine",
    fsin=10.0,
    name="data_source"
)

# Create a control source with specific time segments
# This will produce non-gap buffers only during these segments
segments = ((1000000000, 2000000000), (3000000000, 4000000000))  # 1-2s and 3-4s
control_source = SegmentSource(
    rate=2048,
    segments=segments,
    t0=0,
    end=5,
    name="control_source"
)

# Create a gate transform with control pad specified
gate = Gate(
    control="control",  # Name of the control pad
    name="gate"
)

# Create sink pads for the gate
data_pad = gate.create_pad("data:snk")
control_pad = gate.create_pad("control:snk")

# Connect sources to gate
data_source.add_dest(gate, data_pad)
control_source.add_dest(gate, control_pad)

# Process data
for _ in range(10):
    data_source.process()
    control_source.process()
    gate.process()
    
    # Pull the gated frame
    frame = gate.pull()
    
    # Frame will only contain non-gap buffers during the time segments
    # defined in the control source
"""
```

## How Gate Works

The `Gate` transform splits buffers at the boundaries of non-gap regions in the control signal:

```python
# How Gate works explanation (not tested by mkdocs)
"""
# Gate operation in detail:

# 1. Collect all non-gap slices from the control signal
nongap_slices = TSSlices([buffer.slice for buffer in control_frame if not buffer.is_gap])

# 2. For each buffer in the data signal:
#    - Split the buffer at the boundaries of the non-gap slices
#    - Keep only the parts that overlap with non-gap slices
#    - Discard the parts that don't overlap (or convert them to gaps)

# 3. Return a new frame with the filtered buffers

# This results in data being passed through only when the control
# signal has non-gap buffers
"""
```

## Use Cases

### Conditional Processing

```python
# Conditional processing example (not tested by mkdocs)
"""
from sgnts.transforms import Gate, AmplifyTransform
from sgnts.sources import FakeSeriesSource, SegmentSource
from sgnts.sinks import DumpSeriesSink

# Create a continuous data source
data_source = FakeSeriesSource(
    rate=2048,
    signal_type="sine",
    fsin=10.0
)

# Create a control source that only produces data during specific periods
segments = ((1000000000, 2000000000), (3000000000, 4000000000))  # 1-2s and 3-4s
control_source = SegmentSource(
    rate=2048,
    segments=segments,
    t0=0,
    end=5
)

# Create a gate
gate = Gate(control="control")
data_pad = gate.create_pad("data:snk")
control_pad = gate.create_pad("control:snk")

# Create an amplifier to process gated data
amplifier = AmplifyTransform(factor=2.0)

# Create a sink
sink = DumpSeriesSink(fname="gated_data.txt")

# Connect the pipeline
data_source.add_dest(gate, data_pad)
control_source.add_dest(gate, control_pad)
gate.add_dest(amplifier)
amplifier.add_dest(sink)

# Run the pipeline
for _ in range(10):
    data_source.process()
    control_source.process()
    gate.process()
    amplifier.process()
    sink.process()

# The output file will only contain data from the time periods
# where the control signal was active
"""
```

### Threshold-Based Gating

```python
# Threshold-based gating example (not tested by mkdocs)
"""
from sgnts.transforms import Gate, Threshold
from sgnts.sources import FakeSeriesSource

# Create a data source
data_source = FakeSeriesSource(
    rate=2048,
    signal_type="sine",
    fsin=10.0
)

# Create a control source with varying amplitude
control_source = FakeSeriesSource(
    rate=2048,
    signal_type="sine",
    fsin=0.5  # Slow oscillation
)

# Create a threshold transform to convert the control signal to gaps/non-gaps
threshold = Threshold(
    threshold=0.5,  # Threshold value
    mode="greater"  # Only pass values greater than threshold
)

# Create a gate
gate = Gate(control="control")
data_pad = gate.create_pad("data:snk")
control_pad = gate.create_pad("control:snk")

# Connect the pipeline
data_source.add_dest(gate, data_pad)
control_source.add_dest(threshold)
threshold.add_dest(gate, control_pad)

# Run the pipeline
for _ in range(10):
    data_source.process()
    control_source.process()
    threshold.process()
    gate.process()
    
    # Pull the gated frame
    frame = gate.pull()
    
    # Frame will only contain data when the control signal is above 0.5
"""
```

## Integration with Other Components

```python
# Integration example (not tested by mkdocs)
"""
from sgnts.transforms import Gate, AmplifyTransform, Correlate
from sgnts.sources import FakeSeriesSource
from sgnts.sinks import DumpSeriesSink
import numpy as np

# Create a filter for correlation
filter_width = 32
filter_array = np.hamming(filter_width)

# Create a data source with white noise
data_source = FakeSeriesSource(
    rate=2048,
    signal_type="white"
)

# Create a control source with sine wave
control_source = FakeSeriesSource(
    rate=2048,
    signal_type="sine",
    fsin=0.2  # Very slow oscillation
)

# Create a correlator to process the data
correlator = Correlate(
    sample_rate=2048,
    filters=filter_array
)

# Create a threshold to generate control signal
threshold = Threshold(
    threshold=0.5,
    mode="greater"
)

# Create a gate
gate = Gate(control="control")
data_pad = gate.create_pad("data:snk")
control_pad = gate.create_pad("control:snk")

# Create a sink
sink = DumpSeriesSink(fname="gated_correlation.txt")

# Connect the pipeline
data_source.add_dest(correlator)
correlator.add_dest(gate, data_pad)

control_source.add_dest(threshold)
threshold.add_dest(gate, control_pad)

gate.add_dest(sink)

# Run the pipeline
for _ in range(20):
    data_source.process()
    correlator.process()
    control_source.process()
    threshold.process()
    gate.process()
    sink.process()

# The output file will contain correlated data only when
# the control signal is above the threshold
"""
```

## Best Practices

When using `Gate`:

1. **Match sample rates** - ensure that the data and control sources have compatible sample rates

2. **Consider buffer boundaries** - gates split buffers at the boundaries of non-gap regions, which can result in many small buffers

3. **Use meaningful control signals** - design control signals that correspond to the conditions you want to filter for

4. **Mind performance** - splitting buffers at many points can impact performance

5. **Consider alternatives** - for simple amplitude-based gating, the `Threshold` transform might be sufficient

6. **Chain gates** - you can chain multiple gates for more complex conditional logic