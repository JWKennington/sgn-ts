# Adder Transform

The `Adder` transform combines data from multiple sources by adding them together.

## Overview

`Adder` is a versatile transform that:
- Takes input from multiple sink pads
- Adds the data arrays element-wise
- Supports selective addition through array slicing
- Handles gap buffers appropriately
- Can work with different array backends

## Basic Usage

```python
# Basic usage of Adder (not tested by mkdocs)
"""
from sgnts.transforms import Adder
from sgnts.sources import FakeSeriesSource

# Create two sources with the same rate and buffer size
source1 = FakeSeriesSource(
    rate=2048,
    signal_type="sine",
    fsin=10.0,
    name="sine_source"
)

source2 = FakeSeriesSource(
    rate=2048,
    signal_type="sine",
    fsin=5.0,
    name="second_sine_source"
)

# Create an adder
adder = Adder(name="my_adder")

# Create sink pads for the adder
sink_pad1 = adder.create_pad("input1:snk")
sink_pad2 = adder.create_pad("input2:snk")

# Connect sources to the adder
source1.add_dest(adder, sink_pad1)
source2.add_dest(adder, sink_pad2)

# Create a destination for the adder output
# (omitted for brevity)

# Process data
for _ in range(5):
    source1.process()
    source2.process()
    adder.process()
    # Process next element in the pipeline
"""
```

## Using Array Slices for Selective Addition

The `Adder` can selectively add parts of arrays using the `addslices_map` parameter:

```python
# Selective addition with slices (not tested by mkdocs)
"""
from sgnts.transforms import Adder
from sgnts.sources import FakeSeriesSource
from sgnts.base import NumpyBackend

# Create a source with 2D data (4x2 matrix at each time point)
source1 = FakeSeriesSource(
    rate=2048,
    sample_shape=(4, 2),
    signal_type="white",
    name="matrix_source"
)

# Create a source with smaller shape (2x2 matrix)
source2 = FakeSeriesSource(
    rate=2048,
    sample_shape=(2, 2),
    signal_type="const",
    const=1.0,
    name="const_source"
)

# Create an adder with selective addition
# Only add source2 to a slice of source1 (the first 2 rows)
adder = Adder(
    backend=NumpyBackend,
    addslices_map={"input1": (slice(0, 2), slice(None))},
    name="slice_adder"
)

# Create sink pads
sink_pad1 = adder.create_pad("input1:snk")
sink_pad2 = adder.create_pad("input2:snk")

# Connect sources
source1.add_dest(adder, sink_pad1)
source2.add_dest(adder, sink_pad2)

# Process data (output element omitted for brevity)
for _ in range(5):
    source1.process()
    source2.process()
    adder.process()
"""
```

## Handling Gap Buffers

The `Adder` treats gaps in a special way:
- If all inputs are gaps, the output will be a gap
- If some inputs are gaps, only the non-gap data is used in the addition

```python
# Handling gaps example (not tested by mkdocs)
"""
from sgnts.transforms import Adder
from sgnts.sources import FakeSeriesSource, SegmentSource

# Create a continuous source
source1 = FakeSeriesSource(
    rate=2048,
    signal_type="sine",
    fsin=10.0
)

# Create a source with gaps
segments = ((1000000000, 2000000000), (3000000000, 4000000000))
source2 = SegmentSource(
    rate=2048,
    segments=segments,
    t0=0,
    end=5
)

# Create an adder
adder = Adder()

# Connect sources (sink pad creation omitted for brevity)
source1.add_dest(adder)
source2.add_dest(adder)

# In the output:
# - When source2 produces gap buffers, only source1 data will be in the output
# - When source2 produces non-gap buffers, the data will be added to source1
"""
```

## Using Different Array Backends

```python
# Using different array backends (not tested by mkdocs)
"""
from sgnts.transforms import Adder
from sgnts.sources import FakeSeriesSource
from sgnts.base import NumpyBackend

# Create an adder with the numpy backend
adder = Adder(backend=NumpyBackend)

# Note: If using the TorchBackend, the behavior would be the same
# but operations would be performed using PyTorch tensors internally
"""
```

## Integration in a Complex Pipeline

```python
# Complex pipeline integration (not tested by mkdocs)
"""
from sgnts.transforms import Adder, AmplifyTransform
from sgnts.sources import FakeSeriesSource
from sgnts.sinks import DumpSeriesSink

# Create two sources
source1 = FakeSeriesSource(
    rate=2048,
    signal_type="sine",
    fsin=10.0,
    name="sine_source"
)

source2 = FakeSeriesSource(
    rate=2048,
    signal_type="sine",
    fsin=5.0,
    name="second_sine_source"
)

# Create an amplifier for the second source
amplifier = AmplifyTransform(factor=2.0, name="amplifier")

# Create an adder
adder = Adder(name="adder")

# Create a sink
sink = DumpSeriesSink(fname="combined_signal.txt", name="sink")

# Connect the pipeline
source2.add_dest(amplifier)
source1.add_dest(adder)
amplifier.add_dest(adder)
adder.add_dest(sink)

# Run the pipeline
for _ in range(10):
    source1.process()
    source2.process()
    amplifier.process()
    adder.process()
    sink.process()
"""
```

## Best Practices

When using `Adder`:

1. **Ensure compatible sample rates** - all input sources must have the same sample rate

2. **Align frames** - the buffers from different sources must be aligned in time

3. **Check shape compatibility** - when not using `addslices_map`, all input shapes must be the same

4. **Use appropriate backend** - choose the backend that matches your processing pipeline (numpy, torch, etc.)

5. **Consider synchronization** - in real-time applications, ensure that all inputs are properly synchronized before addition