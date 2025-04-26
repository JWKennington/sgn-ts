# Align Transform

The `Align` transform ensures that frames from multiple sources are properly aligned in time.

## Overview

`Align` is a special transform that:
- Maps frames directly from its sink pads to the corresponding source pads
- Ensures that time-aligned frames from multiple sources stay in sync
- Passes through frames without modifying their content

## Basic Usage

```python
# Basic usage of Align (not tested by mkdocs)
"""
from sgnts.transforms import Align
from sgnts.sources import FakeSeriesSource

# Create two sources with the same rate
source1 = FakeSeriesSource(
    rate=2048,
    signal_type="sine",
    fsin=10.0,
    name="sine_source"
)

source2 = FakeSeriesSource(
    rate=2048,
    signal_type="white",
    name="noise_source"
)

# Create an align transform with matching source and sink pad names
align = Align(
    source_pad_names=["channel1", "channel2"],
    sink_pad_names=["channel1", "channel2"],
    name="aligner"
)

# Create and get pads
align_sink_pad1 = align.sink_pad_dict["aligner:snk:channel1"]
align_sink_pad2 = align.sink_pad_dict["aligner:snk:channel2"]
align_source_pad1 = align.source_pad_dict["aligner:src:channel1"]
align_source_pad2 = align.source_pad_dict["aligner:src:channel2"]

# Connect sources to align transform
source1.add_dest(align, align_sink_pad1)
source2.add_dest(align, align_sink_pad2)

# Connect align transform to downstream elements (not shown)

# Process data
for _ in range(5):
    source1.process()
    source2.process()
    align.process()
"""
```

## How Align Works

The `Align` transform creates a direct mapping between its sink pads and source pads:

```python
# Understanding Align (not tested by mkdocs)
"""
from sgnts.transforms import Align

# Create an align transform
align = Align(
    source_pad_names=["signal1", "signal2"],
    sink_pad_names=["signal1", "signal2"],
    name="align_transform"
)

# During initialization, a pad map is created:
# self.pad_map = {
#    source_pad1: sink_pad1,
#    source_pad2: sink_pad2,
# }

# When new() is called for a source pad, it retrieves the frame
# from the corresponding sink pad and returns it unchanged
"""
```

## Synchronizing Multiple Data Streams

A common use case for `Align` is to synchronize multiple data streams before combining them:

```python
# Synchronizing data streams (not tested by mkdocs)
"""
from sgnts.transforms import Align, Adder
from sgnts.sources import FakeSeriesSource

# Create two sources that might produce data at slightly different times
source1 = FakeSeriesSource(
    rate=2048,
    signal_type="sine",
    fsin=10.0,
    real_time=True,
    name="sine_source"
)

source2 = FakeSeriesSource(
    rate=2048,
    signal_type="sine",
    fsin=5.0,
    real_time=True,
    name="another_sine_source"
)

# Create an align transform
align = Align(
    source_pad_names=["signal1", "signal2"],
    sink_pad_names=["signal1", "signal2"],
    name="aligner"
)

# Create an adder to combine the aligned signals
adder = Adder(name="signal_adder")

# Connect sources to aligner
# (Pad creation code omitted for brevity)
source1.add_dest(align, align.sink_pad_dict["aligner:snk:signal1"])
source2.add_dest(align, align.sink_pad_dict["aligner:snk:signal2"])

# Connect aligned outputs to adder
align.add_dest(adder, from_pad=align.source_pad_dict["aligner:src:signal1"])
align.add_dest(adder, from_pad=align.source_pad_dict["aligner:src:signal2"])

# Process data
for _ in range(10):
    source1.process()
    source2.process()
    align.process()
    adder.process()
"""
```

## Using Align in Complex Pipelines

The `Align` transform is particularly useful in complex pipelines where multiple branches need to be synchronized:

```python
# Complex pipeline example (not tested by mkdocs)
"""
from sgnts.transforms import Align, AmplifyTransform
from sgnts.sources import FakeSeriesSource
from sgnts.sinks import DumpSeriesSink

# Create two sources
source1 = FakeSeriesSource(rate=2048, signal_type="sine", fsin=10.0)
source2 = FakeSeriesSource(rate=2048, signal_type="white")

# Create transforms for each branch
amplify1 = AmplifyTransform(factor=2.0, name="amplify1")
amplify2 = AmplifyTransform(factor=0.5, name="amplify2")

# Create an align transform
align = Align(
    source_pad_names=["branch1", "branch2"],
    sink_pad_names=["branch1", "branch2"],
    name="aligner"
)

# Create sinks
sink1 = DumpSeriesSink(fname="aligned_branch1.txt", name="sink1")
sink2 = DumpSeriesSink(fname="aligned_branch2.txt", name="sink2")

# Connect the pipeline
source1.add_dest(amplify1)
source2.add_dest(amplify2)

amplify1.add_dest(align, align.sink_pad_dict["aligner:snk:branch1"])
amplify2.add_dest(align, align.sink_pad_dict["aligner:snk:branch2"])

align.add_dest(sink1, from_pad=align.source_pad_dict["aligner:src:branch1"])
align.add_dest(sink2, from_pad=align.source_pad_dict["aligner:src:branch2"])

# Process data
for _ in range(10):
    source1.process()
    source2.process()
    amplify1.process()
    amplify2.process()
    align.process()
    sink1.process()
    sink2.process()
"""
```

## Best Practices

When using `Align`:

1. **Match pad names** - ensure that source_pad_names and sink_pad_names contain the same set of names

2. **Keep track of pads** - use the pad dictionaries to access the pads by name

3. **Process in order** - ensure that all upstream elements are processed before the align transform

4. **Consider buffer sizes** - all sources feeding into an align transform should produce compatible buffer sizes

5. **Understand the pass-through nature** - Align doesn't modify data, it just ensures synchronization