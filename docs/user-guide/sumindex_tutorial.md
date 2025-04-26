# SumIndex Transform

The `SumIndex` transform sums array values over specified slices in the first dimension, enabling channel combination, binning, and reduction operations.

## Overview

`SumIndex` is a specialized transform that:
- Takes slices from the first dimension of input arrays and sums them
- Supports different array backends for efficient computation
- Can extract single elements or sum ranges of elements
- Preserves the time structure of frames
- Enables selective channel aggregation

## Basic Usage

```python
# Basic usage of SumIndex (not tested by mkdocs)
"""
import numpy as np
from sgnts.transforms import SumIndex
from sgnts.sources import FakeSeriesSource
from sgnts.base import NumpyBackend

# Create a source with multi-channel data (8 channels)
source = FakeSeriesSource(
    rate=2048,
    sample_shape=(8, 3),  # 8 channels, each with shape (3,)
    signal_type="white"
)

# Create a SumIndex transform to sum specific channel ranges
sumindex = SumIndex(
    sl=[
        slice(0, 2),    # Sum channels 0 and 1
        slice(2, 4),    # Sum channels 2 and 3
        slice(4, 8)     # Sum channels 4 through 7
    ],
    backend=NumpyBackend
)

# Connect source to sumindex transform
source.add_dest(sumindex)

# Process data
source.process()
sumindex.process()

# Pull the transformed frame
frame = sumindex.pull()

# The frame contains data with 3 channels:
# 1. Sum of channels 0 and 1
# 2. Sum of channels 2 and 3
# 3. Sum of channels 4 through 7
# Each channel maintains the original (3,) shape
"""
```

## Channel Selection

`SumIndex` can be used to select specific channels without summing:

```python
# Channel selection example (not tested by mkdocs)
"""
from sgnts.transforms import SumIndex
from sgnts.sources import FakeSeriesSource

# Create a source with 5 channels
source = FakeSeriesSource(
    rate=2048,
    sample_shape=(5,),
    signal_type="sine",
    fsin=10.0
)

# Create a SumIndex transform to select channels 0, 2, and 4
selector = SumIndex(
    sl=[
        slice(0, 1),  # Select only channel 0
        slice(2, 3),  # Select only channel 2
        slice(4, 5)   # Select only channel 4
    ]
)

# Connect and process
source.add_dest(selector)
source.process()
selector.process()

# Pull the frame with selected channels
frame = selector.pull()

# The output has 3 channels corresponding to original channels 0, 2, and 4
"""
```

## Binning and Grouping

`SumIndex` is useful for binning or grouping channels:

```python
# Binning example (not tested by mkdocs)
"""
from sgnts.transforms import SumIndex
from sgnts.sources import FakeSeriesSource

# Create a source with 16 channels
source = FakeSeriesSource(
    rate=2048,
    sample_shape=(16,),
    signal_type="white"
)

# Create a SumIndex to bin channels into groups of 4
binning = SumIndex(
    sl=[
        slice(0, 4),    # Sum channels 0-3
        slice(4, 8),    # Sum channels 4-7
        slice(8, 12),   # Sum channels 8-11
        slice(12, 16)   # Sum channels 12-15
    ]
)

# Connect and process
source.add_dest(binning)
source.process()
binning.process()

# Pull the binned frame
frame = binning.pull()

# The output has 4 channels, each representing the sum of 4 input channels
"""
```

## Combination with Other Transforms

```python
# Transform combination example (not tested by mkdocs)
"""
import numpy as np
from sgnts.transforms import SumIndex, AmplifyTransform
from sgnts.sources import FakeSeriesSource

# Create a source with 6 channels
source = FakeSeriesSource(
    rate=2048,
    sample_shape=(6,),
    signal_type="sine",
    fsin=10.0
)

# Create weights for different channel groups
weights = np.array([0.5, 2.0, 1.0])

# First sum the channels into groups
sumindex = SumIndex(
    sl=[
        slice(0, 2),  # Channels 0-1
        slice(2, 4),  # Channels 2-3
        slice(4, 6)   # Channels 4-5
    ]
)

# Then apply different amplification to each group
amplifier = AmplifyTransform(factor=weights)

# Connect the transforms
source.add_dest(sumindex)
sumindex.add_dest(amplifier)

# Process data
source.process()
sumindex.process()
amplifier.process()

# Pull the result
frame = amplifier.pull()

# The output has 3 channels:
# - Sum of channels 0-1, amplified by 0.5
# - Sum of channels 2-3, amplified by 2.0
# - Sum of channels 4-5, amplified by 1.0
"""
```

## Integration in Processing Pipelines

```python
# Pipeline integration example (not tested by mkdocs)
"""
from sgnts.transforms import SumIndex, Resampler
from sgnts.sources import FakeSeriesSource
from sgnts.sinks import DumpSeriesSink

# Create a source with multi-channel data
source = FakeSeriesSource(
    rate=2048,
    sample_shape=(8,),
    signal_type="white"
)

# Create a SumIndex to combine channels
sumindex = SumIndex(
    sl=[
        slice(0, 4),  # Sum first half of channels
        slice(4, 8)   # Sum second half of channels
    ]
)

# Create a resampler to change sample rate
resampler = Resampler(
    inrate=2048,
    outrate=1024
)

# Create a sink
sink = DumpSeriesSink(fname="processed_data.txt")

# Connect the pipeline
source.add_dest(sumindex)
sumindex.add_dest(resampler)
resampler.add_dest(sink)

# Run the pipeline
for _ in range(10):
    source.process()
    sumindex.process()
    resampler.process()
    sink.process()

# The output file contains the summed and resampled data
"""
```

## Complex Channel Operations

`SumIndex` can be used for more complex operations by carefully defining slices:

```python
# Complex operations example (not tested by mkdocs)
"""
from sgnts.transforms import SumIndex
from sgnts.sources import FakeSeriesSource

# Create a source with matrix data at each time point
source = FakeSeriesSource(
    rate=2048,
    sample_shape=(4, 4),  # 4x4 matrix
    signal_type="white"
)

# Create a SumIndex to perform various operations
operations = SumIndex(
    sl=[
        slice(0, 1),     # Select first row
        slice(0, 4),     # Sum all rows
        slice(1, 4, 2)   # Sum rows 1 and 3 (step of 2)
    ]
)

# Connect and process
source.add_dest(operations)
source.process()
operations.process()

# Pull the processed frame
frame = operations.pull()

# The output has 3 channels:
# 1. The first row of the matrix
# 2. The sum of all rows
# 3. The sum of rows 1 and 3
"""
```

## Best Practices

When using `SumIndex`:

1. **Define slices correctly** - slices are applied to the first dimension of the input data

2. **Check shape compatibility** - ensure that your slice definitions match the shape of your input data

3. **Consider normalization** - summing channels increases amplitude, which might need normalization

4. **Use appropriate backend** - choose the array backend that best matches your computation needs

5. **Mind gap handling** - the transform passes gap buffers through unchanged

6. **Consider computational efficiency** - summing operations are generally fast, but large arrays can impact performance

7. **Remember slice format** - use Python's standard slice objects with start, stop, and optionally step parameters