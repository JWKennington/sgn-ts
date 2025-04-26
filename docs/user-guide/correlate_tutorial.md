# Correlate Transform

The `Correlate` transform performs correlation operations between input data and a set of filters, enabling signal detection, pattern matching, and filtering applications.

## Overview

SGN-TS provides two correlation transforms:

1. `Correlate`: A basic transform that correlates input data with fixed filters
2. `AdaptiveCorrelate`: An advanced transform that can update its filters over time

These transforms are useful for:
- Signal detection and template matching
- Feature extraction from time-series data
- Implementing matched filters
- Time-domain filtering operations

## Basic Correlate Transform

### Overview

The basic `Correlate` transform:
- Applies correlation between input data and predefined filters
- Uses scipy.signal.correlate for the core operation
- Handles both 1D and multi-dimensional filters
- Properly accounts for overlap between frames

### Basic Usage

```python
# Basic usage of Correlate (not tested by mkdocs)
"""
import numpy as np
from sgnts.transforms import Correlate
from sgnts.sources import FakeSeriesSource

# Create a filter - a simple triangular window
filter_width = 32
filter_array = np.bartlett(filter_width)

# Create a source
source = FakeSeriesSource(
    rate=2048,
    signal_type="sine",
    fsin=10.0  # 10 Hz sine wave
)

# Create a correlator
correlator = Correlate(
    sample_rate=2048,           # Must match the source rate
    filters=filter_array        # The filter to correlate with
)

# Connect source to correlator
source.add_dest(correlator)

# Process data
source.process()
correlator.process()

# Pull correlated frame
frame = correlator.pull()

# The frame contains data correlated with the filter
# Note that output length will be len(input) - len(filter) + 1
"""
```

### Multi-dimensional Filters

```python
# Multi-dimensional filters example (not tested by mkdocs)
"""
import numpy as np
from sgnts.transforms import Correlate
from sgnts.sources import FakeSeriesSource

# Create a set of filters
# Each row is a different filter
num_filters = 4
filter_width = 32
filter_array = np.zeros((num_filters, filter_width))

# Create different filters in each row
for i in range(num_filters):
    # Create filters with different frequencies
    t = np.linspace(0, 1, filter_width)
    filter_array[i] = np.sin(2 * np.pi * (i + 1) * t)

# Create a source
source = FakeSeriesSource(
    rate=2048,
    signal_type="white"  # White noise
)

# Create a correlator with multiple filters
correlator = Correlate(
    sample_rate=2048,
    filters=filter_array
)

# Connect and process
source.add_dest(correlator)
source.process()
correlator.process()

# Pull correlated frame
frame = correlator.pull()

# The output data has shape (num_filters, output_length)
# Each row corresponds to correlation with one filter
"""
```

## Adaptive Correlate Transform

### Overview

The `AdaptiveCorrelate` transform extends `Correlate` with the ability to:
- Change filters over time
- Blend between old and new filters using a cosine window
- Receive new filters through a dedicated sink pad

### Basic Usage

```python
# AdaptiveCorrelate example (not tested by mkdocs)
"""
import numpy as np
from sgnts.transforms import AdaptiveCorrelate
from sgnts.sources import FakeSeriesSource
from sgnts.base import EventBuffer, Offset
from sgnts.base.slice_tools import TIME_MAX

# Create initial filters
filter_width = 32
initial_filters = np.bartlett(filter_width)

# Create an event buffer containing the initial filters
init_buffer = EventBuffer(
    ts=0,                  # Start time (t0)
    te=TIME_MAX,           # End time (maximum)
    data=initial_filters   # Filter data
)

# Create an adaptive correlator
adaptive_correlator = AdaptiveCorrelate(
    sample_rate=2048,
    init_filters=init_buffer,
    filter_sink_name="filter_updates"  # Name for the filter update sink pad
)

# Create a source for the main data
data_source = FakeSeriesSource(
    rate=2048,
    signal_type="white"
)

# Connect data source to correlator
data_source.add_dest(adaptive_correlator)

# Process data with initial filters
data_source.process()
adaptive_correlator.process()

# Later, create new filters and send them to the correlator
new_filters = np.hamming(filter_width)

# Create an event buffer for the new filters
new_filter_buffer = EventBuffer(
    ts=Offset.fromsamples(1024, 2048),  # Start using new filters after 1024 samples
    te=TIME_MAX,
    data=new_filters
)

# Create a source for filter updates and send them to the correlator
# (Implementation details omitted for brevity)
"""
```

### Window Blending

When filters are updated, `AdaptiveCorrelate` performs a smooth transition using cosine windows:

```python
# Window blending explanation (not tested by mkdocs)
"""
# During filter update, AdaptiveCorrelate:
# 1. Computes correlation with both old and new filters
# 2. Creates cosine-squared window functions
# 3. Blends the outputs using these window functions

# The blending formula is:
# output = (1 - win_new) * correlation_with_old_filters + win_new * correlation_with_new_filters

# Where win_new is a cosine-squared window that transitions from 0 to 1
# This ensures a smooth transition without discontinuities
"""
```

## Integration in Signal Processing Pipelines

```python
# Pipeline integration example (not tested by mkdocs)
"""
import numpy as np
from sgnts.transforms import Correlate, AmplifyTransform
from sgnts.sources import FakeSeriesSource
from sgnts.sinks import DumpSeriesSink

# Create a filter for detecting a specific pattern
filter_width = 64
t = np.linspace(0, 1, filter_width)
filter_array = np.sin(2 * np.pi * 10 * t) * np.hamming(filter_width)

# Create a source with the pattern embedded in noise
source = FakeSeriesSource(
    rate=2048,
    signal_type="white"
)

# Create a correlator to detect the pattern
correlator = Correlate(
    sample_rate=2048,
    filters=filter_array
)

# Create an amplifier to enhance the correlation result
amplifier = AmplifyTransform(factor=2.0)

# Create a sink to save the results
sink = DumpSeriesSink(fname="correlation_results.txt")

# Connect the pipeline
source.add_dest(correlator)
correlator.add_dest(amplifier)
amplifier.add_dest(sink)

# Run the pipeline
for _ in range(10):
    source.process()
    correlator.process()
    amplifier.process()
    sink.process()
"""
```

## Best Practices

When using correlation transforms:

1. **Match sample rates** - ensure the sample rate specified in the correlator matches the input data

2. **Consider filter size** - larger filters provide more selective matching but reduce output length and increase computation time

3. **Normalize filters** - consider normalizing filters to control correlation amplitude

4. **Mind output size** - correlation output length is len(input) - len(filter) + 1 in "valid" mode

5. **Adapt filter updates** - with `AdaptiveCorrelate`, don't update filters too frequently (only one update per stride is supported)

6. **Use appropriate window functions** - `AdaptiveCorrelate` uses cosine-squared windows for blending, which works well for most applications

7. **Consider computational load** - correlation operations can be computationally intensive, especially with multiple or large filters