# DumpSeriesSink

The `DumpSeriesSink` is a sink component that writes time-series data to a text file.

## Overview

`DumpSeriesSink` is useful when you need to:
- Save the output of a processing pipeline to a file
- Debug the values of processed time-series data
- Export time-series data for analysis in other tools

## Basic Usage

```python
# Basic usage of DumpSeriesSink (not tested by mkdocs)
"""
from sgnts.sinks import DumpSeriesSink
from sgnts.sources import FakeSeriesSource

# Create a source of data
source = FakeSeriesSource(
    rate=2048,
    signal_type="sine",
    fsin=10.0  # 10 Hz sine wave
)

# Create a sink to write the data to a file
sink = DumpSeriesSink(
    fname="output_data.txt",  # Output file name
    verbose=True              # Print frame information
)

# Connect the source to the sink
source.add_dest(sink)

# Run the pipeline for a few iterations
for _ in range(5):
    source.process()
    sink.process()
"""
```

## Output Format

The `DumpSeriesSink` writes data to a text file in a format where:
- The first column represents the time in seconds
- Subsequent columns contain the data values

For multi-dimensional data, the component reshapes the data to ensure it can be written to a file:

```python
# Output format example (not tested by mkdocs)
"""
import numpy as np
from sgnts.sinks import DumpSeriesSink
from sgnts.sources import FakeSeriesSource

# Create a source with 2D data
source = FakeSeriesSource(
    rate=1024,
    sample_shape=(2,),  # 2-channel data
    signal_type="sine",
    fsin=5.0
)

# Create a sink
sink = DumpSeriesSink(fname="multichannel_data.txt")

# Connect and run
source.add_dest(sink)
for _ in range(3):
    source.process()
    sink.process()

# The output file will have format:
# time_in_seconds  channel1_value  channel2_value
# ...
"""
```

## Handling Gaps

The `DumpSeriesSink` only writes non-gap buffers to the output file:

```python
# Handling gaps example (not tested by mkdocs)
"""
from sgnts.sinks import DumpSeriesSink
from sgnts.sources import SegmentSource

# Create a source with gaps
segments = ((1000000000, 2000000000), (3000000000, 4000000000))
source = SegmentSource(rate=2048, segments=segments, t0=0, end=5)

# Create a sink
sink = DumpSeriesSink(fname="segmented_data.txt")

# Connect and run
source.add_dest(sink)
for _ in range(10):
    source.process()
    sink.process()

# The output file will only contain data from the specified segments
# No data will be written for the gap periods
"""
```

## Integration with Transforms

```python
# Integration with transforms example (not tested by mkdocs)
"""
from sgnts.sinks import DumpSeriesSink
from sgnts.sources import FakeSeriesSource
from sgnts.transforms import AmplifyTransform

# Create a pipeline
source = FakeSeriesSource(rate=2048, signal_type="sine", fsin=10.0)
amplifier = AmplifyTransform(factor=2.0)
sink = DumpSeriesSink(fname="amplified_sine.txt")

# Connect the elements
source.add_dest(amplifier)
amplifier.add_dest(sink)

# Run the pipeline
for _ in range(5):
    source.process()
    amplifier.process()
    sink.process()

# The output file will contain the amplified sine wave data
"""
```

## Best Practices

When using `DumpSeriesSink`:

1. **Use descriptive filenames** to identify the data source and processing

2. **Enable verbose mode** (`verbose=True`) during debugging to see frame information

3. **Be aware of file size** - writing high-frequency data for long periods can create very large files

4. **Consider formatting needs** - data is written as plain text, which might not be the most efficient format for large datasets

5. **Check file permissions** - ensure the component has write access to the specified file location

6. **Remember data transformation** - multi-dimensional data is reshaped before writing, which might affect how you need to interpret the file contents later