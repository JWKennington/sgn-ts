# Amplify Transform

The `Amplify` transform multiplies time-series data by a constant factor.

## Overview

`Amplify` is a simple but essential transform that:
- Scales data values by a specified factor
- Preserves the time structure of frames
- Properly handles gap buffers by passing them through unchanged
- Supports both amplification (factor > 1) and attenuation (factor < 1) 

## Basic Usage

```python
# Basic usage of Amplify (not tested by mkdocs)
"""
from sgnts.transforms import Amplify
from sgnts.sources import FakeSeriesSource

# Create a source
source = FakeSeriesSource(
    rate=2048,
    signal_type="sine",
    fsin=10.0  # 10 Hz sine wave
)

# Create an amplifier that doubles the amplitude
amplifier = Amplify(factor=2.0)

# Connect source to amplifier
source.add_dest(amplifier)

# Get a frame from the source and process it
source.process()
amplifier.process()

# Pull the amplified frame
frame = amplifier.pull()

# The data in the frame has been multiplied by 2.0
"""
```

## Amplification and Attenuation

```python
# Amplification and attenuation examples (not tested by mkdocs)
"""
from sgnts.transforms import Amplify

# Create an amplifier (gain > 1)
amplifier = Amplify(factor=5.0)  # Multiply by 5

# Create an attenuator (gain < 1)
attenuator = Amplify(factor=0.5)  # Multiply by 0.5 (halve the amplitude)

# Use negative factors to invert the signal
inverter = Amplify(factor=-1.0)  # Invert the signal
"""
```

## Handling Gap Buffers

The `Amplify` transform preserves gap buffers:

```python
# Gap buffer handling (not tested by mkdocs)
"""
from sgnts.transforms import Amplify
from sgnts.sources import SegmentSource

# Create a source with gaps
segments = ((1000000000, 2000000000), (3000000000, 4000000000))
source = SegmentSource(rate=2048, segments=segments, t0=0, end=5)

# Create an amplifier
amplifier = Amplify(factor=2.0)

# Connect source to amplifier
source.add_dest(amplifier)

# Process frames
source.process()
amplifier.process()

# Pull the frame from the amplifier
frame = amplifier.pull()

# Gap buffers remain as gaps
# Non-gap buffers have been amplified by factor=2.0
for buf in frame:
    if buf.is_gap:
        print(f"Gap buffer at offset {buf.offset}")
    else:
        print(f"Amplified buffer at offset {buf.offset}")
"""
```

## Integration in a Signal Processing Pipeline

```python
# Pipeline integration example (not tested by mkdocs)
"""
from sgnts.transforms import Amplify
from sgnts.sources import FakeSeriesSource
from sgnts.sinks import DumpSeriesSink

# Create a pipeline
source = FakeSeriesSource(
    rate=2048,
    signal_type="sine",
    fsin=10.0
)

# Create an amplifier
amplifier = Amplify(factor=2.5)

# Create a sink to save the amplified data
sink = DumpSeriesSink(fname="amplified_signal.txt")

# Connect the pipeline
source.add_dest(amplifier)
amplifier.add_dest(sink)

# Run the pipeline for multiple iterations
for _ in range(10):
    source.process()
    amplifier.process()
    sink.process()
"""
```

## Chaining Multiple Amplifiers

```python
# Chaining amplifiers example (not tested by mkdocs)
"""
from sgnts.transforms import Amplify
from sgnts.sources import FakeSeriesSource

# Create a source
source = FakeSeriesSource(
    rate=2048,
    signal_type="white"  # White noise
)

# Create two amplifiers in series
first_amplifier = Amplify(factor=2.0, name="first_stage")
second_amplifier = Amplify(factor=1.5, name="second_stage")

# Connect the pipeline
source.add_dest(first_amplifier)
first_amplifier.add_dest(second_amplifier)

# The overall amplification will be 2.0 * 1.5 = 3.0

# Process frames
source.process()
first_amplifier.process()
second_amplifier.process()
"""
```

## Best Practices

When using `Amplify`:

1. **Choose appropriate factors** - excessive amplification may lead to clipping or numerical overflow

2. **Consider precision** - amplification can affect the precision of floating-point data

3. **Use in combination** - combine `Amplify` with other transforms for more complex signal processing

4. **Preserve metadata** - the `Amplify` transform preserves frame metadata, which can be useful for tracking the processing chain

5. **Mind array backend compatibility** - the simple multiplication operation works with all array backends