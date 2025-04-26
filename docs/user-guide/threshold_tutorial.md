# Threshold Transform

The `Threshold` transform selectively passes or blocks data based on its amplitude, converting regions below a threshold to gap buffers.

## Overview

The `Threshold` transform:
- Passes only data that exceeds a specified amplitude threshold
- Can be inverted to pass only data below a threshold
- Supports windows around threshold crossings
- Splits buffers at threshold crossings
- Converts non-qualifying data to gap buffers

## Basic Usage

```python
# Basic usage of Threshold (not tested by mkdocs)
"""
from sgnts.transforms import Threshold
from sgnts.sources import FakeSeriesSource

# Create a source with sine wave data
source = FakeSeriesSource(
    rate=2048,
    signal_type="sine",
    fsin=1.0  # 1 Hz sine wave
)

# Create a threshold transform that only passes data above 0.7
threshold = Threshold(
    threshold=0.7,   # Threshold value
    invert=False,    # Pass data above threshold (default)
    startwn=0,       # No extra samples before crossing
    stopwn=0         # No extra samples after crossing
)

# Connect source to threshold
source.add_dest(threshold)

# Process data
source.process()
threshold.process()

# Pull the thresholded frame
frame = threshold.pull()

# The frame contains data only where the sine wave amplitude exceeds 0.7
# Other regions are converted to gap buffers
"""
```

## Inverted Thresholding

The `invert` parameter allows passing data below a threshold:

```python
# Inverted threshold example (not tested by mkdocs)
"""
from sgnts.transforms import Threshold
from sgnts.sources import FakeSeriesSource

# Create a source with white noise
source = FakeSeriesSource(
    rate=2048,
    signal_type="white"
)

# Create an inverted threshold to pass only low-amplitude data
inverted_threshold = Threshold(
    threshold=1.0,  # Threshold value
    invert=True     # Pass data BELOW threshold
)

# Connect and process
source.add_dest(inverted_threshold)
source.process()
inverted_threshold.process()

# Pull the thresholded frame
frame = inverted_threshold.pull()

# The frame contains data only where the noise amplitude is below 1.0
# Higher amplitude regions are converted to gap buffers
"""
```

## Window Parameters

The `startwn` and `stopwn` parameters allow keeping extra samples around threshold crossings:

```python
# Window parameters example (not tested by mkdocs)
"""
from sgnts.transforms import Threshold
from sgnts.sources import FakeSeriesSource

# Create a source with impulse data
source = FakeSeriesSource(
    rate=2048,
    signal_type="impulse"
)

# Create a threshold with windows
threshold_with_windows = Threshold(
    threshold=0.5,
    startwn=10,  # Keep 10 samples before crossing
    stopwn=20    # Keep 20 samples after crossing
)

# Connect and process
source.add_dest(threshold_with_windows)
source.process()
threshold_with_windows.process()

# Pull the thresholded frame
frame = threshold_with_windows.pull()

# The frame contains the impulse and additional samples
# 10 samples before and 20 samples after the threshold crossing
"""
```

## Threshold Detection Applications

```python
# Threshold detection example (not tested by mkdocs)
"""
from sgnts.transforms import Threshold, AmplifyTransform
from sgnts.sources import FakeSeriesSource
from sgnts.sinks import DumpSeriesSink

# Create a source with a signal containing transients
source = FakeSeriesSource(
    rate=2048,
    signal_type="white"
)

# Create a threshold to detect high-amplitude events
detector = Threshold(
    threshold=2.0,  # High threshold to detect only large transients
    startwn=50,     # Include 50 samples before crossing
    stopwn=100      # Include 100 samples after crossing
)

# Create an amplifier to highlight detected events
amplifier = AmplifyTransform(factor=2.0)

# Create a sink
sink = DumpSeriesSink(fname="detected_events.txt")

# Connect the pipeline
source.add_dest(detector)
detector.add_dest(amplifier)
amplifier.add_dest(sink)

# Run the pipeline
for _ in range(20):
    source.process()
    detector.process()
    amplifier.process()
    sink.process()

# The output file contains only the detected high-amplitude events
# with their surrounding windows, amplified by a factor of 2
"""
```

## Noise Gating

A common use case for `Threshold` is noise gating:

```python
# Noise gating example (not tested by mkdocs)
"""
from sgnts.transforms import Threshold
from sgnts.sources import FakeSeriesSource

# Create a source with audio-like data
source = FakeSeriesSource(
    rate=44100,  # Audio sample rate
    signal_type="sine",
    fsin=440  # 440 Hz tone
)

# Create a noise gate (threshold with windows)
noise_gate = Threshold(
    threshold=0.1,   # Only pass signals above this amplitude
    startwn=441,     # ~10ms attack time
    stopwn=4410      # ~100ms release time
)

# Connect and process
source.add_dest(noise_gate)
source.process()
noise_gate.process()

# Pull the gated frame
frame = noise_gate.pull()

# The frame contains the signal only where it exceeds the threshold,
# with gradual attack and release windows
"""
```

## Integration with Other Components

```python
# Integration example (not tested by mkdocs)
"""
from sgnts.transforms import Threshold, Correlate
from sgnts.sources import FakeSeriesSource
import numpy as np

# Create a filter for matching a specific pattern
filter_width = 32
filter_array = np.hamming(filter_width)

# Create a source with noisy data
source = FakeSeriesSource(
    rate=2048,
    signal_type="white"
)

# Create a correlator to detect patterns
correlator = Correlate(
    sample_rate=2048,
    filters=filter_array
)

# Create a threshold to extract high correlation regions
detector = Threshold(
    threshold=0.8,  # High correlation threshold
    startwn=10,
    stopwn=10
)

# Connect the pipeline
source.add_dest(correlator)
correlator.add_dest(detector)

# Process data
for _ in range(5):
    source.process()
    correlator.process()
    detector.process()
    
    # Pull the detected events
    frame = detector.pull()
    
    # Frame contains only regions where correlation exceeded the threshold
"""
```

## Threshold History

The `Threshold` transform maintains a history of non-gap slices:

```python
# Threshold history example (not tested by mkdocs)
"""
from sgnts.transforms import Threshold
from sgnts.sources import FakeSeriesSource

# Create a source
source = FakeSeriesSource(
    rate=2048,
    signal_type="sine",
    fsin=0.5  # Slow oscillation
)

# Create a threshold
threshold = Threshold(threshold=0.7)

# Connect and process for multiple frames
source.add_dest(threshold)

for _ in range(5):
    source.process()
    threshold.process()
    
    # The threshold maintains a history of slices that crossed the threshold
    # It keeps only relevant slices that may affect current or future data
    # This history is automatically pruned to remove old slices
"""
```

## Best Practices

When using `Threshold`:

1. **Choose appropriate threshold values** - based on the expected amplitude range of your data

2. **Consider window parameters** - `startwn` and `stopwn` help prevent choppy output by including samples around threshold crossings

3. **Mind buffer splitting** - the transform splits buffers at threshold crossings, which can result in many small buffers

4. **Consider inversion** - `invert=True` is useful for removing high-amplitude noise or isolating quiet sections

5. **Test with representative data** - threshold behavior depends heavily on input characteristics

6. **Balance sensitivity** - too low thresholds may pass noise, too high may miss important signals

7. **Check downstream components** - ensure they can handle potentially fragmented buffers after thresholding