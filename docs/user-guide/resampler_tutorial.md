# Resampler Transform

The `Resampler` transform changes the sample rate of time-series data, enabling compatibility between components operating at different rates.

## Overview

`Resampler` is a powerful transform that:
- Upsamples data to higher sample rates
- Downsamples data to lower sample rates
- Uses windowed sinc interpolation for high-quality resampling
- Supports both NumPy and PyTorch backends
- Preserves signal characteristics during rate conversion

## Basic Usage

```python
# Basic usage of Resampler (not tested by mkdocs)
"""
from sgnts.transforms import Resampler
from sgnts.sources import FakeSeriesSource
from sgnts.base import NumpyBackend

# Create a source at 2048 Hz
source = FakeSeriesSource(
    rate=2048,
    signal_type="sine",
    fsin=10.0  # 10 Hz sine wave
)

# Create a resampler to convert to 4096 Hz (upsampling by factor of 2)
upsampler = Resampler(
    inrate=2048,      # Input sample rate
    outrate=4096,     # Output sample rate
    backend=NumpyBackend
)

# Connect source to resampler
source.add_dest(upsampler)

# Process data
source.process()
upsampler.process()

# Pull the resampled frame
frame = upsampler.pull()

# The frame contains data at the new sample rate (4096 Hz)
# with twice as many samples as the original
"""
```

## Upsampling

Upsampling increases the sample rate by interpolating between existing samples:

```python
# Upsampling example (not tested by mkdocs)
"""
from sgnts.transforms import Resampler
from sgnts.sources import FakeSeriesSource

# Create a source at 1000 Hz
source = FakeSeriesSource(
    rate=1000,
    signal_type="sine",
    fsin=10.0
)

# Create a resampler to upsample by factor of 8 (to 8000 Hz)
upsampler = Resampler(
    inrate=1000,
    outrate=8000
)

# Connect and process
source.add_dest(upsampler)
source.process()
upsampler.process()

# Pull the upsampled frame
frame = upsampler.pull()

# The output has 8 times as many samples as the input,
# while preserving the original signal characteristics
"""
```

## Downsampling

Downsampling reduces the sample rate, requiring careful filtering to prevent aliasing:

```python
# Downsampling example (not tested by mkdocs)
"""
from sgnts.transforms import Resampler
from sgnts.sources import FakeSeriesSource

# Create a source at 16384 Hz
source = FakeSeriesSource(
    rate=16384,
    signal_type="sine",
    fsin=100.0  # 100 Hz sine wave
)

# Create a resampler to downsample by factor of 4 (to 4096 Hz)
downsampler = Resampler(
    inrate=16384,
    outrate=4096
)

# Connect and process
source.add_dest(downsampler)
source.process()
downsampler.process()

# Pull the downsampled frame
frame = downsampler.pull()

# The output has 1/4 as many samples as the input,
# but still accurately represents the original 100 Hz sine wave
"""
```

## Using PyTorch Backend

The `Resampler` supports GPU acceleration through PyTorch:

```python
# PyTorch backend example (not tested by mkdocs)
"""
from sgnts.transforms import Resampler
from sgnts.sources import FakeSeriesSource
from sgnts.base import TorchBackend

# Create a source
source = FakeSeriesSource(
    rate=2048,
    signal_type="white"  # White noise
)

# Create a resampler with PyTorch backend
torch_resampler = Resampler(
    inrate=2048,
    outrate=1024,
    backend=TorchBackend  # Use PyTorch for computation
)

# Connect and process
source.add_dest(torch_resampler)
source.process()
torch_resampler.process()

# Pull the resampled frame
frame = torch_resampler.pull()

# Resampling was performed using PyTorch tensors,
# potentially benefiting from GPU acceleration
"""
```

## Multi-channel Resampling

`Resampler` handles multi-channel data correctly:

```python
# Multi-channel resampling example (not tested by mkdocs)
"""
from sgnts.transforms import Resampler
from sgnts.sources import FakeSeriesSource

# Create a source with multi-channel data
source = FakeSeriesSource(
    rate=2048,
    sample_shape=(4,),  # 4-channel data
    signal_type="sine",
    fsin=10.0
)

# Create a resampler
resampler = Resampler(
    inrate=2048,
    outrate=4096
)

# Connect and process
source.add_dest(resampler)
source.process()
resampler.process()

# Pull the resampled frame
frame = resampler.pull()

# All 4 channels have been resampled to the new rate
"""
```

## Integration in Processing Pipelines

```python
# Pipeline integration example (not tested by mkdocs)
"""
from sgnts.transforms import Resampler, AmplifyTransform
from sgnts.sources import FakeSeriesSource
from sgnts.sinks import DumpSeriesSink

# Create a source at 8192 Hz
source = FakeSeriesSource(
    rate=8192,
    signal_type="sine",
    fsin=100.0
)

# Create a resampler to downsample to 2048 Hz
resampler = Resampler(
    inrate=8192,
    outrate=2048
)

# Create an amplifier
amplifier = AmplifyTransform(factor=2.0)

# Create a sink
sink = DumpSeriesSink(fname="resampled_signal.txt")

# Connect the pipeline
source.add_dest(resampler)
resampler.add_dest(amplifier)
amplifier.add_dest(sink)

# Run the pipeline
for _ in range(10):
    source.process()
    resampler.process()
    amplifier.process()
    sink.process()

# The output file contains the downsampled and amplified signal
"""
```

## Resampling Kernel Design

The `Resampler` uses carefully designed kernels for high-quality resampling:

```python
# Understanding resampling kernels (not tested by mkdocs)
"""
# Resampler uses sinc-windowed-sinc kernels:

# For downsampling:
# - Uses longer windows (half_length = DOWN_HALF_LENGTH * factor)
# - Computes a windowed sinc function
# - Applies the kernel via correlation and takes strided output

# For upsampling:
# - Uses shorter windows (half_length = UP_HALF_LENGTH)
# - Creates a set of sub-kernels for efficient computation
# - Inserts zeros in the input data, then applies convolution
"""
```

## Best Practices

When using `Resampler`:

1. **Choose appropriate rates** - both input and output rates must be in the allowed set (powers of 2 times 5)

2. **Consider signal bandwidth** - ensure that downsampling doesn't lose important frequency content (the Nyquist limit will be half the new sample rate)

3. **Mind computational load** - resampling is computationally intensive, especially for high rates and large factors

4. **Use the right backend** - PyTorch backend can be faster for large datasets, especially on GPU

5. **Check for gaps** - resampling preserves gap structure in the original data

6. **Handle boundary effects** - the resampler has internal padding to handle edge effects

7. **Validate results** - especially when downsampling, check that aliasing hasn't introduced artifacts