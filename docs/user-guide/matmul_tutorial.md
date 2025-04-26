# Matmul Transform

The `Matmul` transform performs matrix multiplication between input data and a specified matrix, enabling linear transformations of time-series data.

## Overview

`Matmul` is a versatile transform that:
- Applies a linear transformation to each time slice of the input data
- Supports different array backends for efficient computation
- Handles gap buffers appropriately
- Preserves the time structure of frames

## Basic Usage

```python
# Basic usage of Matmul (not tested by mkdocs)
"""
import numpy as np
from sgnts.transforms import Matmul
from sgnts.sources import FakeSeriesSource
from sgnts.base import NumpyBackend

# Create a matrix for multiplication
# This example creates a 2x4 matrix that will transform 4D inputs to 2D outputs
matrix = np.array([
    [1, 2, 3, 4],   # First output channel
    [5, 6, 7, 8]    # Second output channel
])

# Create a source with matching shape
source = FakeSeriesSource(
    rate=2048,
    sample_shape=(4,),  # 4-dimensional data at each time point
    signal_type="white"
)

# Create a Matmul transform
matmul_transform = Matmul(
    matrix=matrix,
    backend=NumpyBackend
)

# Connect source to matmul transform
source.add_dest(matmul_transform)

# Process data
source.process()
matmul_transform.process()

# Pull the transformed frame
frame = matmul_transform.pull()

# The frame contains data that has been matrix-multiplied with the specified matrix
# Original data shape: (4, N) where N is the number of time samples
# Output data shape: (2, N) - the number of rows in the matrix determines the output dimensions
"""
```

## Dimensionality Reduction

A common use case for `Matmul` is dimensionality reduction:

```python
# Dimensionality reduction example (not tested by mkdocs)
"""
import numpy as np
from sgnts.transforms import Matmul
from sgnts.sources import FakeSeriesSource

# Create a high-dimensional source
source = FakeSeriesSource(
    rate=2048,
    sample_shape=(16,),  # 16-dimensional data
    signal_type="white"
)

# Create a matrix that projects 16D data to 3D
# This could be derived from PCA or other dimensionality reduction techniques
reduction_matrix = np.random.randn(3, 16)  # 3 output dimensions, 16 input dimensions

# Normalize the matrix rows
for i in range(reduction_matrix.shape[0]):
    reduction_matrix[i, :] = reduction_matrix[i, :] / np.linalg.norm(reduction_matrix[i, :])

# Create a Matmul transform for dimensionality reduction
dim_reducer = Matmul(matrix=reduction_matrix)

# Connect and process
source.add_dest(dim_reducer)
source.process()
dim_reducer.process()

# Pull the reduced-dimension frame
frame = dim_reducer.pull()

# The output data has shape (3, N) instead of the original (16, N)
"""
```

## Coordinate Transformations

`Matmul` can be used to apply coordinate transformations:

```python
# Coordinate transformation example (not tested by mkdocs)
"""
import numpy as np
from sgnts.transforms import Matmul
from sgnts.sources import FakeSeriesSource

# Create a source with 3D data (e.g., x, y, z coordinates)
source = FakeSeriesSource(
    rate=2048,
    sample_shape=(3,),
    signal_type="white"
)

# Create a rotation matrix (45 degrees around z-axis)
theta = np.pi / 4  # 45 degrees
rotation_matrix = np.array([
    [np.cos(theta), -np.sin(theta), 0],
    [np.sin(theta), np.cos(theta), 0],
    [0, 0, 1]
])

# Create a Matmul transform
rotator = Matmul(matrix=rotation_matrix)

# Connect and process
source.add_dest(rotator)
source.process()
rotator.process()

# Pull the rotated frame
frame = rotator.pull()

# The data has been rotated 45 degrees around the z-axis
"""
```

## Channel Mixing

`Matmul` can mix channels in multichannel data:

```python
# Channel mixing example (not tested by mkdocs)
"""
import numpy as np
from sgnts.transforms import Matmul
from sgnts.sources import FakeSeriesSource

# Create a source with stereo audio (2 channels)
source = FakeSeriesSource(
    rate=44100,  # Audio sample rate
    sample_shape=(2,),  # Stereo audio
    signal_type="sine",
    fsin=440  # 440 Hz tone
)

# Create a matrix for stereo to mono conversion
# Average the left and right channels (0.5 * left + 0.5 * right)
stereo_to_mono = np.array([[0.5, 0.5]])  # 1x2 matrix

# Create a Matmul transform
mixer = Matmul(matrix=stereo_to_mono)

# Connect and process
source.add_dest(mixer)
source.process()
mixer.process()

# Pull the mixed frame
frame = mixer.pull()

# The output has been converted from stereo (2 channels) to mono (1 channel)
"""
```

## Integration in Processing Pipelines

```python
# Pipeline integration example (not tested by mkdocs)
"""
import numpy as np
from sgnts.transforms import Matmul, AmplifyTransform
from sgnts.sources import FakeSeriesSource
from sgnts.sinks import DumpSeriesSink

# Create a source with multi-channel data
source = FakeSeriesSource(
    rate=2048,
    sample_shape=(4,),
    signal_type="sine",
    fsin=10.0
)

# Create a matrix that extracts specific channels
# This matrix selects channels 0 and 2 from the 4-channel input
extraction_matrix = np.array([
    [1, 0, 0, 0],  # Select channel 0
    [0, 0, 1, 0]   # Select channel 2
])

# Create a Matmul transform
extractor = Matmul(matrix=extraction_matrix)

# Create an amplifier to boost the selected channels
amplifier = AmplifyTransform(factor=2.0)

# Create a sink
sink = DumpSeriesSink(fname="extracted_channels.txt")

# Connect the pipeline
source.add_dest(extractor)
extractor.add_dest(amplifier)
amplifier.add_dest(sink)

# Run the pipeline
for _ in range(10):
    source.process()
    extractor.process()
    amplifier.process()
    sink.process()

# The output file contains only channels 0 and 2, amplified by a factor of 2
"""
```

## Best Practices

When using `Matmul`:

1. **Match matrix dimensions** - ensure that the number of columns in your matrix matches the first dimension of your input data

2. **Consider normalization** - normalize matrix rows to prevent scaling issues, especially for coordinate transforms

3. **Use appropriate backend** - choose the array backend that best matches your computation needs (NumPy, PyTorch, etc.)

4. **Mind computational cost** - matrix multiplication can be computationally intensive for large matrices or high sample rates

5. **Check output shape** - the output shape will be (matrix.shape[0], original_time_samples)

6. **Consider chaining** - multiple matrix multiplications can be combined by multiplying the matrices first, which is more efficient than applying them sequentially