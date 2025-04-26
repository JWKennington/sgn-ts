# Converter Transform

The `Converter` transform changes the data type or device of time-series data, allowing for efficient transitions between different computing backends.

## Overview

`Converter` is a versatile transform that:
- Converts between NumPy arrays and PyTorch tensors
- Changes data precision (e.g., float32 to float16)
- Moves data between devices (e.g., CPU to GPU) for PyTorch tensors
- Preserves the time structure and metadata of frames

## Basic Usage

```python
# Basic usage of Converter (not tested by mkdocs)
"""
from sgnts.transforms import Converter
from sgnts.sources import FakeSeriesSource

# Create a source
source = FakeSeriesSource(
    rate=2048,
    signal_type="sine",
    fsin=10.0  # 10 Hz sine wave
)

# Create a converter to change data type
converter = Converter(
    backend="numpy",  # Target backend
    dtype="float32",  # Target data type
    device="cpu"      # Target device
)

# Connect source to converter
source.add_dest(converter)

# Process data
source.process()
converter.process()

# Pull the converted frame
frame = converter.pull()
"""
```

## Converting Between Backends

```python
# Converting between backends example (not tested by mkdocs)
"""
from sgnts.transforms import Converter
from sgnts.sources import FakeSeriesSource

# Create a source with NumPy backend
source = FakeSeriesSource(
    rate=2048,
    signal_type="sine",
    fsin=10.0
)

# Create a converter to PyTorch
to_torch = Converter(
    backend="torch",  # Convert to PyTorch tensor
    dtype="float32",  # Use float32 precision
    device="cpu"      # Keep on CPU
)

# Connect source to converter
source.add_dest(to_torch)

# Process and get a frame with PyTorch tensor data
source.process()
to_torch.process()
torch_frame = to_torch.pull()

# Create another converter to go back to NumPy
to_numpy = Converter(
    backend="numpy",  # Convert to NumPy array
    dtype="float32"   # Use float32 precision
)

# Connect torch converter to numpy converter
to_torch.add_dest(to_numpy)

# Process and get a frame with NumPy array data
to_numpy.process()
numpy_frame = to_numpy.pull()
"""
```

## Changing Data Precision

```python
# Changing data precision example (not tested by mkdocs)
"""
from sgnts.transforms import Converter
from sgnts.sources import FakeSeriesSource

# Create a source
source = FakeSeriesSource(
    rate=2048,
    signal_type="white"  # White noise
)

# Create a converter to lower precision
to_float16 = Converter(
    backend="numpy",
    dtype="float16"  # Convert to half precision
)

# Connect and process
source.add_dest(to_float16)
source.process()
to_float16.process()

# Get a frame with float16 data
frame = to_float16.pull()
for buf in frame:
    if not buf.is_gap:
        print(f"Data type: {buf.data.dtype}")  # Should be float16
"""
```

## GPU Acceleration with PyTorch

```python
# GPU acceleration example (not tested by mkdocs)
"""
from sgnts.transforms import Converter
from sgnts.sources import FakeSeriesSource

# Create a source
source = FakeSeriesSource(
    rate=2048,
    signal_type="sine",
    fsin=10.0
)

# Create a converter to move data to GPU
to_gpu = Converter(
    backend="torch",
    dtype="float32",
    device="cuda"  # Move to default GPU
)

# For a specific GPU device:
to_specific_gpu = Converter(
    backend="torch",
    dtype="float32",
    device="cuda:0"  # Move to GPU device 0
)

# Connect source to GPU converter
source.add_dest(to_gpu)

# Process data
source.process()
to_gpu.process()

# Get a frame with data on GPU
frame = to_gpu.pull()
"""
```

## Integration in a Processing Pipeline

```python
# Pipeline integration example (not tested by mkdocs)
"""
from sgnts.transforms import Converter, AmplifyTransform
from sgnts.sources import FakeSeriesSource
from sgnts.sinks import DumpSeriesSink

# Create a source
source = FakeSeriesSource(
    rate=2048,
    signal_type="sine",
    fsin=10.0
)

# Create a pipeline with type conversion
to_torch = Converter(backend="torch", dtype="float32", device="cuda")
amplifier = AmplifyTransform(factor=2.0)  # Will operate on torch tensors
to_numpy = Converter(backend="numpy", dtype="float32")  # Convert back for output
sink = DumpSeriesSink(fname="amplified_signal.txt")

# Connect the pipeline
source.add_dest(to_torch)
to_torch.add_dest(amplifier)
amplifier.add_dest(to_numpy)
to_numpy.add_dest(sink)

# Run the pipeline
for _ in range(10):
    source.process()
    to_torch.process()
    amplifier.process()
    to_numpy.process()
    sink.process()
"""
```

## Error Handling

```python
# Error handling example (not tested by mkdocs)
"""
from sgnts.transforms import Converter

# This will fail if PyTorch is not installed
try:
    converter = Converter(
        backend="torch",
        dtype="float32",
        device="cuda"
    )
except ImportError as e:
    print(f"PyTorch not available: {e}")
    # Fall back to NumPy
    converter = Converter(
        backend="numpy",
        dtype="float32"
    )

# This will fail because NumPy can only use the CPU device
try:
    invalid_converter = Converter(
        backend="numpy",
        dtype="float32",
        device="cuda"  # Invalid for NumPy
    )
except ValueError as e:
    print(f"Error: {e}")  # "Converting to numpy only supports device as cpu"
"""
```

## Best Practices

When using `Converter`:

1. **Use appropriate precision** - lower precision (float16) can improve performance but may reduce accuracy

2. **Consider memory transfers** - moving data between CPU and GPU incurs overhead, so minimize unnecessary transfers

3. **Check availability** - ensure PyTorch is installed if using the "torch" backend

4. **Place converters strategically** - put `Converter` transforms at boundaries between processing that benefits from different backends

5. **Batch processing** - for GPU acceleration, process larger batches to amortize transfer costs

6. **Match downstream expectations** - ensure the backend and data type match what downstream components expect