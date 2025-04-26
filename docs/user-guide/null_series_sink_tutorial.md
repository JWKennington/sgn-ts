# NullSeriesSink

The `NullSeriesSink` is a sink component that consumes data without performing any operations on it.

## Overview

`NullSeriesSink` is useful when you need to:
- Terminate a pipeline branch without writing data anywhere
- Monitor data flow for debugging (with `verbose=True`)
- Measure pipeline latency
- Test pipeline performance without I/O overhead

## Basic Usage

```python
# Basic usage of NullSeriesSink (not tested by mkdocs)
"""
from sgnts.sinks import NullSeriesSink
from sgnts.sources import FakeSeriesSource

# Create a source of data
source = FakeSeriesSource(
    rate=2048,
    signal_type="sine",
    fsin=10.0  # 10 Hz sine wave
)

# Create a null sink
sink = NullSeriesSink(
    verbose=True  # Print frame information and latency
)

# Connect the source to the sink
source.add_dest(sink)

# Run the pipeline for a few iterations
for _ in range(5):
    source.process()
    sink.process()
"""
```

## Latency Monitoring

When `verbose=True`, the `NullSeriesSink` can help monitor pipeline latency:

```python
# Latency monitoring example (not tested by mkdocs)
"""
from sgnts.sinks import NullSeriesSink
from sgnts.sources import FakeSeriesSource
from sgnts.transforms import AmplifyTransform

# Create a source with real-time behavior
source = FakeSeriesSource(
    rate=2048,
    signal_type="sine",
    fsin=10.0,
    real_time=True  # Enable real-time mode
)

# Create a transform
amplifier = AmplifyTransform(factor=2.0)

# Create a null sink with verbose output
sink = NullSeriesSink(verbose=True)

# Connect the elements
source.add_dest(amplifier)
amplifier.add_dest(sink)

# Run the pipeline
for _ in range(10):
    source.process()
    amplifier.process()
    sink.process()
    
# The sink will print frame information and latency for each frame
"""
```

## Multiple Sink Pads

Unlike some other sinks, `NullSeriesSink` can handle multiple sink pads:

```python
# Multiple sink pads example (not tested by mkdocs)
"""
from sgnts.sinks import NullSeriesSink
from sgnts.sources import FakeSeriesSource

# Create two sources
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

# Create a null sink with verbose output
sink = NullSeriesSink(
    verbose=True,
    name="multi_sink"
)

# Create and connect sink pads
sink_pad1 = sink.create_pad("sine_input:snk")
sink_pad2 = sink.create_pad("noise_input:snk")

# Connect sources to the respective sink pads
source1.add_dest(sink, sink_pad1)
source2.add_dest(sink, sink_pad2)

# Run the pipeline
for _ in range(5):
    source1.process()
    source2.process()
    sink.process()
    
# The sink will process and print information for both inputs
"""
```

## End of Stream Handling

The `NullSeriesSink` properly handles End of Stream (EOS) markers:

```python
# EOS handling example (not tested by mkdocs)
"""
from sgnts.sinks import NullSeriesSink
from sgnts.sources import FakeSeriesSource

# Create a source with a finite duration
source = FakeSeriesSource(
    rate=2048,
    signal_type="sine",
    fsin=10.0,
    t0=0,
    end=5  # End after 5 seconds
)

# Create a null sink
sink = NullSeriesSink(verbose=True)

# Connect source to sink
source.add_dest(sink)

# Run the pipeline until EOS
while not sink.EOS:
    source.process()
    sink.process()
    
print("Pipeline completed")
"""
```

## Best Practices

When using `NullSeriesSink`:

1. **Enable verbose mode** (`verbose=True`) when debugging to see frame information and latency

2. **Use for performance testing** - `NullSeriesSink` adds minimal overhead, making it ideal for measuring processing performance

3. **Monitor EOS flags** - `NullSeriesSink` properly propagates EOS flags, making it useful for controlled pipeline shutdown

4. **Consider multiple destinations** - you can connect a source to both a `NullSeriesSink` for monitoring and another sink for actual data processing

5. **Check latency patterns** - increasing latency values might indicate performance issues in your pipeline