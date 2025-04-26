# Building Complete Pipelines

This tutorial demonstrates how to build complete signal processing pipelines by connecting sources, transforms, and sinks in SGN-TS.

## Overview

Signal processing pipelines in SGN-TS consist of:
- **Sources**: Components that generate or read time-series data
- **Transforms**: Components that process and modify time-series data
- **Sinks**: Components that consume or output processed data

These components are connected through pads, forming a directed graph that defines how data flows through the system.

## Basic Pipeline Structure

A typical pipeline follows this structure:

```
Source → Transform(s) → Sink
```

Data flows from source to sink, with transforms modifying the data along the way.

## Creating a Simple Pipeline

Let's create a simple pipeline that generates a sine wave, amplifies it, and writes it to a file:

```python
# Simple pipeline example (not tested by mkdocs)
"""
from sgnts.sources import FakeSeriesSource
from sgnts.transforms import AmplifyTransform
from sgnts.sinks import DumpSeriesSink

# Create components
source = FakeSeriesSource(
    rate=2048,
    signal_type="sine",
    fsin=10.0,  # 10 Hz sine wave
    name="sine_source"
)

amplifier = AmplifyTransform(
    factor=2.0,  # Double the amplitude
    name="amplifier"
)

sink = DumpSeriesSink(
    fname="amplified_sine.txt",
    name="file_sink"
)

# Connect components
source.add_dest(amplifier)
amplifier.add_dest(sink)

# Run the pipeline for a few iterations
for _ in range(10):
    source.process()
    amplifier.process()
    sink.process()
"""
```

## Understanding Pads

Pads are the connection points between components:
- **Source pads**: Points where data leaves a component
- **Sink pads**: Points where data enters a component

Components can have multiple pads, allowing complex connection topologies.

```python
# Working with named pads (not tested by mkdocs)
"""
from sgnts.transforms import Adder

# Create an adder with multiple sink pads
adder = Adder(name="my_adder")

# Create named sink pads
input1_pad = adder.create_pad("input1:snk")
input2_pad = adder.create_pad("input2:snk")

# Now you can connect sources to specific pads
source1.add_dest(adder, input1_pad)
source2.add_dest(adder, input2_pad)
"""
```

## Building a Multi-branch Pipeline

Let's build a more complex pipeline with multiple branches:

```python
# Multi-branch pipeline example (not tested by mkdocs)
"""
from sgnts.sources import FakeSeriesSource
from sgnts.transforms import AmplifyTransform, Converter, Adder
from sgnts.sinks import DumpSeriesSink

# Create two sources
source1 = FakeSeriesSource(
    rate=2048,
    signal_type="sine",
    fsin=10.0,
    name="sine_source"
)

source2 = FakeSeriesSource(
    rate=2048,
    signal_type="sine",
    fsin=20.0,  # Different frequency
    name="second_sine_source"
)

# Create transforms for each branch
amplifier1 = AmplifyTransform(factor=1.5, name="amplifier1")
amplifier2 = AmplifyTransform(factor=0.75, name="amplifier2")

# Create an adder to combine the branches
adder = Adder(name="adder")
input1_pad = adder.create_pad("input1:snk")
input2_pad = adder.create_pad("input2:snk")

# Create a sink
sink = DumpSeriesSink(fname="combined_signals.txt", name="file_sink")

# Connect the pipeline
source1.add_dest(amplifier1)
amplifier1.add_dest(adder, input1_pad)

source2.add_dest(amplifier2)
amplifier2.add_dest(adder, input2_pad)

adder.add_dest(sink)

# Run the pipeline
for _ in range(10):
    source1.process()
    source2.process()
    amplifier1.process()
    amplifier2.process()
    adder.process()
    sink.process()

# The output file will contain the sum of:
# - A 10 Hz sine wave amplified by factor 1.5
# - A 20 Hz sine wave amplified by factor 0.75
"""
```

## Pipeline with Different Sample Rates

When components operate at different sample rates, a `Resampler` transform is needed:

```python
# Pipeline with resampling (not tested by mkdocs)
"""
from sgnts.sources import FakeSeriesSource
from sgnts.transforms import Resampler, AmplifyTransform
from sgnts.sinks import DumpSeriesSink

# Create a source at 8192 Hz
source = FakeSeriesSource(
    rate=8192,
    signal_type="sine",
    fsin=100.0,
    name="high_rate_source"
)

# Create a resampler to reduce the rate
resampler = Resampler(
    inrate=8192,
    outrate=2048,
    name="downsampler"
)

# Create an amplifier
amplifier = AmplifyTransform(factor=2.0, name="amplifier")

# Create a sink
sink = DumpSeriesSink(fname="resampled_and_amplified.txt", name="file_sink")

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
"""
```

## Conditional Processing with Gate

The `Gate` transform allows conditional processing based on a control signal:

```python
# Conditional processing example (not tested by mkdocs)
"""
from sgnts.sources import FakeSeriesSource
from sgnts.transforms import Gate, Threshold
from sgnts.sinks import DumpSeriesSink

# Create a data source
data_source = FakeSeriesSource(
    rate=2048,
    signal_type="sine",
    fsin=10.0,
    name="data_source"
)

# Create a control source
control_source = FakeSeriesSource(
    rate=2048,
    signal_type="sine",
    fsin=0.5,  # Slow oscillation
    name="control_source"
)

# Create a threshold for the control signal
threshold = Threshold(
    threshold=0.7,
    name="threshold"
)

# Create a gate
gate = Gate(
    control="control",
    name="gate"
)
data_pad = gate.create_pad("data:snk")
control_pad = gate.create_pad("control:snk")

# Create a sink
sink = DumpSeriesSink(fname="gated_signal.txt", name="file_sink")

# Connect the pipeline
data_source.add_dest(gate, data_pad)

control_source.add_dest(threshold)
threshold.add_dest(gate, control_pad)

gate.add_dest(sink)

# Run the pipeline
for _ in range(20):
    data_source.process()
    control_source.process()
    threshold.process()
    gate.process()
    sink.process()

# The output file will contain the data signal only during periods
# when the control signal is above 0.7
"""
```

## Signal Processing Pipeline

Here's a more advanced signal processing pipeline that demonstrates several transforms:

```python
# Advanced signal processing example (not tested by mkdocs)
"""
import numpy as np
from sgnts.sources import FakeSeriesSource
from sgnts.transforms import Correlate, Threshold, AmplifyTransform
from sgnts.sinks import DumpSeriesSink

# Create a filter for pattern matching
filter_width = 32
t = np.linspace(0, 1, filter_width)
filter_pattern = np.sin(2 * np.pi * 5 * t) * np.hamming(filter_width)

# Create a noisy source
source = FakeSeriesSource(
    rate=2048,
    signal_type="white",
    name="noisy_source"
)

# Create a correlator to detect patterns
correlator = Correlate(
    sample_rate=2048,
    filters=filter_pattern,
    name="pattern_detector"
)

# Create a threshold to extract high correlation regions
detector = Threshold(
    threshold=0.7,
    startwn=16,
    stopwn=16,
    name="threshold_detector"
)

# Create an amplifier to highlight detected patterns
amplifier = AmplifyTransform(
    factor=2.0,
    name="amplifier"
)

# Create a sink
sink = DumpSeriesSink(
    fname="detected_patterns.txt",
    name="file_sink"
)

# Connect the pipeline
source.add_dest(correlator)
correlator.add_dest(detector)
detector.add_dest(amplifier)
amplifier.add_dest(sink)

# Run the pipeline
for _ in range(20):
    source.process()
    correlator.process()
    detector.process()
    amplifier.process()
    sink.process()

# The output file will contain only the detected patterns that
# correlate strongly with the filter_pattern
"""
```

## Managing Resources in Pipelines

When creating pipelines, consider these best practices:

### 1. Processing Order

Components must be processed in the correct order (sources first, then transforms, then sinks):

```python
# Processing order example (not tested by mkdocs)
"""
# Correct processing order
source.process()
transform1.process()
transform2.process()
sink.process()

# Incorrect order would cause transforms to process old data
# or sinks to miss the latest processed data
"""
```

### 2. End of Stream Handling

Check for End of Stream (EOS) conditions to properly terminate processing:

```python
# EOS handling example (not tested by mkdocs)
"""
from sgnts.sources import FakeSeriesSource
from sgnts.sinks import NullSeriesSink

# Create a finite source
source = FakeSeriesSource(
    rate=2048,
    signal_type="sine",
    fsin=10.0,
    t0=0,
    end=5  # End after 5 seconds
)

# Create a sink
sink = NullSeriesSink()

# Connect
source.add_dest(sink)

# Run until EOS
while not sink.EOS:
    source.process()
    sink.process()

print("Pipeline completed")
"""
```

### 3. Named Components

Use descriptive names for components to make debugging easier:

```python
# Named components example (not tested by mkdocs)
"""
# Give each component a descriptive name
source = FakeSeriesSource(name="main_input_signal")
filter1 = AmplifyTransform(name="pre_amplifier")
filter2 = Threshold(name="noise_gate")
sink = DumpSeriesSink(name="output_file_writer")
"""
```

## Best Practices

When building pipelines:

1. **Plan your topology** - sketch out the pipeline structure before implementation

2. **Name your components** - use descriptive names for easier debugging

3. **Check sample rates** - ensure compatible rates or add resamplers

4. **Validate connections** - check that all required pads are connected

5. **Process in order** - always process components in the correct order (source → transform → sink)

6. **Handle EOS properly** - check EOS flags to terminate processing gracefully

7. **Consider performance** - profile your pipeline to identify bottlenecks

8. **Test incremental builds** - build and test the pipeline step by step

9. **Manage resources** - close files and release resources when done

10. **Error handling** - implement appropriate error checking and recovery