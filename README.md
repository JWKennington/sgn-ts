<!-- index.rst content start -->

# SGN-TS (SGN TimeSeries)

SGN-TS is set of extensions to the core library `sgn`, that includes functionality
specific to TimeSeries analysis. This page is for documenatation of the `sgnts` package, but there is a family of
libraries that extend the functionality of SGN in other ways, including:

- [`sgn`](https://docs.ligo.org/greg/sgn/): Base library for SGN
- [`sgn-try`](https://git.ligo.org/greg/sgn-try): Process monitoring and alerting utilities for SGN
- [`sgn-ligo`](https://git.ligo.org/greg/sgn-try): LIGO-specific utilities for SGN

## Installation

To install SGN-TS, simply run:

```bash
pip install sgn-ts
```

More SGN-TS-specific documentation coming soon.

## Design principles

The sgn frame object is modified to include a list of buffers representing time series data.  

## Example

```python
#!/usr/bin/env python3

from dataclasses import dataclass
import numpy
from sgn import Pipeline, NullSink
from sgnts.base import Offset, SeriesBuffer, TSFrame, TSSource, TSSlice, TSSlices
import time

@dataclass
class SineSource(TSSource):
    frequency: float = 32
    rate: int = 2048

    def __post_init__(self):
        super().__post_init__()
        # FIXME turn these into a helper method
        self.num_samples = Offset.sample_stride(self.rate)
        self.shape = (self.num_samples,)

    def new(self, pad):
        time.sleep(1)
        # FIXME turn these into a helper method        
        buf = SeriesBuffer(offset=self.offset[pad], sample_rate=self.rate, data=0, shape=self.shape)
        tarr = numpy.arange(self.num_samples) / buf.sample_rate + buf.t0
        self.offset[pad] += Offset.fromsamples(self.num_samples, self.rate)
        frame = TSFrame(buffers=[buf])

        buf.data[:] = numpy.sin(2 * numpy.pi * self.frequency * tarr)
        return frame

# FIXME don't require names
src = SineSource(name="sine", source_pad_names = ["H1"], frequency=32, rate=2048)
sink = NullSink(name="fakesink", sink_pad_names = ["H1"])

# Create the Pipeline
p = Pipeline()

# FIXME simplify linking
p.insert(
    src,
    sink,
    link_map={"fakesink:sink:H1": "sine:src:H1"},
)
```
