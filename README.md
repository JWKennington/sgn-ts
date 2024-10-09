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
