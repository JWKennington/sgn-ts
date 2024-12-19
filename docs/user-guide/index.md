# User's guide

Before reading this guide you should carefully read and understand the [SGN
developers guide](https://greg.docs.ligo.org/sgn/#developer-s-guide).   

The core motivation with SGN TS (`sgnts`) is to build Time Series (TS) handling
into SGN.  This is appropriate for e.g., signal processing applications.  Of
course nothing is stopping you from doing any of these things with just SGN,
but you will likely have to deal with some of the conceptual and technical
hurdles that this library solves.  That being said, there are many limitations
of `sgnts` and you should understand those carefully in the context of your
project.  We are open to making changes that reach a wider audience, so please
let us know your thoughts. 

## New Concepts beyond `sgn`

- **Data**: data are now rigidly defined to be uniformly sampled time series.  There is
  an expectation that elements will deal with data in a synchronous way.
- **Syncing**: synchronization means that the continuity equation must be satisfied.  Data
  cannot be produced at a higher rate in one source element than another,
  otherwise synchronous operations will be impossible without data "piling up"
  somewhere.
- **Temporal Bookkeeping**: time index accuracy is important. The library aims to keep single
  sample point timing accuracies even for applications that are designed to run
  uninterrupted for years.  This requires a bit of rigidity in bookkeeping, but we
  try to hide as much as possible from the casual developer and user.
