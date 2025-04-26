# Understanding Time Constants

## Introduction

The `time` module in SGN-TS provides fundamental constants for working with time at different scales. Though small in code size, this module plays a crucial role in ensuring consistent time handling across the library.

This tutorial will guide you through:

1. Understanding the time constants and their roles
2. Using time constants for conversions
3. Integrating time constants with other SGN-TS components
4. Best practices for time representation

## The Time Enum

The `Time` class is an enumeration that defines standard time units in terms of nanoseconds. It inherits from both `int` and `Enum`, allowing the values to be used directly in numeric calculations.

```python
# Example of the Time enum (not tested by mkdocs)
"""
from sgnts.base.time import Time

# Time constants
SECONDS = Time.SECONDS         # 1,000,000,000 nanoseconds
MILLISECONDS = Time.MILLISECONDS  # 1,000,000 nanoseconds
MICROSECONDS = Time.MICROSECONDS  # 1,000 nanoseconds
NANOSECONDS = Time.NANOSECONDS    # 1 nanosecond
"""
```

Each constant represents the number of nanoseconds in the respective time unit:

| Constant | Value | Description |
|----------|-------|-------------|
| `Time.SECONDS` | 1,000,000,000 | Number of nanoseconds in a second |
| `Time.MILLISECONDS` | 1,000,000 | Number of nanoseconds in a millisecond |
| `Time.MICROSECONDS` | 1,000 | Number of nanoseconds in a microsecond |
| `Time.NANOSECONDS` | 1 | Number of nanoseconds in a nanosecond |

## Using Time Constants for Conversions

One of the primary uses of the `Time` constants is to convert between different time units consistently:

```python
# Example of time unit conversions (not tested by mkdocs)
"""
from sgnts.base.time import Time

# Convert 2.5 seconds to nanoseconds
seconds = 2.5
nanoseconds = seconds * Time.SECONDS  # 2,500,000,000 nanoseconds

# Convert 500 milliseconds to nanoseconds
milliseconds = 500
nanoseconds = milliseconds * Time.MILLISECONDS  # 500,000,000 nanoseconds

# Convert nanoseconds back to seconds
nanoseconds = 3_750_000_000
seconds = nanoseconds / Time.SECONDS  # 3.75 seconds
"""
```

The `Time` constants ensure that conversions are consistent and reduce the chance of errors from manually calculating conversion factors.

## Integration with SGN-TS Components

The `Time` constants are used throughout SGN-TS to provide consistent time representation. Here are some examples of how they integrate with other components:

### Integration with Offset

The `Offset` class uses `Time.SECONDS` for converting between offsets and nanoseconds:

```python
# Example of Time integration with Offset (not tested by mkdocs)
"""
from sgnts.base.time import Time
from sgnts.base.offset import Offset

# Converting from nanoseconds to offsets
timestamp_ns = 1_500_000_000  # 1.5 seconds in nanoseconds
offset = Offset.fromns(timestamp_ns)  # Uses Time.SECONDS internally

# Converting from offsets to nanoseconds
offset = 24576  # An offset value
nanoseconds = Offset.tons(offset)  # Uses Time.SECONDS internally
"""
```

### Specifying Timeouts and Durations

Time constants are useful for specifying timeouts, durations, and other time-related parameters:

```python
# Example of using Time constants for parameters (not tested by mkdocs)
"""
from sgnts.base.time import Time
from sgnts.base import TSTransform

# Create a transform with a 5-second maximum age
transform = TSTransform(
    sink_pad_names=["input"],
    source_pad_names=["output"],
    max_age=5 * Time.SECONDS  # 5 seconds in nanoseconds
)

# Specify a timeout for data retrieval
timeout_ms = 500
timeout_ns = timeout_ms * Time.MILLISECONDS  # Convert to nanoseconds
"""
```

## Best Practices for Time Representation

### Consistent Time Units

Always use the most appropriate time constant for your use case to ensure consistency:

```python
# Example of consistent time units (not tested by mkdocs)
"""
from sgnts.base.time import Time

# Good: Clear and consistent time unit usage
timeout_ns = 500 * Time.MILLISECONDS  # 500 milliseconds in nanoseconds
duration_ns = 2 * Time.SECONDS  # 2 seconds in nanoseconds

# Avoid: Mixed or unclear time units
timeout_ns = 500_000_000  # What unit is this? Is it 500 ms or 0.5 seconds?
duration_ns = 2_000_000_000  # Is this 2 seconds or 2 billion something else?
"""
```

### Nanosecond Precision for Integer Calculations

When working with time values that need to maintain precise integer values, use nanoseconds as the base unit:

```python
# Example of precision with nanoseconds (not tested by mkdocs)
"""
from sgnts.base.time import Time

# Integer calculation with nanosecond precision
duration1_ns = 1 * Time.SECONDS + 500 * Time.MILLISECONDS  # 1.5 seconds
duration2_ns = 750 * Time.MILLISECONDS  # 0.75 seconds

# Calculate total duration in nanoseconds, then convert to seconds
total_ns = duration1_ns + duration2_ns  # 2,250,000,000 nanoseconds
total_seconds = total_ns / Time.SECONDS  # 2.25 seconds

# Avoid using floating-point for precise time calculations
# float_duration1 = 1.5  # seconds
# float_duration2 = 0.75  # seconds
# total = float_duration1 + float_duration2  # 2.25, but may have precision issues
"""
```

### Handling Large Time Values

When dealing with very large time values (like Unix timestamps), be mindful of potential overflow:

```python
# Example of handling large time values (not tested by mkdocs)
"""
from sgnts.base.time import Time
import time

# Current Unix timestamp in seconds
unix_timestamp = time.time()  # e.g., 1682719937.123456

# Convert to nanoseconds (still within int64 range)
unix_timestamp_ns = int(unix_timestamp * Time.SECONDS)

# For extremely large times, consider working with more manageable units
# or using specialized data structures for time representation
"""
```

## Advanced Uses of Time Constants

### Creating Custom Time Utility Functions

You can create utility functions that leverage the `Time` constants for common conversions:

```python
# Example of custom time utility functions (not tested by mkdocs)
"""
from sgnts.base.time import Time

def seconds_to_ns(seconds):
    \"\"\"Convert seconds to nanoseconds.\"\"\"
    return int(seconds * Time.SECONDS)

def ms_to_ns(milliseconds):
    \"\"\"Convert milliseconds to nanoseconds.\"\"\"
    return int(milliseconds * Time.MILLISECONDS)

def ns_to_seconds(nanoseconds):
    \"\"\"Convert nanoseconds to seconds.\"\"\"
    return nanoseconds / Time.SECONDS

def ns_to_ms(nanoseconds):
    \"\"\"Convert nanoseconds to milliseconds.\"\"\"
    return nanoseconds / Time.MILLISECONDS

# Usage example
duration_ns = seconds_to_ns(1.5) + ms_to_ns(500)  # 2,000,000,000 ns
print(f"Duration: {ns_to_seconds(duration_ns)} seconds")  # 2.0 seconds
"""
```

### Working with Time Ranges

Time constants can be used to define time ranges and intervals:

```python
# Example of time ranges with Time constants (not tested by mkdocs)
"""
from sgnts.base.time import Time
from sgnts.base.slice_tools import TSSlice
from sgnts.base.offset import Offset

# Define a 5-second window centered at time t
center_time_ns = 1_000_000_000  # 1 second
window_size_ns = 5 * Time.SECONDS
start_time_ns = center_time_ns - (window_size_ns // 2)
end_time_ns = center_time_ns + (window_size_ns // 2)

# Convert to offsets and create a time slice
start_offset = Offset.fromns(start_time_ns)
end_offset = Offset.fromns(end_time_ns)
time_window = TSSlice(start_offset, end_offset)

# Process data within the time window
# ...
"""
```

## Conclusion

The `Time` constants provide a simple yet powerful foundation for time handling in SGN-TS. By using these constants consistently, you can:

1. Ensure clear and consistent time unit conversions
2. Avoid errors from manual calculation of conversion factors
3. Maintain precise integer-based time calculations
4. Integrate smoothly with other SGN-TS components

While the implementation is minimal, the impact of these constants on code clarity and correctness is significant. Always prefer using these constants over hardcoded values when working with time in SGN-TS.

## Next Steps

- Learn about [Offsets](offset_tutorial.md) for time representation in SGN-TS
- Explore [Time Slices](slice_tools_tutorial.md) for working with time intervals
- Study the [Core Components](base_tutorial.md) to see how time is used in the pipeline architecture