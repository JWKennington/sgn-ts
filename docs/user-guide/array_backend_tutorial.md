# Working with Array Backends

## Introduction

The `ArrayBackend` system in SGN-TS provides a flexible abstraction for array operations, allowing your code to run on different computational backends without modifications. This tutorial will guide you through:

1. Understanding the array backend architecture
2. Using the built-in backends (NumPy)
3. Creating custom backends
4. Switching between backends in your application

## Understanding Array Backend Architecture

SGN-TS uses a backend abstraction pattern to separate the interface of array operations from their implementations. This allows the same high-level code to run on different computational frameworks.

The core of this system is the `ArrayBackend` abstract base class, which defines the API for array operations such as `zeros`, `ones`, `matmul`, etc. Concrete implementations (like `NumpyBackend`) provide the actual functionality.

### Why Use Backend Abstraction?

- **Portability**: Write code once, run it on different hardware (CPU, GPU) using the appropriate backend
- **Flexibility**: Switch between backends without changing your application code
- **Performance**: Choose the backend that offers the best performance for your specific use case
- **Future-proofing**: As new array libraries emerge, you can add support without rewriting your application

## Using Built-in Backends

### NumPy Backend

The most straightforward backend in SGN-TS is the `NumpyBackend`, which implements array operations using NumPy.

```python
# Example of how to use NumpyBackend (not tested by mkdocs)
"""
from sgnts.base.numpy_backend import NumpyBackend

# Create arrays
zeros_array = NumpyBackend.zeros((3, 4))  # 3x4 array of zeros
ones_array = NumpyBackend.ones((2, 2))    # 2x2 array of ones
range_array = NumpyBackend.arange(0, 10, 0.5)  # Array from 0 to 9.5 with step 0.5

# Perform operations
filled_array = NumpyBackend.full((3, 3), 5)  # 3x3 array filled with 5
matrix_a = NumpyBackend.full((2, 3), 2)
matrix_b = NumpyBackend.full((3, 2), 3)
result = NumpyBackend.matmul(matrix_a, matrix_b)  # Matrix multiplication
"""
```

### Working with Arrays from Different Backends

Each backend operation returns arrays specific to that backend. For example, `NumpyBackend` operations return `numpy.ndarray` objects.

```python
# Example (not tested by mkdocs)
"""
# NumPy backend operations return numpy arrays
array1 = NumpyBackend.zeros((3, 3))
print(type(array1))  # <class 'numpy.ndarray'>

# You can use backend operations on arrays from the same backend
array2 = NumpyBackend.ones((3, 3))
summed = NumpyBackend.sum(array1 + array2)  # Converts to 9.0
"""
```

## Creating Custom Backends

You can create your own backend by subclassing `ArrayBackend` and implementing its methods. Here's a minimal example of a custom backend that uses NumPy but applies a scaling factor to all operations:

```python
# Example of custom backend (not tested by mkdocs)
"""
from sgnts.base.array_backend import ArrayBackend
from sgnts.base.numpy_backend import NumpyBackend
import numpy as np

class ScalingBackend(ArrayBackend):
    \"\"\"A custom backend that scales all created arrays by a factor.\"\"\"
    
    SCALE_FACTOR = 2.0
    
    @staticmethod
    def zeros(shape):
        \"\"\"Create an array of zeros scaled by SCALE_FACTOR.\"\"\"
        return NumpyBackend.zeros(shape) * ScalingBackend.SCALE_FACTOR
    
    @staticmethod
    def ones(shape):
        \"\"\"Create an array of ones scaled by SCALE_FACTOR.\"\"\"
        return NumpyBackend.ones(shape) * ScalingBackend.SCALE_FACTOR
    
    # Implement other methods similarly...
    
    @staticmethod
    def arange(stop, start=0, step=1):
        \"\"\"Create an arange array scaled by SCALE_FACTOR.\"\"\"
        return NumpyBackend.arange(stop, start, step) * ScalingBackend.SCALE_FACTOR
"""
```

### Backend Implementation Requirements

When implementing a custom backend, you must:

1. Inherit from the `ArrayBackend` base class
2. Implement all abstract methods defined in `ArrayBackend`
3. Ensure that your methods follow the same API contract (parameters and return types)
4. Make sure your array types are compatible with the rest of the SGN-TS library


### Global Backend Configuration

In some cases, you might want to set a default backend for your entire application. While SGN-TS doesn't have a built-in global backend setting, you can create one:

```python
# Example of global configuration (not tested by mkdocs)
"""
# In your application configuration
from sgnts.base.numpy_backend import NumpyBackend

# Set the default backend
DEFAULT_BACKEND = NumpyBackend

# Then in your components
def create_component(backend=None):
    backend = backend or DEFAULT_BACKEND
    # Create component using the specified or default backend
"""
```

## Performance Considerations

Different backends can offer significant performance differences depending on:

1. The size and dimensionality of your data
2. The hardware you're running on (CPU vs. GPU)
3. The specific operations being performed

It's worth benchmarking your application with different backends to determine which offers the best performance for your particular use case.

## Advanced Usage: Extending Existing Backends

You can extend existing backends to add new functionality or modify behavior:

```python
# Example of extending backends (not tested by mkdocs)
"""
from sgnts.base.numpy_backend import NumpyBackend
import numpy as np

class ExtendedNumpyBackend(NumpyBackend):
    \"\"\"Extended NumPy backend with additional operations.\"\"\"
    
    @staticmethod
    def softmax(x, axis=None):
        \"\"\"Compute softmax values for the array.\"\"\"
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    @staticmethod
    def norm(x, ord=None, axis=None):
        \"\"\"Compute the norm of the array.\"\"\"
        return np.linalg.norm(x, ord=ord, axis=axis)
"""
```

## Conclusion

The `ArrayBackend` system in SGN-TS provides a powerful abstraction for array operations, allowing you to write code that is portable across different computational frameworks. By understanding and leveraging this system, you can create more flexible and maintainable applications.

Remember to choose the right backend for your specific needs, and consider implementing custom backends when specialized functionality is required.

## Next Steps

- Explore the API reference for the full list of operations supported by `ArrayBackend`
- Look at the implementation of different backend classes to understand how they work
- Try benchmarking your application with different backends to identify performance bottlenecks
