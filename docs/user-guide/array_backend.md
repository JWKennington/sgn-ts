# Array Backend Documentation

## Overview

The `ArrayBackend` class provides an interface for implementing array operations across multiple backends (e.g., NumPy, PyTorch, TensorFlow). This abstraction allows writing generic code that can execute seamlessly on different array computation frameworks.

The class defines a set of fundamental array operations as static methods. These methods must be implemented in subclasses specific to each backend.

## Available Operations

### `arange`
Creates a 1-D array with values ranging from `start` to `stop` (exclusive) with a given `step`.

**Parameters:**
- `stop` (float): The stopping value for the set of points.
- `start` (float, default `0`): The starting value for the set of points.
- `step` (float, default `1`): The difference between consecutive elements.

**Returns:**
- `Array`: A 1-D array of evenly spaced values.

### `cat`
Concatenates multiple arrays along a specified axis.

**Parameters:**
- `data` (Sequence[Array]): The list of arrays to concatenate.
- `axis` (int): The axis along which to concatenate.

**Returns:**
- `Array`: A concatenated array.

### `full`
Creates an array filled with a specified value.

**Parameters:**
- `shape` (Tuple[int, ...]): The shape of the output array.
- `fill_value` (Any): The value to fill the array with.

**Returns:**
- `Array`: An array filled with `fill_value`.

### `matmul`
Performs matrix multiplication on two arrays.

**Parameters:**
- `a` (Array): The first input array.
- `b` (Array): The second input array.

**Returns:**
- `Array`: The result of the matrix multiplication.

### `ones`
Creates an array filled with ones.

**Parameters:**
- `shape` (Tuple[int, ...]): The shape of the output array.

**Returns:**
- `Array`: An array of ones.

### `pad`
Pads an array with zeros.

**Parameters:**
- `data` (Array): The input array to pad.
- `pad_samples` (Tuple[int, int]): The number of zeros to pad at each end.

**Returns:**
- `Array`: The padded array.

### `stack`
Stacks multiple arrays along a new axis.

**Parameters:**
- `data` (Sequence[Array]): The list of arrays to stack.
- `axis` (int, default `0`): The axis along which to stack.

**Returns:**
- `Array`: A stacked array.

### `sum`
Computes the sum of array elements along a specified axis.

**Parameters:**
- `a` (Array): The input array.
- `axis` (Optional[Union[int, Tuple[int, ...]]], default `None`): The axis or axes along which to compute the sum.

**Returns:**
- `Array`: The sum of elements along the given axis.

### `zeros`
Creates an array filled with zeros.

**Parameters:**
- `shape` (Tuple[int, ...]): The shape of the output array.

**Returns:**
- `Array`: An array of zeros.

## Implementation Notes
- All methods are defined as static methods and must be implemented in backend-specific subclasses.
- The `stack` method utilizes `cat` internally.
- The design allows compatibility across different array-processing libraries by enforcing a common API.
