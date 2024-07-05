import numpy as np

class Math:
    def cat_func(xs, axis):
        return np.concatenate(xs, axis=axis)

    def pad_func(data, pad_samples: tuple):
        npad = [(0, 0)] * data.ndim
        npad[-1] = pad_samples
        return np.pad(data, npad, "constant")

    def full_func(shape, fill_value):
        return np.full(shape, fill_value)

    zeros_func = np.zeros
