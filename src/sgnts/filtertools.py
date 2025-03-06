"""Utilities for creating, modifying, and applying filters to data.

Largely wrappers around scipy.signal functions, but with some additional
functionality for working with filters in the context of sgnts.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy


def sinc_sample_size(f_cutoff: int, f_sample: int, n_zeros: int) -> int:
    """Compute the sample size for a sinc filter with the given cutoff
    frequency, sample rate, and number of zeros.

    Args:
        f_cutoff:
            int, the cutoff frequency of the sinc filter
        f_sample:
            int, the sample rate of the data to which the filter will be applied
        n_zeros:
            int, the number of zeros in the sinc filter

    Returns:
        int, the sample size of the sinc filter
    """
    boundary = n_zeros / (f_cutoff / f_sample) + 1
    boundary = int(numpy.ceil(boundary))
    if boundary % 2 == 0:
        boundary += 1
    return boundary


def low_pass_filter(
    f_cutoff: int,
    f_sample: int,
    size: Optional[int] = None,
    n_zeros: Optional[int] = None,
    win_func: Optional[Callable] = None,
    normalize: bool = True,
    fix_size: Optional[int] = None,
):
    """

    Args:
        f_cutoff:
        f_sample:
        size:
        n_zeros:
        win_func:
            Callable, a window function to apply to the filter. Typically
            a function from scipy.signal.windows. Default is None, which
            applies no window function. Function MUST accept a single
            argument, M, which is the number of taps in the filter.

    Returns:

    """
    # Check size is known
    if size is None and n_zeros is None:
        raise ValueError("Either size or n_zeros must be provided.")

    # Compute size from n_zeros
    if size is None:
        assert n_zeros is not None
        size = sinc_sample_size(f_cutoff, f_sample, n_zeros)

    # Compute the raw sample size
    n = numpy.arange(0, size, 1)
    filt = numpy.sinc(2 * (f_cutoff / f_sample) * (n - (size - 1) // 2))

    # if a window function is provided, apply it
    if win_func is not None:
        filt *= win_func(M=size)

    # Normalize the filter
    if normalize:
        filt /= numpy.sum(filt)

    # Fix the size of the filter
    if fix_size is not None:

        # If the filter is too small, pad with zeros centered on the filter
        if size < fix_size:
            pad = (fix_size - size) // 2
            filt = numpy.pad(filt, (pad, pad), "constant")

        # If the filter is too large, truncate the filter centered on the filter
        elif size > fix_size:
            pad = (size - fix_size) // 2
            filt = filt[pad:-pad]

    return filt
