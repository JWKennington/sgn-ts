"""Plotting utilities for SGN-TS time-series data.

This module provides plotting functionality for SeriesBuffer and TSFrame objects.
Requires matplotlib as an optional dependency.

Install with: pip install sgn-ts[plot]
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Optional, Sequence, Union

if TYPE_CHECKING:  # pragma: no cover
    from sgnts.base.buffer import SeriesBuffer, TSFrame

# Lazy import of matplotlib to avoid import errors when not installed
_plt = None
_matplotlib_available = None


def _check_matplotlib():
    """Check if matplotlib is available and import it.

    Returns:
        module: The matplotlib.pyplot module

    Raises:
        ImportError: If matplotlib is not installed
    """
    global _plt, _matplotlib_available

    if _matplotlib_available is None:
        try:
            import matplotlib.pyplot as plt

            _plt = plt
            _matplotlib_available = True
        except ImportError:
            _matplotlib_available = False

    if not _matplotlib_available:
        raise ImportError(
            "matplotlib is required for plotting functionality. "
            "Install it with: pip install sgn-ts[plot]"
        )

    return _plt


def plot_buffer(
    buffer: "SeriesBuffer",
    ax=None,
    label: Optional[str] = None,
    channel: Optional[Union[int, tuple[int, ...]]] = None,
    gap_color: str = "red",
    gap_alpha: float = 0.3,
    show_gaps: bool = True,
    time_unit: Literal["s", "ms", "ns", "gps"] = "gps",
    **kwargs,
):
    """Plot a SeriesBuffer's time-series data.

    Args:
        buffer:
            SeriesBuffer, the buffer to plot
        ax:
            matplotlib Axes, optional. If None, creates a new figure and axes.
        label:
            str, optional. Legend label for this buffer's data line.
        channel:
            int or tuple[int, ...], optional. For multi-dimensional data,
            specifies which channel(s) to plot. If None and data is 1D, plots
            the data directly. If None and data is multi-dimensional, plots
            all channels.
        gap_color:
            str, color for gap region shading. Default 'red'.
        gap_alpha:
            float, alpha transparency for gap region shading. Default 0.3.
        show_gaps:
            bool, whether to show gap indicators. Default True.
        time_unit:
            str, time unit for x-axis: 's' (seconds since start), 'ms',
            'ns', or 'gps' (absolute GPS time). Default 'gps'.
        **kwargs:
            Additional keyword arguments passed to ax.plot().

    Returns:
        tuple: (fig, ax) matplotlib figure and axes objects
    """
    plt = _check_matplotlib()
    from sgnts.base.time import Time

    # Create figure/axes if needed
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    # Calculate time array based on time_unit
    # buffer.tarr gives absolute GPS time in seconds
    if time_unit == "gps":
        tarr = buffer.tarr
        ax.set_xlabel("GPS Time (s)")
    elif time_unit == "s":
        # Use absolute time in seconds (same as GPS but clearer label)
        tarr = buffer.tarr
        ax.set_xlabel("Time (s)")
    elif time_unit == "ms":
        tarr = buffer.tarr * 1000
        ax.set_xlabel("Time (ms)")
    elif time_unit == "ns":
        tarr = buffer.tarr * 1e9
        ax.set_xlabel("Time (ns)")
    else:
        raise ValueError(
            f"Unknown time_unit: {time_unit}. Use 's', 'ms', 'ns', or 'gps'"
        )

    # Handle gap buffer
    if buffer.is_gap:
        if show_gaps:
            # Get time bounds for the gap region (use absolute time)
            t_start_s = buffer.t0 / Time.SECONDS
            t_end_s = buffer.end / Time.SECONDS

            if time_unit == "gps" or time_unit == "s":
                t_start = t_start_s
                t_end = t_end_s
            elif time_unit == "ms":
                t_start = t_start_s * 1000
                t_end = t_end_s * 1000
            elif time_unit == "ns":
                t_start = t_start_s * 1e9
                t_end = t_end_s * 1e9

            ax.axvspan(t_start, t_end, color=gap_color, alpha=gap_alpha, label=label)
        return fig, ax

    # Get data to plot (we know it's not None since we checked is_gap above)
    # After SeriesBuffer.__post_init__, data is always an Array, not int
    data = buffer.data
    assert data is not None and not isinstance(data, int)  # Help type checker

    # Handle channel selection for multi-dimensional data
    if len(buffer.sample_shape) > 0:
        if channel is not None:
            if isinstance(channel, int):
                data = data[channel]
            else:
                data = data[channel]
        else:
            # Plot all channels - data shape is (channels..., samples)
            # Flatten all but last dimension and plot each
            flat_shape = (-1, buffer.samples)
            flat_data = data.reshape(flat_shape)
            for i, ch_data in enumerate(flat_data):
                ch_label = f"{label} ch{i}" if label else f"ch{i}"
                ax.plot(tarr, ch_data, label=ch_label, **kwargs)
            return fig, ax

    # Plot single channel data
    ax.plot(tarr, data, label=label, **kwargs)

    return fig, ax


def plot_frame(
    frame: "TSFrame",
    ax=None,
    label: Optional[str] = None,
    channel: Optional[Union[int, tuple[int, ...]]] = None,
    gap_color: str = "red",
    gap_alpha: float = 0.3,
    show_gaps: bool = True,
    time_unit: Literal["s", "ms", "ns", "gps"] = "gps",
    multichannel: Literal["overlay", "subplots"] = "overlay",
    **kwargs,
):
    """Plot a TSFrame's time-series data.

    Args:
        frame:
            TSFrame, the frame to plot
        ax:
            matplotlib Axes, optional. If None, creates a new figure and axes.
            Ignored if multichannel='subplots' and channel is None.
        label:
            str, optional. Legend label for this frame's data line.
        channel:
            int or tuple[int, ...], optional. For multi-dimensional data,
            specifies which channel(s) to plot. If None and data is 1D, plots
            the data directly. If None and data is multi-dimensional, behavior
            depends on multichannel parameter.
        gap_color:
            str, color for gap region shading. Default 'red'.
        gap_alpha:
            float, alpha transparency for gap region shading. Default 0.3.
        show_gaps:
            bool, whether to show gap indicators. Default True.
        time_unit:
            str, time unit for x-axis: 's' (seconds since start), 'ms',
            'ns', or 'gps' (absolute GPS time). Default 'gps'.
        multichannel:
            str, how to handle multi-channel data when channel is None:
            'overlay' plots all channels on the same axes,
            'subplots' creates a subplot for each channel. Default 'overlay'.
        **kwargs:
            Additional keyword arguments passed to ax.plot().

    Returns:
        tuple: (fig, ax) matplotlib figure and axes objects. If multichannel='subplots',
               ax is an array of axes.
    """
    plt = _check_matplotlib()

    # Determine number of channels
    sample_shape = frame.sample_shape
    n_channels = 1
    if len(sample_shape) > 0:
        import numpy as np

        n_channels = int(np.prod(sample_shape))

    # Handle subplots for multi-channel data
    if multichannel == "subplots" and channel is None and n_channels > 1:
        fig, axes = plt.subplots(
            n_channels, 1, sharex=True, figsize=(10, 2 * n_channels)
        )
        # n_channels > 1 guaranteed by guard above, so axes is always a list

        for i, ax_i in enumerate(axes):
            # Plot each buffer in the frame for this channel
            for buf in frame.buffers:
                plot_buffer(
                    buf,
                    ax=ax_i,
                    label=label if buf == frame.buffers[0] else None,
                    channel=i,
                    gap_color=gap_color,
                    gap_alpha=gap_alpha,
                    show_gaps=show_gaps,
                    time_unit=time_unit,
                    **kwargs,
                )
            ax_i.set_ylabel(f"Channel {i}")

        # Only set xlabel on bottom subplot
        axes[-1].set_xlabel(_get_time_label(time_unit))
        return fig, axes

    # Create figure/axes if needed
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    # Track if we've added the label (only label first non-gap buffer)
    label_used = False
    # Track the color used for the first data buffer so all buffers use the same color
    line_color = kwargs.pop("color", None)

    # Plot each buffer
    for buf in frame.buffers:
        # Only use label for first data buffer
        buf_label = None
        if not label_used and not buf.is_gap:
            buf_label = label
            label_used = True
        elif buf.is_gap and show_gaps and not label_used:
            # For gap-only frames, label the first gap
            buf_label = label
            label_used = True

        # Build kwargs for this buffer, including color if we have one
        buf_kwargs = dict(kwargs)
        if line_color is not None:
            buf_kwargs["color"] = line_color

        plot_buffer(
            buf,
            ax=ax,
            label=buf_label,
            channel=channel,
            gap_color=gap_color,
            gap_alpha=gap_alpha,
            show_gaps=show_gaps,
            time_unit=time_unit,
            **buf_kwargs,
        )

        # Capture the color from the first plotted data buffer
        if line_color is None and not buf.is_gap:
            lines = ax.get_lines()
            if lines:
                line_color = lines[-1].get_color()

    ax.set_xlabel(_get_time_label(time_unit))

    return fig, ax


def plot_frames(
    frames: Sequence["TSFrame"],
    ax=None,
    labels: Optional[Sequence[Optional[str]]] = None,
    gap_color: str = "red",
    gap_alpha: float = 0.3,
    show_gaps: bool = True,
    time_unit: Literal["s", "ms", "ns", "gps"] = "gps",
    **kwargs,
):
    """Plot multiple TSFrames on the same axes.

    Args:
        frames:
            Sequence[TSFrame], the frames to plot
        ax:
            matplotlib Axes, optional. If None, creates a new figure and axes.
        labels:
            Sequence[str], optional. Legend labels for each frame.
        gap_color:
            str, color for gap region shading. Default 'red'.
        gap_alpha:
            float, alpha transparency for gap region shading. Default 0.3.
        show_gaps:
            bool, whether to show gap indicators. Default True.
        time_unit:
            str, time unit for x-axis. Default 'gps'.
        **kwargs:
            Additional keyword arguments passed to ax.plot().

    Returns:
        tuple: (fig, ax) matplotlib figure and axes objects
    """
    plt = _check_matplotlib()

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    if labels is None:
        labels = [None] * len(frames)

    for frame, label in zip(frames, labels):
        plot_frame(
            frame,
            ax=ax,
            label=label,
            gap_color=gap_color,
            gap_alpha=gap_alpha,
            show_gaps=show_gaps,
            time_unit=time_unit,
            **kwargs,
        )

    return fig, ax


def _get_time_label(time_unit: str) -> str:
    """Get the appropriate axis label for the time unit."""
    labels = {
        "gps": "GPS Time (s)",
        "s": "Time (s)",
        "ms": "Time (ms)",
        "ns": "Time (ns)",
    }
    return labels.get(time_unit, "Time")
