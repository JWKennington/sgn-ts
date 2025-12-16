"""Plotting sink for SGN-TS pipelines.

This module provides TSPlotSink, a sink that collects frames during pipeline
execution and provides convenient plotting methods for visualization.

Requires matplotlib as an optional dependency.
Install with: pip install sgn-ts[plot]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Optional, Sequence

from sgnts.plotting import _check_matplotlib
from sgnts.sinks.collect import TSFrameCollectSink

if TYPE_CHECKING:  # pragma: no cover
    pass


@dataclass
class TSPlotSink(TSFrameCollectSink):
    """Sink that collects frames and provides plotting methods.

    Extends TSFrameCollectSink with convenient plotting capabilities
    for visualizing collected data after pipeline completion.

    Example::

        sink = TSPlotSink(name="detector_data", sink_pad_names=("H1", "L1"))
        pipeline.connect(source, sink)
        pipeline.run()

        # Both pads overlaid, labeled "H1" and "L1"
        fig, ax = sink.plot(time_unit="s")
        ax.legend()
        plt.show()

    Args:
        name: Element name (used as default title for multi-pad plots)
        sink_pad_names: Names of input pads (used as default labels)
    """

    def plot(
        self,
        pads: Optional[Sequence[str]] = None,
        layout: Literal["overlay", "subplots"] = "overlay",
        labels: Optional[dict[str, str]] = None,
        title: Optional[str] = None,
        time_unit: Literal["s", "ms", "ns", "gps"] = "gps",
        show_gaps: bool = True,
        gap_color: str = "red",
        gap_alpha: float = 0.3,
        figsize: Optional[tuple[float, float]] = None,
        ax=None,
        **kwargs,
    ):
        """Plot collected frames.

        Args:
            pads:
                Which pads to plot. Default: all pads in order they were defined.
            layout:
                How to arrange multiple pads:

                - ``"overlay"``: All pads on same axes (default)
                - ``"subplots"``: Vertical stack with shared x-axis
            labels:
                Custom labels for pads as ``{pad_name: label}``.
                Default: use pad names.
            title:
                Figure title. Default: element name if multiple pads,
                None for single pad.
            time_unit:
                Time unit for x-axis: ``"s"``, ``"ms"``, ``"ns"``, or ``"gps"``.
            show_gaps:
                Show gap regions as shaded areas.
            gap_color:
                Color for gap shading.
            gap_alpha:
                Transparency for gap shading.
            figsize:
                Figure size as ``(width, height)``. Default: auto-calculated
                based on layout and number of pads.
            ax:
                Existing matplotlib axes to plot on. Only used when
                ``layout="overlay"``. If provided, plots on this axes
                instead of creating a new figure.
            **kwargs:
                Additional arguments passed to matplotlib ``plot()``.

        Returns:
            tuple: ``(fig, ax)`` for overlay layout, ``(fig, axes)`` for
            subplots layout where ``axes`` is a list.

        Raises:
            ValueError: If an unknown pad name is specified.
            ImportError: If matplotlib is not installed.
        """
        _check_matplotlib()

        # Get collected frames
        frames = self.out_frames()

        # Determine which pads to plot
        if pads is None:
            # Use all pads in the order they were defined
            pads_to_plot = list(self.sink_pad_names)
        else:
            pads_to_plot = list(pads)

        # Validate pad names
        available_pads = set(frames.keys())
        for pad_name in pads_to_plot:
            if pad_name not in available_pads:
                raise ValueError(
                    f"Unknown pad '{pad_name}'. "
                    f"Available pads: {sorted(available_pads)}"
                )

        n_pads = len(pads_to_plot)

        # Determine labels (default to pad names)
        if labels is None:
            labels = {}
        pad_labels = {pad: labels.get(pad, pad) for pad in pads_to_plot}

        # Determine title (default to element name for multiple pads)
        if title is None and n_pads > 1:
            title = self.name

        # Handle different layouts
        if layout == "subplots" and n_pads > 1:
            return self._plot_subplots(
                frames=frames,
                pads_to_plot=pads_to_plot,
                pad_labels=pad_labels,
                title=title,
                time_unit=time_unit,
                show_gaps=show_gaps,
                gap_color=gap_color,
                gap_alpha=gap_alpha,
                figsize=figsize,
                **kwargs,
            )
        else:
            return self._plot_overlay(
                frames=frames,
                pads_to_plot=pads_to_plot,
                pad_labels=pad_labels,
                title=title,
                time_unit=time_unit,
                show_gaps=show_gaps,
                gap_color=gap_color,
                gap_alpha=gap_alpha,
                figsize=figsize,
                ax=ax,
                **kwargs,
            )

    def _plot_overlay(
        self,
        frames,
        pads_to_plot,
        pad_labels,
        title,
        time_unit,
        show_gaps,
        gap_color,
        gap_alpha,
        figsize,
        ax,
        **kwargs,
    ):
        """Plot all pads on the same axes."""
        plt = _check_matplotlib()
        from sgnts.plotting import plot_frame

        # Create figure if needed
        if ax is None:
            if figsize is None:
                figsize = (10, 4)
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        # Plot each pad
        for pad_name in pads_to_plot:
            frame = frames[pad_name]
            plot_frame(
                frame,
                ax=ax,
                label=pad_labels[pad_name],
                time_unit=time_unit,
                show_gaps=show_gaps,
                gap_color=gap_color,
                gap_alpha=gap_alpha,
                **kwargs,
            )

        if title:
            ax.set_title(title)

        return fig, ax

    def _plot_subplots(
        self,
        frames,
        pads_to_plot,
        pad_labels,
        title,
        time_unit,
        show_gaps,
        gap_color,
        gap_alpha,
        figsize,
        **kwargs,
    ):
        """Plot each pad in its own subplot."""
        plt = _check_matplotlib()
        from sgnts.plotting import plot_frame

        n_pads = len(pads_to_plot)

        # Calculate figure size
        if figsize is None:
            figsize = (10, 2.5 * n_pads)

        fig, axes = plt.subplots(n_pads, 1, sharex=True, figsize=figsize, squeeze=False)
        axes = [ax[0] for ax in axes]  # Flatten from 2D array

        # Plot each pad in its subplot
        for i, pad_name in enumerate(pads_to_plot):
            frame = frames[pad_name]
            plot_frame(
                frame,
                ax=axes[i],
                label=pad_labels[pad_name],
                time_unit=time_unit,
                show_gaps=show_gaps,
                gap_color=gap_color,
                gap_alpha=gap_alpha,
                **kwargs,
            )
            axes[i].set_ylabel(pad_labels[pad_name])

        if title:
            fig.suptitle(title)

        return fig, axes
