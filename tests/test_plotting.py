"""Tests for plotting functionality."""

import numpy as np
import pytest

from sgnts.base.buffer import SeriesBuffer, TSFrame
from sgnts.sources import FakeSeriesSource

# Check if matplotlib is available
try:
    import matplotlib

    matplotlib.use("Agg")  # Use non-interactive backend for testing
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

pytestmark = pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")


class TestSeriesBufferPlot:
    """Tests for SeriesBuffer.plot() method."""

    def test_plot_basic(self):
        """Test basic buffer plotting."""
        buf = SeriesBuffer(
            offset=0,
            sample_rate=256,
            data=np.sin(2 * np.pi * 2 * np.arange(256) / 256),
            shape=(256,),
        )
        fig, ax = buf.plot()
        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_plot_with_label(self):
        """Test buffer plotting with label."""
        buf = SeriesBuffer(
            offset=0,
            sample_rate=256,
            data=np.ones(256),
            shape=(256,),
        )
        fig, ax = buf.plot(label="test_label")
        lines = ax.get_lines()
        assert len(lines) == 1
        assert lines[0].get_label() == "test_label"
        plt.close(fig)

    def test_plot_gap_buffer(self):
        """Test plotting a gap buffer."""
        buf = SeriesBuffer(
            offset=0,
            sample_rate=256,
            data=None,
            shape=(256,),
        )
        fig, ax = buf.plot(show_gaps=True, gap_color="red", gap_alpha=0.5)
        # Gap should create an axvspan, not a line
        assert len(ax.get_lines()) == 0
        assert len(ax.patches) == 1  # axvspan creates a patch
        plt.close(fig)

    def test_plot_gap_buffer_no_show(self):
        """Test plotting a gap buffer with show_gaps=False."""
        buf = SeriesBuffer(
            offset=0,
            sample_rate=256,
            data=None,
            shape=(256,),
        )
        fig, ax = buf.plot(show_gaps=False)
        assert len(ax.patches) == 0
        plt.close(fig)

    def test_plot_time_units(self):
        """Test different time units."""
        buf = SeriesBuffer(
            offset=0,
            sample_rate=256,
            data=np.ones(256),
            shape=(256,),
        )

        for unit in ["s", "ms", "ns", "gps"]:
            fig, ax = buf.plot(time_unit=unit)
            assert ax.get_xlabel() != ""
            plt.close(fig)

    def test_plot_multichannel(self):
        """Test plotting multi-channel buffer."""
        buf = SeriesBuffer(
            offset=0,
            sample_rate=256,
            data=np.random.randn(3, 256),
            shape=(3, 256),
        )
        fig, ax = buf.plot()
        # Should have 3 lines for 3 channels
        assert len(ax.get_lines()) == 3
        plt.close(fig)

    def test_plot_specific_channel(self):
        """Test plotting specific channel."""
        buf = SeriesBuffer(
            offset=0,
            sample_rate=256,
            data=np.random.randn(3, 256),
            shape=(3, 256),
        )
        fig, ax = buf.plot(channel=1)
        # Should have 1 line for 1 channel
        assert len(ax.get_lines()) == 1
        plt.close(fig)

    def test_plot_on_existing_axes(self):
        """Test plotting on existing axes."""
        fig, ax = plt.subplots()
        buf = SeriesBuffer(
            offset=0,
            sample_rate=256,
            data=np.ones(256),
            shape=(256,),
        )
        fig2, ax2 = buf.plot(ax=ax)
        assert ax2 is ax
        assert fig2 is fig
        plt.close(fig)


class TestTSFramePlot:
    """Tests for TSFrame.plot() method."""

    def test_plot_basic(self):
        """Test basic frame plotting."""
        buf = SeriesBuffer(
            offset=0,
            sample_rate=256,
            data=np.sin(2 * np.pi * 2 * np.arange(256) / 256),
            shape=(256,),
        )
        frame = TSFrame(buffers=[buf])
        fig, ax = frame.plot()
        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_plot_with_gaps(self):
        """Test frame plotting with gap buffers."""
        buf1 = SeriesBuffer(
            offset=0,
            sample_rate=256,
            data=np.ones(256),
            shape=(256,),
        )
        # Gap buffer
        buf2 = SeriesBuffer(
            offset=buf1.end_offset,
            sample_rate=256,
            data=None,
            shape=(256,),
        )
        buf3 = SeriesBuffer(
            offset=buf2.end_offset,
            sample_rate=256,
            data=np.ones(256),
            shape=(256,),
        )
        frame = TSFrame(buffers=[buf1, buf2, buf3])

        fig, ax = frame.plot(show_gaps=True)
        # Should have lines for data buffers and patch for gap
        assert len(ax.get_lines()) >= 1
        assert len(ax.patches) == 1  # One gap
        plt.close(fig)

    def test_plot_multichannel_overlay(self):
        """Test multi-channel frame plotting with overlay."""
        buf = SeriesBuffer(
            offset=0,
            sample_rate=256,
            data=np.random.randn(3, 256),
            shape=(3, 256),
        )
        frame = TSFrame(buffers=[buf])
        fig, ax = frame.plot(multichannel="overlay")
        assert len(ax.get_lines()) == 3
        plt.close(fig)

    def test_plot_multichannel_subplots(self):
        """Test multi-channel frame plotting with subplots."""
        buf = SeriesBuffer(
            offset=0,
            sample_rate=256,
            data=np.random.randn(3, 256),
            shape=(3, 256),
        )
        frame = TSFrame(buffers=[buf])
        fig, axes = frame.plot(multichannel="subplots")
        assert len(axes) == 3
        for ax in axes:
            assert len(ax.get_lines()) == 1
        plt.close(fig)

    def test_plot_multiple_frames_same_axes(self):
        """Test plotting multiple frames on same axes."""
        buf1 = SeriesBuffer(
            offset=0,
            sample_rate=256,
            data=np.sin(2 * np.pi * 2 * np.arange(256) / 256),
            shape=(256,),
        )
        buf2 = SeriesBuffer(
            offset=0,
            sample_rate=256,
            data=np.sin(2 * np.pi * 5 * np.arange(256) / 256),
            shape=(256,),
        )
        frame1 = TSFrame(buffers=[buf1])
        frame2 = TSFrame(buffers=[buf2])

        fig, ax = frame1.plot(label="Frame 1")
        frame2.plot(ax=ax, label="Frame 2")

        assert len(ax.get_lines()) == 2
        plt.close(fig)


class TestPlotFramesFunction:
    """Tests for plot_frames utility function."""

    def test_plot_frames(self):
        """Test plotting multiple frames with utility function."""
        from sgnts.plotting import plot_frames

        buf1 = SeriesBuffer(
            offset=0,
            sample_rate=256,
            data=np.ones(256),
            shape=(256,),
        )
        buf2 = SeriesBuffer(
            offset=0,
            sample_rate=256,
            data=np.ones(256) * 2,
            shape=(256,),
        )
        frame1 = TSFrame(buffers=[buf1])
        frame2 = TSFrame(buffers=[buf2])

        fig, ax = plot_frames([frame1, frame2], labels=["F1", "F2"])
        assert len(ax.get_lines()) == 2
        plt.close(fig)


class TestPlotFramesEdgeCases:
    """Edge case tests for plot_frames."""

    def test_plot_frames_with_existing_ax(self):
        """Test plot_frames with pre-existing axes."""
        from sgnts.plotting import plot_frames

        fig, ax = plt.subplots()
        buf = SeriesBuffer(offset=0, sample_rate=256, data=np.ones(256), shape=(256,))
        frame = TSFrame(buffers=[buf])

        fig2, ax2 = plot_frames([frame], ax=ax)
        assert ax2 is ax
        assert fig2 is fig
        plt.close(fig)

    def test_plot_frames_no_labels(self):
        """Test plot_frames without labels (uses None for each)."""
        from sgnts.plotting import plot_frames

        buf = SeriesBuffer(offset=0, sample_rate=256, data=np.ones(256), shape=(256,))
        frame = TSFrame(buffers=[buf])

        fig, ax = plot_frames([frame])  # No labels provided
        assert len(ax.get_lines()) == 1
        plt.close(fig)


class TestGapTimeUnits:
    """Tests for gap buffer plotting with different time units."""

    def test_gap_buffer_time_unit_ms(self):
        """Test gap buffer with millisecond time unit."""
        buf = SeriesBuffer(offset=0, sample_rate=256, data=None, shape=(256,))
        fig, ax = buf.plot(time_unit="ms", show_gaps=True)
        assert len(ax.patches) == 1
        plt.close(fig)

    def test_gap_buffer_time_unit_ns(self):
        """Test gap buffer with nanosecond time unit."""
        buf = SeriesBuffer(offset=0, sample_rate=256, data=None, shape=(256,))
        fig, ax = buf.plot(time_unit="ns", show_gaps=True)
        assert len(ax.patches) == 1
        plt.close(fig)


class TestInvalidTimeUnit:
    """Test invalid time unit handling."""

    def test_invalid_time_unit_raises(self):
        """Test that invalid time_unit raises ValueError."""
        from sgnts.plotting import plot_buffer

        buf = SeriesBuffer(offset=0, sample_rate=256, data=np.ones(256), shape=(256,))
        with pytest.raises(ValueError, match="Unknown time_unit"):
            plot_buffer(buf, time_unit="invalid")


class TestTupleChannelSelection:
    """Test tuple-based channel selection."""

    def test_tuple_channel_selection(self):
        """Test selecting channel with tuple index."""
        buf = SeriesBuffer(
            offset=0,
            sample_rate=256,
            data=np.random.randn(2, 3, 256),
            shape=(2, 3, 256),
        )
        fig, ax = buf.plot(channel=(1, 2))  # Select specific sub-channel
        assert len(ax.get_lines()) == 1
        plt.close(fig)


class TestGapOnlyFrame:
    """Test frames containing only gap buffers."""

    def test_gap_only_frame_labeling(self):
        """Test that gap-only frames get labeled correctly."""
        gap_buf = SeriesBuffer(offset=0, sample_rate=256, data=None, shape=(256,))
        frame = TSFrame(buffers=[gap_buf])

        fig, ax = frame.plot(label="Gap frame", show_gaps=True)
        # The gap should be labeled
        assert len(ax.patches) == 1
        plt.close(fig)


class TestSingleChannelSubplots:
    """Test single-channel data with subplots mode."""

    def test_single_channel_subplots_returns_list(self):
        """Test that single-channel subplots returns axes as list."""
        # 1-channel data with subplots should still work
        # but the n_channels==1 guard won't trigger since we need >1 for subplots mode
        buf = SeriesBuffer(
            offset=0,
            sample_rate=256,
            data=np.random.randn(1, 256),
            shape=(1, 256),
        )
        frame = TSFrame(buffers=[buf])
        # With 1 channel, subplots mode falls back to overlay (n_channels > 1 check)
        fig, ax = frame.plot(multichannel="subplots")
        plt.close(fig)


class TestIntegrationWithSources:
    """Integration tests with FakeSeriesSource."""

    def test_plot_fake_series_source_output(self):
        """Test plotting output from FakeSeriesSource."""
        source = FakeSeriesSource(
            name="source",
            source_pad_names=("data",),
            rate=256,
            signal_type="sine",
            fsin=5.0,
            t0=0,
            end=2.0,
        )
        pad = source.srcs["data"]
        frame = source.new(pad)

        fig, ax = frame.plot(label="Sine wave")
        assert len(ax.get_lines()) == 1
        plt.close(fig)

    def test_plot_fake_series_with_gaps(self):
        """Test plotting FakeSeriesSource output with gaps."""
        source = FakeSeriesSource(
            name="source",
            source_pad_names=("data",),
            rate=256,
            signal_type="sine",
            fsin=5.0,
            ngap=2,  # Gap every 2nd buffer
            t0=0,
            end=3.0,
        )

        pad = source.srcs["data"]

        # Collect a few frames
        buffers = []
        for _ in range(3):
            frame = source.new(pad)
            buffers.extend(frame.buffers)
            if frame.EOS:
                break

        combined_frame = TSFrame(buffers=buffers)
        fig, ax = combined_frame.plot(show_gaps=True)
        plt.close(fig)


class TestMatplotlibNotInstalled:
    """Test behavior when matplotlib is not installed."""

    def test_check_matplotlib_import_error(self):
        """Test that ImportError is raised when matplotlib is not available."""
        import builtins
        import sgnts.plotting as plotting_module
        from unittest.mock import patch

        # Save original state
        original_plt = plotting_module._plt
        original_available = plotting_module._matplotlib_available
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "matplotlib.pyplot":
                raise ImportError("No module named 'matplotlib'")
            return original_import(name, *args, **kwargs)

        try:
            # Reset module state to simulate fresh import check
            plotting_module._plt = None
            plotting_module._matplotlib_available = None

            # Mock import to fail for matplotlib.pyplot
            with patch.object(builtins, "__import__", side_effect=mock_import):
                with pytest.raises(ImportError, match="matplotlib is required"):
                    plotting_module._check_matplotlib()

            # Verify state was set correctly
            assert plotting_module._matplotlib_available is False
        finally:
            # Restore original state
            plotting_module._plt = original_plt
            plotting_module._matplotlib_available = original_available
