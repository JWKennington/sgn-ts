"""Tests for TSPlotSink."""

import pytest

from sgn import Pipeline
from sgnts.sinks import TSPlotSink
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


class TestTSPlotSinkBasic:
    """Basic tests for TSPlotSink."""

    def test_single_pad_plot(self):
        """Test plotting with a single pad."""
        source = FakeSeriesSource(
            name="source",
            source_pad_names=("data",),
            rate=256,
            signal_type="sine",
            fsin=5.0,
            t0=0,
            end=1.0,
        )
        sink = TSPlotSink(name="sink", sink_pad_names=("data",))

        pipeline = Pipeline()
        pipeline.connect(source, sink)
        pipeline.run()

        fig, ax = sink.plot(time_unit="s")
        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_multi_pad_overlay(self):
        """Test plotting multiple pads on same axes."""
        source = FakeSeriesSource(
            name="source",
            source_pad_names=("H1", "L1"),
            rate=256,
            signal_type="sine",
            fsin=5.0,
            t0=0,
            end=1.0,
        )
        sink = TSPlotSink(name="Detectors", sink_pad_names=("H1", "L1"))

        pipeline = Pipeline()
        pipeline.connect(source, sink)
        pipeline.run()

        fig, ax = sink.plot(time_unit="s")
        assert fig is not None
        assert ax is not None
        # Should have lines for both pads
        assert len(ax.get_lines()) >= 2
        plt.close(fig)

    def test_multi_pad_subplots(self):
        """Test plotting multiple pads in subplots."""
        source = FakeSeriesSource(
            name="source",
            source_pad_names=("H1", "L1", "V1"),
            rate=256,
            signal_type="sine",
            fsin=5.0,
            t0=0,
            end=1.0,
        )
        sink = TSPlotSink(name="Detectors", sink_pad_names=("H1", "L1", "V1"))

        pipeline = Pipeline()
        pipeline.connect(source, sink)
        pipeline.run()

        fig, axes = sink.plot(layout="subplots", time_unit="s")
        assert fig is not None
        assert len(axes) == 3
        for ax in axes:
            assert len(ax.get_lines()) >= 1
        plt.close(fig)


class TestTSPlotSinkLabels:
    """Tests for labeling functionality."""

    def test_default_labels_are_pad_names(self):
        """Test that default labels come from pad names."""
        source = FakeSeriesSource(
            name="source",
            source_pad_names=("H1", "L1"),
            rate=256,
            signal_type="sine",
            fsin=5.0,
            t0=0,
            end=1.0,
        )
        sink = TSPlotSink(name="sink", sink_pad_names=("H1", "L1"))

        pipeline = Pipeline()
        pipeline.connect(source, sink)
        pipeline.run()

        fig, ax = sink.plot()
        lines = ax.get_lines()
        labels = [line.get_label() for line in lines]
        assert "H1" in labels
        assert "L1" in labels
        plt.close(fig)

    def test_custom_labels(self):
        """Test custom labels override pad names."""
        source = FakeSeriesSource(
            name="source",
            source_pad_names=("H1", "L1"),
            rate=256,
            signal_type="sine",
            fsin=5.0,
            t0=0,
            end=1.0,
        )
        sink = TSPlotSink(name="sink", sink_pad_names=("H1", "L1"))

        pipeline = Pipeline()
        pipeline.connect(source, sink)
        pipeline.run()

        fig, ax = sink.plot(labels={"H1": "Hanford", "L1": "Livingston"})
        lines = ax.get_lines()
        labels = [line.get_label() for line in lines]
        assert "Hanford" in labels
        assert "Livingston" in labels
        plt.close(fig)

    def test_default_title_multi_pad(self):
        """Test that multi-pad plots get element name as title."""
        source = FakeSeriesSource(
            name="source",
            source_pad_names=("H1", "L1"),
            rate=256,
            signal_type="sine",
            fsin=5.0,
            t0=0,
            end=1.0,
        )
        sink = TSPlotSink(name="Detector Data", sink_pad_names=("H1", "L1"))

        pipeline = Pipeline()
        pipeline.connect(source, sink)
        pipeline.run()

        fig, ax = sink.plot()
        assert ax.get_title() == "Detector Data"
        plt.close(fig)

    def test_custom_title(self):
        """Test custom title overrides default."""
        source = FakeSeriesSource(
            name="source",
            source_pad_names=("data",),
            rate=256,
            signal_type="sine",
            fsin=5.0,
            t0=0,
            end=1.0,
        )
        sink = TSPlotSink(name="sink", sink_pad_names=("data",))

        pipeline = Pipeline()
        pipeline.connect(source, sink)
        pipeline.run()

        fig, ax = sink.plot(title="My Custom Title")
        assert ax.get_title() == "My Custom Title"
        plt.close(fig)


class TestTSPlotSinkPadSelection:
    """Tests for pad selection."""

    def test_plot_subset_of_pads(self):
        """Test plotting only some pads."""
        source = FakeSeriesSource(
            name="source",
            source_pad_names=("H1", "L1", "V1"),
            rate=256,
            signal_type="sine",
            fsin=5.0,
            t0=0,
            end=1.0,
        )
        sink = TSPlotSink(name="sink", sink_pad_names=("H1", "L1", "V1"))

        pipeline = Pipeline()
        pipeline.connect(source, sink)
        pipeline.run()

        fig, ax = sink.plot(pads=["H1", "V1"])
        lines = ax.get_lines()
        labels = [line.get_label() for line in lines]
        assert "H1" in labels
        assert "V1" in labels
        assert "L1" not in labels
        plt.close(fig)

    def test_invalid_pad_raises(self):
        """Test that invalid pad name raises ValueError."""
        source = FakeSeriesSource(
            name="source",
            source_pad_names=("data",),
            rate=256,
            signal_type="sine",
            fsin=5.0,
            t0=0,
            end=1.0,
        )
        sink = TSPlotSink(name="sink", sink_pad_names=("data",))

        pipeline = Pipeline()
        pipeline.connect(source, sink)
        pipeline.run()

        with pytest.raises(ValueError, match="Unknown pad 'invalid'"):
            sink.plot(pads=["invalid"])


class TestTSPlotSinkExistingAxes:
    """Tests for plotting on existing axes."""

    def test_plot_on_existing_axes(self):
        """Test plotting on pre-existing axes."""
        source = FakeSeriesSource(
            name="source",
            source_pad_names=("data",),
            rate=256,
            signal_type="sine",
            fsin=5.0,
            t0=0,
            end=1.0,
        )
        sink = TSPlotSink(name="sink", sink_pad_names=("data",))

        pipeline = Pipeline()
        pipeline.connect(source, sink)
        pipeline.run()

        fig, ax = plt.subplots()
        ax.axhline(0, color="gray")  # Add something first

        fig2, ax2 = sink.plot(ax=ax)
        assert ax2 is ax
        assert fig2 is fig
        # Should have reference line plus data
        assert len(ax.get_lines()) >= 2
        plt.close(fig)


class TestTSPlotSinkWithGaps:
    """Tests for gap visualization."""

    def test_plot_with_gaps(self):
        """Test that gaps are shown."""
        source = FakeSeriesSource(
            name="source",
            source_pad_names=("data",),
            rate=256,
            signal_type="sine",
            fsin=5.0,
            ngap=2,
            t0=0,
            end=2.0,
        )
        sink = TSPlotSink(name="sink", sink_pad_names=("data",))

        pipeline = Pipeline()
        pipeline.connect(source, sink)
        pipeline.run()

        fig, ax = sink.plot(show_gaps=True, gap_color="red")
        # Should have patches for gaps
        assert len(ax.patches) >= 1
        plt.close(fig)

    def test_plot_without_gaps(self):
        """Test gaps can be hidden."""
        source = FakeSeriesSource(
            name="source",
            source_pad_names=("data",),
            rate=256,
            signal_type="sine",
            fsin=5.0,
            ngap=2,
            t0=0,
            end=2.0,
        )
        sink = TSPlotSink(name="sink", sink_pad_names=("data",))

        pipeline = Pipeline()
        pipeline.connect(source, sink)
        pipeline.run()

        fig, ax = sink.plot(show_gaps=False)
        assert len(ax.patches) == 0
        plt.close(fig)


class TestTSPlotSinkFigsize:
    """Tests for figure size handling."""

    def test_custom_figsize_overlay(self):
        """Test custom figure size for overlay."""
        source = FakeSeriesSource(
            name="source",
            source_pad_names=("data",),
            rate=256,
            signal_type="sine",
            fsin=5.0,
            t0=0,
            end=1.0,
        )
        sink = TSPlotSink(name="sink", sink_pad_names=("data",))

        pipeline = Pipeline()
        pipeline.connect(source, sink)
        pipeline.run()

        fig, ax = sink.plot(figsize=(12, 6))
        assert fig.get_figwidth() == 12
        assert fig.get_figheight() == 6
        plt.close(fig)

    def test_custom_figsize_subplots(self):
        """Test custom figure size for subplots."""
        source = FakeSeriesSource(
            name="source",
            source_pad_names=("H1", "L1"),
            rate=256,
            signal_type="sine",
            fsin=5.0,
            t0=0,
            end=1.0,
        )
        sink = TSPlotSink(name="sink", sink_pad_names=("H1", "L1"))

        pipeline = Pipeline()
        pipeline.connect(source, sink)
        pipeline.run()

        fig, axes = sink.plot(layout="subplots", figsize=(14, 8))
        assert fig.get_figwidth() == 14
        assert fig.get_figheight() == 8
        plt.close(fig)


class TestTSPlotSinkSubplotsLabels:
    """Tests for subplot ylabel handling."""

    def test_subplots_ylabel_from_pad_names(self):
        """Test that subplots get pad names as ylabels."""
        source = FakeSeriesSource(
            name="source",
            source_pad_names=("H1", "L1"),
            rate=256,
            signal_type="sine",
            fsin=5.0,
            t0=0,
            end=1.0,
        )
        sink = TSPlotSink(name="sink", sink_pad_names=("H1", "L1"))

        pipeline = Pipeline()
        pipeline.connect(source, sink)
        pipeline.run()

        fig, axes = sink.plot(layout="subplots")
        assert axes[0].get_ylabel() == "H1"
        assert axes[1].get_ylabel() == "L1"
        plt.close(fig)

    def test_subplots_ylabel_custom_labels(self):
        """Test that subplots use custom labels for ylabels."""
        source = FakeSeriesSource(
            name="source",
            source_pad_names=("H1", "L1"),
            rate=256,
            signal_type="sine",
            fsin=5.0,
            t0=0,
            end=1.0,
        )
        sink = TSPlotSink(name="sink", sink_pad_names=("H1", "L1"))

        pipeline = Pipeline()
        pipeline.connect(source, sink)
        pipeline.run()

        fig, axes = sink.plot(
            layout="subplots", labels={"H1": "Hanford", "L1": "Livingston"}
        )
        assert axes[0].get_ylabel() == "Hanford"
        assert axes[1].get_ylabel() == "Livingston"
        plt.close(fig)


class TestTSPlotSinkSinglePadSubplots:
    """Test edge case of single pad with subplots layout."""

    def test_single_pad_subplots_falls_back_to_overlay(self):
        """Test that single pad with subplots layout returns single axes."""
        source = FakeSeriesSource(
            name="source",
            source_pad_names=("data",),
            rate=256,
            signal_type="sine",
            fsin=5.0,
            t0=0,
            end=1.0,
        )
        sink = TSPlotSink(name="sink", sink_pad_names=("data",))

        pipeline = Pipeline()
        pipeline.connect(source, sink)
        pipeline.run()

        # Single pad with subplots should fall back to overlay
        fig, ax = sink.plot(layout="subplots")
        # Should return single axes, not list
        assert not isinstance(ax, list)
        plt.close(fig)
