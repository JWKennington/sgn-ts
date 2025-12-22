"""Tests for TSAppSink."""

import pytest

from sgn import Pipeline
from sgn.base import SinkPad

from sgnts.base import SeriesBuffer
from sgnts.sinks import TSAppSink
from sgnts.sources import FakeSeriesSource


class TestTSAppSinkBasic:
    """Basic tests for TSAppSink callback invocation."""

    def test_single_pad_callback(self):
        """Test that callback is invoked for each buffer on a single pad."""
        received_buffers = []

        def callback(pad: SinkPad, buf: SeriesBuffer) -> None:
            received_buffers.append((pad, buf))

        source = FakeSeriesSource(
            name="source",
            source_pad_names=("data",),
            rate=256,
            signal_type="sine",
            fsin=5.0,
            t0=0,
            end=1.0,
        )
        sink = TSAppSink(
            name="sink",
            sink_pad_names=("data",),
            callbacks={"data": callback},
        )

        pipeline = Pipeline()
        pipeline.connect(source, sink)
        pipeline.run()

        # Should have received buffers
        assert len(received_buffers) > 0
        # All buffers should be from "data" pad
        for pad, buf in received_buffers:
            assert pad.pad_name == "data"
            assert isinstance(buf, SeriesBuffer)

    def test_multi_pad_different_callbacks(self):
        """Test that different callbacks are invoked for different pads."""
        h1_buffers = []
        l1_buffers = []

        def h1_callback(pad: SinkPad, buf: SeriesBuffer) -> None:
            h1_buffers.append(buf)

        def l1_callback(pad: SinkPad, buf: SeriesBuffer) -> None:
            l1_buffers.append(buf)

        source = FakeSeriesSource(
            name="source",
            source_pad_names=("H1", "L1"),
            rate=256,
            signal_type="sine",
            fsin=5.0,
            t0=0,
            end=1.0,
        )
        sink = TSAppSink(
            name="sink",
            sink_pad_names=("H1", "L1"),
            callbacks={"H1": h1_callback, "L1": l1_callback},
        )

        pipeline = Pipeline()
        pipeline.connect(source, sink)
        pipeline.run()

        # Both pads should have received buffers
        assert len(h1_buffers) > 0
        assert len(l1_buffers) > 0

    def test_pad_without_callback_is_ignored(self):
        """Test that pads without callbacks silently ignore buffers."""
        received_pads = []

        def callback(pad: SinkPad, buf: SeriesBuffer) -> None:
            received_pads.append(pad)

        source = FakeSeriesSource(
            name="source",
            source_pad_names=("H1", "L1", "V1"),
            rate=256,
            signal_type="sine",
            fsin=5.0,
            t0=0,
            end=1.0,
        )
        sink = TSAppSink(
            name="sink",
            sink_pad_names=("H1", "L1", "V1"),
            callbacks={"H1": callback},  # Only H1 has a callback
        )

        pipeline = Pipeline()
        pipeline.connect(source, sink)
        pipeline.run()

        # Only H1 should have triggered callbacks
        assert all(pad.pad_name == "H1" for pad in received_pads)


class TestTSAppSinkCallbackRegistration:
    """Tests for callback registration methods."""

    def test_set_callback_method(self):
        """Test registering callback via set_callback method."""
        received_buffers = []

        def callback(pad: SinkPad, buf: SeriesBuffer) -> None:
            received_buffers.append(buf)

        source = FakeSeriesSource(
            name="source",
            source_pad_names=("data",),
            rate=256,
            signal_type="sine",
            fsin=5.0,
            t0=0,
            end=1.0,
        )
        sink = TSAppSink(
            name="sink",
            sink_pad_names=("data",),
        )
        sink.set_callback("data", callback)

        pipeline = Pipeline()
        pipeline.connect(source, sink)
        pipeline.run()

        assert len(received_buffers) > 0

    def test_remove_callback(self):
        """Test removing a callback."""
        received_buffers = []

        def callback(pad: SinkPad, buf: SeriesBuffer) -> None:
            received_buffers.append(buf)

        sink = TSAppSink(
            name="sink",
            sink_pad_names=("data",),
            callbacks={"data": callback},
        )

        # Verify callback is registered
        assert sink.get_callback("data") is callback

        # Remove it
        sink.remove_callback("data")

        # Verify it's gone
        assert sink.get_callback("data") is None

    def test_get_callback(self):
        """Test getting a registered callback."""

        def callback(pad: SinkPad, buf: SeriesBuffer) -> None:
            pass

        sink = TSAppSink(
            name="sink",
            sink_pad_names=("H1", "L1"),
            callbacks={"H1": callback},
        )

        assert sink.get_callback("H1") is callback
        assert sink.get_callback("L1") is None
        assert sink.get_callback("invalid") is None

    def test_set_callback_invalid_pad_raises(self):
        """Test that set_callback raises for invalid pad name."""
        sink = TSAppSink(
            name="sink",
            sink_pad_names=("data",),
        )

        with pytest.raises(ValueError, match="Unknown pad name 'invalid'"):
            sink.set_callback("invalid", lambda p, b: None)

    def test_constructor_invalid_pad_raises(self):
        """Test that constructor raises for invalid pad name in callbacks."""
        with pytest.raises(ValueError, match="Unknown pad name 'invalid'"):
            TSAppSink(
                name="sink",
                sink_pad_names=("data",),
                callbacks={"invalid": lambda p, b: None},
            )


class TestTSAppSinkGapHandling:
    """Tests for gap buffer handling."""

    def test_gaps_emitted_by_default(self):
        """Test that gap buffers are emitted by default (emit_gaps=True)."""
        received_buffers = []

        def callback(pad: SinkPad, buf: SeriesBuffer) -> None:
            received_buffers.append(buf)

        source = FakeSeriesSource(
            name="source",
            source_pad_names=("data",),
            rate=256,
            signal_type="sine",
            fsin=5.0,
            ngap=2,  # Create gaps
            t0=0,
            end=2.0,
        )
        sink = TSAppSink(
            name="sink",
            sink_pad_names=("data",),
            callbacks={"data": callback},
            # emit_gaps=True is the default
        )

        pipeline = Pipeline()
        pipeline.connect(source, sink)
        pipeline.run()

        # Should have received both gap and non-gap buffers
        gap_buffers = [b for b in received_buffers if b.is_gap]
        non_gap_buffers = [b for b in received_buffers if not b.is_gap]

        assert len(gap_buffers) > 0, "Expected gap buffers by default"
        assert len(non_gap_buffers) > 0, "Expected non-gap buffers"

    def test_gaps_skipped_when_disabled(self):
        """Test that gap buffers are skipped when emit_gaps=False."""
        received_buffers = []

        def callback(pad: SinkPad, buf: SeriesBuffer) -> None:
            received_buffers.append(buf)

        source = FakeSeriesSource(
            name="source",
            source_pad_names=("data",),
            rate=256,
            signal_type="sine",
            fsin=5.0,
            ngap=2,  # Create gaps
            t0=0,
            end=2.0,
        )
        sink = TSAppSink(
            name="sink",
            sink_pad_names=("data",),
            callbacks={"data": callback},
            emit_gaps=False,
        )

        pipeline = Pipeline()
        pipeline.connect(source, sink)
        pipeline.run()

        # Should not have received any gap buffers
        for buf in received_buffers:
            assert not buf.is_gap


class TestTSAppSinkCallbackArgs:
    """Tests for callback argument correctness."""

    def test_callback_receives_correct_pad(self):
        """Test that callback receives the correct SinkPad object."""
        received_pads = set()

        def callback(pad: SinkPad, buf: SeriesBuffer) -> None:
            received_pads.add(pad)

        source = FakeSeriesSource(
            name="source",
            source_pad_names=("H1", "L1"),
            rate=256,
            signal_type="sine",
            fsin=5.0,
            t0=0,
            end=1.0,
        )
        sink = TSAppSink(
            name="sink",
            sink_pad_names=("H1", "L1"),
            callbacks={"H1": callback, "L1": callback},
        )

        pipeline = Pipeline()
        pipeline.connect(source, sink)
        pipeline.run()

        # Should have received pads for both H1 and L1
        pad_names = {p.pad_name for p in received_pads}
        assert "H1" in pad_names
        assert "L1" in pad_names

    def test_callback_receives_buffer_with_data(self):
        """Test that callback receives buffer with actual data."""
        received_buffers = []

        def callback(pad: SinkPad, buf: SeriesBuffer) -> None:
            received_buffers.append(buf)

        source = FakeSeriesSource(
            name="source",
            source_pad_names=("data",),
            rate=256,
            signal_type="sine",
            fsin=5.0,
            t0=0,
            end=1.0,
        )
        sink = TSAppSink(
            name="sink",
            sink_pad_names=("data",),
            callbacks={"data": callback},
        )

        pipeline = Pipeline()
        pipeline.connect(source, sink)
        pipeline.run()

        # Verify buffer has data
        for buf in received_buffers:
            assert buf.data is not None
            assert len(buf.data) > 0
