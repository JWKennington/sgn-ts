"""Tests for sgnts.compose module - composable time-series elements."""

import pytest

from sgn.apps import Pipeline
from sgn.sinks import NullSink
from sgn.sources import IterSource
from sgn.transforms import CallableTransform

from sgnts.compose import (
    TSCompose,
    TSComposedSinkElement,
    TSComposedSourceElement,
    TSComposedTransformElement,
)
from sgnts.sinks import TSFrameCollectSink
from sgnts.sources import FakeSeriesSource
from sgnts.transforms import Amplify


class TestTSComposedSourceElement:
    """Tests for TSComposedSourceElement."""

    def test_source_only(self):
        """A single TS source can be wrapped as a composed source."""
        source = FakeSeriesSource(
            name="src",
            source_pad_names=["out"],
            rate=256,
            signal_type="const",
            const=1.0,
            end=1,
        )

        composed = TSCompose(source).as_source(name="composed_src")

        assert isinstance(composed, TSComposedSourceElement)
        assert "out" in composed.srcs
        assert len(composed.source_pads) == 1

    def test_source_plus_transform(self):
        """TSSource + TSTransform composes into a source."""
        source = FakeSeriesSource(
            name="src",
            source_pad_names=["data"],
            rate=256,
            signal_type="const",
            const=5.0,
            ngap=0,
            end=2,
        )

        transform = Amplify(
            name="double",
            source_pad_names=["data"],
            sink_pad_names=["data"],
            factor=2.0,
        )

        composed = TSCompose().connect(source, transform).as_source(name="doubled_src")

        assert isinstance(composed, TSComposedSourceElement)
        assert "data" in composed.srcs

        # Test in pipeline
        sink = TSFrameCollectSink(name="sink", sink_pad_names=["data"])
        pipeline = Pipeline()
        pipeline.connect(composed, sink)
        pipeline.run()

        out_frame = sink.out_frames()["data"]
        # Should be 5.0 * 2.0 = 10.0
        assert all(out_frame.filleddata() == 10.0)

    def test_source_plus_multiple_transforms(self):
        """TSSource + multiple transforms compose correctly."""
        source = FakeSeriesSource(
            name="src",
            source_pad_names=["x"],
            rate=256,
            signal_type="const",
            const=2.0,
            ngap=0,
            end=1,
        )

        double = Amplify(
            name="double",
            source_pad_names=["x"],
            sink_pad_names=["x"],
            factor=2.0,
        )

        triple = Amplify(
            name="triple",
            source_pad_names=["x"],
            sink_pad_names=["x"],
            factor=3.0,
        )

        # 2.0 * 2.0 * 3.0 = 12.0
        composed = (
            TSCompose()
            .connect(source, double)
            .connect(double, triple)
            .as_source(name="processed")
        )

        sink = TSFrameCollectSink(name="sink", sink_pad_names=["x"])
        pipeline = Pipeline()
        pipeline.connect(composed, sink)
        pipeline.run()

        out_frame = sink.out_frames()["x"]
        assert all(out_frame.filleddata() == 12.0)

    def test_source_composition_with_multi_pad(self):
        """Composed source with multiple pads works correctly."""
        source = FakeSeriesSource(
            name="src",
            source_pad_names=["H1", "L1"],
            rate=256,
            signal_type="const",
            const=1.0,
            ngap=0,
            end=1,
        )

        # Transform that doubles H1 and triples L1
        double_h1 = Amplify(
            name="double_h1",
            source_pad_names=["H1"],
            sink_pad_names=["H1"],
            factor=2.0,
        )

        triple_l1 = Amplify(
            name="triple_l1",
            source_pad_names=["L1"],
            sink_pad_names=["L1"],
            factor=3.0,
        )

        composed = (
            TSCompose()
            .connect(source, double_h1)
            .connect(source, triple_l1)
            .as_source(name="processed")
        )

        assert set(composed.srcs.keys()) == {"H1", "L1"}

        sink = TSFrameCollectSink(name="sink", sink_pad_names=["H1", "L1"])
        pipeline = Pipeline()
        pipeline.connect(composed, sink)
        pipeline.run()

        h1_frame = sink.out_frames()["H1"]
        l1_frame = sink.out_frames()["L1"]
        assert all(h1_frame.filleddata() == 2.0)
        assert all(l1_frame.filleddata() == 3.0)

    @pytest.mark.xfail(
        reason="TS validation temporarily disabled to allow Latency element"
    )
    def test_invalid_source_composition_non_ts_element(self):
        """Source composition with non-TS elements is rejected."""
        # Regular SGN source (not TS)
        sgn_source = IterSource(
            name="src",
            source_pad_names=["data"],
            iters={"data": iter([1, 2, 3])},
        )

        with pytest.raises(TypeError, match="Non-TS elements found"):
            TSCompose(sgn_source).as_source()

    def test_invalid_source_composition_with_sink(self):
        """Source composition cannot contain a SinkElement."""
        source = FakeSeriesSource(
            name="src",
            source_pad_names=["data"],
            rate=256,
            end=1,
        )

        transform = Amplify(
            name="t",
            source_pad_names=["data"],
            sink_pad_names=["data"],
            factor=1.0,
        )

        sink = TSFrameCollectSink(name="sink", sink_pad_names=["data"])

        with pytest.raises(TypeError, match="cannot contain SinkElement"):
            (
                TSCompose()
                .connect(source, transform)
                .connect(transform, sink)
                .as_source()
            )


class TestTSComposedTransformElement:
    """Tests for TSComposedTransformElement."""

    def test_single_transform(self):
        """A single transform can be wrapped."""
        transform = Amplify(
            name="double",
            source_pad_names=["out"],
            sink_pad_names=["in"],
            factor=2.0,
        )

        composed = TSCompose(transform).as_transform(name="composed_transform")

        assert isinstance(composed, TSComposedTransformElement)
        assert "in" in composed.snks
        assert "out" in composed.srcs

    def test_transform_chain(self):
        """Multiple transforms compose into a single transform."""
        t1 = Amplify(
            name="double",
            source_pad_names=["data"],
            sink_pad_names=["data"],
            factor=2.0,
        )

        t2 = Amplify(
            name="triple",
            source_pad_names=["data"],
            sink_pad_names=["data"],
            factor=3.0,
        )

        composed = TSCompose().connect(t1, t2).as_transform(name="times_six")

        assert "data" in composed.snks
        assert "data" in composed.srcs

        # Test in pipeline
        source = FakeSeriesSource(
            name="src",
            source_pad_names=["data"],
            rate=256,
            signal_type="const",
            const=1.0,
            ngap=0,
            end=1,
        )
        sink = TSFrameCollectSink(name="sink", sink_pad_names=["data"])

        pipeline = Pipeline()
        pipeline.connect(source, composed)
        pipeline.connect(composed, sink)
        pipeline.run()

        out_frame = sink.out_frames()["data"]
        # 1.0 * 2.0 * 3.0 = 6.0
        assert all(out_frame.filleddata() == 6.0)

    def test_invalid_transform_with_source(self):
        """Transform composition cannot contain SourceElement."""
        source = FakeSeriesSource(
            name="src",
            source_pad_names=["data"],
            rate=256,
            end=1,
        )

        with pytest.raises(TypeError, match="can only contain TSTransform"):
            TSCompose(source).as_transform()

    def test_invalid_transform_with_sink(self):
        """Transform composition cannot contain SinkElement."""
        transform = Amplify(
            name="t",
            source_pad_names=["data"],
            sink_pad_names=["data"],
            factor=1.0,
        )
        sink = TSFrameCollectSink(name="sink", sink_pad_names=["data"])

        with pytest.raises(TypeError, match="can only contain TSTransform"):
            TSCompose().connect(transform, sink).as_transform()

    @pytest.mark.xfail(
        reason="TS validation temporarily disabled to allow Latency element"
    )
    def test_invalid_transform_non_ts_element(self):
        """Transform composition with non-TS elements is rejected."""
        sgn_transform = CallableTransform.from_callable(
            name="t",
            callable=lambda frame: frame.data,
            output_pad_name="data",
            sink_pad_names=["data"],
        )

        with pytest.raises(TypeError, match="Non-TS elements found"):
            TSCompose(sgn_transform).as_transform()


class TestTSComposedSinkElement:
    """Tests for TSComposedSinkElement."""

    def test_sink_only(self):
        """A single TS sink can be wrapped."""
        sink = TSFrameCollectSink(name="sink", sink_pad_names=["data"])

        composed = TSCompose(sink).as_sink(name="composed_sink")

        assert isinstance(composed, TSComposedSinkElement)
        assert "data" in composed.snks

    def test_transform_plus_sink(self):
        """TSTransform + TSSink composes into a sink."""
        transform = Amplify(
            name="double",
            source_pad_names=["data"],
            sink_pad_names=["data"],
            factor=2.0,
        )

        collect = TSFrameCollectSink(name="collect", sink_pad_names=["data"])

        composed = TSCompose().connect(transform, collect).as_sink(name="doubling_sink")

        assert isinstance(composed, TSComposedSinkElement)
        assert "data" in composed.snks

        # Test in pipeline
        source = FakeSeriesSource(
            name="src",
            source_pad_names=["data"],
            rate=256,
            signal_type="const",
            const=5.0,
            ngap=0,
            end=1,
        )

        pipeline = Pipeline()
        pipeline.connect(source, composed)
        pipeline.run()

        # Access internal sink's data
        out_frame = collect.out_frames()["data"]
        assert all(out_frame.filleddata() == 10.0)

    def test_invalid_sink_with_source_first(self):
        """Sink composition cannot contain SourceElement."""
        source = FakeSeriesSource(
            name="src",
            source_pad_names=["data"],
            rate=256,
            end=1,
        )

        sink = TSFrameCollectSink(name="sink", sink_pad_names=["data"])

        with pytest.raises(TypeError, match="cannot contain SourceElement"):
            TSCompose().connect(source, sink).as_sink()

    def test_invalid_sink_not_ending_with_sink(self):
        """Sink composition must contain at least one SinkElement."""
        transform = Amplify(
            name="t",
            source_pad_names=["data"],
            sink_pad_names=["data"],
            factor=1.0,
        )

        with pytest.raises(TypeError, match="requires at least one SinkElement"):
            TSCompose(transform).as_sink()

    @pytest.mark.xfail(
        reason="TS validation temporarily disabled to allow Latency element"
    )
    def test_invalid_sink_non_ts_element(self):
        """Sink composition with non-TS elements is rejected."""
        sgn_sink = NullSink(name="sink", sink_pad_names=["data"])

        with pytest.raises(TypeError, match="Non-TS elements found"):
            TSCompose(sgn_sink).as_sink()


class TestNestedComposition:
    """Tests for nested composition (composed elements containing composed elements)."""

    def test_nested_transform_in_source(self):
        """A composed transform can be nested inside a composed source."""
        # Create inner composed transform
        t1 = Amplify(
            name="double",
            source_pad_names=["data"],
            sink_pad_names=["data"],
            factor=2.0,
        )
        t2 = Amplify(
            name="triple",
            source_pad_names=["data"],
            sink_pad_names=["data"],
            factor=3.0,
        )

        inner_transform = (
            TSCompose().connect(t1, t2).as_transform(name="inner_transform")
        )

        # Create outer composed source with inner transform
        source = FakeSeriesSource(
            name="src",
            source_pad_names=["data"],
            rate=256,
            signal_type="const",
            const=1.0,
            ngap=0,
            end=1,
        )

        outer_source = (
            TSCompose().connect(source, inner_transform).as_source(name="nested_source")
        )

        assert isinstance(outer_source, TSComposedSourceElement)

        # Test in pipeline
        sink = TSFrameCollectSink(name="sink", sink_pad_names=["data"])
        pipeline = Pipeline()
        pipeline.connect(outer_source, sink)
        pipeline.run()

        out_frame = sink.out_frames()["data"]
        # 1.0 * 2.0 * 3.0 = 6.0
        assert all(out_frame.filleddata() == 6.0)

    def test_nested_transform_in_transform(self):
        """A composed transform can be nested inside another composed transform."""
        # Create inner composed transform
        inner_t1 = Amplify(
            name="inner_double",
            source_pad_names=["data"],
            sink_pad_names=["data"],
            factor=2.0,
        )
        inner_t2 = Amplify(
            name="inner_triple",
            source_pad_names=["data"],
            sink_pad_names=["data"],
            factor=3.0,
        )
        inner = TSCompose().connect(inner_t1, inner_t2).as_transform(name="inner")

        # Create outer composed transform with inner transform
        # Note: element name must differ from composed element name
        # to avoid pad conflicts
        outer_t = Amplify(
            name="times_ten",
            source_pad_names=["data"],
            sink_pad_names=["data"],
            factor=10.0,
        )
        outer = TSCompose().connect(inner, outer_t).as_transform(name="outer")

        assert isinstance(outer, TSComposedTransformElement)

        # Test in pipeline
        source = FakeSeriesSource(
            name="src",
            source_pad_names=["data"],
            rate=256,
            signal_type="const",
            const=1.0,
            ngap=0,
            end=1,
        )
        sink = TSFrameCollectSink(name="sink", sink_pad_names=["data"])

        pipeline = Pipeline()
        pipeline.connect(source, outer)
        pipeline.connect(outer, sink)
        pipeline.run()

        out_frame = sink.out_frames()["data"]
        # 1.0 * 2.0 * 3.0 * 10.0 = 60.0
        assert all(out_frame.filleddata() == 60.0)

    def test_nested_transform_in_sink(self):
        """A composed transform can be nested inside a composed sink."""
        # Create inner composed transform
        t1 = Amplify(
            name="double",
            source_pad_names=["data"],
            sink_pad_names=["data"],
            factor=2.0,
        )
        t2 = Amplify(
            name="triple",
            source_pad_names=["data"],
            sink_pad_names=["data"],
            factor=3.0,
        )
        inner_transform = (
            TSCompose().connect(t1, t2).as_transform(name="inner_transform")
        )

        # Create outer composed sink with inner transform
        collect = TSFrameCollectSink(name="collect", sink_pad_names=["data"])
        outer_sink = (
            TSCompose().connect(inner_transform, collect).as_sink(name="nested_sink")
        )

        assert isinstance(outer_sink, TSComposedSinkElement)

        # Test in pipeline
        source = FakeSeriesSource(
            name="src",
            source_pad_names=["data"],
            rate=256,
            signal_type="const",
            const=1.0,
            ngap=0,
            end=1,
        )

        pipeline = Pipeline()
        pipeline.connect(source, outer_sink)
        pipeline.run()

        out_frame = collect.out_frames()["data"]
        # 1.0 * 2.0 * 3.0 = 6.0
        assert all(out_frame.filleddata() == 6.0)


class TestTSComposeBuilder:
    """Tests for the TSCompose builder class."""

    def test_compose_preserves_element_identity(self):
        """Internal elements are stored by reference."""
        source = FakeSeriesSource(
            name="src",
            source_pad_names=["data"],
            rate=256,
            end=1,
        )

        composed = TSCompose(source).as_source()

        assert composed.internal_elements[0] is source

    def test_connect_auto_inserts_elements(self):
        """connect() automatically inserts elements not yet in composition."""
        source = FakeSeriesSource(
            name="src",
            source_pad_names=["data"],
            rate=256,
            signal_type="const",
            const=1.0,
            ngap=0,
            end=1,
        )

        transform = Amplify(
            name="t",
            source_pad_names=["data"],
            sink_pad_names=["data"],
            factor=5.0,
        )

        # Don't call insert() - connect() should handle it
        composed = TSCompose().connect(source, transform).as_source()

        sink = TSFrameCollectSink(name="sink", sink_pad_names=["data"])
        pipeline = Pipeline()
        pipeline.connect(composed, sink)
        pipeline.run()

        out_frame = sink.out_frames()["data"]
        assert all(out_frame.filleddata() == 5.0)


class TestEOSPropagation:
    """Tests for End-of-Stream propagation in composed elements."""

    def test_composed_source_eos(self):
        """EOS propagates correctly through composed source."""
        source = FakeSeriesSource(
            name="src",
            source_pad_names=["data"],
            rate=256,
            signal_type="const",
            const=1.0,
            ngap=0,
            end=2,
        )

        transform = Amplify(
            name="t",
            source_pad_names=["data"],
            sink_pad_names=["data"],
            factor=1.0,
        )

        composed = TSCompose().connect(source, transform).as_source()

        sink = TSFrameCollectSink(name="sink", sink_pad_names=["data"])
        pipeline = Pipeline()
        pipeline.connect(composed, sink)
        pipeline.run()

        # Should complete normally with EOS
        out_frame = sink.out_frames()["data"]
        assert out_frame.samples == 256 * 2


class TestIntegration:
    """Integration tests for composed elements with full pipelines."""

    def test_composed_elements_in_complex_pipeline(self):
        """Composed elements work in complex pipeline configurations."""
        # Create a composed source (source + amplify)
        raw_source = FakeSeriesSource(
            name="raw",
            source_pad_names=["signal"],
            rate=256,
            signal_type="const",
            const=1.0,
            ngap=0,
            end=1,
        )

        preprocess = Amplify(
            name="preprocess",
            source_pad_names=["signal"],
            sink_pad_names=["signal"],
            factor=2.0,
        )

        composed_source = (
            TSCompose()
            .connect(raw_source, preprocess)
            .as_source(name="preprocessed_source")
        )

        # Create a composed transform (amplify chain)
        t1 = Amplify(
            name="t1",
            source_pad_names=["signal"],
            sink_pad_names=["signal"],
            factor=3.0,
        )

        t2 = Amplify(
            name="t2",
            source_pad_names=["signal"],
            sink_pad_names=["signal"],
            factor=5.0,
        )

        composed_transform = (
            TSCompose().connect(t1, t2).as_transform(name="processing_chain")
        )

        # Create a composed sink (amplify + collect)
        final_transform = Amplify(
            name="final",
            source_pad_names=["signal"],
            sink_pad_names=["signal"],
            factor=7.0,
        )

        collector = TSFrameCollectSink(name="collector", sink_pad_names=["signal"])

        composed_sink = (
            TSCompose().connect(final_transform, collector).as_sink(name="output_sink")
        )

        # Build pipeline: composed_source -> composed_transform -> composed_sink
        pipeline = Pipeline()
        pipeline.connect(composed_source, composed_transform)
        pipeline.connect(composed_transform, composed_sink)
        pipeline.run()

        # Calculation: 1.0 * 2.0 * 3.0 * 5.0 * 7.0 = 210.0
        out_frame = collector.out_frames()["signal"]
        assert all(out_frame.filleddata() == 210.0)

    def test_mixed_composed_and_regular_elements(self):
        """Composed elements work alongside regular TS elements."""
        # Regular source
        source = FakeSeriesSource(
            name="src",
            source_pad_names=["data"],
            rate=256,
            signal_type="const",
            const=10.0,
            ngap=0,
            end=1,
        )

        # Composed transform
        t1 = Amplify(
            name="half",
            source_pad_names=["data"],
            sink_pad_names=["data"],
            factor=0.5,
        )

        t2 = Amplify(
            name="double",
            source_pad_names=["data"],
            sink_pad_names=["data"],
            factor=2.0,
        )

        composed_transform = TSCompose().connect(t1, t2).as_transform()

        # Regular sink
        sink = TSFrameCollectSink(name="sink", sink_pad_names=["data"])

        pipeline = Pipeline()
        pipeline.connect(source, composed_transform)
        pipeline.connect(composed_transform, sink)
        pipeline.run()

        # 10.0 * 0.5 * 2.0 = 10.0
        out_frame = sink.out_frames()["data"]
        assert all(out_frame.filleddata() == 10.0)


class TestEdgeCases:
    """Tests for edge cases and validation errors."""

    def test_composed_source_empty_elements(self):
        """TSComposedSourceElement with empty elements raises ValueError."""
        with pytest.raises(ValueError, match="requires at least one element"):
            TSComposedSourceElement(
                name="empty",
                internal_elements=[],
                internal_links={},
            )

    def test_composed_transform_empty_elements(self):
        """TSComposedTransformElement with empty elements raises ValueError."""
        with pytest.raises(ValueError, match="requires at least one element"):
            TSComposedTransformElement(
                name="empty",
                internal_elements=[],
                internal_links={},
            )

    def test_composed_sink_empty_elements(self):
        """TSComposedSinkElement with empty elements raises ValueError."""
        with pytest.raises(ValueError, match="requires at least one element"):
            TSComposedSinkElement(
                name="empty",
                internal_elements=[],
                internal_links={},
            )
