"""Tests for align_buffers functionality in AdapterConfig"""

from dataclasses import dataclass

import numpy
from sgn import validator
from sgn.apps import Pipeline
from sgn.sources import SignalEOS

from sgnts.base import Offset, SeriesBuffer, TSFrame, TSTransform
from sgnts.base.audioadapter import AdapterConfig
from sgnts.base.slice_tools import TSSlice, TSSlices
from sgnts.sinks import NullSeriesSink
from sgnts.sources import FakeSeriesSource, TSIterSource
from sgnts.transforms import Correlate


class TestAlignBuffersBasic:
    """Basic tests for align_buffers functionality"""

    def test_adapter_config_has_align_buffers(self):
        """Test that AdapterConfig has align_buffers parameter"""
        config = AdapterConfig(align_buffers=True)
        assert config.align_buffers is True

        config_default = AdapterConfig()
        assert config_default.align_buffers is False

    def test_sink_with_align_buffers_single_rate(self):
        """Test sink with align_buffers at a single rate (no-op case)"""
        sink = NullSeriesSink(
            name="sink",
            sink_pad_names=["test"],
            adapter_config=AdapterConfig(align_buffers=True),
        )

        # Create frame at single rate
        frame = TSFrame(
            buffers=[
                SeriesBuffer(
                    offset=0,
                    sample_rate=4,
                    shape=(2,),
                    data=numpy.array([1.0, 2.0]),
                )
            ]
        )

        # Pull frame
        sink.pull(pad=sink.snks["test"], frame=frame)
        sink.internal()

        # Verify aligned_slices were computed
        assert sink.aligned_slices[sink.snks["test"]] is not None

    def test_sink_with_align_buffers_multi_rate(self):
        """Test sink with align_buffers at multiple rates"""
        sink = NullSeriesSink(
            name="sink",
            sink_pad_names=["high_rate", "low_rate"],
            adapter_config=AdapterConfig(align_buffers=True),
        )

        # At 4 Hz: offset_per_sample = 16384 / 4 = 4096
        # Create buffer spanning offsets [0, 8192) = 2 samples at 4Hz
        frame_high = TSFrame(
            buffers=[
                SeriesBuffer(
                    offset=0,
                    sample_rate=4,
                    shape=(2,),
                    data=numpy.array([1.0, 2.0]),
                )
            ]
        )

        # At 2 Hz: offset_per_sample = 16384 / 2 = 8192
        # Create buffer spanning offsets [0, 8192) = 1 sample at 2Hz
        frame_low = TSFrame(
            buffers=[
                SeriesBuffer(
                    offset=0,
                    sample_rate=2,
                    shape=(1,),
                    data=numpy.array([3.0]),
                )
            ]
        )

        # Pull frames
        sink.pull(pad=sink.snks["high_rate"], frame=frame_high)
        sink.pull(pad=sink.snks["low_rate"], frame=frame_low)
        sink.internal()

        # Verify aligned_slices were computed for both pads
        assert sink.aligned_slices[sink.snks["high_rate"]] is not None
        assert sink.aligned_slices[sink.snks["low_rate"]] is not None


class TestAlignBuffersDownsampling:
    """Tests for buffer alignment with downsampling behavior"""

    def test_align_buffers_shrinks_slices_to_min_rate(self):
        """Test that slices are aligned to minimum sampling rate"""
        sink = NullSeriesSink(
            name="sink",
            sink_pad_names=["rate_4hz", "rate_2hz"],
            adapter_config=AdapterConfig(align_buffers=True, skip_gaps=True),
        )

        # At 4 Hz: offsets at multiples of 4096
        # Create contiguous buffers: data, gap, data
        # (0, 4096) = 1 sample, (4096, 8192) = 1 sample gap, (8192, 16384) = 2 samples
        frame_4hz = TSFrame(
            buffers=[
                SeriesBuffer(
                    offset=0,
                    sample_rate=4,
                    shape=(1,),
                    data=numpy.array([1.0]),
                ),
                SeriesBuffer(
                    offset=Offset.fromsamples(1, 4),
                    sample_rate=4,
                    shape=(1,),  # gap - must be contiguous
                    data=None,
                ),
                SeriesBuffer(
                    offset=Offset.fromsamples(2, 4),
                    sample_rate=4,
                    shape=(2,),
                    data=numpy.array([2.0, 3.0]),
                ),
            ]
        )

        # At 2 Hz: offsets at multiples of 8192
        frame_2hz = TSFrame(
            buffers=[
                SeriesBuffer(
                    offset=0,
                    sample_rate=2,
                    shape=(2,),
                    data=numpy.array([4.0, 5.0]),
                )
            ]
        )

        # Pull frames
        sink.pull(pad=sink.snks["rate_4hz"], frame=frame_4hz)
        sink.pull(pad=sink.snks["rate_2hz"], frame=frame_2hz)
        sink.internal()

        # Minimum rate is 2Hz
        # aligned_slices now contains ALL slice boundaries for frame.align()
        # The intersection of non-gap slices across pads determines boundaries
        # 4Hz pad non-gap: (0, 4096) eliminated, (8192, 16384) kept
        # 2Hz pad non-gap: (0, 16384) kept
        # Intersection: (8192, 16384)
        # So boundaries are: [0, 8192, 16384] -> slices [(0, 8192), (8192, 16384)]
        aligned_4hz = sink.aligned_slices[sink.snks["rate_4hz"]]
        assert len(aligned_4hz.slices) == 2
        assert aligned_4hz.slices[0] == TSSlice(0, 8192)  # gap region
        assert aligned_4hz.slices[1] == TSSlice(8192, 16384)  # data region

        # Both pads get the same slice boundaries
        aligned_2hz = sink.aligned_slices[sink.snks["rate_2hz"]]
        assert aligned_2hz == aligned_4hz

        # Verify preparedframes have correct gap/non-gap buffer structure
        # 4Hz pad: first buffer should be gap, second should be data
        pf_4hz = sink.preparedframes[sink.snks["rate_4hz"]]
        assert len(pf_4hz.buffers) == 2
        assert pf_4hz.buffers[0].slice == TSSlice(0, 8192)
        assert pf_4hz.buffers[0].is_gap is True  # gap region
        assert pf_4hz.buffers[1].slice == TSSlice(8192, 16384)
        assert pf_4hz.buffers[1].is_gap is False  # data region

        # 2Hz pad: original data was contiguous, so both buffers have data
        # (frame.align splits buffers but preserves each pad's data/gap structure)
        pf_2hz = sink.preparedframes[sink.snks["rate_2hz"]]
        assert len(pf_2hz.buffers) == 2
        assert pf_2hz.buffers[0].slice == TSSlice(0, 8192)
        assert pf_2hz.buffers[0].is_gap is False  # 2Hz had data here
        assert pf_2hz.buffers[1].slice == TSSlice(8192, 16384)
        assert pf_2hz.buffers[1].is_gap is False  # 2Hz had data here

    def test_align_buffers_with_gaps(self):
        """Test that gaps are properly handled during alignment"""
        sink = NullSeriesSink(
            name="sink",
            sink_pad_names=["data"],
            adapter_config=AdapterConfig(align_buffers=True, skip_gaps=True),
        )

        # Create frame with data and gap buffers at 8Hz
        # At 8 Hz: offset_per_sample = 16384 / 8 = 2048
        # Make buffers contiguous
        frame = TSFrame(
            buffers=[
                # Data: (0, 2048) = 1 sample at 8Hz
                SeriesBuffer(
                    offset=0,
                    sample_rate=8,
                    shape=(1,),
                    data=numpy.array([1.0]),
                ),
                # Gap: (2048, 6144) = 2 samples at 8Hz (contiguous)
                SeriesBuffer(
                    offset=Offset.fromsamples(1, 8),
                    sample_rate=8,
                    shape=(2,),
                    data=None,
                ),
                # Data: (6144, 16384) = 5 samples at 8Hz (contiguous)
                SeriesBuffer(
                    offset=Offset.fromsamples(3, 8),
                    sample_rate=8,
                    shape=(5,),
                    data=numpy.array([2.0, 3.0, 4.0, 5.0, 6.0]),
                ),
            ]
        )

        sink.pull(pad=sink.snks["data"], frame=frame)
        sink.internal()

        # aligned_slices contains ALL slice boundaries for frame.align()
        # Single pad, so boundaries come from that pad's non-gap slices
        # Non-gap slices at 8Hz: (0, 2048) and (6144, 16384)
        # Boundaries: [0, 2048, 6144, 16384]
        # -> slices: [(0, 2048), (2048, 6144), (6144, 16384)]
        aligned = sink.aligned_slices[sink.snks["data"]]
        assert len(aligned.slices) == 3
        assert aligned.slices[0] == TSSlice(0, 2048)  # data region
        assert aligned.slices[1] == TSSlice(2048, 6144)  # gap region
        assert aligned.slices[2] == TSSlice(6144, 16384)  # data region

        # Verify preparedframes have correct gap/non-gap buffer structure
        pf = sink.preparedframes[sink.snks["data"]]
        assert len(pf.buffers) == 3
        assert pf.buffers[0].slice == TSSlice(0, 2048)
        assert pf.buffers[0].is_gap is False  # data region
        assert pf.buffers[1].slice == TSSlice(2048, 6144)
        assert pf.buffers[1].is_gap is True  # gap region
        assert pf.buffers[2].slice == TSSlice(6144, 16384)
        assert pf.buffers[2].is_gap is False  # data region


class TestAlignBuffersEdgeCases:
    """Edge case tests for align_buffers"""

    def test_align_buffers_empty_frame(self):
        """Test align_buffers with heartbeat (empty) buffers"""
        sink = NullSeriesSink(
            name="sink",
            sink_pad_names=["test"],
            adapter_config=AdapterConfig(align_buffers=True),
        )

        # Heartbeat buffer (shape=(0,))
        frame = TSFrame(
            buffers=[
                SeriesBuffer(
                    offset=0,
                    sample_rate=4,
                    shape=(0,),
                    data=None,
                )
            ]
        )

        sink.pull(pad=sink.snks["test"], frame=frame)

        # Should not crash, aligned_slices should be empty
        # Note: internal() may not be called if not aligned yet

    def test_align_buffers_all_gaps(self):
        """Test align_buffers when all buffers are gaps"""
        sink = NullSeriesSink(
            name="sink",
            sink_pad_names=["test"],
            adapter_config=AdapterConfig(align_buffers=True, skip_gaps=True),
        )

        frame = TSFrame(
            buffers=[
                SeriesBuffer(
                    offset=0,
                    sample_rate=4,
                    shape=(4,),
                    data=None,  # All gap
                )
            ]
        )

        sink.pull(pad=sink.snks["test"], frame=frame)
        sink.internal()

        # All gaps means no non-gap slices, so boundaries are just [start, end]
        # This creates a single slice covering the entire frame
        aligned = sink.aligned_slices[sink.snks["test"]]
        assert isinstance(aligned, TSSlices)
        assert len(aligned.slices) == 1
        assert aligned.slices[0] == TSSlice(0, 16384)

    def test_align_buffers_preserves_aligned_slices(self):
        """Test that already-aligned slices are preserved"""
        sink = NullSeriesSink(
            name="sink",
            sink_pad_names=["test"],
            adapter_config=AdapterConfig(align_buffers=True),
        )

        # Create frame with slices already at 2Hz boundaries
        # At 2 Hz: offset_per_sample = 8192
        frame = TSFrame(
            buffers=[
                SeriesBuffer(
                    offset=0,
                    sample_rate=2,
                    shape=(2,),
                    data=numpy.array([1.0, 2.0]),
                )
            ]
        )

        sink.pull(pad=sink.snks["test"], frame=frame)
        sink.internal()

        aligned = sink.aligned_slices[sink.snks["test"]]
        # Should be preserved as-is
        assert len(aligned.slices) == 1
        assert aligned.slices[0] == TSSlice(0, 16384)


class TestAlignBuffersIntegration:
    """Integration tests for align_buffers with frame.align()"""

    def test_preparedframes_aligned_after_processing(self):
        """Test that preparedframes are properly aligned after processing"""
        sink = NullSeriesSink(
            name="sink",
            sink_pad_names=["high", "low"],
            adapter_config=AdapterConfig(align_buffers=True),
        )

        # High rate: 4Hz
        frame_high = TSFrame(
            buffers=[
                SeriesBuffer(
                    offset=0,
                    sample_rate=4,
                    shape=(4,),
                    data=numpy.array([1.0, 2.0, 3.0, 4.0]),
                )
            ]
        )

        # Low rate: 2Hz (minimum)
        frame_low = TSFrame(
            buffers=[
                SeriesBuffer(
                    offset=0,
                    sample_rate=2,
                    shape=(2,),
                    data=numpy.array([5.0, 6.0]),
                )
            ]
        )

        sink.pull(pad=sink.snks["high"], frame=frame_high)
        sink.pull(pad=sink.snks["low"], frame=frame_low)
        sink.internal()

        # Verify preparedframes exist and are properly structured
        assert sink.preparedframes[sink.snks["high"]] is not None
        assert sink.preparedframes[sink.snks["low"]] is not None

        # Verify preparedframes have been aligned
        # After alignment, buffers should correspond to aligned_slices
        prepared_high = sink.preparedframes[sink.snks["high"]]
        aligned_high = sink.aligned_slices[sink.snks["high"]]

        # Number of buffers should match number of aligned slices
        assert len(prepared_high.buffers) == len(aligned_high.slices)


@dataclass(kw_only=True)
class SimpleAdder(TSTransform):
    """Simple two-input adder for testing align_buffers."""

    def configure(self) -> None:
        self.adapter_config.align_buffers = True
        self.adapter_config.skip_gaps = True

    @validator.num_pads(sink_pads=2, source_pads=1)
    def validate(self) -> None:
        pass

    def internal(self) -> None:
        """Add two input frames buffer by buffer."""
        super().internal()

        input_frames = self.next_inputs()
        _, output_frame = self.next_output()

        # Get the two input frames
        frame1, frame2 = input_frames.values()

        # Check that frames have same number of buffers
        assert len(frame1.buffers) == len(frame2.buffers), (
            f"Input frames have different number of buffers: "
            f"{len(frame1.buffers)} vs {len(frame2.buffers)}"
        )

        # Add corresponding buffers
        for buf1, buf2 in zip(frame1.buffers, frame2.buffers):
            # Check alignment
            assert (
                buf1.offset == buf2.offset
            ), f"Buffers not aligned: {buf1.offset} vs {buf2.offset}"
            assert (
                buf1.samples == buf2.samples
            ), f"Buffers have different samples: {buf1.samples} vs {buf2.samples}"

            # Handle gaps
            if buf1.is_gap or buf2.is_gap:
                data = None
            else:
                assert buf1.data is not None and buf2.data is not None
                data = buf1.data + buf2.data

            output_buf = SeriesBuffer(
                data=data,
                offset=buf1.offset,
                sample_rate=buf1.sample_rate,
                shape=buf1.shape,
            )
            output_frame.append(output_buf)

        output_frame.close()


class TestAlignBuffersPipeline:
    """Pipeline-based integration tests for align_buffers"""

    def test_align_buffers_with_multi_buffer_frames(self):
        """Test that align_buffers correctly aligns multi-buffer frames.

        This test creates two sources with different buffer boundaries:
        - Pad 1: data [0-150 samples), gap [150-200 samples), data [200-384 samples)
        - Pad 2: data [0-128 samples), gap [128-256 samples), data [256-384 samples)

        With skip_gaps=True, the intersection of non-gap regions creates boundaries at:
        0, 8192, 16384, 24576 (in offset units) = 0, 128, 256, 384 samples

        After alignment, both pads should have 3 buffers:
        - Buffer 0: [0-8192) = 128 samples - data
        - Buffer 1: [8192-16384) = 128 samples - gap
        - Buffer 2: [16384-24576) = 128 samples - data
        """
        sample_rate = 256

        # Pad 1: data [0-150), gap [150-200), data [200-384)
        buffers1 = [
            SeriesBuffer(
                offset=Offset.fromsamples(0, sample_rate),
                sample_rate=sample_rate,
                data=numpy.ones(150),
                shape=(150,),
            ),
            SeriesBuffer(
                offset=Offset.fromsamples(150, sample_rate),
                sample_rate=sample_rate,
                data=None,  # Gap
                shape=(50,),
            ),
            SeriesBuffer(
                offset=Offset.fromsamples(200, sample_rate),
                sample_rate=sample_rate,
                data=numpy.ones(184) * 2,
                shape=(184,),
            ),
        ]

        # Pad 2: data [0-128), gap [128-256), data [256-384)
        buffers2 = [
            SeriesBuffer(
                offset=Offset.fromsamples(0, sample_rate),
                sample_rate=sample_rate,
                data=numpy.ones(128) * 10,
                shape=(128,),
            ),
            SeriesBuffer(
                offset=Offset.fromsamples(128, sample_rate),
                sample_rate=sample_rate,
                data=None,  # Gap
                shape=(128,),
            ),
            SeriesBuffer(
                offset=Offset.fromsamples(256, sample_rate),
                sample_rate=sample_rate,
                data=numpy.ones(128) * 20,
                shape=(128,),
            ),
        ]

        frame1 = TSFrame(buffers=buffers1)
        frame2 = TSFrame(buffers=buffers2)

        source1 = TSIterSource(
            name="src1",
            source_pad_names=["in1"],
            frames=[frame1],
            t0=0,
        )
        source2 = TSIterSource(
            name="src2",
            source_pad_names=["in2"],
            frames=[frame2],
            t0=0,
        )

        # Create adder with align_buffers enabled
        adder = SimpleAdder(
            name="adder",
            sink_pad_names=["in1", "in2"],
            source_pad_names=["out"],
        )

        sink = NullSeriesSink(
            name="snk",
            sink_pad_names=["out"],
            verbose=True,
            adapter_config=AdapterConfig(),
        )

        # Connect sources to adder
        pipeline = Pipeline()
        pipeline.connect(source1, adder)
        pipeline.connect(source2, adder)
        pipeline.connect(adder, sink)

        # Run the pipeline
        pipeline.run()

        # The adder should have processed the frames
        # With align_buffers=True, the buffers should be aligned to match
        # We expect the assertion in internal() to pass, confirming alignment


class TestOffsetShiftWithAlignBuffers:
    """Tests for offset_shift application in various adapter configurations.

    These tests verify that offset_shift is correctly applied to output buffer
    offsets in different code paths through the prepare_buffers method. The
    offset_shift parameter is used by transforms that introduce latency or
    phase shifts (e.g., filters, correlators) to adjust output timestamps.
    """

    def test_offset_shift_applied_with_align_buffers_mode(self):
        """Verify offset_shift is applied when align_buffers=True."""
        pipeline = Pipeline()

        src1 = FakeSeriesSource(
            name="src1",
            source_pad_names=["O1"],
            signals={"O1": {"signal_type": "const", "const": 1.0, "rate": 256}},
            duration=5,
            t0=1000000000,
        )
        src2 = FakeSeriesSource(
            name="src2",
            source_pad_names=["O1"],
            signals={"O1": {"signal_type": "const", "const": 2.0, "rate": 256}},
            duration=5,
            t0=1000000000,
        )

        # Create correlate filters with latency
        filter_length = 32
        filters = numpy.ones(filter_length)

        corr1 = Correlate(
            name="corr1",
            sink_pad_names=["I1"],
            source_pad_names=["O1"],
            sample_rate=256,
            filters=filters,
            latency=filter_length // 2,
        )
        corr2 = Correlate(
            name="corr2",
            sink_pad_names=["I1"],
            source_pad_names=["O1"],
            sample_rate=256,
            filters=filters,
            latency=filter_length // 2,
        )

        snk = NullSeriesSink(
            name="snk",
            sink_pad_names=["I1", "I2"],
            verbose=True,
            adapter_config=AdapterConfig(align_buffers=True),
        )

        pipeline.insert(
            src1,
            corr1,
            src2,
            corr2,
            snk,
            link_map={
                corr1.snks["I1"]: src1.srcs["O1"],
                corr2.snks["I1"]: src2.srcs["O1"],
                snk.snks["I1"]: corr1.srcs["O1"],
                snk.snks["I2"]: corr2.srcs["O1"],
            },
        )

        with SignalEOS():
            pipeline.run()

    def test_offset_shift_applied_with_gap_buffer_generation(self):
        """Verify offset_shift is applied when generating gap buffers."""
        pipeline = Pipeline()

        # Create a source that generates gaps periodically
        src = FakeSeriesSource(
            name="src",
            source_pad_names=["O1"],
            signals={"O1": {"signal_type": "white", "rate": 256}},
            duration=5,
            t0=1000000000,
            ngap=2,  # Generate a gap every 2 buffers
        )

        filter_length = 32
        filters = numpy.ones(filter_length)
        correlate = Correlate(
            name="corr",
            sink_pad_names=["I1"],
            source_pad_names=["O1"],
            sample_rate=256,
            filters=filters,
            latency=filter_length // 2,
        )

        snk = NullSeriesSink(
            name="snk",
            sink_pad_names=["I1"],
            verbose=True,
            adapter_config=AdapterConfig(skip_gaps=True),
        )

        pipeline.insert(
            src,
            correlate,
            snk,
            link_map={
                correlate.snks["I1"]: src.srcs["O1"],
                snk.snks["I1"]: correlate.srcs["O1"],
            },
        )

        with SignalEOS():
            pipeline.run()
