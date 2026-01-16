"""Unit test for correlate transforms"""

from __future__ import annotations

import numpy
import pytest
import scipy.signal.windows

from sgn import CollectSink, IterSource, Pipeline
from sgn import Frame, SinkPad
from sgnts import filtertools
from sgnts.base import (
    AdapterConfig,
    EventBuffer,
    EventFrame,
    Offset,
    SeriesBuffer,
    TSFrame,
)
from sgnts.base.buffer import Event
from sgnts.base.slice_tools import TIME_MAX
from sgnts.base.time import Time
from sgnts.sinks import DumpSeriesSink
from sgnts.sources import FakeSeriesSource
from sgnts.transforms import Correlate
from sgnts.transforms.correlate import AdaptiveCorrelate


class IsGapCollectSink(CollectSink):
    """Like CollectSink but the collection ignores gap frames."""

    def __post_init__(self):
        self.skip_empty = False
        super().__post_init__()

    def pull(self, pad: SinkPad, frame: Frame) -> None:
        if frame.is_gap:
            return
        super().pull(pad, frame)


class TestCorrelate:
    """Unit tests for Correlate transform element"""

    def test_init(self):
        """Create a Correlate transform element"""
        crl = Correlate(
            filters=numpy.array([[1, 2, 3]]),
            sample_rate=4096,
            source_pad_names=["O1"],
            sink_pad_names=["I1"],
        )
        assert isinstance(crl, Correlate)

    def test_corr(self):
        """Test the corr method"""
        # Create correlate element
        sample_rate = 1
        crl = Correlate(
            filters=numpy.array([[1, 2, 3]]),
            sample_rate=sample_rate,
            source_pad_names=["O1"],
            sink_pad_names=["I1"],
        )

        # Create SeriesBuffer
        frame = TSFrame(
            buffers=[
                SeriesBuffer(
                    offset=0,
                    data=numpy.array([1, 2, 3, 4, 5, 6, 7]),
                    sample_rate=1,
                    shape=(7,),
                ),
            ]
        )

        # Pull onto sink pad
        crl.pull(pad=crl.snks["I1"], frame=frame)

        # Call internal
        crl.internal()

        # Call new
        res = crl.new(pad=crl.srcs["O1"])

        assert res is not None
        assert res[0].data.shape == (1, 5)
        assert res[0].offset == Offset.fromsamples(2, sample_rate)
        numpy.testing.assert_almost_equal(
            res[0].data, numpy.array([[14, 20, 26, 32, 38]])
        )

    def test_corr_latency(self):
        """Test the corr method with nonzero latency"""
        # Create correlate element
        crl = Correlate(
            filters=numpy.array([[1, 2, 3]]),
            sample_rate=1,
            latency=2,
            source_pad_names=["O1"],
            sink_pad_names=["I1"],
        )

        # Create SeriesBuffer
        frame = TSFrame(
            buffers=[
                SeriesBuffer(
                    offset=0,
                    data=numpy.array([1, 2, 3, 4, 5, 6, 7]),
                    sample_rate=1,
                    shape=(7,),
                ),
            ]
        )

        # Pull onto sink pad
        crl.pull(pad=crl.snks["I1"], frame=frame)

        # Call internal
        crl.internal()

        # Call new
        res = crl.new(pad=crl.srcs["O1"])

        assert res is not None
        assert res[0].data.shape == (1, 5)
        assert res[0].offset == 0
        numpy.testing.assert_almost_equal(
            res[0].data, numpy.array([[14, 20, 26, 32, 38]])
        )

    def test_corr_null_ic(self):
        """Test the corr method with null initial conditions (filters=None).
        It should produce gap buffers without error.
        """
        # Create correlate element with no filters
        crl = Correlate(
            sample_rate=1,
            filters=None,
            source_pad_names=["O1"],
            sink_pad_names=["I1"],
        )

        # Create SeriesBuffer
        frame = TSFrame(
            buffers=[
                SeriesBuffer(
                    offset=0,
                    data=numpy.array([1, 2, 3, 4, 5]),
                    sample_rate=1,
                    shape=(5,),
                ),
            ]
        )

        # Pull onto sink pad
        crl.pull(pad=crl.snks["I1"], frame=frame)

        # Call internal
        crl.internal()

        # Call new
        res = crl.new(pad=crl.srcs["O1"])

        # Expectation: Output is a Gap buffer because filters are None
        assert res is not None
        assert len(res) == 1
        assert res[0].is_gap
        assert res[0].data is None

    def test_corr_err_null_filteres(self):
        """Test the corr method with null filters (filters=None).
        It should raise an error if we try to correlate with null filters.
        """
        # Create correlate element with no filters
        crl = Correlate(
            sample_rate=1,
            filters=None,
            source_pad_names=["O1"],
            sink_pad_names=["I1"],
        )

        # Call new and expect error
        with pytest.raises(ValueError, match="Cannot correlate without filters"):
            crl.corr(data=None)


class TestAdaptiveCorrelate:
    """Unit tests for Correlate transform element"""

    def test_init(self):
        """Create a Correlate transform element"""
        init_filters = numpy.array([[1, 2, 3]])
        crl = AdaptiveCorrelate(
            filters=init_filters,
            sample_rate=4096,
            source_pad_names=["O1"],
            sink_pad_names=["I1"],
        )
        assert isinstance(crl, AdaptiveCorrelate)
        assert crl.sink_pad_names == ["I1", "filters"]

    def test_corr_no_adapt(self):
        """Test the corr method"""
        # Create correlate element
        init_filters = numpy.array([[1, 2, 3]])
        crl = AdaptiveCorrelate(
            filters=init_filters,
            sample_rate=1,
            source_pad_names=["O1"],
            sink_pad_names=["I1"],
        )

        # Check intial filters

        # Create SeriesBuffer
        frame = TSFrame(
            buffers=[
                SeriesBuffer(
                    offset=0,
                    data=numpy.array([1, 2, 3, 4, 5, 6, 7, 8, 9]),
                    sample_rate=1,
                    shape=(9,),
                ),
            ]
        )

        # Create EventFrame for new filters
        event = Event.from_time(time=2)
        buf = EventBuffer.from_span(
            start=2,
            end=int(TIME_MAX),
            data=[event],
        )
        f_nonew_filt = EventFrame(data=[buf])

        # Pull onto sink pads (no new filters)
        crl.pull(pad=crl.snks["I1"], frame=frame)
        crl.pull(pad=crl.snks["filters"], frame=f_nonew_filt)

        # Call internal
        crl.internal()

        # Call new
        res = crl.new(pad=crl.srcs["O1"])

        assert res is not None
        assert res[0].data.shape == (1, 7)
        numpy.testing.assert_almost_equal(
            res[0].data[0],
            numpy.array(
                [14, 20, 26, 32, 38, 44, 50],
            ),
        )

    def test_corr_adapt(self):
        """Test the corr method"""
        # Create correlate element
        init_filters = numpy.array([[1, 2, 3]])
        crl = AdaptiveCorrelate(
            filters=init_filters,
            sample_rate=1,
            source_pad_names=["O1"],
            sink_pad_names=["I1"],
            verbose=True,
        )

        # Check intial filters

        # Create SeriesBuffer
        frame = TSFrame(
            buffers=[
                SeriesBuffer(
                    offset=0,
                    data=numpy.array([1, 2, 3, 4, 5, 6, 7, 8, 9]),
                    sample_rate=1,
                    shape=(9,),
                ),
            ]
        )

        # Create EventFrame for new filters
        event = Event.from_time(time=2, data=numpy.array([[4, 5, 6]]))
        buf = EventBuffer.from_span(
            start=2,
            end=int(TIME_MAX),
            data=[event],
        )
        f_new_filt = EventFrame(data=[buf])

        # Pull onto sink pads (no new filters)
        crl.pull(pad=crl.snks["I1"], frame=frame)
        crl.pull(pad=crl.snks["filters"], frame=f_new_filt)

        # Call internal
        crl.internal()

        # Call new
        res = crl.new(pad=crl.srcs["O1"])

        assert res is not None
        assert res[0].data.shape == (1, 7)
        numpy.testing.assert_almost_equal(
            res[0].data[0],
            numpy.array(
                [
                    14.2256488,
                    22.945275,
                    36.1900927,
                    54.5,
                    76.714861,
                    100.1276917,
                    121.0974048,
                ]
            ),
        )

    def test_pipeline_simple(self):
        """Test the AdaptiveCorrelate element in a white noise pipeline,
        with periodic filter updates
        """

        pipeline = Pipeline()
        t0 = 0.0
        duration = 3  # seconds
        # Run pipeline
        data_source = FakeSeriesSource(
            name="NoiseSrc",
            source_pad_names=("C1",),
            rate=1,
            t0=t0,
            end=20 * duration,
            real_time=False,
        )

        def make_filters_frame(EOS: bool, data: tuple | None, **kwargs):
            # Handle case of no data left
            if data is None:
                t0, arr = 0, None
                filt = None
            else:
                t0, arr = data
                filt = None if arr is None else numpy.array([arr])
            event = Event.from_time(time=t0, data=filt)
            return EventFrame(
                data=[
                    EventBuffer.from_span(
                        start=t0,
                        end=int(TIME_MAX),
                        data=[event],
                    )
                ],
                EOS=EOS,
            )

        filter_source = IterSource(
            name="FilterSrc",
            source_pad_names=["filters"],
            iters={
                "filters": [
                    (7, None),
                    (8, None),
                    (9, None),
                    (7, None),
                    (8, None),
                    (9, None),
                    (1, [1, 2, 3]),
                    (7, None),
                    (8, None),
                    (9, None),
                    (7, None),
                    (8, None),
                    (9, None),
                    (10, [7, 8, 9]),
                    (12, None),
                    (13, None),
                    (14, None),
                ]
            },
            frame_factory=make_filters_frame,
        )

        init_filters = numpy.array([[1, 2, 3]])
        afilter = AdaptiveCorrelate(
            filters=init_filters,
            sample_rate=1,
            source_pad_names=["C1"],
            sink_pad_names=["C1"],
            filter_sink_name="filters",
            adapter_config=AdapterConfig(
                stride=Offset.fromsamples(6, sample_rate=1),
            ),
        )

        csink = IsGapCollectSink(
            name="CollectSink",
            sink_pad_names=["C1"],
            extract_data=False,
        )

        pipeline.insert(
            data_source,
            filter_source,
            afilter,
            csink,
            link_map={
                afilter.snks["C1"]: data_source.srcs["C1"],
                afilter.snks["filters"]: filter_source.srcs["filters"],
                csink.snks["C1"]: afilter.srcs["C1"],
            },
        )
        pipeline.run()

        assert len(csink.collects["C1"]) > 0

    def test_pipeline_simple_err_too_many_updates(self):
        """Test the AdaptiveCorrelate element in a white noise pipeline,
        with periodic filter updates

        Error case: uploading multiple new filters in a single stride
        """

        pipeline = Pipeline()
        t0 = 0.0
        duration = 3  # seconds
        # Run pipeline
        data_source = FakeSeriesSource(
            name="NoiseSrc",
            source_pad_names=("C1",),
            rate=1,
            t0=t0,
            end=20 * duration,
            real_time=False,
        )

        def make_filters_frame(EOS: bool, data: tuple | None, **kwargs):
            # Handle case of no data left
            if data is None:
                t0, arr = 0, None
                filt = None
            else:
                t0, arr = data
                filt = numpy.array([arr])
            event = Event.from_time(time=t0, data=filt)
            return EventFrame(
                data=[
                    EventBuffer.from_span(
                        start=t0,
                        end=int(TIME_MAX),
                        data=[event],
                    )
                ],
                EOS=EOS,
            )

        filter_source = IterSource(
            name="FilterSrc",
            source_pad_names=["filters"],
            iters={
                "filters": [
                    (1, [1, 2, 3]),
                    (3, [7, 8, 9]),
                    (6, [4, 5, 6]),
                ]
            },
            frame_factory=make_filters_frame,
        )

        init_filters = numpy.array([[1, 2, 3]])
        afilter = AdaptiveCorrelate(
            filters=init_filters,
            sample_rate=1,
            source_pad_names=["C1"],
            sink_pad_names=["C1"],
            filter_sink_name="filters",
            adapter_config=AdapterConfig(
                stride=Offset.fromsamples(6, sample_rate=1),
            ),
        )

        csink = CollectSink(
            name="CollectSink",
            sink_pad_names=["C1"],
            extract_data=False,
        )

        pipeline.insert(
            data_source,
            filter_source,
            afilter,
            csink,
            link_map={
                afilter.snks["C1"]: data_source.srcs["C1"],
                afilter.snks["filters"]: filter_source.srcs["filters"],
                csink.snks["C1"]: afilter.srcs["C1"],
            },
        )

        with pytest.raises(ValueError):
            pipeline.run()

    def test_pipeline_simple_err_mismatched_shapes(self):
        """Test the AdaptiveCorrelate element in a white noise pipeline,
        with periodic filter updates

        Error case: uploading multiple new filters in a single stride
        """
        pipeline = Pipeline()

        # Run pipeline
        data_source = FakeSeriesSource(
            name="test",
            rate=1,
            signal_type="sin",
            fsin=1,
            t0=0,
            duration=10,
            source_pad_names=["C1"],
        )

        def make_filters_frame(EOS: bool, data: tuple | None, **kwargs):
            # Handle case of no data left
            if data is None:
                t0, arr = 0, None
                filt = None
            else:
                t0, arr = data
                filt = numpy.array([arr])
            event = Event.from_time(time=t0, data=filt)
            return EventFrame(
                data=[
                    EventBuffer.from_span(
                        start=t0,
                        end=int(TIME_MAX),
                        data=[event],
                    )
                ],
                EOS=EOS,
            )

        filter_source = IterSource(
            name="FilterSrc",
            source_pad_names=["filters"],
            iters={
                "filters": [
                    (0, None),
                    (0, None),
                    (0, None),
                    (0, [1, 2, 3]),
                    (0, None),
                    (0, None),
                    (0, None),
                    (5 * Time.SECONDS, [7, 8, 9, 10]),
                ]
            },
            frame_factory=make_filters_frame,
        )

        init_filters = numpy.array([[1, 2, 3]])
        afilter = AdaptiveCorrelate(
            filters=init_filters,
            sample_rate=1,
            source_pad_names=["C1"],
            sink_pad_names=["C1"],
            filter_sink_name="filters",
            adapter_config=AdapterConfig(
                stride=Offset.fromsamples(3, sample_rate=1),
            ),
        )

        csink = CollectSink(
            name="CollectSink",
            sink_pad_names=["C1"],
            extract_data=False,
        )

        pipeline.insert(
            data_source,
            filter_source,
            afilter,
            csink,
            link_map={
                afilter.snks["C1"]: data_source.srcs["C1"],
                afilter.snks["filters"]: filter_source.srcs["filters"],
                csink.snks["C1"]: afilter.srcs["C1"],
            },
        )

        with pytest.raises(ValueError):
            pipeline.run()

    def test_pipeline_simple_err_multiple_events(self):
        """Test the AdaptiveCorrelate element in a white noise pipeline,
        with periodic filter updates

        Error case: uploading new filters as multiple events
        """
        pipeline = Pipeline()

        # Run pipeline
        data_source = FakeSeriesSource(
            name="test",
            rate=1,
            signal_type="sin",
            fsin=1,
            t0=0,
            duration=10,
            source_pad_names=["C1"],
        )

        def make_filters_frame(EOS: bool, data: tuple | None, **kwargs):
            # Handle case of no data left
            if data is None:
                t0, arr = 0, None
                filt = None
            else:
                t0, arr = data
                filt = numpy.array([arr])
            event = Event.from_time(time=t0, data=filt)
            return EventFrame(
                data=[
                    EventBuffer.from_span(
                        start=t0,
                        end=int(TIME_MAX),
                        data=[event, event],
                    )
                ],
                EOS=EOS,
            )

        filter_source = IterSource(
            name="FilterSrc",
            source_pad_names=["filters"],
            iters={
                "filters": [
                    (0, None),
                    (0, None),
                    (0, None),
                    (0, [1, 2, 3]),
                    (0, None),
                    (0, None),
                    (0, None),
                    (5 * Time.SECONDS, [4, 5, 6]),
                ]
            },
            frame_factory=make_filters_frame,
        )

        init_filters = numpy.array([[1, 2, 3]])
        afilter = AdaptiveCorrelate(
            filters=init_filters,
            sample_rate=1,
            source_pad_names=["C1"],
            sink_pad_names=["C1"],
            filter_sink_name="filters",
            adapter_config=AdapterConfig(
                stride=Offset.fromsamples(3, sample_rate=1),
            ),
        )

        csink = CollectSink(
            name="CollectSink",
            sink_pad_names=["C1"],
            extract_data=False,
        )

        pipeline.insert(
            data_source,
            filter_source,
            afilter,
            csink,
            link_map={
                afilter.snks["C1"]: data_source.srcs["C1"],
                afilter.snks["filters"]: filter_source.srcs["filters"],
                csink.snks["C1"]: afilter.srcs["C1"],
            },
        )

        with pytest.raises(ValueError):
            pipeline.run()

    def test_pipeline_sine(self, tmp_path):
        """Test the AdaptiveCorrelate element in a sine pipeline at frequency
        f_src, with a two different low pass filters (f1, f2) applied
        in sequence (in time) using an adaptive filter, such that f1> f_src > f2.
        Expected result is that the output of the adaptive filter will be
        a sine wave at f_src, with the amplitude of the sine wave
        decreasing as the filter adapts to f2.
        """
        out_file = str(tmp_path / "out.txt")

        # Parameters of test
        t0 = 0.0
        duration = 3
        f_sample = 1024
        f_source = 32
        f_cutoff1 = 64
        f_cutoff2 = 16
        n_zeros = 5

        # Create pipeline
        pipeline = Pipeline()

        # Create data source
        data_source = FakeSeriesSource(
            name="SineSrc",
            source_pad_names=("C1",),
            rate=f_sample,
            t0=t0,
            end=10 * duration,
            real_time=False,
            signal_type="sine",
            fsin=f_source,
        )

        def make_filters_frame(EOS: bool, data: tuple | None, **kwargs):
            # Handle case of no data left
            if data is None:
                t0, params = 0, {"f_cutoff": f_cutoff1}
                filt = None
            else:
                t0, params = data
                if params is None:
                    filt = None
                else:
                    # Make filter
                    filt = filtertools.low_pass_filter(
                        f_cutoff=params["f_cutoff"],
                        f_sample=f_sample,
                        n_zeros=n_zeros,
                        normalize=True,
                        win_func=scipy.signal.windows.blackman,
                        fix_size=321,
                    )

            event = Event.from_time(time=t0, data=filt)
            return EventFrame(
                data=[
                    EventBuffer.from_span(
                        start=t0,
                        end=int(TIME_MAX),
                        data=[event],
                    )
                ],
                EOS=EOS,
            )

        filter_source = IterSource(
            name="FilterSrc",
            source_pad_names=["filters"],
            iters={
                "filters": [
                    (0, {"f_cutoff": f_cutoff1}),
                    (0, None),
                    (0, None),
                    (0, None),
                    (0, {"f_cutoff": f_cutoff2}),
                    (0, None),
                    (0, None),
                    (0, None),
                    (0, None),
                    (0, None),
                    (0, None),
                    (0, None),
                    (0, {"f_cutoff": f_cutoff1}),
                    (0, None),
                    (0, None),
                    (0, None),
                    (0, None),
                ]
            },
            frame_factory=make_filters_frame,
        )

        init_filter = make_filters_frame(False, (0, {"f_cutoff": f_cutoff1}))
        afilter = AdaptiveCorrelate(
            filters=init_filter.events[0].data,
            sample_rate=f_sample,
            source_pad_names=["C1"],
            sink_pad_names=["C1"],
            filter_sink_name="filters",
            adapter_config=AdapterConfig(
                stride=Offset.fromsec(duration),
            ),
        )

        csink = CollectSink(
            name="CollectSink",
            sink_pad_names=["C1"],
            extract_data=False,
        )

        dsink = DumpSeriesSink(
            name="DumpSink",
            sink_pad_names=["C1"],
            fname=out_file,
            verbose=True,
        )

        pipeline.insert(
            data_source,
            filter_source,
            afilter,
            csink,
            dsink,
            link_map={
                afilter.snks["C1"]: data_source.srcs["C1"],
                afilter.snks["filters"]: filter_source.srcs["filters"],
                csink.snks["C1"]: afilter.srcs["C1"],
                dsink.snks["C1"]: afilter.srcs["C1"],
            },
        )
        pipeline.run()

        res = numpy.loadtxt(out_file)
        _, data = res[:, 0], res[:, 1]

        # Assert that max value in beginning is near 1
        numpy.testing.assert_almost_equal(numpy.max(data[:100]), 1, decimal=3)

        # Assert that max value in middle is near 0
        numpy.testing.assert_almost_equal(
            numpy.max(data[len(data) // 2 - 50 : len(data) // 2 + 50]), 0, decimal=3
        )

        # Assert that max value in end is near 1
        numpy.testing.assert_almost_equal(numpy.max(data[-100:]), 1, decimal=3)

        # Uncomment below for making plot
        # df = pandas.DataFrame(res, columns=["time", "data"])
        # fig = express.line(df, x="time", y="data")
        # fig.show()

    def test_pipeline_sine_no_overlap(self, tmp_path):
        """Test the similar case as test_pipeline_sine, but with no overlap
        between the new filter and the data (so there should be no change in behavior)
        """
        out_file = str(tmp_path / "out.txt")

        # Parameters of test
        t0 = 0.0
        duration = 3
        f_sample = 1024
        f_source = 32
        f_cutoff1 = 64
        n_zeros = 5

        # Create pipeline
        pipeline = Pipeline()

        # Create data source
        data_source = FakeSeriesSource(
            name="SineSrc",
            source_pad_names=("C1",),
            rate=f_sample,
            t0=t0,
            end=10 * duration,
            real_time=False,
            signal_type="sine",
            fsin=f_source,
        )

        def make_filters_frame(EOS: bool, data: tuple | None, **kwargs):
            # Handle case of no data left
            if data is None:
                t0, params = 0, {"f_cutoff": f_cutoff1}
                filt = None
            else:
                t0, params = data
                if params is None:
                    filt = None
                else:
                    # Make filter
                    filt = filtertools.low_pass_filter(
                        f_cutoff=params["f_cutoff"],
                        f_sample=f_sample,
                        n_zeros=n_zeros,
                        normalize=True,
                        win_func=scipy.signal.windows.blackman,
                        fix_size=321,
                    )

            t_start = int(TIME_MAX) - 100
            event = Event.from_time(time=t_start, data=filt)
            return EventFrame(
                data=[
                    EventBuffer.from_span(
                        start=t_start,
                        end=int(TIME_MAX),
                        data=[event],
                    )
                ],
                EOS=EOS,
            )

        filter_source = IterSource(
            name="FilterSrc",
            source_pad_names=["filters"],
            iters={
                "filters": [
                    (0, {"f_cutoff": f_cutoff1}),
                    (0, None),
                    (0, None),
                    (0, None),
                    (0, None),
                    (0, None),
                    (0, None),
                    (0, None),
                    (0, None),
                    (0, None),
                    (0, None),
                    (0, None),
                    (0, None),
                    (0, None),
                    (0, None),
                ]
            },
            frame_factory=make_filters_frame,
        )

        init_filter = make_filters_frame(False, (0, {"f_cutoff": f_cutoff1}))
        afilter = AdaptiveCorrelate(
            filters=init_filter.events[0].data,
            sample_rate=f_sample,
            source_pad_names=["C1"],
            sink_pad_names=["C1"],
            filter_sink_name="filters",
            adapter_config=AdapterConfig(
                stride=Offset.fromsec(duration),
            ),
        )

        csink = CollectSink(
            name="CollectSink",
            sink_pad_names=["C1"],
            extract_data=False,
        )

        dsink = DumpSeriesSink(
            name="DumpSink",
            sink_pad_names=["C1"],
            fname=out_file,
            verbose=True,
        )

        pipeline.insert(
            data_source,
            filter_source,
            afilter,
            csink,
            dsink,
            link_map={
                afilter.snks["C1"]: data_source.srcs["C1"],
                afilter.snks["filters"]: filter_source.srcs["filters"],
                csink.snks["C1"]: afilter.srcs["C1"],
                dsink.snks["C1"]: afilter.srcs["C1"],
            },
        )
        pipeline.run()

        res = numpy.loadtxt(out_file)
        _, data = res[:, 0], res[:, 1]

        # Assert that max value in beginning is near 1
        numpy.testing.assert_almost_equal(numpy.max(data[:100]), 1, decimal=3)

        # Assert that max value in middle is near 0
        numpy.testing.assert_almost_equal(
            numpy.max(data[len(data) // 2 - 50 : len(data) // 2 + 50]), 1, decimal=3
        )

        # Assert that max value in end is near 1
        numpy.testing.assert_almost_equal(numpy.max(data[-100:]), 1, decimal=3)

        # Uncomment below for making plot
        # df = pandas.DataFrame(res, columns=["time", "data"])
        # fig = express.line(df, x="time", y="data", title="No Overlap")
        # fig.show()

    def test_can_adapt_when_not_adapting(self):
        """Test can_adapt() returns False when is_adapting is False"""
        # Create an AdaptiveCorrelate that's not adapting
        init_filters = numpy.ones((1, 256))
        correlator = AdaptiveCorrelate(
            filters=init_filters,
            sample_rate=256,
            sink_pad_names=["input"],
            source_pad_names=["output"],
        )

        # Ensure filter_deque has only one element (is_adapting will be False)
        assert len(correlator.filter_deque) == 1
        assert not correlator.is_adapting

        # Create a dummy frame
        frame = TSFrame(
            buffers=[
                SeriesBuffer(
                    offset=0,
                    sample_rate=256,
                    data=numpy.zeros(256),
                    shape=(256,),
                )
            ]
        )

        # can_adapt should return False when not adapting
        assert not correlator.can_adapt(frame)

    def test_adapt_null_ic_startup(self):
        """Test AdaptiveCorrelate startup behavior with filters=None.
        It should produce gaps until sufficient filters are provided to the sink pad.
        """
        # Initialize with None
        crl = AdaptiveCorrelate(
            filters=None,
            sample_rate=1,
            source_pad_names=["O1"],
            sink_pad_names=["I1"],
            filter_sink_name="filters",
            adapter_config=AdapterConfig(
                stride=Offset.fromsamples(10, sample_rate=1),
            ),
        )

        # 1. Feed Data, No Filters -> Expect Gap output
        # 10 samples at 1Hz = 10 * 16384 offsets (assuming MAX_RATE=16384)
        data_frame = TSFrame(
            buffers=[
                SeriesBuffer(
                    offset=0,
                    data=numpy.ones(10),
                    sample_rate=1,
                    shape=(10,),
                ),
            ]
        )
        crl.pull(pad=crl.snks["I1"], frame=data_frame)
        crl.internal()
        res = crl.new(pad=crl.srcs["O1"])
        assert res[0].is_gap

        # 2. Feed First Filter -> Primes the deque
        filt_event = Event.from_time(time=0, data=numpy.array([[1, 1, 1]]))
        filt_frame = EventFrame(
            data=[EventBuffer.from_span(start=0, end=int(TIME_MAX), data=[filt_event])]
        )
        crl.pull(pad=crl.snks["filters"], frame=filt_frame)

        # 3. Feed Second Filter -> Triggers adaptation
        filt_event_2 = Event.from_time(time=0, data=numpy.array([[2, 2, 2]]))
        filt_frame_2 = EventFrame(
            data=[
                EventBuffer.from_span(start=0, end=int(TIME_MAX), data=[filt_event_2])
            ]
        )
        crl.pull(pad=crl.snks["filters"], frame=filt_frame_2)

        # 4. Feed NEXT chunk of data to trigger processing
        # We must respect continuity. Previous frame ended at offset
        # corresponding to 10 samples.
        next_offset = Offset.fromsamples(10, sample_rate=1)
        data_frame_2 = TSFrame(
            buffers=[
                SeriesBuffer(
                    offset=next_offset,
                    data=numpy.ones(10),
                    sample_rate=1,
                    shape=(10,),
                ),
            ]
        )

        crl.pull(pad=crl.snks["I1"], frame=data_frame_2)
        crl.internal()
        res = crl.new(pad=crl.srcs["O1"])

        # Expectation: Now that we are adapting, we should have valid data
        assert not res[0].is_gap
        assert res[0].data is not None
        # Verify simple correlation result occurred (all 1s data vs filters)
        assert numpy.any(res[0].data != 0)

    def test_null_ic_filters_cur(self):
        """Test that filters=None is handled correctly in corr()
        when called from new()"""
        crl = AdaptiveCorrelate(
            filters=None,
            sample_rate=1,
            source_pad_names=["O1"],
            sink_pad_names=["I1"],
            filter_sink_name="filters",
        )

        # Call new without ever providing filters
        assert crl.filters_cur is None
        assert not crl.can_adapt(frame=TSFrame())

    def test_pull_gap_frame(self):
        """Test that pulling a gap frame on the filter sink pad is handled correctly"""
        crl = AdaptiveCorrelate(
            filters=None,
            sample_rate=1,
            source_pad_names=["O1"],
            sink_pad_names=["I1"],
            filter_sink_name="filters",
        )

        # Pull a gap frame on the filter sink pad
        gap_frame = TSFrame(
            buffers=[
                SeriesBuffer(
                    offset=0,
                    data=None,
                    sample_rate=1,
                    shape=(0,),
                )
            ]
        )
        crl.pull(pad=crl.snks["filters"], frame=gap_frame)

        # Expectation: filters_cur should be set to None, and we
        # should not be able to adapt
        assert crl.filters_cur is None
        assert not crl.can_adapt(frame=TSFrame())

    def test_ignore_rapid_updates_verbose(self):
        """Test that rapid filter updates are ignored and logged when verbose=True"""
        crl = AdaptiveCorrelate(
            filters=numpy.array([[1, 2, 3]]),
            sample_rate=1,
            source_pad_names=["O1"],
            sink_pad_names=["I1"],
            filter_sink_name="filters",
            adapter_config=AdapterConfig(
                stride=Offset.fromsamples(10, sample_rate=1),
            ),
            verbose=True,
            ignore_rapid_updates=True,
        )

        # Create two filter frames that arrive within the stride period
        filt_event_1 = Event.from_time(time=0, data=numpy.array([[1, 2, 3]]))
        filt_frame_1 = EventFrame(
            data=[
                EventBuffer.from_span(start=0, end=int(TIME_MAX), data=[filt_event_1])
            ]
        )

        filt_event_2 = Event.from_time(time=5, data=numpy.array([[4, 5, 6]]))
        filt_frame_2 = EventFrame(
            data=[
                EventBuffer.from_span(start=5, end=int(TIME_MAX), data=[filt_event_2])
            ]
        )

        # Pull first filter frame (should be accepted)
        crl.pull(pad=crl.snks["filters"], frame=filt_frame_1)
        assert crl.filters_cur is not None

        # Pull second filter frame (should be ignored due to rapid update)
        with pytest.warns(RuntimeWarning, match="Ignoring rapid filter update at"):
            crl.pull(pad=crl.snks["filters"], frame=filt_frame_2)
