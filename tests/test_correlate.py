"""Unit test for correlate transforms"""

import numpy
import pytest
import scipy.signal.windows

from sgn import CollectSink, IterSource, Pipeline
from sgn import Frame
from sgnts import filtertools
from sgnts.base import (
    AdapterConfig,
    EventBuffer,
    EventFrame,
    Offset,
    SeriesBuffer,
    TSFrame,
)
from sgnts.base.slice_tools import TIME_MAX
from sgnts.sinks import DumpSeriesSink
from sgnts.sources import FakeSeriesSource
from sgnts.transforms import Correlate
from sgnts.transforms.correlate import AdaptiveCorrelate


class IsGapCollectSink(CollectSink):
    """Stupid subclass to fix out-of-repo bug in CollectSink.
    Decisions are still being made on how to handle frame aliases
    for the .data attribute in the sgnts library.
    """

    def internal(self) -> None:
        """Internal action is to append all most recent frames to the associated
        collections, then empty the inputs dict.

        Args:
            pad:

        Returns:
        """
        self.inputs: dict[str, Frame]

        for pad_name, frame in self.inputs.items():

            if not frame.is_gap:
                self.collects[pad_name].append(
                    frame.data if self.extract_data else frame
                )

        self.inputs = {}


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
        crl = Correlate(
            filters=numpy.array([[1, 2, 3]]),
            sample_rate=1,
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
        numpy.testing.assert_almost_equal(
            res[0].data, numpy.array([[14, 20, 26, 32, 38]])
        )


class TestAdaptiveCorrelate:
    """Unit tests for Correlate transform element"""

    def test_init(self):
        """Create a Correlate transform element"""
        init_filters = EventBuffer(
            ts=0, te=int(TIME_MAX), data=numpy.array([[1, 2, 3]])
        )
        crl = AdaptiveCorrelate(
            init_filters=init_filters,
            sample_rate=4096,
            source_pad_names=["O1"],
            sink_pad_names=["I1"],
        )
        assert isinstance(crl, AdaptiveCorrelate)
        assert crl.sink_pad_names == ["I1", "filters"]

    def test_init_unaligned(self):
        """Test creating without specifying filter_sink_name in the unaligned pads"""
        init_filters = EventBuffer(
            ts=0, te=int(TIME_MAX), data=numpy.array([[1, 2, 3]])
        )
        crl = AdaptiveCorrelate(
            init_filters=init_filters,
            sample_rate=4096,
            source_pad_names=["O1"],
            sink_pad_names=["I1", "OtherUnaligned"],
            unaligned=["OtherUnaligned"],
            filter_sink_name="MissingUnaligned",
        )
        assert isinstance(crl, AdaptiveCorrelate)
        assert "MissingUnaligned" in crl.unaligned

    def test_corr_no_adapt(self):
        """Test the corr method"""
        # Create correlate element
        init_filters = EventBuffer(
            ts=0, te=int(TIME_MAX), data=numpy.array([[1, 2, 3]])
        )
        crl = AdaptiveCorrelate(
            init_filters=init_filters,
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
        f_nonew_filt = EventFrame(
            events={
                "events": [
                    EventBuffer(
                        ts=2,
                        te=int(TIME_MAX),
                        data=None,
                    ),
                ],
            }
        )

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
        init_filters = EventBuffer(
            ts=0, te=int(TIME_MAX), data=numpy.array([[1, 2, 3]])
        )
        crl = AdaptiveCorrelate(
            init_filters=init_filters,
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
        f_new_filt = EventFrame(
            events={
                "events": [
                    EventBuffer(
                        ts=2,
                        te=int(TIME_MAX),
                        data=numpy.array([[4, 5, 6]]),
                    ),
                ],
            }
        )

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

        def make_filters_frame(EOS: bool, data: tuple):
            # Handle case of no data left
            if data is None:
                t0, arr = 0, None
            else:
                t0, arr = data
            return EventFrame(
                events={
                    "events": [
                        EventBuffer(
                            ts=t0,
                            te=int(TIME_MAX),
                            data=None if arr is None else numpy.array([arr]),
                        ),
                    ],
                },
                EOS=EOS,
            )

        filter_source = IterSource(
            name="FilterSrc",
            source_pad_names=["filters"],
            iters={
                "FilterSrc:src:filters": [
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

        afilter = AdaptiveCorrelate(
            init_filters=EventBuffer(
                ts=0, te=int(TIME_MAX), data=numpy.array([[1, 2, 3]])
            ),
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
            collects={
                "CollectSink:snk:C1": [],
            },
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

        assert len(csink.collects["CollectSink:snk:C1"]) > 0

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

        def make_filters_frame(EOS: bool, data: tuple):
            # Handle case of no data left
            if data is None:
                t0, arr = 0, None
            else:
                t0, arr = data
            return EventFrame(
                events={
                    "events": [
                        EventBuffer(
                            ts=t0,
                            te=int(TIME_MAX),
                            data=None if arr is None else numpy.array([arr]),
                        ),
                    ],
                },
                EOS=EOS,
            )

        filter_source = IterSource(
            name="FilterSrc",
            source_pad_names=["filters"],
            iters={
                "FilterSrc:src:filters": [
                    (1, [1, 2, 3]),
                    (3, [7, 8, 9]),
                    (6, [4, 5, 6]),
                ]
            },
            frame_factory=make_filters_frame,
        )

        afilter = AdaptiveCorrelate(
            init_filters=EventBuffer(
                ts=0, te=int(TIME_MAX), data=numpy.array([[1, 2, 3]])
            ),
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
            collects={
                "CollectSink:snk:C1": [],
            },
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

        def make_filters_frame(EOS: bool, data: tuple):
            # Handle case of no data left
            if data is None:
                t0, arr = 0, None
            else:
                t0, arr = data
            return EventFrame(
                events={
                    "events": [
                        EventBuffer(
                            ts=t0,
                            te=int(TIME_MAX),
                            data=None if arr is None else numpy.array([arr]),
                        ),
                    ],
                },
                EOS=EOS,
            )

        filter_source = IterSource(
            name="FilterSrc",
            source_pad_names=["filters"],
            iters={
                "FilterSrc:src:filters": [
                    (0, [1, 2, 3]),
                    (0, None),
                    (0, None),
                    (0, None),
                    (0, None),
                    (0, None),
                    (0, None),
                    (0, [7, 8, 9, 10]),
                ]
            },
            frame_factory=make_filters_frame,
        )

        afilter = AdaptiveCorrelate(
            init_filters=EventBuffer(
                ts=0, te=int(TIME_MAX), data=numpy.array([[1, 2, 3]])
            ),
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
            collects={
                "CollectSink:snk:C1": [],
            },
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

    def test_pipeline_sine(self):
        """Test the AdaptiveCorrelate element in a sine pipeline at frequency
        f_src, with a two different low pass filters (f1, f2) applied
        in sequence (in time) using an adaptive filter, such that f1> f_src > f2.
        Expected result is that the output of the adaptive filter will be
        a sine wave at f_src, with the amplitude of the sine wave
        decreasing as the filter adapts to f2.
        """

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

        def make_filters_frame(EOS: bool, data: tuple):
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

            return EventFrame(
                events={
                    "events": [
                        EventBuffer(
                            ts=t0,
                            te=int(TIME_MAX),
                            data=filt,
                        ),
                    ],
                },
                EOS=EOS,
            )

        filter_source = IterSource(
            name="FilterSrc",
            source_pad_names=["filters"],
            iters={
                "FilterSrc:src:filters": [
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
            init_filters=init_filter.events["events"][0],
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
            collects={
                "CollectSink:snk:C1": [],
            },
            extract_data=False,
        )

        def fname(buf: SeriesBuffer):
            return f"test-{buf.offset}-{buf.duration}-{buf.is_gap}.txt"

        dsink = DumpSeriesSink(
            name="DumpSink",
            sink_pad_names=["C1"],
            fname="out.txt",
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

        res = numpy.loadtxt("out.txt")
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

    def test_pipeline_sine_no_overlap(self):
        """Test the similar case as test_pipeline_sine, but with no overlap
        between the new filter and the data (so there should be no change in behavior)
        """

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

        def make_filters_frame(EOS: bool, data: tuple):
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

            return EventFrame(
                events={
                    "events": [
                        EventBuffer(
                            ts=int(TIME_MAX) - 100,
                            te=int(TIME_MAX),
                            data=filt,
                        ),
                    ],
                },
                EOS=EOS,
            )

        filter_source = IterSource(
            name="FilterSrc",
            source_pad_names=["filters"],
            iters={
                "FilterSrc:src:filters": [
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
            init_filters=init_filter.events["events"][0],
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
            collects={
                "CollectSink:snk:C1": [],
            },
            extract_data=False,
        )

        def fname(buf: SeriesBuffer):
            return f"test-{buf.offset}-{buf.duration}-{buf.is_gap}.txt"

        dsink = DumpSeriesSink(
            name="DumpSink",
            sink_pad_names=["C1"],
            fname="out.txt",
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

        res = numpy.loadtxt("out.txt")
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
