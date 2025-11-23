"""Tests for nary ops"""

import numpy
import pytest
from sgn import Pipeline
from test_correlate import IsGapCollectSink

from sgnts.base import AdapterConfig, Offset, SeriesBuffer, TSFrame
from sgnts.sources import FakeSeriesSource
from sgnts.transforms import nary
from sgnts.transforms.nary import Multiply, NaryTransform, Real


class TestNaryTransform:
    """Test group for binary operations"""

    def test_init(self):
        """Test creating a binop"""
        b = NaryTransform(
            sink_pad_names=["I1", "I2"],
            source_pad_names=["O1"],
            op=lambda a, b: a + b,
        )
        assert isinstance(b, NaryTransform)

    def test_validate_op_finite(self):
        """Test validating the operator"""
        b = NaryTransform(
            sink_pad_names=["I1", "I2"],
            source_pad_names=["O1"],
            op=lambda a, b: a + b,
        )
        b._validate_op()

    def test_validate_op_var_args(self):
        """Test validating the operator"""
        b = NaryTransform(
            sink_pad_names=["I1", "I2"],
            source_pad_names=["O1"],
            op=lambda *args: sum(args),
        )
        b._validate_op()

    def test_validate_op_err_finite_mismatch(self):
        """Test validating the operator"""
        with pytest.raises(AssertionError):
            NaryTransform(
                sink_pad_names=["I1", "I2"],
                source_pad_names=["O1"],
                op=lambda a: a,
            )

    def test_binop_no_pipeline(self):
        """Test binop example"""
        b = NaryTransform(
            sink_pad_names=["I1", "I2"],
            source_pad_names=["O1"],
            op=lambda a, b: a + b,
        )

        frame1 = TSFrame(
            buffers=[
                SeriesBuffer(data=numpy.array([1, 2, 3]), offset=0, sample_rate=1),
            ]
        )

        frame2 = TSFrame(
            buffers=[
                SeriesBuffer(data=numpy.array([-1, 0, 3]), offset=0, sample_rate=1),
            ]
        )

        # Pull frames onto pads
        b.pull(b.sink_pads[0], frame1)
        b.pull(b.sink_pads[1], frame2)

        # Call internal method
        b.internal()

        # Call new method
        out = b.new(b.source_pads[0])

        # Check output
        numpy.testing.assert_almost_equal(out.buffers[0].data, numpy.array([0, 2, 6]))

    def test_binop_no_pipeline_gap(self):
        """Test binop example"""
        b = NaryTransform(
            sink_pad_names=["I1", "I2"],
            source_pad_names=["O1"],
            op=lambda a, b: a + b,
        )

        frame1 = TSFrame(
            buffers=[
                SeriesBuffer(data=None, offset=0, sample_rate=1, shape=(3,)),
            ]
        )

        frame2 = TSFrame(
            buffers=[
                SeriesBuffer(data=None, offset=0, sample_rate=1, shape=(3,)),
            ]
        )

        # Pull frames onto pads
        b.pull(b.sink_pads[0], frame1)
        b.pull(b.sink_pads[1], frame2)

        # Call internal method
        b.internal()

        # Call new method
        out = b.new(b.source_pads[0])

        # Check output
        assert out.data is None

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_pipeline_simple(self, n: int):
        """Test the AdaptiveCorrelate element in a white noise pipeline,
        with periodic filter updates
        """
        pipeline = Pipeline()
        # Run pipeline
        src = FakeSeriesSource(
            name="NoiseSrc",
            source_pad_names=["O1"],
            rate=4,
            t0=0,
            end=20,
            real_time=False,
        )

        nt = NaryTransform(
            sink_pad_names=[f"I{k}" for k in range(n)],
            source_pad_names=["O1"],
            adapter_config=AdapterConfig(
                stride=Offset.fromsamples(4, sample_rate=4),
            ),
            op=lambda *args: sum(args),
        )

        snk = IsGapCollectSink(
            name="snk1",
            sink_pad_names=[f"I{k}" for k in range(n)] + ["IRes"],
            extract_data=False,
        )

        links = {
            snk.snks["IRes"]: nt.srcs["O1"],
        }
        for k in range(n):
            links[nt.snks[f"I{k}"]] = src.srcs["O1"]
            links[snk.snks[f"I{k}"]] = src.srcs["O1"]

        pipeline.insert(src, nt, snk, link_map=links)
        pipeline.run()

        data_inputs = [
            numpy.array(snk.collects[f"I{k}"][0].buffers[0].data) for k in range(n)
        ]
        data_res = numpy.array(snk.collects["IRes"][0].buffers[0].data)
        expected = sum(data_inputs)
        numpy.testing.assert_almost_equal(data_res, expected)


class TestMultiply:
    """Test multiple transform"""

    def test_init(self):
        """Test creating a binop"""
        b = Multiply(
            sink_pad_names=["I1", "I2"],
            source_pad_names=["O1"],
        )
        assert isinstance(b, Multiply)

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test__multiply(self, n: int):
        """Test multiply"""
        data = [numpy.array([1, 2, 3]) for _ in range(n)]
        expected = numpy.array([1, 2, 3]) ** n
        res = nary._multiply(*data)
        numpy.testing.assert_almost_equal(res, expected)

    def test_pipeline(self):
        """Test multiply transform in a pipeline"""
        pipeline = Pipeline()
        # Run pipeline
        src = FakeSeriesSource(
            name="NoiseSrc",
            source_pad_names=["O1"],
            rate=4,
            t0=0,
            duration=1,
            real_time=False,
        )

        nt = Multiply(
            sink_pad_names=["I1", "I2"],
            source_pad_names=["O1"],
            adapter_config=AdapterConfig(
                stride=Offset.fromsamples(4, sample_rate=4),
            ),
        )

        snk = IsGapCollectSink(
            name="snk1",
            sink_pad_names=["I1", "I2", "IRes"],
            extract_data=False,
        )

        links = {
            # Set the links
            nt.snks["I1"]: src.srcs["O1"],
            nt.snks["I2"]: src.srcs["O1"],
            # Set the links for the snk
            snk.snks["IRes"]: nt.srcs["O1"],
            snk.snks["I1"]: src.srcs["O1"],
            snk.snks["I2"]: src.srcs["O1"],
        }

        pipeline.insert(src, nt, snk, link_map=links)
        pipeline.run()

        data1 = numpy.array(snk.collects["I1"][0].buffers[0].data)
        data2 = numpy.array(snk.collects["I2"][0].buffers[0].data)
        data_res = numpy.array(snk.collects["IRes"][0].buffers[0].data)
        expected = data1 * data2
        numpy.testing.assert_almost_equal(data_res, expected)


class TestReal:
    """Test real transform"""

    def test_init(self):
        """Test creating a binop"""
        r = Real(
            sink_pad_names=["R1", "R2"],
            source_pad_names=["O1"],
        )
        assert isinstance(r, Real)

    def test_real(self):
        """Test real"""
        data = [numpy.array([1 + 1j, 2 + 2j, 3 + 3j])]
        expected = numpy.array([1, 2, 3])
        res = nary._real(*data)
        numpy.testing.assert_almost_equal(res, expected)

    def test_pipeline_real(self):
        """Test real transform in a pipeline"""
        pipeline = Pipeline()
        # Run pipeline
        src = FakeSeriesSource(
            name="ConstSrc",
            source_pad_names=["O1"],
            signal_type="const",
            const=1 + 1j,
            rate=4,
            t0=0,
            duration=1,
            real_time=False,
        )

        r = Real(
            name="real",
            sink_pad_names=["I1"],
            source_pad_names=["O1"],
            adapter_config=AdapterConfig(
                stride=Offset.fromsamples(4, sample_rate=4),
            ),
        )

        snk = IsGapCollectSink(
            name="snk1",
            sink_pad_names=["I1", "IRes"],
            extract_data=False,
        )

        links = {
            # Set the links
            r.snks["I1"]: src.srcs["O1"],
            # Set the links for the snk
            snk.snks["IRes"]: r.srcs["O1"],
            snk.snks["I1"]: src.srcs["O1"],
        }

        pipeline.insert(src, r, snk, link_map=links)
        pipeline.run()

        data_res = numpy.array(snk.collects["IRes"][0].buffers[0].data)
        expected = numpy.ones(4)
        numpy.testing.assert_almost_equal(data_res, expected)
