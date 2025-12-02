#!/usr/bin/env python3
import re

import pytest
from sgn.apps import Pipeline

from sgnts.base import AdapterConfig, Offset
from sgnts.sources import FakeSeriesSource
from sgnts.sinks import DumpSeriesSink
from sgnts.sinks import NullSeriesSink
from sgnts.transforms import Resampler


def test_multi_sink(capsys):

    pipeline = Pipeline()

    #
    #       ----------    -------   --------
    #      | src1     |  | src2  | | src3   |
    #       ----------    -------   --------
    #              \         |      /
    #           H1  \     L1 |     / V1
    #               ----------------
    #              | sink           |
    #               ----------------

    inrate = 256

    t0 = 0
    end = 10

    pipeline.insert(
        FakeSeriesSource(
            name="src1",
            source_pad_names=("H1",),
            rate=inrate,
            t0=t0,
            end=end,
        ),
        FakeSeriesSource(
            name="src2",
            source_pad_names=("L1",),
            rate=inrate,
            t0=t0,
            end=end,
            signal_type="impulse",
            impulse_position=1,
        ),
        FakeSeriesSource(
            name="src3",
            source_pad_names=("V1",),
            rate=inrate,
            t0=t0,
            end=end,
            signal_type="impulse",
            impulse_position=-1,
        ),
        NullSeriesSink(
            name="snk3",
            sink_pad_names=(
                "H1",
                "L1",
                "V1",
            ),
            verbose=True,
        ),
        link_map={
            "snk3:snk:H1": "src1:src:H1",
            "snk3:snk:L1": "src2:src:L1",
            "snk3:snk:V1": "src3:src:V1",
        },
    )

    pipeline.run()


def test_invalid_fake_series():
    pipeline = Pipeline()
    src = FakeSeriesSource(
        name="blah",
        source_pad_names=("V1",),
        rate=2048,
        t0=0,
        end=1,
        signal_type="blah",
    )
    sink = NullSeriesSink(
        name="blah2",
        sink_pad_names=("V1",),
    )
    pipeline.insert(src, sink, link_map={sink.snks["V1"]: src.srcs["V1"]})
    with pytest.raises(ValueError):
        pipeline.run()


def test_invalid_dump_series_pads():
    with pytest.raises(AssertionError):
        DumpSeriesSink(
            fname="test.txt",
            sink_pad_names=("H1", "L1"),
        )


@pytest.mark.parametrize(
    "ngap,skip_gaps", [(0, False), (1, False), (3, False), (3, True)]
)
def test_sink_with_alignment(capsys, ngap, skip_gaps):
    """Test sink receiving misaligned data from Resampler and requesting aligned input.

    Pipeline: FakeSeriesSource -> Resampler -> NullSeriesSink

    The Resampler introduces offset misalignment due to filter group delay.
    The NullSeriesSink requests aligned input via AdapterConfig with align_to.
    This tests that the sink's adapter correctly realigns the misaligned input.

    Args:
        capsys: Pytest fixture to capture stdout/stderr
        ngap: Number of buffers between gaps (0=no gaps, 3=gap every 3 buffers)
        skip_gaps: Whether to skip gaps in aligned segments
    """
    pipeline = Pipeline()

    inrate = 16384
    outrate = 2048
    t0 = 0
    duration = 10

    # Create source producing aligned data (with optional gaps)
    source = FakeSeriesSource(
        name="source",
        source_pad_names=("H1",),
        rate=inrate,
        t0=t0,
        end=t0 + duration,
        ngap=ngap,  # 0 for no gaps, 3 for periodic gaps
    )

    # Create resampler that introduces misalignment
    # The FIR filter's group delay causes offset adjustment
    resampler = Resampler(
        name="resampler",
        sink_pad_names=("H1",),
        source_pad_names=("H1",),
        inrate=inrate,
        outrate=outrate,
    )

    # Create sink with alignment configured and verbose output
    sink = NullSeriesSink(
        name="sink",
        sink_pad_names=("H1",),
        adapter_config=AdapterConfig(
            stride=Offset.fromsec(1),  # 1 second stride
            align_to=Offset.fromsec(1),  # Align to 1 second boundaries
            skip_gaps=skip_gaps,  # Test both gap handling modes
        ),
        verbose=True,  # Enable verbose output to verify alignment
    )

    pipeline.insert(
        source,
        resampler,
        sink,
        link_map={
            resampler.snks["H1"]: source.srcs["H1"],
            sink.snks["H1"]: resampler.srcs["H1"],
        },
    )

    # Run pipeline
    pipeline.run()

    # Capture and verify output
    captured = capsys.readouterr()
    output = captured.out

    # Parse output to verify offsets are aligned
    # NullSeriesSink prints "offset=<value>" in verbose mode
    one_second = Offset.fromsec(1)

    offset_pattern = re.compile(r"offset=(\d+)")
    offsets = [int(m.group(1)) for m in offset_pattern.finditer(output)]

    # Verify all offsets are aligned to 1-second boundaries
    for offset in offsets:
        assert offset % one_second == 0, (
            f"Offset {offset} is not aligned to 1-second boundary "
            f"(one_second={one_second}, remainder={offset % one_second})"
        )


if __name__ == "__main__":
    test_multi_sink(None)
