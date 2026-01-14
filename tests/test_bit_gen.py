"""Tests for BitVector element."""

import numpy as np
from pathlib import Path
from sgn.apps import Pipeline

from sgnts.sources.segment import SegmentSource
from sgnts.transforms.bit_gen import BitVector
from sgnts.sinks import DumpSeriesSink


def test_bit_gen_basic(tmp_path):
    """Test basic BitVector functionality with two inputs."""
    p = Pipeline()

    # Source 1: has data from 0.1-0.5s (gap from 0-0.1s and 0.5-1s)
    # Source 2: has data from 0.3-1.0s (gap from 0-0.3s)
    # Expected BitVector output (as integers from binary state):
    # - 0.0-0.1s: 0 (binary 00 = both gaps)
    # - 0.1-0.3s: 2 (binary 10 = only src1 has buffer)
    # - 0.3-0.5s: 3 (binary 11 = both have buffers)
    # - 0.5-1.0s: 1 (binary 01 = only src2 has buffer)
    p.insert(
        SegmentSource(
            name="src1",
            source_pad_names=("src",),
            rate=16,
            t0=0.0,
            duration=1.0,
            segments=((int(0.125 * 1e9), int(0.5 * 1e9)),),
            values=(1,),
        ),
        SegmentSource(
            name="src2",
            source_pad_names=("src",),
            rate=16,
            t0=0.0,
            duration=1.0,
            segments=((int(0.25 * 1e9), int(1.0 * 1e9)),),
            values=(2,),
        ),
        BitVector(
            name="bitvector",
            sink_pad_names=("input1", "input2"),
            source_pad_names=("src",),
        ),
        DumpSeriesSink(
            name="sink", fname=str(tmp_path / "output.txt"), sink_pad_names=("snk",)
        ),
        link_map={
            "bitvector:snk:input1": "src1:src:src",
            "bitvector:snk:input2": "src2:src:src",
            "sink:snk:snk": "bitvector:src:src",
        },
    )

    # Run pipeline
    p.run()

    # Load and verify output
    output_data = np.loadtxt(str(tmp_path / "output.txt"))

    # Output should have 2 columns: time, integer_value
    assert output_data.shape[1] == 2, f"Expected 2 columns, got {output_data.shape[1]}"

    # Verify data types - should be uint8 (stored as float in text file)
    # Check that values are in valid range for 2 inputs (0-3)
    assert np.all(
        np.isin(output_data[:, 1], [0, 1, 2, 3])
    ), "Values should be 0, 1, 2, or 3"

    # Verify state transitions at different time points

    # At t=0.05s (in region 1): should be 0 (binary 00)
    idx = np.argmin(np.abs(output_data[:, 0] - 0.05))
    assert output_data[idx, 1] == 0, "At t=0.05s, value should be 0 (binary 00)"

    # At t=0.4375s (in region 2, both have buffers): should be 3 (binary 11)
    idx = np.argmin(np.abs(output_data[:, 0] - 0.4375))
    assert output_data[idx, 1] == 3, "At t=0.4375s, value should be 3 (binary 11)"

    # At t=0.75s (in region 3, only src2 has buffer): should be 1 (binary 01)
    idx = np.argmin(np.abs(output_data[:, 0] - 0.75))
    assert output_data[idx, 1] == 1, "At t=0.75s, value should be 1 (binary 01)"


def test_bit_gen_three_inputs(tmp_path):
    """Test BitVector with three inputs of different rates."""
    p = Pipeline()

    # Source 1: data from 0.1-0.2s, 0.4-0.7s, 0.8-1.0s
    # Source 2: data from 0.15-0.5s, 0.65-1.0s
    # Source 3: data from 0.1-0.3s, 0.6-0.9s

    p.insert(
        SegmentSource(
            name="src1",
            source_pad_names=("src",),
            rate=256,
            t0=0.0,
            duration=1.0,
            segments=(
                (int(0.1 * 1e9), int(0.2 * 1e9)),
                (int(0.4 * 1e9), int(0.7 * 1e9)),
                (int(0.8 * 1e9), int(1.0 * 1e9)),
            ),
            values=(1, 1, 1),
        ),
        SegmentSource(
            name="src2",
            source_pad_names=("src",),
            rate=128,
            t0=0.0,
            duration=1.0,
            segments=(
                (int(0.15 * 1e9), int(0.5 * 1e9)),
                (int(0.65 * 1e9), int(1.0 * 1e9)),
            ),
            values=(1, 1),
        ),
        SegmentSource(
            name="src3",
            source_pad_names=("src",),
            rate=64,
            t0=0.0,
            duration=1.0,
            segments=(
                (int(0.1 * 1e9), int(0.3 * 1e9)),
                (int(0.6 * 1e9), int(0.9 * 1e9)),
            ),
            values=(1, 1),
        ),
        BitVector(
            name="bitvector",
            sink_pad_names=("input1", "input2", "input3"),
            source_pad_names=("src",),
        ),
        DumpSeriesSink(
            name="sink", fname=str(tmp_path / "output.txt"), sink_pad_names=("snk",)
        ),
        link_map={
            "bitvector:snk:input1": "src1:src:src",
            "bitvector:snk:input2": "src2:src:src",
            "bitvector:snk:input3": "src3:src:src",
            "sink:snk:snk": "bitvector:src:src",
        },
    )

    p.run()

    # Verify output
    output_data = np.loadtxt(str(tmp_path / "output.txt"))

    # Should have 2 columns: time + integer_value
    assert output_data.shape[1] == 2, f"Expected 2 columns, got {output_data.shape[1]}"

    # Verify some state combinations
    # At t=0.17s: 7 (binary 111 = all have data during 0.15-0.2s)
    idx = np.argmin(np.abs(output_data[:, 0] - 0.17))
    assert output_data[idx, 1] == 7, "Value should be 7 (binary 111)"

    # At t=0.67s: 7 (binary 111 = all have data during 0.65-0.7s)
    idx = np.argmin(np.abs(output_data[:, 0] - 0.67))
    assert output_data[idx, 1] == 7, "Value should be 7 (binary 111)"


def test_bit_gen_all_gaps(tmp_path):
    """Test BitVector when all inputs have only gaps."""
    p = Pipeline()

    # Create sources with only gaps (no segments)
    # Unlike ANDTransform, BitVector should output buffers (not gaps)
    # with all zeros
    p.insert(
        SegmentSource(
            name="src1",
            source_pad_names=("src",),
            rate=128,
            t0=0.0,
            duration=1.0,
            segments=(),  # Empty segments = all gaps
            values=(),
        ),
        SegmentSource(
            name="src2",
            source_pad_names=("src",),
            rate=64,
            t0=0.0,
            duration=1.0,
            segments=(),  # Empty segments = all gaps
            values=(),
        ),
        BitVector(
            name="bitvector",
            sink_pad_names=("input1", "input2"),
            source_pad_names=("src",),
        ),
        DumpSeriesSink(
            name="sink", fname=str(tmp_path / "output.txt"), sink_pad_names=("snk",)
        ),
        link_map={
            "bitvector:snk:input1": "src1:src:src",
            "bitvector:snk:input2": "src2:src:src",
            "sink:snk:snk": "bitvector:src:src",
        },
    )

    p.run()

    # BitVector outputs buffers (not gaps), so file should have data
    output_data = np.loadtxt(str(tmp_path / "output.txt"))

    # Verify output is all zeros (binary 00 = 0)
    assert np.all(output_data[:, 1] == 0), "Value should be all zeros (binary 00)"


def test_bit_gen_all_buffers(tmp_path):
    """Test BitVector when all inputs have continuous data."""
    p = Pipeline()

    # Both sources have continuous data from 0-1s
    # Output should be all 3s (binary 11)
    p.insert(
        SegmentSource(
            name="src1",
            source_pad_names=("src",),
            rate=128,
            t0=0.0,
            duration=1.0,
            segments=((0, int(1e9)),),
            values=(1,),
        ),
        SegmentSource(
            name="src2",
            source_pad_names=("src",),
            rate=64,
            t0=0.0,
            duration=1.0,
            segments=((0, int(1e9)),),
            values=(1,),
        ),
        BitVector(
            name="bitvector",
            sink_pad_names=("input1", "input2"),
            source_pad_names=("src",),
        ),
        DumpSeriesSink(
            name="sink", fname=str(tmp_path / "output.txt"), sink_pad_names=("snk",)
        ),
        link_map={
            "bitvector:snk:input1": "src1:src:src",
            "bitvector:snk:input2": "src2:src:src",
            "sink:snk:snk": "bitvector:src:src",
        },
    )

    p.run()

    # Verify output is all 3s (binary 11)
    output_data = np.loadtxt(str(tmp_path / "output.txt"))
    assert np.all(output_data[:, 1] == 3), "Value should be all 3s (binary 11)"


def test_bit_gen_single_input(tmp_path):
    """Test BitVector with only one input stream."""
    p = Pipeline()

    p.insert(
        SegmentSource(
            name="src1",
            source_pad_names=("src",),
            rate=128,
            t0=0.0,
            duration=1.0,
            segments=((int(0.2 * 1e9), int(0.8 * 1e9)),),
            values=(1,),
        ),
        BitVector(
            name="bitvector",
            sink_pad_names=("input1",),
            source_pad_names=("src",),
        ),
        DumpSeriesSink(
            name="sink", fname=str(tmp_path / "output.txt"), sink_pad_names=("snk",)
        ),
        link_map={
            "bitvector:snk:input1": "src1:src:src",
            "sink:snk:snk": "bitvector:src:src",
        },
    )

    p.run()

    # Verify output
    output_data = np.loadtxt(str(tmp_path / "output.txt"))

    # Should have 2 columns: time + integer_value
    assert output_data.shape[1] == 2, f"Expected 2 columns, got {output_data.shape[1]}"

    # Verify state matches input buffer/gap pattern
    # At t=0.1s (gap): should be 0 (binary 0)
    idx = np.argmin(np.abs(output_data[:, 0] - 0.1))
    assert output_data[idx, 1] == 0, "Should be 0 (binary 0) in gap region"

    # At t=0.5s (buffer): should be 1 (binary 1)
    idx = np.argmin(np.abs(output_data[:, 0] - 0.5))
    assert output_data[idx, 1] == 1, "Should be 1 (binary 1) in buffer region"

    # At t=0.9s (gap): should be 0 (binary 0)
    idx = np.argmin(np.abs(output_data[:, 0] - 0.9))
    assert output_data[idx, 1] == 0, "Should be 0 (binary 0) in gap region"


def test_bit_gen_different_rates(tmp_path):
    """Test BitVector with inputs at very different sample rates."""
    p = Pipeline()

    # Source 1 at high rate: 256 Hz
    # Source 2 at low rate: 16 Hz
    # Output should use minimum rate (16 Hz) by default
    p.insert(
        SegmentSource(
            name="src1",
            source_pad_names=("src",),
            rate=256,
            t0=0.0,
            duration=1.0,
            segments=((int(0.1 * 1e9), int(0.5 * 1e9)),),
            values=(1,),
        ),
        SegmentSource(
            name="src2",
            source_pad_names=("src",),
            rate=16,
            t0=0.0,
            duration=1.0,
            segments=((int(0.3 * 1e9), int(0.7 * 1e9)),),
            values=(1,),
        ),
        BitVector(
            name="bitvector",
            sink_pad_names=("input1", "input2"),
            source_pad_names=("src",),
        ),
        DumpSeriesSink(
            name="sink", fname=str(tmp_path / "output.txt"), sink_pad_names=("snk",)
        ),
        link_map={
            "bitvector:snk:input1": "src1:src:src",
            "bitvector:snk:input2": "src2:src:src",
            "sink:snk:snk": "bitvector:src:src",
        },
    )

    p.run()

    # Verify output exists and has correct structure
    output_data = np.loadtxt(str(tmp_path / "output.txt"))
    assert output_data.shape[1] == 2, f"Expected 2 columns, got {output_data.shape[1]}"

    # Check sample rate - time differences should be ~1/16 seconds
    time_diffs = np.diff(output_data[:, 0])
    avg_diff = np.mean(time_diffs)
    expected_diff = 1.0 / 16  # ~0.0625 seconds
    assert (
        abs(avg_diff - expected_diff) < 0.001
    ), f"Expected ~{expected_diff}s between samples, got {avg_diff}s"


def test_bit_gen_output_rate(tmp_path):
    """Test BitVector with configured output_rate."""
    p = Pipeline()

    # Sources at rates 128 and 64 Hz
    # Configure output at 256 Hz (higher than max input rate)
    p.insert(
        SegmentSource(
            name="src1",
            source_pad_names=("src",),
            rate=128,
            t0=0.0,
            duration=1.0,
            segments=((int(0.1 * 1e9), int(0.3 * 1e9)),),
            values=(1,),
        ),
        SegmentSource(
            name="src2",
            source_pad_names=("src",),
            rate=64,
            t0=0.0,
            duration=1.0,
            segments=((int(0.2 * 1e9), int(0.4 * 1e9)),),
            values=(1,),
        ),
        BitVector(
            name="bitvector",
            sink_pad_names=("input1", "input2"),
            source_pad_names=("src",),
            output_rate=256,  # Specify output rate
        ),
        DumpSeriesSink(
            name="sink", fname=str(tmp_path / "output.txt"), sink_pad_names=("snk",)
        ),
        link_map={
            "bitvector:snk:input1": "src1:src:src",
            "bitvector:snk:input2": "src2:src:src",
            "sink:snk:snk": "bitvector:src:src",
        },
    )

    p.run()

    # Verify output
    output_data = np.loadtxt(str(tmp_path / "output.txt"))

    # Check sample rate - time differences should be ~1/256 seconds
    time_diffs = np.diff(output_data[:, 0])
    avg_diff = np.mean(time_diffs)
    expected_diff = 1.0 / 256  # ~0.0039 seconds
    assert (
        abs(avg_diff - expected_diff) < 0.001
    ), f"Expected ~{expected_diff}s between samples, got {avg_diff}s"

    # Verify state transitions based on aligned buffer boundaries
    # Note: align_buffers=True creates coarse-grained boundaries,
    # so states reflect the aligned buffer regions, not individual segment boundaries

    # At t=0.1s (in first aligned buffer 0-0.203125s): 0 (binary 00)
    idx = np.argmin(np.abs(output_data[:, 0] - 0.1))
    assert output_data[idx, 1] == 0, "Value should be 0 (binary 00)"

    # At t=0.25s (in second aligned buffer 0.203125-0.296875s): 3 (binary 11)
    idx = np.argmin(np.abs(output_data[:, 0] - 0.25))
    assert output_data[idx, 1] == 3, "Value should be 3 (binary 11)"

    # At t=0.4s (in third aligned buffer 0.296875-0.5s): 0 (binary 00)
    idx = np.argmin(np.abs(output_data[:, 0] - 0.4))
    assert output_data[idx, 1] == 0, "Value should be 0 (binary 00)"


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Run all tests
        test_bit_gen_basic(tmp_path)

        test_bit_gen_three_inputs(tmp_path)

        test_bit_gen_all_gaps(tmp_path)

        test_bit_gen_all_buffers(tmp_path)

        test_bit_gen_single_input(tmp_path)

        test_bit_gen_different_rates(tmp_path)

        test_bit_gen_output_rate(tmp_path)
