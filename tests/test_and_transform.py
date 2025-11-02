"""Tests for ANDTransform element."""

import numpy as np
from pathlib import Path
from sgn.apps import Pipeline

from sgnts.sources.segment import SegmentSource
from sgnts.transforms.and_transform import ANDTransform
from sgnts.sinks import DumpSeriesSink


def test_and_transform_basic(tmp_path):
    """Test basic ANDTransform functionality with two inputs."""
    p = Pipeline()

    # Source 1: has data from 0.1-0.5s (gap from 0-0.1s and 0.5-1s)
    # Source 2: has data from 0.3-1.0s (gap from 0-0.3s)
    # Expected AND output: high from 0.3-0.5s (where both have data)
    p.insert(
        SegmentSource(
            name="src1",
            source_pad_names=("src",),
            rate=128,
            t0=0.0,
            duration=1.0,
            segments=((int(0.1 * 1e9), int(0.5 * 1e9)),),
            values=(1,),
        ),
        SegmentSource(
            name="src2",
            source_pad_names=("src",),
            rate=64,
            t0=0.0,
            duration=1.0,
            segments=((int(0.3 * 1e9), int(1.0 * 1e9)),),
            values=(2,),
        ),
        ANDTransform(
            name="detector",
            sink_pad_names=("input1", "input2"),
            source_pad_names=("src",),
        ),
        DumpSeriesSink(
            name="sink", fname=str(tmp_path / "output.txt"), sink_pad_names=("snk",)
        ),
        link_map={
            "detector:snk:input1": "src1:src:src",
            "detector:snk:input2": "src2:src:src",
            "sink:snk:snk": "detector:src:src",
        },
    )

    # Run pipeline
    p.run()

    # Load and verify output
    output_data = np.loadtxt(str(tmp_path / "output.txt"))

    # Find where output is 1 (all inputs have data)
    high_indices = np.where(output_data[:, 1] == 1)[0]

    if len(high_indices) > 0:
        # Check that high region is approximately 0.3-0.5s
        first_high_time = output_data[high_indices[0], 0]
        last_high_time = output_data[high_indices[-1], 0]

        assert (
            0.28 < first_high_time < 0.32
        ), f"Expected first high around 0.3s, got {first_high_time}"
        assert (
            0.48 < last_high_time < 0.52
        ), f"Expected last high around 0.5s, got {last_high_time}"


def test_and_transform_three_inputs(tmp_path):
    """Test ANDTransform with three inputs of different rates."""
    p = Pipeline()

    # Source 1: data from 0.1-0.2s, 0.4-0.7s, 0.8-1.0s
    # Source 2: data from 0.15-0.5s, 0.65-1.0s
    # Source 3: data from 0.1-0.3s, 0.6-0.9s
    # Expected AND: 0.15-0.2s, 0.65-0.7s, 0.8-0.9s

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
        ANDTransform(
            name="detector",
            sink_pad_names=("input1", "input2", "input3"),
            source_pad_names=("src",),
        ),
        DumpSeriesSink(
            name="sink", fname=str(tmp_path / "output.txt"), sink_pad_names=("snk",)
        ),
        link_map={
            "detector:snk:input1": "src1:src:src",
            "detector:snk:input2": "src2:src:src",
            "detector:snk:input3": "src3:src:src",
            "sink:snk:snk": "detector:src:src",
        },
    )

    p.run()

    # Verify output
    output_data = np.loadtxt(str(tmp_path / "output.txt"))
    high_data = output_data[output_data[:, 1] == 1]

    # Should have three distinct high regions
    if len(high_data) > 0:
        # Find continuous regions
        time_diffs = np.diff(high_data[:, 0])
        gap_threshold = 2.0 / 256  # Two samples at max rate
        gap_indices = np.where(time_diffs > gap_threshold)[0]

        # Should have 2 gaps, so 3 regions
        assert (
            len(gap_indices) == 2
        ), f"Expected 3 high regions, found {len(gap_indices) + 1}"


def test_and_transform_no_overlap(tmp_path):
    """Test ANDTransform when inputs have no overlapping data."""
    p = Pipeline()

    # Source 1: data from 0-0.4s
    # Source 2: data from 0.6-1.0s
    # No overlap - output should be all gaps

    p.insert(
        SegmentSource(
            name="src1",
            source_pad_names=("src",),
            rate=128,
            t0=0.0,
            duration=1.0,
            segments=((0, int(0.4 * 1e9)),),
            values=(1,),
        ),
        SegmentSource(
            name="src2",
            source_pad_names=("src",),
            rate=64,
            t0=0.0,
            duration=1.0,
            segments=((int(0.6 * 1e9), int(1.0 * 1e9)),),
            values=(1,),
        ),
        ANDTransform(
            name="detector",
            sink_pad_names=("input1", "input2"),
            source_pad_names=("src",),
        ),
        DumpSeriesSink(
            name="sink", fname=str(tmp_path / "output.txt"), sink_pad_names=("snk",)
        ),
        link_map={
            "detector:snk:input1": "src1:src:src",
            "detector:snk:input2": "src2:src:src",
            "sink:snk:snk": "detector:src:src",
        },
    )

    p.run()

    # Verify output is empty (gap buffers aren't written to file)
    with open(tmp_path / "output.txt") as f:
        content = f.read()
    assert content == "", "Expected empty output (gap) when no overlap"


def test_and_transform_all_data(tmp_path):
    """Test ANDTransform when all inputs have continuous data."""
    p = Pipeline()

    # Both sources have continuous data from 0-1s
    # Output should be all ones

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
        ANDTransform(
            name="detector",
            sink_pad_names=("input1", "input2"),
            source_pad_names=("src",),
        ),
        DumpSeriesSink(
            name="sink", fname=str(tmp_path / "output.txt"), sink_pad_names=("snk",)
        ),
        link_map={
            "detector:snk:input1": "src1:src:src",
            "detector:snk:input2": "src2:src:src",
            "sink:snk:snk": "detector:src:src",
        },
    )

    p.run()

    # Verify output is all ones
    output_data = np.loadtxt(str(tmp_path / "output.txt"))
    assert np.all(output_data[:, 1] == 1), "Expected all ones when all inputs have data"


def test_and_transform_output_shape(tmp_path):
    """Test ANDTransform with custom output shape."""
    p = Pipeline()

    p.insert(
        SegmentSource(
            name="src1",
            source_pad_names=("src",),
            rate=128,
            t0=0.0,
            duration=0.5,
            segments=((0, int(0.5 * 1e9)),),
            values=(1,),
        ),
        SegmentSource(
            name="src2",
            source_pad_names=("src",),
            rate=64,
            t0=0.0,
            duration=0.5,
            segments=((0, int(0.5 * 1e9)),),
            values=(1,),
        ),
        ANDTransform(
            name="detector",
            sink_pad_names=("input1", "input2"),
            source_pad_names=("src",),
            output_shape=(2,),  # Vector output
        ),
        DumpSeriesSink(
            name="sink", fname=str(tmp_path / "output.txt"), sink_pad_names=("snk",)
        ),
        link_map={
            "detector:snk:input1": "src1:src:src",
            "detector:snk:input2": "src2:src:src",
            "sink:snk:snk": "detector:src:src",
        },
    )

    p.run()

    # Verify output shape
    output_data = np.loadtxt(str(tmp_path / "output.txt"))
    # Output should have 3 columns: time, value1, value2
    assert (
        output_data.shape[1] == 3
    ), f"Expected 3 columns for vector output, got {output_data.shape[1]}"


def test_and_transform_visualization(tmp_path):
    """Test ANDTransform with visualization (only when run as standalone)."""
    import sys

    # Only generate plot if running as standalone script
    if __name__ == "__main__" and not sys.argv[0].endswith("pytest"):
        _create_visualization(tmp_path)

    # Always run the basic test
    p = Pipeline()

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
        ANDTransform(
            name="detector",
            sink_pad_names=("input1", "input2"),
            source_pad_names=("src",),
        ),
        DumpSeriesSink(
            name="sink", fname=str(tmp_path / "output.txt"), sink_pad_names=("snk",)
        ),
        link_map={
            "detector:snk:input1": "src1:src:src",
            "detector:snk:input2": "src2:src:src",
            "sink:snk:snk": "detector:src:src",
        },
    )

    p.run()

    # Basic verification
    output_data = np.loadtxt(str(tmp_path / "output.txt"))
    assert len(output_data) > 0, "Output should have data"


def _create_visualization(tmp_path):
    """Create visualization of ANDTransform operation."""
    import matplotlib.pyplot as plt

    print("Generating ANDTransform visualization...")

    # Create pipeline
    p = Pipeline()

    # Define segments for visualization
    src1_segments = (
        (int(0.1 * 1e9), int(0.2 * 1e9)),
        (int(0.4 * 1e9), int(0.7 * 1e9)),
        (int(0.8 * 1e9), int(1.0 * 1e9)),
    )

    src2_segments = (
        (int(0.15 * 1e9), int(0.5 * 1e9)),
        (int(0.65 * 1e9), int(1.0 * 1e9)),
    )

    src3_segments = (
        (int(0.1 * 1e9), int(0.3 * 1e9)),
        (int(0.6 * 1e9), int(0.9 * 1e9)),
    )

    p.insert(
        SegmentSource(
            name="src1",
            source_pad_names=("src",),
            rate=256,
            t0=0.0,
            duration=1.0,
            segments=src1_segments,
            values=(1, 1, 1),
        ),
        SegmentSource(
            name="src2",
            source_pad_names=("src",),
            rate=128,
            t0=0.0,
            duration=1.0,
            segments=src2_segments,
            values=(1, 1),
        ),
        SegmentSource(
            name="src3",
            source_pad_names=("src",),
            rate=64,
            t0=0.0,
            duration=1.0,
            segments=src3_segments,
            values=(1, 1),
        ),
        ANDTransform(
            name="detector",
            sink_pad_names=("input1", "input2", "input3"),
            source_pad_names=("src",),
        ),
        DumpSeriesSink(
            name="sink1",
            fname=str(tmp_path / "src1_output.txt"),
            sink_pad_names=("snk",),
        ),
        DumpSeriesSink(
            name="sink2",
            fname=str(tmp_path / "src2_output.txt"),
            sink_pad_names=("snk",),
        ),
        DumpSeriesSink(
            name="sink3",
            fname=str(tmp_path / "src3_output.txt"),
            sink_pad_names=("snk",),
        ),
        DumpSeriesSink(
            name="sink_detector",
            fname=str(tmp_path / "detector_output.txt"),
            sink_pad_names=("snk",),
        ),
        link_map={
            "detector:snk:input1": "src1:src:src",
            "detector:snk:input2": "src2:src:src",
            "detector:snk:input3": "src3:src:src",
            "sink1:snk:snk": "src1:src:src",
            "sink2:snk:snk": "src2:src:src",
            "sink3:snk:snk": "src3:src:src",
            "sink_detector:snk:snk": "detector:src:src",
        },
    )

    # Run pipeline
    p.run()

    # Load output data
    # Note: ANDTransform outputs gap buffers where any input has gaps,
    # and DumpSeriesSink only writes non-gap data to file, so we only see the 1s
    detector_data = np.loadtxt(str(tmp_path / "detector_output.txt"))

    # Create figure
    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(
        "AND Transform: Input Streams and Output", fontsize=14, fontweight="bold"
    )

    def create_gap_visualization(times, segments_ns):
        """Create binary signal showing where segments have data."""
        signal = np.zeros_like(times)
        for seg_start, seg_end in segments_ns:
            seg_start_s = seg_start / 1e9
            seg_end_s = seg_end / 1e9
            mask = (times >= seg_start_s) & (times < seg_end_s)
            signal[mask] = 1
        return signal

    # Create time array
    vis_times = np.linspace(0, 1.0, 1000)

    # Plot Source 1
    signal1 = create_gap_visualization(vis_times, src1_segments)
    axes[0].fill_between(vis_times, 0, signal1, step="post", alpha=0.7, color="blue")
    axes[0].set_ylabel("Source 1\n(256 Hz)", fontweight="bold")
    axes[0].set_ylim(-0.1, 1.1)
    axes[0].set_yticks([0, 1])
    axes[0].set_yticklabels(["Gap", "Data"])
    axes[0].grid(True, alpha=0.3)

    # Plot Source 2
    signal2 = create_gap_visualization(vis_times, src2_segments)
    axes[1].fill_between(vis_times, 0, signal2, step="post", alpha=0.7, color="green")
    axes[1].set_ylabel("Source 2\n(128 Hz)", fontweight="bold")
    axes[1].set_ylim(-0.1, 1.1)
    axes[1].set_yticks([0, 1])
    axes[1].set_yticklabels(["Gap", "Data"])
    axes[1].grid(True, alpha=0.3)

    # Plot Source 3
    signal3 = create_gap_visualization(vis_times, src3_segments)
    axes[2].fill_between(vis_times, 0, signal3, step="post", alpha=0.7, color="orange")
    axes[2].set_ylabel("Source 3\n(64 Hz)", fontweight="bold")
    axes[2].set_ylim(-0.1, 1.1)
    axes[2].set_yticks([0, 1])
    axes[2].set_yticklabels(["Gap", "Data"])
    axes[2].grid(True, alpha=0.3)

    # Plot Detector Output (only non-gap data is written to file)
    # We need to detect discontinuities to avoid connecting segments
    if len(detector_data) > 0:
        times = detector_data[:, 0]
        values = detector_data[:, 1]

        # Find discontinuities (gaps larger than 2 sample periods)
        if len(times) > 1:
            sample_period = times[1] - times[0]
            time_diffs = np.diff(times)
            gap_indices = np.where(time_diffs > sample_period * 1.5)[0]

            # Split into continuous segments
            segments = []
            start_idx = 0
            for gap_idx in gap_indices:
                segments.append((start_idx, gap_idx + 1))
                start_idx = gap_idx + 1
            segments.append((start_idx, len(times)))

            # Plot each segment separately to avoid connecting gaps
            for start_idx, end_idx in segments:
                axes[3].fill_between(
                    times[start_idx:end_idx],
                    0,
                    values[start_idx:end_idx],
                    step="post",
                    alpha=0.7,
                    color="red",
                )
        else:
            # Single point
            axes[3].fill_between(
                times,
                0,
                values,
                step="post",
                alpha=0.7,
                color="red",
            )
    axes[3].set_ylabel("AND\nOutput\n(256 Hz)", fontweight="bold")
    axes[3].set_ylim(-0.1, 1.1)
    axes[3].set_yticks([0, 1])
    axes[3].set_yticklabels(["Gap (not shown)", "All Data"])
    axes[3].set_xlabel("Time (seconds)", fontweight="bold")
    axes[3].grid(True, alpha=0.3)

    # Add vertical lines to show boundaries
    segment_times = [0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.8]
    for t in segment_times:
        for ax in axes:
            ax.axvline(x=t, color="gray", linestyle="--", alpha=0.3)

    # Highlight overlap regions
    overlap_regions = [(0.15, 0.2), (0.65, 0.7), (0.8, 0.9)]
    for start, end in overlap_regions:
        axes[3].axvspan(start, end, alpha=0.2, color="yellow")

    # Add legend
    handles = [
        plt.Rectangle((0, 0), 1, 1, fc="blue", alpha=0.7, label="Source 1"),
        plt.Rectangle((0, 0), 1, 1, fc="green", alpha=0.7, label="Source 2"),
        plt.Rectangle((0, 0), 1, 1, fc="orange", alpha=0.7, label="Source 3"),
        plt.Rectangle((0, 0), 1, 1, fc="red", alpha=0.7, label="All have data"),
    ]
    axes[0].legend(handles=handles, loc="upper right", ncol=4, fontsize=8)

    plt.tight_layout()

    # Save figure
    output_path = "and_transform_visualization.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to: {output_path}")

    # Show plot
    try:
        plt.show(block=False)
    except Exception:  # noqa: S110
        pass

    # Print analysis
    print("\n=== AND Transform Analysis ===")
    print("Source 1 data: 0.1-0.2s, 0.4-0.7s, 0.8-1.0s")
    print("Source 2 data: 0.15-0.5s, 0.65-1.0s")
    print("Source 3 data: 0.1-0.3s, 0.6-0.9s")
    print("Expected output HIGH: 0.15-0.2s, 0.65-0.7s, 0.8-0.9s")

    # Verify detector output
    detector_high = detector_data[detector_data[:, 1] == 1]
    if len(detector_high) > 0:
        print("\nActual output HIGH periods:")
        time_diffs = np.diff(detector_high[:, 0])
        gap_threshold = 2.0 / 256
        gap_indices = np.where(time_diffs > gap_threshold)[0]

        start_idx = 0
        for gap_idx in gap_indices:
            end_idx = gap_idx
            print(
                f"  {detector_high[start_idx, 0]:.3f}s - "
                f"{detector_high[end_idx, 0]:.3f}s"
            )
            start_idx = end_idx + 1
        print(f"  {detector_high[start_idx, 0]:.3f}s - " f"{detector_high[-1, 0]:.3f}s")


def test_and_transform_all_gaps(tmp_path):
    """Test ANDTransform when all inputs have only gaps (covers line 97)."""
    p = Pipeline()

    # Create sources with only gaps (no segments)
    p.insert(
        SegmentSource(
            name="src1",
            source_pad_names=("src",),
            rate=128,
            t0=0.0,
            duration=0.125,  # 1/8 second for proper alignment
            segments=(),  # Empty segments = all gaps
            values=(),
        ),
        SegmentSource(
            name="src2",
            source_pad_names=("src",),
            rate=64,
            t0=0.0,
            duration=0.125,  # 1/8 second for proper alignment
            segments=(),  # Empty segments = all gaps
            values=(),
        ),
        ANDTransform(
            name="detector",
            sink_pad_names=("input1", "input2"),
            source_pad_names=("src",),
        ),
        DumpSeriesSink(
            name="sink", fname=str(tmp_path / "output.txt"), sink_pad_names=("snk",)
        ),
        link_map={
            "detector:snk:input1": "src1:src:src",
            "detector:snk:input2": "src2:src:src",
            "sink:snk:snk": "detector:src:src",
        },
    )

    p.run()

    # Output should be empty (gap buffers aren't written to file)
    with open(tmp_path / "output.txt") as f:
        content = f.read()
    assert content == "", "Expected empty output (gaps) for all-gap inputs"


def test_and_transform_heartbeat(tmp_path):
    """Test ANDTransform heartbeat buffer case (covers lines 131-133)."""
    p = Pipeline()

    # Create sources with extremely small duration (< 1 sample at max rate)
    p.insert(
        SegmentSource(
            name="src1",
            source_pad_names=("src",),
            rate=16384,  # Max rate
            t0=0.0,
            duration=1e-10,  # Less than one sample duration
            segments=(),
            values=(),
        ),
        SegmentSource(
            name="src2",
            source_pad_names=("src",),
            rate=16384,
            t0=0.0,
            duration=1e-10,
            segments=(),
            values=(),
        ),
        ANDTransform(
            name="detector",
            sink_pad_names=("input1", "input2"),
            source_pad_names=("src",),
            output_shape=(3,),  # Test with non-scalar shape
        ),
        DumpSeriesSink(
            name="sink", fname=str(tmp_path / "output.txt"), sink_pad_names=("snk",)
        ),
        link_map={
            "detector:snk:input1": "src1:src:src",
            "detector:snk:input2": "src2:src:src",
            "sink:snk:snk": "detector:src:src",
        },
    )

    p.run()

    # File should exist (may be empty for heartbeat)
    assert (tmp_path / "output.txt").exists()


if __name__ == "__main__":
    import tempfile
    import sys

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # If running as standalone script, generate visualization
        if not sys.argv[0].endswith("pytest"):
            _create_visualization(tmp_path)
        else:
            # Run all tests
            test_and_transform_basic(tmp_path)
            test_and_transform_three_inputs(tmp_path)
            test_and_transform_no_overlap(tmp_path)
            test_and_transform_all_data(tmp_path)
            test_and_transform_output_shape(tmp_path)
            test_and_transform_visualization(tmp_path)
            test_and_transform_all_gaps(tmp_path)
            test_and_transform_heartbeat(tmp_path)
            print("All tests passed!")
