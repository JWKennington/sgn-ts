import numpy as np
from sgn.apps import Pipeline
from sgnts.sinks import DumpSeriesSink
from sgnts.sources import FakeSeriesSource
from sgnts.transforms import Adder, Amplify, Average


def get_expected_output(input_dict):

    end = input_dict["end"]
    inrate = input_dict["inrate"]
    default_value = input_dict["default_value"]
    initialize_array = input_dict["initialize_array"]
    avg_overlap_samples = input_dict["avg_overlap_samples"]
    real_f = input_dict["real_f"]
    imag_f = input_dict["imag_f"]
    real_amp = input_dict["real_amp"]
    imag_amp = input_dict["imag_amp"]
    ngap = input_dict["ngap"]
    reject_zeros = input_dict["reject_zeros"]
    default_to_avg = input_dict["default_to_avg"]

    t = np.linspace(0, end - 1.0 / inrate, end * inrate)
    indata = real_amp * np.sin(2 * np.pi * real_f * t)
    if abs(imag_amp) > 0.0:
        data_is_complex = True
        indata = indata.astype(complex) + 1j * imag_amp * np.sin(2 * np.pi * imag_f * t)
    else:
        data_is_complex = False

    # Compute the expected timestamps
    t_start = -avg_overlap_samples[1] / 16
    t_end = t_start + end - 1.0 / inrate
    expected_times = np.linspace(t_start, t_end, end * inrate)

    # Compute the expected output data
    expected_outdata = np.empty(
        end * inrate, dtype=np.complex128 if data_is_complex else np.float64
    )
    n_mean = 1 + sum(avg_overlap_samples)
    current_mean = default_value
    if data_is_complex:
        current_mean += 0j
    mean_array = np.tile(current_mean, n_mean)
    valid_samples = 0
    for idx in range(end * inrate):
        if (
            (1 + idx // inrate) % ngap == 0
            or np.isnan(indata[idx])
            or np.isinf(indata[idx])
            or indata[idx] == 0
            and reject_zeros
        ):
            # This is either gap data or the values are undesirable
            if default_to_avg:
                mean_array[idx % n_mean] = current_mean
            else:
                mean_array[idx % n_mean] = default_value
        else:
            # This is good nongap data
            valid_samples += 1
            mean_array[idx % n_mean] = indata[idx]
        if not initialize_array and valid_samples < n_mean:
            mean_array_subset = np.roll(mean_array, valid_samples - (idx + 1) % n_mean)[
                : max(1, valid_samples)
            ]
            current_mean = np.mean(mean_array_subset.real)
            if data_is_complex:
                current_mean += 1j * np.mean(mean_array_subset.imag)
        else:
            current_mean = np.mean(mean_array.real)
            if data_is_complex:
                current_mean += 1j * np.mean(mean_array.imag)
        expected_outdata[idx] = current_mean

    return expected_times, expected_outdata


def test_mean_complex(tmp_path):

    # Options for input data and element configuration
    input_dict1 = {}
    input_dict1["end"] = 8
    input_dict1["inrate"] = 16
    input_dict1["ngap"] = 5
    input_dict1["default_value"] = 3 - 1j
    input_dict1["initialize_array"] = True
    input_dict1["avg_overlap_samples"] = (8, 16)
    input_dict1["real_f"] = 0.125
    input_dict1["imag_f"] = 1.0
    input_dict1["real_amp"] = 2.0
    input_dict1["imag_amp"] = -5.0
    input_dict1["reject_zeros"] = True
    input_dict1["default_to_avg"] = True

    # Try different combinations of options
    input_dict2 = input_dict1.copy()
    input_dict2["initialize_array"] = False
    input_dict3 = input_dict1.copy()
    input_dict3["reject_zeros"] = False
    input_dict4 = input_dict1.copy()
    input_dict4["initialize_array"] = False
    input_dict4["default_to_avg"] = False

    pipeline = Pipeline()

    #
    #       ----------
    #      | src1     |
    #       ----------
    #              \
    #           H1  \ SR2
    #           ------------
    #          |  Average   |
    #           ------------
    #                 \
    #             H1   \ SR2
    #             ---------
    #            | snk1    |
    #             ---------

    pipeline.insert(
        FakeSeriesSource(
            name="rsrc",
            source_pad_names=("src",),
            rate=input_dict1["inrate"],
            ngap=input_dict1["ngap"],
            signal_type="sin",
            fsin=input_dict1["real_f"],
            end=input_dict1["end"],
        ),
        FakeSeriesSource(
            name="isrc",
            source_pad_names=("src",),
            rate=input_dict1["inrate"],
            ngap=input_dict1["ngap"],
            signal_type="sin",
            fsin=input_dict1["imag_f"],
            end=input_dict1["end"],
        ),
        Amplify(
            name="ramp",
            source_pad_names=("src",),
            sink_pad_names=("snk",),
            factor=input_dict1["real_amp"] + 0j,
        ),
        Amplify(
            name="iamp",
            source_pad_names=("src",),
            sink_pad_names=("snk",),
            factor=1j * input_dict1["imag_amp"],
        ),
        Adder(
            name="adder",
            sink_pad_names=("rsnk", "isnk"),
            source_pad_names=("src",),
        ),
        Average(
            name="mean1",
            method="mean",
            source_pad_names=("src",),
            sink_pad_names=("snk",),
            avg_overlap_samples=input_dict1["avg_overlap_samples"],
            default_value=input_dict1["default_value"],
            initialize_array=input_dict1["initialize_array"],
            reject_zeros=input_dict1["reject_zeros"],
            default_to_avg=input_dict1["default_to_avg"],
        ),
        Average(
            name="mean2",
            method="mean",
            source_pad_names=("src",),
            sink_pad_names=("snk",),
            avg_overlap_samples=input_dict2["avg_overlap_samples"],
            default_value=input_dict2["default_value"],
            initialize_array=input_dict2["initialize_array"],
            reject_zeros=input_dict2["reject_zeros"],
            default_to_avg=input_dict2["default_to_avg"],
        ),
        Average(
            name="mean3",
            method="mean",
            source_pad_names=("src",),
            sink_pad_names=("snk",),
            avg_overlap_samples=input_dict3["avg_overlap_samples"],
            default_value=input_dict3["default_value"],
            initialize_array=input_dict3["initialize_array"],
            reject_zeros=input_dict3["reject_zeros"],
            default_to_avg=input_dict3["default_to_avg"],
        ),
        Average(
            name="mean4",
            method="mean",
            source_pad_names=("src",),
            sink_pad_names=("snk",),
            avg_overlap_samples=input_dict3["avg_overlap_samples"],
            default_value=input_dict4["default_value"],
            initialize_array=input_dict4["initialize_array"],
            reject_zeros=input_dict4["reject_zeros"],
            default_to_avg=input_dict4["default_to_avg"],
        ),
        DumpSeriesSink(
            name="snk1",
            fname=str(tmp_path / "meanc1.txt"),
            sink_pad_names=("snk",),
        ),
        DumpSeriesSink(
            name="snk2",
            fname=str(tmp_path / "meanc2.txt"),
            sink_pad_names=("snk",),
        ),
        DumpSeriesSink(
            name="snk3",
            fname=str(tmp_path / "meanc3.txt"),
            sink_pad_names=("snk",),
        ),
        DumpSeriesSink(
            name="snk4",
            fname=str(tmp_path / "meanc4.txt"),
            sink_pad_names=("snk",),
        ),
        link_map={
            "ramp:snk:snk": "rsrc:src:src",
            "iamp:snk:snk": "isrc:src:src",
            "adder:snk:rsnk": "ramp:src:src",
            "adder:snk:isnk": "iamp:src:src",
            "mean1:snk:snk": "adder:src:src",
            "snk1:snk:snk": "mean1:src:src",
            "mean2:snk:snk": "adder:src:src",
            "snk2:snk:snk": "mean2:src:src",
            "mean3:snk:snk": "adder:src:src",
            "snk3:snk:snk": "mean3:src:src",
            "mean4:snk:snk": "adder:src:src",
            "snk4:snk:snk": "mean4:src:src",
        },
    )

    pipeline.run()

    # Get the output data
    outdata1 = np.loadtxt(str(tmp_path / "meanc1.txt"), dtype=np.complex128)
    t1 = np.real(np.transpose(outdata1)[0])
    outdata1 = np.transpose(outdata1)[1]
    expected_t1, expected_outdata1 = get_expected_output(input_dict1)
    np.testing.assert_almost_equal(t1, expected_t1)
    np.testing.assert_almost_equal(outdata1.real, expected_outdata1.real)
    np.testing.assert_almost_equal(outdata1.imag, expected_outdata1.imag)

    outdata2 = np.loadtxt(str(tmp_path / "meanc2.txt"), dtype=np.complex128)
    t2 = np.real(np.transpose(outdata2)[0])
    outdata2 = np.transpose(outdata2)[1]
    expected_t2, expected_outdata2 = get_expected_output(input_dict2)
    np.testing.assert_almost_equal(t2, expected_t2)
    np.testing.assert_almost_equal(outdata2.real, expected_outdata2.real)
    np.testing.assert_almost_equal(outdata2.imag, expected_outdata2.imag)

    outdata3 = np.loadtxt(str(tmp_path / "meanc3.txt"), dtype=np.complex128)
    t3 = np.real(np.transpose(outdata3)[0])
    outdata3 = np.transpose(outdata3)[1]
    expected_t3, expected_outdata3 = get_expected_output(input_dict3)
    np.testing.assert_almost_equal(t3, expected_t3)
    np.testing.assert_almost_equal(outdata3.real, expected_outdata3.real)
    np.testing.assert_almost_equal(outdata3.imag, expected_outdata3.imag)

    outdata4 = np.loadtxt(str(tmp_path / "meanc4.txt"), dtype=np.complex128)
    t4 = np.real(np.transpose(outdata4)[0])
    outdata4 = np.transpose(outdata4)[1]
    expected_t4, expected_outdata4 = get_expected_output(input_dict4)
    np.testing.assert_almost_equal(t4, expected_t4)
    np.testing.assert_almost_equal(outdata4.real, expected_outdata4.real)
    np.testing.assert_almost_equal(outdata4.imag, expected_outdata4.imag)


def test_mean_real(tmp_path):

    # Options for input data and element configuration
    input_dict1 = {}
    input_dict1["end"] = 8
    input_dict1["inrate"] = 16
    input_dict1["ngap"] = 5
    input_dict1["default_value"] = 0.0
    input_dict1["initialize_array"] = True
    input_dict1["avg_overlap_samples"] = (8, 16)
    input_dict1["real_f"] = 0.125
    input_dict1["imag_f"] = 1.0
    input_dict1["real_amp"] = 2.0
    input_dict1["imag_amp"] = 0.0
    input_dict1["reject_zeros"] = True
    input_dict1["default_to_avg"] = True

    # Try different combinations of options
    input_dict2 = input_dict1.copy()
    input_dict2["initialize_array"] = False
    input_dict3 = input_dict1.copy()
    input_dict3["reject_zeros"] = False
    input_dict4 = input_dict1.copy()
    input_dict4["initialize_array"] = False
    input_dict4["default_to_avg"] = False

    pipeline = Pipeline()

    #
    #       ----------
    #      | src1     |
    #       ----------
    #              \
    #           H1  \ SR2
    #           ------------
    #          |  Average   |
    #           ------------
    #                 \
    #             H1   \ SR2
    #             ---------
    #            | snk1    |
    #             ---------

    pipeline.insert(
        FakeSeriesSource(
            name="rsrc",
            source_pad_names=("src",),
            rate=input_dict1["inrate"],
            ngap=input_dict1["ngap"],
            signal_type="sin",
            fsin=input_dict1["real_f"],
            end=input_dict1["end"],
        ),
        Amplify(
            name="ramp",
            source_pad_names=("src",),
            sink_pad_names=("snk",),
            factor=input_dict1["real_amp"],
        ),
        Average(
            name="mean1",
            method="mean",
            source_pad_names=("src",),
            sink_pad_names=("snk",),
            avg_overlap_samples=input_dict1["avg_overlap_samples"],
            default_value=input_dict1["default_value"],
            initialize_array=input_dict1["initialize_array"],
            reject_zeros=input_dict1["reject_zeros"],
            default_to_avg=input_dict1["default_to_avg"],
        ),
        Average(
            name="mean2",
            method="mean",
            source_pad_names=("src",),
            sink_pad_names=("snk",),
            avg_overlap_samples=input_dict2["avg_overlap_samples"],
            default_value=input_dict2["default_value"],
            initialize_array=input_dict2["initialize_array"],
            reject_zeros=input_dict2["reject_zeros"],
            default_to_avg=input_dict2["default_to_avg"],
        ),
        Average(
            name="mean3",
            method="mean",
            source_pad_names=("src",),
            sink_pad_names=("snk",),
            avg_overlap_samples=input_dict3["avg_overlap_samples"],
            default_value=input_dict3["default_value"],
            initialize_array=input_dict3["initialize_array"],
            reject_zeros=input_dict3["reject_zeros"],
            default_to_avg=input_dict3["default_to_avg"],
        ),
        Average(
            name="mean4",
            method="mean",
            source_pad_names=("src",),
            sink_pad_names=("snk",),
            avg_overlap_samples=input_dict3["avg_overlap_samples"],
            default_value=input_dict4["default_value"],
            initialize_array=input_dict4["initialize_array"],
            reject_zeros=input_dict4["reject_zeros"],
            default_to_avg=input_dict4["default_to_avg"],
        ),
        DumpSeriesSink(
            name="snk1",
            fname=str(tmp_path / "meanr1.txt"),
            sink_pad_names=("snk",),
        ),
        DumpSeriesSink(
            name="snk2",
            fname=str(tmp_path / "meanr2.txt"),
            sink_pad_names=("snk",),
        ),
        DumpSeriesSink(
            name="snk3",
            fname=str(tmp_path / "meanr3.txt"),
            sink_pad_names=("snk",),
        ),
        DumpSeriesSink(
            name="snk4",
            fname=str(tmp_path / "meanr4.txt"),
            sink_pad_names=("snk",),
        ),
        link_map={
            "ramp:snk:snk": "rsrc:src:src",
            "mean1:snk:snk": "ramp:src:src",
            "snk1:snk:snk": "mean1:src:src",
            "mean2:snk:snk": "ramp:src:src",
            "snk2:snk:snk": "mean2:src:src",
            "mean3:snk:snk": "ramp:src:src",
            "snk3:snk:snk": "mean3:src:src",
            "mean4:snk:snk": "ramp:src:src",
            "snk4:snk:snk": "mean4:src:src",
        },
    )

    pipeline.run()

    # Get the output data
    outdata1 = np.loadtxt(str(tmp_path / "meanr1.txt"))
    t1 = np.real(np.transpose(outdata1)[0])
    outdata1 = np.transpose(outdata1)[1]
    expected_t1, expected_outdata1 = get_expected_output(input_dict1)
    np.testing.assert_almost_equal(t1, expected_t1)
    np.testing.assert_almost_equal(outdata1, expected_outdata1)

    outdata2 = np.loadtxt(str(tmp_path / "meanr2.txt"))
    t2 = np.real(np.transpose(outdata2)[0])
    outdata2 = np.transpose(outdata2)[1]
    expected_t2, expected_outdata2 = get_expected_output(input_dict2)
    np.testing.assert_almost_equal(t2, expected_t2)
    np.testing.assert_almost_equal(outdata2, expected_outdata2)

    outdata3 = np.loadtxt(str(tmp_path / "meanr3.txt"))
    t3 = np.real(np.transpose(outdata3)[0])
    outdata3 = np.transpose(outdata3)[1]
    expected_t3, expected_outdata3 = get_expected_output(input_dict3)
    np.testing.assert_almost_equal(t3, expected_t3)
    np.testing.assert_almost_equal(outdata3, expected_outdata3)

    outdata4 = np.loadtxt(str(tmp_path / "meanr4.txt"))
    t4 = np.real(np.transpose(outdata4)[0])
    outdata4 = np.transpose(outdata4)[1]
    expected_t4, expected_outdata4 = get_expected_output(input_dict4)
    np.testing.assert_almost_equal(t4, expected_t4)
    np.testing.assert_almost_equal(outdata4, expected_outdata4)


def test_real_data_complex_avg_array_warning(tmp_path, capsys):
    """Test warning when real data is sent but avg_array was initialized as complex
    This covers lines 189-196 in average.py
    """
    pipeline = Pipeline()

    # Initialize with complex default value to force complex avg_array
    pipeline.insert(
        FakeSeriesSource(
            name="src",
            source_pad_names=("src",),
            rate=16,
            signal_type="sin",
            fsin=0.125,
            end=2,
        ),
        Average(
            name="avg",
            method="mean",
            source_pad_names=("src",),
            sink_pad_names=("snk",),
            avg_overlap_samples=(4, 4),
            default_value=1.0 + 2.0j,  # Complex default value
            initialize_array=True,
        ),
        DumpSeriesSink(
            name="snk",
            fname=str(tmp_path / "real_complex_warning.txt"),
            sink_pad_names=("snk",),
        ),
        link_map={
            "avg:snk:snk": "src:src:src",
            "snk:snk:snk": "avg:src:src",
        },
    )

    pipeline.run()

    # Check that the warning was printed
    captured = capsys.readouterr()
    assert "WARNING: Average: data is real; discarding imaginary" in captured.out

    # Check that output was created
    outdata = np.loadtxt(str(tmp_path / "real_complex_warning.txt"))
    assert len(outdata) > 0


def test_complex_data_real_avg_array_conversion(tmp_path):
    """Test conversion of avg_array to complex when complex data arrives
    This covers line 200 in average.py
    """
    pipeline = Pipeline()

    # Start with real default value, then send complex data
    pipeline.insert(
        # Real source
        FakeSeriesSource(
            name="rsrc",
            source_pad_names=("src",),
            rate=16,
            signal_type="sin",
            fsin=0.125,
            end=2,
        ),
        # Imaginary source
        FakeSeriesSource(
            name="isrc",
            source_pad_names=("src",),
            rate=16,
            signal_type="sin",
            fsin=0.25,
            end=2,
        ),
        # Make real part
        Amplify(
            name="ramp",
            source_pad_names=("src",),
            sink_pad_names=("snk",),
            factor=1.0 + 0j,
        ),
        # Make imaginary part
        Amplify(
            name="iamp",
            source_pad_names=("src",),
            sink_pad_names=("snk",),
            factor=1j,
        ),
        # Add to create complex signal
        Adder(
            name="adder",
            sink_pad_names=("rsnk", "isnk"),
            source_pad_names=("src",),
        ),
        Average(
            name="avg",
            method="mean",
            source_pad_names=("src",),
            sink_pad_names=("snk",),
            avg_overlap_samples=(4, 4),
            default_value=1.0,  # Real default value
            initialize_array=True,
        ),
        DumpSeriesSink(
            name="snk",
            fname=str(tmp_path / "complex_conversion.txt"),
            sink_pad_names=("snk",),
        ),
        link_map={
            "ramp:snk:snk": "rsrc:src:src",
            "iamp:snk:snk": "isrc:src:src",
            "adder:snk:rsnk": "ramp:src:src",
            "adder:snk:isnk": "iamp:src:src",
            "avg:snk:snk": "adder:src:src",
            "snk:snk:snk": "avg:src:src",
        },
    )

    pipeline.run()

    # Check that output was created and is complex
    outdata = np.loadtxt(str(tmp_path / "complex_conversion.txt"), dtype=np.complex128)
    assert len(outdata) > 0
    # Verify we have complex data (some imaginary parts should be non-zero)
    assert np.any(np.imag(outdata[:, 1]) != 0)


def test_zero_length_gap_mean(tmp_path):
    """Test handling of zero-length gap buffer in mean
    This covers line 161 in average.py
    """
    from dataclasses import dataclass
    from sgnts.base import TSSource, TSSlice

    @dataclass
    class ZeroLengthGapSource(TSSource):
        """Custom source that produces a zero-length gap buffer"""

        def __post_init__(self):
            super().__post_init__()
            self._counter = 0
            # Set up buffer params for the source pad
            for pad in self.source_pads:
                self.set_pad_buffer_params(pad=pad, sample_shape=(), rate=16)

        def new(self, pad):
            # Get the next frame
            frame = self.prepare_frame(pad)

            # For the second buffer, create a zero-length gap
            self._counter += 1

            if self._counter == 2:
                # Create a zero-length gap buffer
                buf = frame[0]
                # Set the buffer to have zero length by making offset == end_offset
                gap_buf = buf.new(
                    TSSlice(buf.offset, buf.offset),  # Zero length
                    data=None,  # Gap buffer
                )
                frame.set_buffers([gap_buf])
            elif self._counter < 2:
                # Normal data buffer
                frame[0].set_data(np.ones(frame[0].shape))
            else:
                # Normal data buffer
                frame[0].set_data(2 * np.ones(frame[0].shape))

            return frame

    pipeline = Pipeline()

    pipeline.insert(
        ZeroLengthGapSource(
            name="src",
            source_pad_names=("src",),
            t0=0,
            end=2,
        ),
        Average(
            name="avg",
            method="mean",
            source_pad_names=("src",),
            sink_pad_names=("snk",),
            avg_overlap_samples=(0, 0),  # No overlap to simplify
            default_value=0.0,
        ),
        DumpSeriesSink(
            name="snk",
            fname=str(tmp_path / "mean_zero_gap.txt"),
            sink_pad_names=("snk",),
        ),
        link_map={
            "avg:snk:snk": "src:src:src",
            "snk:snk:snk": "avg:src:src",
        },
    )

    pipeline.run()

    # Check that output was created
    outdata = np.loadtxt(str(tmp_path / "mean_zero_gap.txt"))
    assert len(outdata) > 0
