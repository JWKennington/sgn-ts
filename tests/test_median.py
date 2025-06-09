#!/usr/bin/env python3

import numpy as np
from sgn.apps import Pipeline
from sgnts.sinks import DumpSeriesSink
from sgnts.sources import FakeSeriesSource
from sgnts.transforms import Adder, Amplify, Median
from sgnts.transforms.median import get_new_median


def test_get_new_median():
    arr = 100 * (np.random.rand(201) - 0.5)
    current_median = arr[0]
    for i in range(1, 1000):
        if i < len(arr):
            # The array is still filling up
            arr_sub = arr[:i]
        else:
            # The array is full, so replace a value
            arr_sub = arr
            arr_sub[i % len(arr_sub)] = 100 * (np.random.rand() - 0.5)
        if len(arr_sub) % 2:
            current_median = get_new_median(arr_sub, current_median)
            assert current_median == np.median(arr_sub)
        else:
            current_median = np.median(arr_sub)


def get_expected_output(input_dict):

    end = input_dict["end"]
    inrate = input_dict["inrate"]
    default_value = input_dict["default_value"]
    initialize_array = input_dict["initialize_array"]
    median_overlap_samples = input_dict["median_overlap_samples"]
    real_f = input_dict["real_f"]
    imag_f = input_dict["imag_f"]
    real_amp = input_dict["real_amp"]
    imag_amp = input_dict["imag_amp"]
    ngap = input_dict["ngap"]
    reject_zeros = input_dict["reject_zeros"]
    default_to_median = input_dict["default_to_median"]

    t = np.linspace(0, end - 1.0 / inrate, end * inrate)
    indata = real_amp * np.sin(2 * np.pi * real_f * t)
    if abs(imag_amp) > 0.0:
        data_is_complex = True
        indata = indata.astype(complex) + 1j * imag_amp * np.sin(2 * np.pi * imag_f * t)
    else:
        data_is_complex = False

    # Compute the expected timestamps
    t_start = -median_overlap_samples[1] / 16
    t_end = t_start + end - 1.0 / inrate
    expected_times = np.linspace(t_start, t_end, end * inrate)

    # Compute the expected output data
    expected_outdata = np.empty(
        end * inrate, dtype=np.complex128 if data_is_complex else np.float64
    )
    n_median = 1 + sum(median_overlap_samples)
    current_median = default_value
    if data_is_complex:
        current_median += 0j
    median_array = np.tile(current_median, n_median)
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
            if default_to_median:
                median_array[idx % n_median] = current_median
            else:
                median_array[idx % n_median] = default_value
        else:
            # This is good nongap data
            valid_samples += 1
            median_array[idx % n_median] = indata[idx]
        if not initialize_array and valid_samples < n_median:
            median_array_subset = np.roll(
                median_array, valid_samples - (idx + 1) % n_median
            )[: max(1, valid_samples)]
            current_median = np.median(median_array_subset.real)
            if data_is_complex:
                current_median += 1j * np.median(median_array_subset.imag)
        else:
            current_median = np.median(median_array.real)
            if data_is_complex:
                current_median += 1j * np.median(median_array.imag)
        expected_outdata[idx] = current_median

    return expected_times, expected_outdata


def test_median_complex():

    # Options for input data and element configuration
    input_dict1 = {}
    input_dict1["end"] = 8
    input_dict1["inrate"] = 16
    input_dict1["ngap"] = 5
    input_dict1["default_value"] = 3 - 1j
    input_dict1["initialize_array"] = True
    input_dict1["median_overlap_samples"] = (8, 16)
    input_dict1["real_f"] = 0.125
    input_dict1["imag_f"] = 1.0
    input_dict1["real_amp"] = 2.0
    input_dict1["imag_amp"] = -5.0
    input_dict1["reject_zeros"] = True
    input_dict1["default_to_median"] = True

    # Try different combinations of options
    input_dict2 = input_dict1.copy()
    input_dict2["initialize_array"] = False
    input_dict3 = input_dict1.copy()
    input_dict3["reject_zeros"] = False
    input_dict4 = input_dict1.copy()
    input_dict4["initialize_array"] = False
    input_dict4["default_to_median"] = False

    pipeline = Pipeline()

    #
    #       ----------
    #      | src1     |
    #       ----------
    #              \
    #           H1  \ SR2
    #           ------------
    #          |   Median   |
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
        Median(
            name="median1",
            source_pad_names=("src",),
            sink_pad_names=("snk",),
            median_overlap_samples=input_dict1["median_overlap_samples"],
            default_value=input_dict1["default_value"],
            initialize_array=input_dict1["initialize_array"],
            reject_zeros=input_dict1["reject_zeros"],
            default_to_median=input_dict1["default_to_median"],
        ),
        Median(
            name="median2",
            source_pad_names=("src",),
            sink_pad_names=("snk",),
            median_overlap_samples=input_dict2["median_overlap_samples"],
            default_value=input_dict2["default_value"],
            initialize_array=input_dict2["initialize_array"],
            reject_zeros=input_dict2["reject_zeros"],
            default_to_median=input_dict2["default_to_median"],
        ),
        Median(
            name="median3",
            source_pad_names=("src",),
            sink_pad_names=("snk",),
            median_overlap_samples=input_dict3["median_overlap_samples"],
            default_value=input_dict3["default_value"],
            initialize_array=input_dict3["initialize_array"],
            reject_zeros=input_dict3["reject_zeros"],
            default_to_median=input_dict3["default_to_median"],
        ),
        Median(
            name="median4",
            source_pad_names=("src",),
            sink_pad_names=("snk",),
            median_overlap_samples=input_dict3["median_overlap_samples"],
            default_value=input_dict4["default_value"],
            initialize_array=input_dict4["initialize_array"],
            reject_zeros=input_dict4["reject_zeros"],
            default_to_median=input_dict4["default_to_median"],
        ),
        DumpSeriesSink(
            name="snk1",
            fname="output1.txt",
            sink_pad_names=("snk",),
        ),
        DumpSeriesSink(
            name="snk2",
            fname="output2.txt",
            sink_pad_names=("snk",),
        ),
        DumpSeriesSink(
            name="snk3",
            fname="output3.txt",
            sink_pad_names=("snk",),
        ),
        DumpSeriesSink(
            name="snk4",
            fname="output4.txt",
            sink_pad_names=("snk",),
        ),
        link_map={
            "ramp:snk:snk": "rsrc:src:src",
            "iamp:snk:snk": "isrc:src:src",
            "adder:snk:rsnk": "ramp:src:src",
            "adder:snk:isnk": "iamp:src:src",
            "median1:snk:snk": "adder:src:src",
            "snk1:snk:snk": "median1:src:src",
            "median2:snk:snk": "adder:src:src",
            "snk2:snk:snk": "median2:src:src",
            "median3:snk:snk": "adder:src:src",
            "snk3:snk:snk": "median3:src:src",
            "median4:snk:snk": "adder:src:src",
            "snk4:snk:snk": "median4:src:src",
        },
    )

    pipeline.run()

    # Get the output data
    outdata1 = np.loadtxt("output1.txt", dtype=np.complex128)
    t1 = np.real(np.transpose(outdata1)[0])
    outdata1 = np.transpose(outdata1)[1]
    expected_t1, expected_outdata1 = get_expected_output(input_dict1)
    np.testing.assert_almost_equal(t1, expected_t1)
    np.testing.assert_almost_equal(outdata1.real, expected_outdata1.real)
    np.testing.assert_almost_equal(outdata1.imag, expected_outdata1.imag)

    outdata2 = np.loadtxt("output2.txt", dtype=np.complex128)
    t2 = np.real(np.transpose(outdata2)[0])
    outdata2 = np.transpose(outdata2)[1]
    expected_t2, expected_outdata2 = get_expected_output(input_dict2)
    np.testing.assert_almost_equal(t2, expected_t2)
    np.testing.assert_almost_equal(outdata2.real, expected_outdata2.real)
    np.testing.assert_almost_equal(outdata2.imag, expected_outdata2.imag)

    outdata3 = np.loadtxt("output3.txt", dtype=np.complex128)
    t3 = np.real(np.transpose(outdata3)[0])
    outdata3 = np.transpose(outdata3)[1]
    expected_t3, expected_outdata3 = get_expected_output(input_dict3)
    np.testing.assert_almost_equal(t3, expected_t3)
    np.testing.assert_almost_equal(outdata3.real, expected_outdata3.real)
    np.testing.assert_almost_equal(outdata3.imag, expected_outdata3.imag)

    outdata4 = np.loadtxt("output4.txt", dtype=np.complex128)
    t4 = np.real(np.transpose(outdata4)[0])
    outdata4 = np.transpose(outdata4)[1]
    expected_t4, expected_outdata4 = get_expected_output(input_dict4)
    np.testing.assert_almost_equal(t4, expected_t4)
    np.testing.assert_almost_equal(outdata4.real, expected_outdata4.real)
    np.testing.assert_almost_equal(outdata4.imag, expected_outdata4.imag)


def test_median_real():

    # Options for input data and element configuration
    input_dict1 = {}
    input_dict1["end"] = 8
    input_dict1["inrate"] = 16
    input_dict1["ngap"] = 5
    input_dict1["default_value"] = 0.0
    input_dict1["initialize_array"] = True
    input_dict1["median_overlap_samples"] = (8, 16)
    input_dict1["real_f"] = 0.125
    input_dict1["imag_f"] = 1.0
    input_dict1["real_amp"] = 2.0
    input_dict1["imag_amp"] = 0.0
    input_dict1["reject_zeros"] = True
    input_dict1["default_to_median"] = True

    # Try different combinations of options
    input_dict2 = input_dict1.copy()
    input_dict2["initialize_array"] = False
    input_dict3 = input_dict1.copy()
    input_dict3["reject_zeros"] = False
    input_dict4 = input_dict1.copy()
    input_dict4["initialize_array"] = False
    input_dict4["default_to_median"] = False

    pipeline = Pipeline()

    #
    #       ----------
    #      | src1     |
    #       ----------
    #              \
    #           H1  \ SR2
    #           ------------
    #          |   Median   |
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
        Median(
            name="median1",
            source_pad_names=("src",),
            sink_pad_names=("snk",),
            median_overlap_samples=input_dict1["median_overlap_samples"],
            default_value=input_dict1["default_value"],
            initialize_array=input_dict1["initialize_array"],
            reject_zeros=input_dict1["reject_zeros"],
            default_to_median=input_dict1["default_to_median"],
        ),
        Median(
            name="median2",
            source_pad_names=("src",),
            sink_pad_names=("snk",),
            median_overlap_samples=input_dict2["median_overlap_samples"],
            default_value=input_dict2["default_value"],
            initialize_array=input_dict2["initialize_array"],
            reject_zeros=input_dict2["reject_zeros"],
            default_to_median=input_dict2["default_to_median"],
        ),
        Median(
            name="median3",
            source_pad_names=("src",),
            sink_pad_names=("snk",),
            median_overlap_samples=input_dict3["median_overlap_samples"],
            default_value=input_dict3["default_value"],
            initialize_array=input_dict3["initialize_array"],
            reject_zeros=input_dict3["reject_zeros"],
            default_to_median=input_dict3["default_to_median"],
        ),
        Median(
            name="median4",
            source_pad_names=("src",),
            sink_pad_names=("snk",),
            median_overlap_samples=input_dict3["median_overlap_samples"],
            default_value=input_dict4["default_value"],
            initialize_array=input_dict4["initialize_array"],
            reject_zeros=input_dict4["reject_zeros"],
            default_to_median=input_dict4["default_to_median"],
        ),
        DumpSeriesSink(
            name="snk1",
            fname="output1.txt",
            sink_pad_names=("snk",),
        ),
        DumpSeriesSink(
            name="snk2",
            fname="output2.txt",
            sink_pad_names=("snk",),
        ),
        DumpSeriesSink(
            name="snk3",
            fname="output3.txt",
            sink_pad_names=("snk",),
        ),
        DumpSeriesSink(
            name="snk4",
            fname="output4.txt",
            sink_pad_names=("snk",),
        ),
        link_map={
            "ramp:snk:snk": "rsrc:src:src",
            "median1:snk:snk": "ramp:src:src",
            "snk1:snk:snk": "median1:src:src",
            "median2:snk:snk": "ramp:src:src",
            "snk2:snk:snk": "median2:src:src",
            "median3:snk:snk": "ramp:src:src",
            "snk3:snk:snk": "median3:src:src",
            "median4:snk:snk": "ramp:src:src",
            "snk4:snk:snk": "median4:src:src",
        },
    )

    pipeline.run()

    # Get the output data
    outdata1 = np.loadtxt("output1.txt")
    t1 = np.real(np.transpose(outdata1)[0])
    outdata1 = np.transpose(outdata1)[1]
    expected_t1, expected_outdata1 = get_expected_output(input_dict1)
    np.testing.assert_almost_equal(t1, expected_t1)
    np.testing.assert_almost_equal(outdata1, expected_outdata1)

    outdata2 = np.loadtxt("output2.txt")
    t2 = np.real(np.transpose(outdata2)[0])
    outdata2 = np.transpose(outdata2)[1]
    expected_t2, expected_outdata2 = get_expected_output(input_dict2)
    np.testing.assert_almost_equal(t2, expected_t2)
    np.testing.assert_almost_equal(outdata2, expected_outdata2)

    outdata3 = np.loadtxt("output3.txt")
    t3 = np.real(np.transpose(outdata3)[0])
    outdata3 = np.transpose(outdata3)[1]
    expected_t3, expected_outdata3 = get_expected_output(input_dict3)
    np.testing.assert_almost_equal(t3, expected_t3)
    np.testing.assert_almost_equal(outdata3, expected_outdata3)

    outdata4 = np.loadtxt("output4.txt")
    t4 = np.real(np.transpose(outdata4)[0])
    outdata4 = np.transpose(outdata4)[1]
    expected_t4, expected_outdata4 = get_expected_output(input_dict4)
    np.testing.assert_almost_equal(t4, expected_t4)
    np.testing.assert_almost_equal(outdata4, expected_outdata4)


if __name__ == "__main__":
    test_get_new_median()
    test_median_complex()
    test_median_real()
