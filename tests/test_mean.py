#!/usr/bin/env python3

import numpy as np
from sgn.apps import Pipeline
from sgnts.sinks import DumpSeriesSink
from sgnts.sources import FakeSeriesSource
from sgnts.transforms import Adder, Amplify, Mean


def test_mean():

    end = 8
    inrate = 16
    default_value = 3 - 1j
    mean_overlap_samples = (3, 0)
    real_f = 0.125
    imag_f = 1.0
    real_amp = 2.0
    imag_amp = -5.0
    t = np.linspace(0, end - 1.0 / inrate, end * inrate)
    indata_real = real_amp * np.sin(2 * np.pi * real_f * t)
    indata_imag = imag_amp * np.sin(2 * np.pi * imag_f * t)
    indata = indata_real + 1j * indata_imag
    indata[0] = default_value

    pipeline = Pipeline()

    #
    #       ----------
    #      | src1     |
    #       ----------
    #              \
    #           H1  \ SR2
    #           ------------
    #          |    Mean    |
    #           ------------
    #                 \
    #             H1   \ SR2
    #             ---------
    #            | snk1    |
    #             ---------

    ngap = end // 2 + 1

    pipeline.insert(
        FakeSeriesSource(
            name="rsrc",
            source_pad_names=("src",),
            rate=inrate,
            ngap=ngap,
            signal_type="sin",
            fsin=real_f,
            end=end,
        ),
        FakeSeriesSource(
            name="isrc",
            source_pad_names=("src",),
            rate=inrate,
            ngap=ngap,
            signal_type="sin",
            fsin=imag_f,
            end=end,
        ),
        Amplify(
            name="ramp",
            source_pad_names=("src",),
            sink_pad_names=("snk",),
            factor=real_amp + 0j,
        ),
        Amplify(
            name="iamp",
            source_pad_names=("src",),
            sink_pad_names=("snk",),
            factor=1j * imag_amp,
        ),
        Adder(
            name="adder",
            sink_pad_names=("rsnk", "isnk"),
            source_pad_names=("src",),
        ),
        Mean(
            name="mean",
            source_pad_names=("src",),
            sink_pad_names=("snk",),
            mean_overlap_samples=mean_overlap_samples,
            default_value=default_value,
        ),
        DumpSeriesSink(
            name="snk",
            fname="output.txt",
            sink_pad_names=("snk",),
        ),
        link_map={
            "ramp:snk:snk": "rsrc:src:src",
            "iamp:snk:snk": "isrc:src:src",
            "adder:snk:rsnk": "ramp:src:src",
            "adder:snk:isnk": "iamp:src:src",
            "mean:snk:snk": "adder:src:src",
            "snk:snk:snk": "mean:src:src",
        },
    )

    pipeline.run()

    # Get the output data
    outdata = np.loadtxt("output.txt", dtype=np.complex128)
    t = np.real(np.transpose(outdata)[0])
    outdata = np.transpose(outdata)[1]

    # Check that the times include the expected latency
    t_start = -mean_overlap_samples[1] / 16
    t_end = t_start + end - 1.0 / inrate
    np.testing.assert_almost_equal(t, np.linspace(t_start, t_end, end * inrate))

    # Compute the expected output data
    expected_outdata = np.empty(end * inrate, dtype=np.complex128)
    n_mean = 1 + sum(mean_overlap_samples)
    current_mean = default_value
    mean_array = np.tile(current_mean, n_mean)
    # Before the gap
    for idx in range((ngap - 1) * inrate):
        mean_array[idx % n_mean] = indata[idx]
        current_mean = np.mean(mean_array)
        expected_outdata[idx] = current_mean
    # During the gap
    for idx in range((ngap - 1) * inrate, ngap * inrate):
        mean_array[idx % n_mean] = current_mean
        current_mean = np.mean(mean_array)
        expected_outdata[idx] = current_mean
    # After the gap
    for idx in range(ngap * inrate, end * inrate):
        mean_array[idx % n_mean] = indata[idx]
        current_mean = np.mean(mean_array)
        expected_outdata[idx] = current_mean

    np.testing.assert_almost_equal(outdata.real, expected_outdata.real)
    np.testing.assert_almost_equal(outdata.imag, expected_outdata.imag)


if __name__ == "__main__":
    test_mean()
