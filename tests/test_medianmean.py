#!/usr/bin/env python3

import numpy as np
from sgn.apps import Pipeline
from sgnts.sinks import DumpSeriesSink
from sgnts.sources import FakeSeriesSource
from sgnts.transforms import MedianMean


def test_medianmean():

    indata = (
        1 + np.sin(np.linspace(0, 7, 128)) + 1j * (-1 - np.cos(np.linspace(0, 23, 128)))
    )
    end = 8
    inrate = 16
    default_real = 3
    default_imag = -1
    median_overlap_samples = (8, 16)
    mean_overlap_samples = (3, 0)

    pipeline = Pipeline()

    #
    #       ----------
    #      | src1     |
    #       ----------
    #              \
    #           H1  \ SR2
    #           ------------
    #          | MedianMean |
    #           ------------
    #                 \
    #             H1   \ SR2
    #             ---------
    #            | snk1    |
    #             ---------

    ngap = end // 2 + 1

    pipeline.insert(
        FakeSeriesSource(
            name="src",
            source_pad_names=("src",),
            rate=inrate,
            ngap=ngap,
            signal_type="const",
            const=indata,
            end=end,
        ),
        MedianMean(
            name="medianmean",
            source_pad_names=("src",),
            sink_pad_names=("snk",),
            median_overlap_samples=median_overlap_samples,
            mean_overlap_samples=mean_overlap_samples,
            default_real=default_real,
            default_imag=default_imag,
        ),
        DumpSeriesSink(
            name="snk",
            fname="output.txt",
            sink_pad_names=("snk",),
        ),
        link_map={
            "medianmean:snk:snk": "src:src:src",
            "snk:snk:snk": "medianmean:src:src",
        },
    )

    pipeline.run()

    # Get the output data
    outdata = np.loadtxt("output.txt", dtype=np.complex_)
    t = np.real(np.transpose(outdata)[0])
    outdata = np.transpose(outdata)[1]

    # Check that the times include the expected latency
    t_start = -(median_overlap_samples[1] + mean_overlap_samples[1]) / 16
    t_end = t_start + end - 1.0 / inrate
    np.testing.assert_almost_equal(t, np.linspace(t_start, t_end, end * inrate))

    # Compute the expected output data
    expected_outdata = np.empty(end * inrate, dtype=np.complex128)
    n_median = 1 + sum(median_overlap_samples)
    n_mean = 1 + sum(mean_overlap_samples)
    current_median = default_real + 1j * default_imag
    median_array = np.tile(current_median, n_median)
    mean_array = np.tile(current_median, n_mean)
    # Before the gap
    for idx in range((ngap - 1) * inrate):
        median_array[idx % n_median] = indata[idx]
        current_median = np.median(median_array.real) + 1j * np.median(
            median_array.imag
        )
        mean_array[idx % n_mean] = current_median
        expected_outdata[idx] = np.mean(mean_array)
    # During the gap
    for idx in range((ngap - 1) * inrate, ngap * inrate):
        median_array[idx % n_median] = current_median
        mean_array[idx % n_mean] = current_median
        expected_outdata[idx] = np.mean(mean_array)
    # After the gap
    for idx in range(ngap * inrate, end * inrate):
        median_array[idx % n_median] = indata[idx]
        current_median = np.median(median_array.real) + 1j * np.median(
            median_array.imag
        )
        mean_array[idx % n_mean] = current_median
        expected_outdata[idx] = np.mean(mean_array)

    np.testing.assert_almost_equal(outdata.real, expected_outdata.real)
    np.testing.assert_almost_equal(outdata.imag, expected_outdata.imag)


if __name__ == "__main__":
    test_medianmean()
