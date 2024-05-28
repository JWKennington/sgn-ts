#!/usr/bin/env python3

from sgn.apps import Pipeline

from sgnts.sinks import FakeSeriesSink
from sgnts.transforms import Sync
from sgnts.sources import FakeSeriesSrc


def test_sync(capsys):

    pipeline = Pipeline()

    #
    #       ----------    -------   --------
    #      | src1     |  | src2  | | src3   |
    #       ----------    -------   --------
    #              \         |      /
    #           H1  \     L1 |     / V1
    #               ----------------
    #              | sync           |
    #               ----------------
    #                 |        |    \
    #             H1  |      L1|     \ V1
    #           ---------   -------   --------
    #          | snk1    | | snk2  | |  snk3  |
    #           ---------   -------   --------

    mode = "pad"
    inrate = 256
    duration = 1
    num_buffers = 6
    H1_t0 = 0
    L1_t0 = 3
    V1_t0 = 5
    print(f"{mode=} {H1_t0=}, {L1_t0=}, {V1_t0=}")
    if mode == "pad":
        print_message = "f' duration {bufs[-1].duration} data_is_all_zeros {not np.any(bufs[-1].data)}'"
    elif mode == "drop":
        print_message = (
            "f' duration {bufs[-1].duration} data_is_none {bufs[-1].data is None}'"
        )

    H1_duration = 1
    L1_duration = 1
    V1_duration = 1
    pipeline.insert(
        FakeSeriesSrc(
            name="src1",
            source_pad_names=("H1",),
            num_buffers=num_buffers,
            rate=inrate,
            duration=H1_duration,
            t0=H1_t0,
        ),
        Sync(
            name="trans1",
            pad_names_map={"H1":"H1","L1":"L1","V1":"V1"},
            mode=mode,
        ),
        FakeSeriesSink(
            name="snk1",
            sink_pad_names=("H1",),
            print_message=print_message,
        ),
        FakeSeriesSrc(
            name="src2",
            source_pad_names=("L1",),
            num_buffers=num_buffers,
            rate=inrate,
            duration=L1_duration,
            t0=L1_t0,
        ),
        FakeSeriesSink(
            name="snk2",
            sink_pad_names=("L1",),
            print_message=print_message,
        ),
        FakeSeriesSrc(
            name="src3",
            source_pad_names=("V1",),
            num_buffers=num_buffers,
            rate=inrate,
            duration=V1_duration,
            t0=V1_t0,
        ),
        FakeSeriesSink(
            name="snk3",
            sink_pad_names=("V1",),
            print_message=print_message,
        ),
        link_map={
            "trans1:sink:H1": "src1:src:H1",
            "snk1:sink:H1": "trans1:src:H1",
            "trans1:sink:L1": "src2:src:L1",
            "snk2:sink:L1": "trans1:src:L1",
            "trans1:sink:V1": "src3:src:V1",
            "snk3:sink:V1": "trans1:src:V1",
        }
    )

    pipeline.run()
    if capsys is not None:
        captured = capsys.readouterr()
        if mode == "pad":
            assert (
                captured.out.strip()
                == """
mode='pad' H1_t0=0, L1_t0=3, V1_t0=5
buffer flow:  'src1:src:H1'+'src2:src:L1'+'src3:src:V1' -> 'trans1:src:H1' -> 'snk1:sink:H1' offset 0 time 0 duration 1000000000 data_is_all_zeros False
buffer flow:  'src1:src:H1'+'src2:src:L1'+'src3:src:V1' -> 'trans1:src:L1' -> 'snk2:sink:L1' offset 0 time 0 duration 1000000000 data_is_all_zeros True
buffer flow:  'src1:src:H1'+'src2:src:L1'+'src3:src:V1' -> 'trans1:src:V1' -> 'snk3:sink:V1' offset 0 time 0 duration 1000000000 data_is_all_zeros True
buffer flow:  'src1:src:H1'+'src2:src:L1'+'src3:src:V1' -> 'trans1:src:H1' -> 'snk1:sink:H1' offset 16384 time 1000000000 duration 1000000000 data_is_all_zeros False
buffer flow:  'src1:src:H1'+'src2:src:L1'+'src3:src:V1' -> 'trans1:src:L1' -> 'snk2:sink:L1' offset 16384 time 1000000000 duration 1000000000 data_is_all_zeros True
buffer flow:  'src1:src:H1'+'src2:src:L1'+'src3:src:V1' -> 'trans1:src:V1' -> 'snk3:sink:V1' offset 16384 time 1000000000 duration 1000000000 data_is_all_zeros True
buffer flow:  'src1:src:H1'+'src2:src:L1'+'src3:src:V1' -> 'trans1:src:H1' -> 'snk1:sink:H1' offset 32768 time 2000000000 duration 1000000000 data_is_all_zeros False
buffer flow:  'src1:src:H1'+'src2:src:L1'+'src3:src:V1' -> 'trans1:src:L1' -> 'snk2:sink:L1' offset 32768 time 2000000000 duration 1000000000 data_is_all_zeros True
buffer flow:  'src1:src:H1'+'src2:src:L1'+'src3:src:V1' -> 'trans1:src:V1' -> 'snk3:sink:V1' offset 32768 time 2000000000 duration 1000000000 data_is_all_zeros True
buffer flow:  'src1:src:H1'+'src2:src:L1'+'src3:src:V1' -> 'trans1:src:H1' -> 'snk1:sink:H1' offset 49152 time 3000000000 duration 1000000000 data_is_all_zeros False
buffer flow:  'src1:src:H1'+'src2:src:L1'+'src3:src:V1' -> 'trans1:src:L1' -> 'snk2:sink:L1' offset 49152 time 3000000000 duration 1000000000 data_is_all_zeros False
buffer flow:  'src1:src:H1'+'src2:src:L1'+'src3:src:V1' -> 'trans1:src:V1' -> 'snk3:sink:V1' offset 49152 time 3000000000 duration 1000000000 data_is_all_zeros True
buffer flow:  'src1:src:H1'+'src2:src:L1'+'src3:src:V1' -> 'trans1:src:H1' -> 'snk1:sink:H1' offset 65536 time 4000000000 duration 1000000000 data_is_all_zeros False
buffer flow:  'src1:src:H1'+'src2:src:L1'+'src3:src:V1' -> 'trans1:src:L1' -> 'snk2:sink:L1' offset 65536 time 4000000000 duration 1000000000 data_is_all_zeros False
buffer flow:  'src1:src:H1'+'src2:src:L1'+'src3:src:V1' -> 'trans1:src:V1' -> 'snk3:sink:V1' offset 65536 time 4000000000 duration 1000000000 data_is_all_zeros True
buffer flow:  'src1:src:H1'+'src2:src:L1'+'src3:src:V1' -> 'trans1:src:H1' -> 'snk1:sink:H1' offset 81920 time 5000000000 duration 1000000000 data_is_all_zeros False
buffer flow:  'src1:src:H1'+'src2:src:L1'+'src3:src:V1' -> 'trans1:src:L1' -> 'snk2:sink:L1' offset 81920 time 5000000000 duration 1000000000 data_is_all_zeros False
buffer flow:  'src1:src:H1'+'src2:src:L1'+'src3:src:V1' -> 'trans1:src:V1' -> 'snk3:sink:V1' offset 81920 time 5000000000 duration 1000000000 data_is_all_zeros False
buffer flow:  'src1:src:H1'+'src2:src:L1'+'src3:src:V1' -> 'trans1:src:H1' -> 'snk1:sink:H1' offset 98304 time 6000000000 duration 1000000000 data_is_all_zeros False
buffer flow:  'src1:src:H1'+'src2:src:L1'+'src3:src:V1' -> 'trans1:src:L1' -> 'snk2:sink:L1' offset 98304 time 6000000000 duration 1000000000 data_is_all_zeros False
buffer flow:  'src1:src:H1'+'src2:src:L1'+'src3:src:V1' -> 'trans1:src:V1' -> 'snk3:sink:V1' offset 98304 time 6000000000 duration 1000000000 data_is_all_zeros False
""".strip()
            )


if __name__ == "__main__":
    test_sync(None)
