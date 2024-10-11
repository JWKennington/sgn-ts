from dataclasses import dataclass

import numpy as np

from ..base import Offset, SeriesBuffer, TSFrame, TSSource, TSSlice, TSSlices
from .fake_realtime import RealTimeWhiteNoiseSrc


@dataclass
class FakeSeriesSrc(TSSource):
    """
    A time-series source that generates fake data in fixed-size buffers.

    Parameters:
    -----------
    num_buffers: int
        is required and sets how many buffers will be created before setting "EOS"
    rate: int
        the sample rate of the data
    channels: tuple
        the number channels of the data in each dimension except the last, i.e.,
        channels = data.shape[:-1]. If data has shape (A, B, N), then channels =
        (A, B). Note that if data is one dimensional and has shape (N,), channels
        would be an empty tuple ().
    signal_type: str
        currently supported types: (1) 'white': white noise data. (2) 'sin' or 'sine':
        sine wave data. (3) 'impulse': creates an impulse data, where the value is one
        at one sample point, and everywhere else is zero
    fsin: float
        frequency of the sine wave if signal_type = 'sin'
    ngap: int
        Frequency of gap buffers, will generate a gap buffer every ngap buffers.
        ngap=0: do not generate gap buffers.
        ngap=-1: generates gap buffers randomly.
    random_seed: int
        set the random seed, used for signal_type = 'white' or 'impulse'
    impulse_position: int
        The sample point position to place the impulse. If None, then the impulse
        will be generated randomly.
    """

    num_buffers: int = 0
    rate: int = 2048
    channels: tuple = ()
    signal_type: str = "white"
    fsin: float = 5
    ngap: int = 0
    random_seed: int = None
    impulse_position: int = None
    verbose: bool = False

    def __post_init__(self):
        super().__post_init__()
        self.cnt = {p: 0 for p in self.source_pads}
        self.shape = self.channels + (self.num_samples,)
        if self.random_seed is not None and (
            self.signal_type == "white" or self.signal_type == "impulse"
        ):
            np.random.seed(self.random_seed)
        if self.signal_type == "impulse":
            assert len(self.shape) == 1
            # self.current_samples = 0
            if self.impulse_position is None:
                self.impulse_position = np.random.randint(
                    0, self.num_buffers * self.num_samples
                )
            if self.verbose:
                print("Placing impulse at sample point", self.impulse_position)

    def create_impulse_data(self, offset):
        data = np.zeros(self.num_samples)
        current_samples = Offset.tosamples(offset, self.rate)
        if (
            current_samples <= self.impulse_position
            and self.impulse_position < current_samples + self.num_samples
        ):
            if self.verbose:
                print("Creating the impulse")
            data[self.impulse_position - current_samples] = 1
        return data

    def create_data(self, offset):
        if self.signal_type == "white":
            return np.random.randn(*self.shape)
        elif self.signal_type == "sin" or self.signal_type == "sine":
            t0 = Offset.tosec(offset)
            duration = self.num_samples / self.rate
            return np.sin(
                self.fsin
                * np.tile(
                    np.linspace(t0, t0 + duration, self.shape[-1], endpoint=False),
                    self.channels + (1,),
                )
            )
        elif self.signal_type == "impulse":
            return self.create_impulse_data(offset)
        else:
            raise ValueError("Unknown signal type")

    def new(self, pad):
        """
        New buffers are created on "pad" with an instance specific count and a
        name derived from the pad name. "EOS" is set if we have surpassed the requested
        number of buffers.
        """
        self.cnt[pad] += 1
        ngap = self.ngap
        if (ngap == -1 and np.random.rand(1) > 0.5) or (
            ngap > 0 and self.cnt[pad] % ngap == 0
        ):
            data = None
        else:
            data = self.create_data(self.offset[pad])

        outbuf = SeriesBuffer(
            offset=self.offset[pad], sample_rate=self.rate, data=data, shape=self.shape
        )

        self.offset[pad] += Offset.fromsamples(self.num_samples, self.rate)
        metadata = {"cnt": self.cnt, "name": "'%s'" % pad.name}
        if self.impulse_position is not None:
            metadata["impulse_offset"] = Offset.fromsamples(
                self.impulse_position, self.rate
            )

        return TSFrame(
            buffers=[outbuf],
            metadata=metadata,
            EOS=self.cnt[pad] > self.num_buffers,
        )


@dataclass
class SegmentSrc(TSSource):
    """

    Parameters:
    -----------
    rate: int
        the sample rate of the data
    segments: tuple
        A tuple of segment tuples corresponding to time in ns
    end: int
        The time at which to stop producing buffers
    """

    rate: int = 2048
    segments: tuple = None
    end: float = None

    def __post_init__(self):
        assert isinstance(self.end, float)
        assert self.segments is not None
        super().__post_init__()
        # FIXME
        self.segments = TSSlices(
            TSSlice(Offset.fromns(s[0]), Offset.fromns(s[1]))
            for s in self.segments
            if (s[0] >= self.t0 * 1e9 and s[1] <= self.end * 1e9)
        ).simplify()

    def new(self, pad):
        """ """
        frame_slice = TSSlice(
            self.offset[pad],
            self.offset[pad] + Offset.fromsamples(self.num_samples, self.rate),
        )
        nongap_slices = self.segments.search(frame_slice)
        gap_slices = nongap_slices.invert(frame_slice)
        outbufs = [
            SeriesBuffer.fromoffsetslice(s, self.rate) for s in gap_slices.slices
        ]
        outbufs.extend(
            [
                SeriesBuffer.fromoffsetslice(s, self.rate, data=1)
                for s in nongap_slices.slices
            ]
        )
        outbufs = sorted(outbufs)

        self.offset[pad] = frame_slice.stop
        return TSFrame(buffers=outbufs, EOS=outbufs[-1].end >= self.end * 1e9)  # FIXME
