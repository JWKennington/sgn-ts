from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
from sgn.base import SourcePad, get_sgn_logger

from sgnts.base import Array, Offset, SeriesBuffer, TSFrame, TSSource
from sgnts.utils import gpsnow

LOGGER = get_sgn_logger("sgn-ts")


@dataclass
class FakeSeriesSource(TSSource):
    """A time-series source that generates fake data in fixed-size buffers.

    If `t0` is not specified the current GPS time will be used as the
    start time.

    Args:
        rate:
            int, the sample rate of the data
        sample_shape:
            tuple[int, ...], the shape of a sample of data, or the
            shape of the data in each dimension except the last (time)
            dimension, i.e., sample_shape = data.shape[:-1]. For
            example, if the data is a multi-dimensional array and has
            shape=(2, 4, 16) then sample_shape = (2, 4).  Note that if
            data is one dimensional and has shape (16,), sample_shape
            would be an empty tuple ().
        signal_type:
            str, currently supported types: (1) 'white': white noise data. (2) 'sin' or
            'sine': sine wave data. (3) 'impulse': creates an impulse data, where the
            value is one at one sample point, and everywhere else is zero
        fsin:
            float, the frequency of the sine wave if signal_type = 'sin'
        ngap:
            int, the frequency to generate gap buffers, will generate a gap buffer every
            ngap buffers. ngap=0: do not generate gap buffers. ngap=-1: generates gap
            buffers randomly.
        random_seed:
            int, set the random seed, used for signal_type = 'white' or 'impulse'
        impulse_position:
            int, the sample point position to place the impulse. If -1, then the
            impulse will be generated randomly.
        real_time:
            bool, run the source in "real time", such that frames are
            produced at the rate corresponding to their relative
            offsets.  In real-time mode, t0 will default to the
            current GPS time if not otherwise specified.

    """

    rate: int = 2048
    sample_shape: tuple[int, ...] = ()
    signal_type: str = "white"
    fsin: float = 5
    ngap: int = 0
    random_seed: Optional[int] = None
    impulse_position: int = -1
    real_time: bool = False
    verbose: bool = False

    def __post_init__(self):
        # this variable is used for real time tracking when t0 is not
        # the current time
        self._start_offset_from_realtime = None

        if self.real_time and self.t0 is None:
            self.t0 = int(gpsnow())
            self._start_offset_from_realtime = 0

        super().__post_init__()

        self.cnt = {p: 0 for p in self.source_pads}

        # setup buffers
        for pad in self.source_pads:
            self.set_pad_buffer_params(
                pad=pad, sample_shape=self.sample_shape, rate=self.rate
            )

        if self.random_seed is not None and (
            self.signal_type == "white" or self.signal_type == "impulse"
        ):
            np.random.seed(self.random_seed)
        if self.signal_type == "impulse":
            assert self.sample_shape == ()
            assert len(self.source_pads) == 1
            if self.impulse_position == -1:
                self.impulse_position = np.random.randint(0, int(self.end * self.rate))
            if self.verbose:
                print("Placing impulse at sample point", self.impulse_position)

        if self._start_offset_from_realtime is None:
            self._start_offset_from_realtime = gpsnow() - self.t0

    def create_impulse_data(self, offset: int, num_samples: int, rate: int) -> Array:
        """Create the impulse data, where data is zero everywhere, and equals one at one
        sample point.

        Args:
            offset:
                int, the offset of the current buffer, used for checking whether the
                the impulse is to be placed in the current buffer
            num_samples:
                int, the number of samples the data should have
            rate:
                int, the sample rate of the data

        Returns:
            Array, the impulse data
        """
        data = np.zeros(num_samples)
        current_samples = Offset.tosamples(offset, rate)
        if (
            current_samples <= self.impulse_position
            and self.impulse_position < current_samples + num_samples
        ):
            if self.verbose:
                print("Creating the impulse")
            data[self.impulse_position - current_samples] = 1
        return data

    def create_data(self, buf: SeriesBuffer, cnt: int) -> Array:
        """Create the fake data, can be (1) white noise (2) sine wave (3) impulse data.

        Args:
            buf:
                SeriesBuffer, the buffer to create the data for
            cnt:
                int, the number of buffers the source pad has generated, used for
                determining whether to generate gap buffers

        Returns:
            Array, the fake data array
        """
        offset = buf.offset
        ngap = self.ngap
        if (ngap == -1 and np.random.rand(1) > 0.5) or (ngap > 0 and cnt % ngap == 0):
            return None
        elif self.signal_type == "white":
            return np.random.randn(*buf.shape)
        elif self.signal_type == "sin" or self.signal_type == "sine":
            t0 = Offset.tosec(offset)
            duration = buf.duration
            return np.sin(
                self.fsin
                * np.tile(
                    np.linspace(t0, t0 + duration, buf.samples, endpoint=False),
                    self.sample_shape + (1,),
                )
            )
        elif self.signal_type == "impulse":
            return self.create_impulse_data(offset, buf.samples, buf.sample_rate)
        else:
            raise ValueError("Unknown signal type")

    def internal(self):
        if self.real_time:
            # in real-time mode we want to "release" the data after
            # the time of the last sample in the output frame. this
            # should correspond to the current offset value plus the
            # offset stride (what will be the start time of the next
            # frame)
            #
            # all pads should have the same offset so just take the
            # offset from the first pad.  FIXME: is there a better
            # way to get the current frame offset?
            next_offset = (
                list(self.offset.values())[0] + Offset.SAMPLE_STRIDE_AT_MAX_RATE
            )
            next_time = Offset.tosec(next_offset)
            # we then need to shift the next time to account for t0 that
            # are not "now"
            sleep = next_time + self._start_offset_from_realtime - gpsnow()
            if sleep < 0:
                LOGGER.warning("Warning: FakeSeriesSource falling behind real time")
            else:
                time.sleep(sleep)

    def new(self, pad: SourcePad) -> TSFrame:
        """New buffers are created on "pad" with an instance specific count and a name
        derived from the pad name. "EOS" is set if we have surpassed the requested
        end time.

        Args:
            Pad:
                SourcePad, the source pad to generate TSFrames

        Returns:
            TSFrame, the TSFrame that carries the buffers with fake data
        """
        self.cnt[pad] += 1

        # setup metadata
        metadata = {"cnt": self.cnt, "name": "'%s'" % pad.name}
        if self.impulse_position is not None:
            metadata["impulse_offset"] = Offset.fromsamples(
                self.impulse_position, self.rate
            )

        frame = self.prepare_frame(pad, data=None, metadata=metadata)
        for buf in frame:
            buf.set_data(self.create_data(buf, self.cnt[pad]))

        return frame
