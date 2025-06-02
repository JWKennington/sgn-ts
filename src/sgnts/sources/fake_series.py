from __future__ import annotations

import collections
import time
from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np
from sgn.base import SourcePad, get_sgn_logger

from sgnts.base import Array, Offset, SeriesBuffer, TSFrame, TSSource
from sgnts.base.time import Time
from sgnts.utils import gpsnow

LOGGER = get_sgn_logger("sgn-ts")


@dataclass
class FakeSeriesSource(TSSource):
    """A time-series source that generates fake data in fixed-size buffers.

    If `t0` is not specified the current GPS time will be used as the
    start time.

    Args:
        signals:
            dict, keyed by source pad name, defining the signals to be
            produced on that pad.  The expected values are:

            signal_type:
                str, currently supported types: (1) 'white': white
                noise data. (2) 'sin' or 'sine': sine wave data. (3)
                'impulse': creates an impulse data, where the value is
                one at one sample point, and everywhere else is zero.
                (4) 'const': constant values as specified by user.
            rate:
                int, the sample rate of the data
            sample_shape:
                tuple[int, ...], the shape of a sample of data, or the
                shape of the data in each dimension except the last
                (time) dimension, i.e., sample_shape =
                data.shape[:-1]. For example, if the data is a
                multi-dimensional array and has shape=(2, 4, 16) then
                sample_shape = (2, 4).  Note that if data is one
                dimensional and has shape (16,), sample_shape would be
                an empty tuple ().
            fsin:
                float, sine wave frequency for 'sin' signals.
            impulse_position:
                int, impulse position for 'impulse' signals. If -1,
                then the impulse position will be random.
            const:
                int | float | tuple[float, ...], the constant int or
                float for output. If a tuple is given, the outputs
                will form a piecewise function, with an equal amount
                of time spent on each value in the tuple.

            These parameters may be specified directly as keyword
            arguments during class init, in which case they will be
            used as the defaults for undefined parameters in the
            signals dict.
        ngap:
            int, the frequency to generate gap buffers, will generate
            a gap buffer every ngap buffers. ngap=0: do not generate
            gap buffers. ngap=-1: generates gap buffers randomly.
        random_seed:
            int, set the random seed, used for 'white' and 'impulse'
            signals.
        real_time:
            bool, run the source in "real time", such that frames are
            produced at the rate corresponding to their relative
            offsets.  In real-time mode, t0 will default to the
            current GPS time if not otherwise specified.

    """

    signals: dict[str, dict[str, Any]] | None = None
    signal_type: str = "white"
    rate: int = 2048
    sample_shape: tuple[int, ...] = ()
    fsin: float = 5
    impulse_position: int = -1
    const: Union[int, float, tuple[float, ...]] = 1
    ngap: int = 0
    random_seed: Optional[int] = None
    real_time: bool = False

    def __post_init__(self):
        if self.signals is None:
            self.signals = {}

        if self.t0 is None:
            if self.real_time:
                # FIXME make this aligned to a general buffer (which depends on
                # rate) integers will always be aligned.
                self.t0 = int(gpsnow())
            else:
                self.t0 = 0

        super().__post_init__()

        self.cnt = {p: 0 for p in self.source_pads}

        # setup buffers this gives us the first timestamp / offset too
        for pad in self.source_pads:
            signal = self.signals.get(self.rsrcs[pad], {})
            sample_rate = signal.get("rate", self.rate)
            sample_shape = signal.get("sample_shape", self.sample_shape)
            self.set_pad_buffer_params(
                pad=pad, sample_shape=sample_shape, rate=sample_rate
            )

        # This is gauranteed to be the t0 of the element at this point
        self._start_time = self.current_t0

        if self.random_seed is not None:
            np.random.seed(self.random_seed)

    def create_data(self, pad: SourcePad, buf: SeriesBuffer) -> Array:
        """Create the fake data

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
        end_offset = buf.end_offset
        ngap = self.ngap
        cnt = self.cnt[pad]
        metadata: dict[str, int] = {}

        assert self.signals is not None

        signal = self.signals.get(self.rsrcs[pad], {})
        signal_type = signal.get("signal_type", self.signal_type)

        if (ngap == -1 and np.random.rand(1) > 0.5) or (ngap > 0 and cnt % ngap == 0):
            data = None
        elif signal_type == "white":
            data = np.random.randn(*buf.shape)
        elif signal_type in ["sin", "sine"]:
            data = np.sin(
                2
                * np.pi
                * signal.get("fsin", self.fsin)
                * np.tile(
                    buf.tarr,
                    buf.sample_shape + (1,),
                ),
            )
        elif signal_type == "impulse":
            # return self.create_impulse_data(offset, buf.samples, buf.sample_rate)
            impulse_position = signal.get("impulse_position", self.impulse_position)
            if impulse_position == -1 and self.end is not None:
                impulse_position = np.random.randint(0, int(self.end * buf.sample_rate))
            data = np.zeros(buf.samples)
            current_samples = Offset.tosamples(offset, buf.sample_rate)
            if (
                current_samples <= impulse_position
                and impulse_position < current_samples + buf.samples
            ):
                data[impulse_position - current_samples] = 1
            metadata["impulse_offset"] = Offset.fromsamples(
                impulse_position, buf.sample_rate
            )
        elif signal_type == "const":
            if isinstance(self.const, collections.abc.Iterable):
                data = np.array([])
                buffer_t0 = Offset.tosec(offset)
                buffer_end = Offset.tosec(end_offset)
                if self.t0 is not None:
                    t0 = self.t0
                else:
                    t0 = 0.0
                if self.end is not None:
                    piecewise_cycle_dur = self.end - t0
                else:
                    piecewise_cycle_dur = 100.0
                const_dur = piecewise_cycle_dur / len(self.const)
                first_transition = buffer_t0 - (buffer_t0 - t0) % const_dur
                last_transition = buffer_end + const_dur - (buffer_end - t0) % const_dur
                transition_times = np.arange(
                    first_transition, last_transition + 1, const_dur
                )
                for i in range(len(transition_times)):
                    t_now = buffer_t0 + len(data) / buf.sample_rate
                    if t_now >= transition_times[i] and t_now < transition_times[i + 1]:
                        samples_til_next_transition = int(
                            (transition_times[i + 1] - t_now) * buf.sample_rate
                        )
                        current_const = self.const[
                            int(
                                (
                                    (transition_times[i] + transition_times[i + 1]) / 2
                                    - t0
                                )
                                / const_dur
                            )
                            % len(self.const)
                        ]
                        remaining_buf_samples = buf.samples - len(data)
                        if samples_til_next_transition >= remaining_buf_samples:
                            data = np.append(
                                data, np.tile(current_const, remaining_buf_samples)
                            )
                            break
                        else:
                            data = np.append(
                                data,
                                np.tile(current_const, samples_til_next_transition),
                            )
                if len(data) < buf.samples:
                    data = np.append(
                        data, np.tile(self.const[-1], buf.samples - len(data))
                    )
            else:
                data = np.full(buf.shape, self.const)
        else:
            msg = f"Unknown signal type '{signal_type}'."
            raise ValueError(msg)

        return data, metadata

    def internal(self):
        super().internal()

        if self.real_time:
            # in real-time mode we want to "release" the data after
            # the time of the last sample in the output frame.
            sleep = self.current_end / Time.SECONDS - gpsnow()
            if sleep < 0:
                if sleep < -1:
                    LOGGER.warning(
                        "Warning: FakeSeriesSource falling behind real time (%.2f s)",
                        sleep,
                    )
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

        metadata = {"name": f"{self.rsrcs[pad]}", "cnt": self.cnt[pad]}

        frame = self.prepare_frame(pad, data=None, metadata=metadata)
        for buf in frame:
            data, _metadata = self.create_data(pad, buf)
            buf.set_data(data)
            metadata.update(_metadata)

        return frame
