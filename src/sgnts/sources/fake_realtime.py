import time
from dataclasses import dataclass

import numpy as np

from ..base import Offset, SeriesBuffer, TSFrame, TSSource


@dataclass
class RealTimeWhiteNoiseSrc(TSSource):
    """
    A time-series source that generates fake data in fixed-size buffers in real-time
    """

    num_buffers: int = 10

    def __post_init__(self):
        super().__post_init__()
        self.cnt = {p: 0 for p in self.source_pads}
        self.shape = (self.num_samples,)
        if self.rate != self.num_samples:
            raise ValueError(
                f"Can only produce 1 second buffers, rate={self.rate}"
                f" num_samples={self.num_samples}"
            )
        self.next_time = None

    def new(self, pad):
        self.cnt[pad] += 1

        # Produce buffers at every integer second time
        now = time.time()
        if self.next_time is None:
            self.next_time = int(now) + 1

        sleep = self.next_time - now
        if sleep > 0:
            # There might be cases where sleep < 0 and we are behind? In that case
            # don't sleep
            time.sleep(sleep)
        self.next_time = int(time.time()) + 1

        data = np.random.randn(self.num_samples)

        outbuf = SeriesBuffer(
            offset=self.offset[pad], sample_rate=self.rate, data=data, shape=self.shape
        )

        self.offset[pad] += Offset.fromsamples(self.num_samples, self.rate)
        metadata = {"cnt": self.cnt, "name": "'%s'" % pad.name}

        return TSFrame(
            buffers=[outbuf],
            metadata=metadata,
            EOS=self.cnt[pad] >= self.num_buffers,
        )
