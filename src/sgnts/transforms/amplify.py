from dataclasses import dataclass

from sgn import validator

from sgnts.base import TSCollectFrame, TSFrame, TSTransform
from sgnts.decorators import transform


@dataclass
class Amplify(TSTransform):
    """Amplify data by a factor.

    Args:
        factor:
            float, the factor to multiply the data with
    """

    factor: float = 1

    @validator.one_to_one
    def validate(self) -> None:
        pass

    @transform.one_to_one
    def process(self, input_frame: TSFrame, output_frame: TSCollectFrame) -> None:
        """Amplify non-gap data by the factor."""
        for buf in input_frame:
            if not buf.is_gap:
                assert buf.data is not None
                data = buf.data * self.factor
                buf = buf.copy(data=data)
            output_frame.append(buf)
