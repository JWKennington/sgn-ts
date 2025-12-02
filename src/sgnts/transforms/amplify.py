from dataclasses import dataclass

from sgn import validator

from sgnts.base import TSTransform


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

    def internal(self) -> None:
        """Amplify non-gap data by the factor."""
        super().internal()

        _, input_frame = self.next_input()
        _, output_frame = self.next_output()

        for buf in input_frame:
            if not buf.is_gap:
                assert buf.data is not None
                data = buf.data * self.factor
                buf = buf.replace(data=data)
            output_frame.append(buf)
