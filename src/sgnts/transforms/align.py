from dataclasses import dataclass

from sgn import validator
from sgn.base import SourcePad

from sgnts.base import TSFrame, TSTransform


@dataclass
class Align(TSTransform):
    """Align frames from multiple sink pads."""

    def configure(self) -> None:
        self.pad_map = {
            src_pad: self.snks[src_pad.pad_name] for src_pad in self.source_pads
        }

    @validator.pad_names_match
    def validate(self) -> None:
        pass

    def new(self, pad: SourcePad) -> TSFrame:
        out = self.preparedframes[self.pad_map[pad]]
        self.preparedframes[self.pad_map[pad]] = None
        return out
