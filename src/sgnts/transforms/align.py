from dataclasses import dataclass

from sgn import validator
from sgn.base import SinkPad, SourcePad

from sgnts.base import TSCollectFrame, TSFrame, TSTransform


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

    def process(
        self,
        input_frames: dict[SinkPad, TSFrame],
        output_frames: dict[SourcePad, TSCollectFrame],
    ) -> None:
        """Pass through frames from sink to source."""
        # just pass through frames from sink to source
        for src_pad, sink_pad in self.pad_map.items():
            output_frames[src_pad].extend(input_frames[sink_pad])
