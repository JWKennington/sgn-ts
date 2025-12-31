from dataclasses import dataclass, field
from typing import Callable

from sgn.base import SinkPad

from sgnts.base import SeriesBuffer, TSFrame, TSSink

# Type alias for callback signature
BufferCallback = Callable[[SinkPad, SeriesBuffer], None]


@dataclass
class TSAppSink(TSSink):
    """
    Application sink that executes user-defined callbacks on each buffer.

    Similar to GStreamer's appsink, this element allows application code
    to receive and process buffers via callbacks.

    Callbacks are registered per-pad and are invoked for each SeriesBuffer
    that arrives on that pad.

    Example:
        def my_callback(pad: SinkPad, buf: SeriesBuffer) -> None:
            print(f"Received buffer on {pad.name}: {buf.slice}")

        # Via constructor
        sink = TSAppSink(
            name="app_sink",
            sink_pad_names=("H1", "L1"),
            callbacks={"H1": my_callback}
        )

        # Via method
        sink.set_callback("L1", another_callback)

    Args:
        callbacks: Optional dict mapping pad names to callback functions.
        emit_gaps: If True, callbacks are also invoked for gap buffers.
                   Default is True (gap buffers are included).
    """

    callbacks: dict[str, BufferCallback] = field(default_factory=dict)
    emit_gaps: bool = True

    # Internal storage for registered callbacks
    _callbacks: dict[SinkPad, BufferCallback] = field(
        default_factory=dict, init=False, repr=False
    )

    def configure(self) -> None:
        """Register callbacks from constructor dict."""
        super().configure()

        # Convert name-based callbacks to pad-based
        for pad_name, callback in self.callbacks.items():
            if pad_name not in self.snks:
                raise ValueError(
                    f"Unknown pad name '{pad_name}'. "
                    f"Available pads: {list(self.snks.keys())}"
                )
            self._callbacks[self.snks[pad_name]] = callback

    def set_callback(self, pad_name: str, callback: BufferCallback) -> None:
        """
        Register a callback for a specific pad.

        Args:
            pad_name: Name of the sink pad.
            callback: Function to call for each buffer on this pad.
                      Signature: (pad: SinkPad, buf: SeriesBuffer) -> None

        Raises:
            ValueError: If pad_name is not a valid sink pad.
        """
        if pad_name not in self.snks:
            raise ValueError(
                f"Unknown pad name '{pad_name}'. "
                f"Available pads: {list(self.snks.keys())}"
            )
        self._callbacks[self.snks[pad_name]] = callback

    def remove_callback(self, pad_name: str) -> None:
        """
        Remove the callback for a specific pad.

        Args:
            pad_name: Name of the sink pad.
        """
        if pad_name in self.snks:
            pad = self.snks[pad_name]
            self._callbacks.pop(pad, None)

    def get_callback(self, pad_name: str) -> BufferCallback | None:
        """
        Get the callback registered for a pad.

        Args:
            pad_name: Name of the sink pad.

        Returns:
            The callback function or None if not registered.
        """
        if pad_name not in self.snks:
            return None
        return self._callbacks.get(self.snks[pad_name])

    def process(self, input_frames: dict[SinkPad, TSFrame]) -> None:
        """Process incoming frames by invoking callbacks for each buffer."""
        for pad, frame in input_frames.items():
            callback = self._callbacks.get(pad)

            if callback is None:
                # No callback registered for this pad - skip
                continue

            for buf in frame:
                # Skip gaps unless emit_gaps is True
                if buf.is_gap and not self.emit_gaps:
                    continue

                callback(pad, buf)

            # Handle EOS
            if frame.EOS:
                self.mark_eos(pad)
