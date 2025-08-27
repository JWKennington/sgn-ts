"""Simple tests for TimeSeriesMixin factory function."""

from sgn.base import SinkElement, TransformElement

from sgnts.base.audioadapter import AdapterConfig
from sgnts.base.base import make_ts_element, TimeSeriesMixin


class SimpleSGNSink(SinkElement):
    """Simple SGN sink for testing factory functions."""

    def pull(self, pad, frame):
        pass

    def internal(self):
        pass


class SimpleSGNTransform(TransformElement):
    """Simple SGN transform for testing factory functions."""

    def pull(self, pad, frame):
        pass

    def internal(self):
        pass

    def new(self, pad):
        pass


def test_make_ts_element_basic():
    """Test basic factory function creates TS-enabled element."""
    TSElement = make_ts_element(SimpleSGNSink)
    element = TSElement(name="test", sink_pad_names=["input"])

    # Should have both SGN and TS capabilities
    assert isinstance(element, SimpleSGNSink)
    assert isinstance(element, TimeSeriesMixin)
    assert hasattr(element, "sink_pads")
    assert hasattr(element, "is_aligned")


def test_make_ts_element_default_config():
    """Test factory function uses basic default config."""
    TSElement = make_ts_element(SimpleSGNSink)
    element = TSElement(name="test", sink_pad_names=["input"])

    # Should have basic default adapter config
    assert element.adapter_config is not None
    assert element.adapter_config.stride == 0  # Default AdapterConfig
    assert element.adapter_config.overlap == (0, 0)  # Default AdapterConfig


def test_make_ts_element_config_override():
    """Test factory function allows config override at instantiation."""
    TSElement = make_ts_element(SimpleSGNSink)

    # Can override the default config at instantiation
    override_config = AdapterConfig(stride=1024, skip_gaps=True)
    element = TSElement(
        name="test", sink_pad_names=["input"], adapter_config=override_config
    )

    assert element.adapter_config == override_config
    assert element.adapter_config.stride == 1024
    assert element.adapter_config.skip_gaps


def test_make_ts_element_sink_and_transform():
    """Test factory function works with both sinks and transforms."""
    # Test with sink
    TSSink = make_ts_element(SimpleSGNSink)
    sink = TSSink(name="test", sink_pad_names=["input"])
    assert isinstance(sink, SimpleSGNSink)
    assert isinstance(sink, TimeSeriesMixin)

    # Test with transform
    TSTransform = make_ts_element(SimpleSGNTransform)
    transform = TSTransform(
        name="test", sink_pad_names=["input"], source_pad_names=["output"]
    )
    assert isinstance(transform, SimpleSGNTransform)
    assert isinstance(transform, TimeSeriesMixin)


def test_factory_created_class_names():
    """Test that factory function creates properly named classes."""
    TSSink = make_ts_element(SimpleSGNSink)
    TSTransform = make_ts_element(SimpleSGNTransform)

    assert TSSink.__name__ == "TSSimpleSGNSink"
    assert TSTransform.__name__ == "TSSimpleSGNTransform"
