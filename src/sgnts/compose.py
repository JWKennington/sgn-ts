"""Composable time-series elements for SGN-TS.

This module provides TS-aware composition functionality that ensures
time-series semantics are preserved across composed element boundaries.

Example usage:
    >>> from sgnts.compose import TSCompose
    >>>
    >>> # Create a composed source from source + transforms
    >>> composed = (
    ...     TSCompose()
    ...     .connect(ts_source, transform1)
    ...     .connect(transform1, transform2)
    ...     .as_source(name="my_composed_ts_source")
    ... )
    >>>
    >>> # Use like any other source element
    >>> pipeline.connect(composed, my_sink)

    >>> # Nested composition: build from smaller composed units
    >>> inner_transform = (
    ...     TSCompose()
    ...     .connect(filter1, filter2)
    ...     .as_transform(name="filter_chain")
    ... )
    >>>
    >>> outer_source = (
    ...     TSCompose()
    ...     .connect(ts_source, inner_transform)  # composed inside composed
    ...     .as_source(name="filtered_source")
    ... )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from sgn.base import Element
from sgn.compose import (
    Compose,
    ComposedSinkElement,
    ComposedSourceElement,
    ComposedTransformElement,
)

# These imports are used by disabled validation code (see FIXME in _is_ts_element)
from sgnts.base.base import TimeSeriesMixin, TSTransform, _TSSource  # noqa: F401
from sgnts.sinks.collect import TSFrameCollectSink  # noqa: F401

if TYPE_CHECKING:
    pass


def _is_ts_element(elem: Element) -> bool:
    """Check if an element is a time-series element.

    An element is considered TS-compatible if it:
    - Inherits from TimeSeriesMixin (TSTransform, TSSink), OR
    - Inherits from _TSSource (TSSource, TSResourceSource), OR
    - Is a TSFrameCollectSink (TS-compatible collect sink), OR
    - Is a TSComposed* element (for nested composition)

    Args:
        elem: Element to check

    Returns:
        True if the element is TS-compatible
    """
    # FIXME: Validation temporarily disabled to allow EventFrame-producing
    # transforms (e.g., Latency) in TSComposedSourceElement. To re-enable:
    # 1. Rewrite Latency as a TSTransform that produces TSFrame, OR
    # 2. Add a registration mechanism for EventFrame-compatible transforms, OR
    # 3. Check for TimeSpanFrame output capability (duck typing)
    return True

    # Original validation (disabled):
    # return isinstance(
    #     elem,
    #     (
    #         TimeSeriesMixin,  # TSTransform, TSSink
    #         _TSSource,  # TSSource, TSResourceSource
    #         TSFrameCollectSink,  # TS-compatible collect sink
    #         TSComposedSourceElement,  # Nested composition
    #         TSComposedTransformElement,
    #         TSComposedSinkElement,
    #     ),
    # )


def _validate_ts_elements(elements: list[Element], context: str) -> None:
    """Validate that all elements are time-series compatible.

    Args:
        elements: List of elements to validate
        context: Name of the composed element class for error messages

    Raises:
        TypeError: If any element is not TS-compatible
    """
    non_ts = [e.name for e in elements if not _is_ts_element(e)]
    if non_ts:  # pragma: no cover (validation currently disabled)
        raise TypeError(
            f"{context} requires all elements to be TS elements "
            f"(TSSource, TSTransform, TSSink, or TSComposed*). "
            f"Non-TS elements found: {non_ts}"
        )


@dataclass(repr=False, kw_only=True)
class TSComposedSourceElement(ComposedSourceElement):
    """A composed time-series source element.

    Like ComposedSourceElement but validates that all internal elements
    are TS-compatible and ensures TSFrame semantics are preserved.

    Created from: TSSource + TSTransform(s)
    Exposes: Source pads producing TSFrame

    Internal elements maintain their own alignment/buffering via their
    individual AdapterConfig settings. The composed element acts as a
    transparent wrapper that merges internal graphs into the Pipeline.

    The `also_expose_source_pads` parameter (inherited from ComposedSourceElement)
    allows source pads to be exposed externally even when they are also connected
    to internal sinks. This enables multilink patterns where a single source feeds
    both internal elements (e.g., latency tracking) and external consumers.

    Example:
        >>> composed = TSComposedSourceElement(
        ...     name="my_composed_source",
        ...     internal_elements=[ts_source, ts_transform],
        ...     internal_links={"transform:snk:data": "source:src:data"},
        ... )

    Example with exposed internal source:
        >>> # H1 pad is connected to latency internally but also exposed
        >>> composed = TSComposedSourceElement(
        ...     name="source_with_latency",
        ...     internal_elements=[strain_source, latency_element],
        ...     internal_links={"latency:snk:data": "strain:src:H1"},
        ...     also_expose_source_pads=["strain:src:H1"],
        ... )
    """

    # Inherited from ComposedSourceElement:
    # internal_elements: list[Element]
    # internal_links: dict[str, str]
    # also_expose_source_pads: list[str]

    def __post_init__(self) -> None:
        # Validate before parent init
        _validate_ts_elements(self.internal_elements, "TSComposedSourceElement")
        # Parent handles also_expose_source_pads
        super().__post_init__()


@dataclass(repr=False, kw_only=True)
class TSComposedTransformElement(ComposedTransformElement):
    """A composed time-series transform element.

    Like ComposedTransformElement but validates that all internal elements
    are TS-compatible and ensures TSFrame semantics are preserved.

    Created from: TSTransform(s) only
    Exposes: Sink pads accepting TSFrame, Source pads producing TSFrame

    Each internal TSTransform maintains its own alignment/buffering
    configuration via its AdapterConfig. This allows for sophisticated
    multi-stage processing pipelines where each stage can have different
    overlap/stride requirements.

    Example:
        >>> composed = TSComposedTransformElement(
        ...     name="my_processing_chain",
        ...     internal_elements=[transform1, transform2],
        ...     internal_links={"t2:snk:data": "t1:src:data"},
        ... )
    """

    # Inherited from ComposedTransformElement:
    # internal_elements: list[Element]
    # internal_links: dict[str, str]

    def __post_init__(self) -> None:
        # Validate before parent init
        _validate_ts_elements(self.internal_elements, "TSComposedTransformElement")

        # Additional validation: all must be transforms
        # (TSTransform or TSComposedTransformElement)
        for elem in self.internal_elements:
            if not isinstance(elem, (TSTransform, TSComposedTransformElement)):
                raise TypeError(
                    f"TSComposedTransformElement can only contain TSTransform "
                    f"or TSComposedTransformElement elements, "
                    f"got {type(elem).__name__}: {elem.name}"
                )
        super().__post_init__()


@dataclass(kw_only=True)
class TSComposedSinkElement(ComposedSinkElement):
    """A composed time-series sink element.

    Like ComposedSinkElement but validates that all internal elements
    are TS-compatible and ensures TSFrame semantics are preserved.

    Created from: TSTransform(s) + TSSink(s)
    Exposes: Sink pads accepting TSFrame

    Supports both linear chains (transform -> sink) and non-linear graphs
    (transform that fans out to multiple sinks).

    Example:
        >>> composed = TSComposedSinkElement(
        ...     name="my_output_stage",
        ...     internal_elements=[transform, sink1, sink2],
        ...     internal_links={
        ...         "sink1:snk:data": "transform:src:out1",
        ...         "sink2:snk:data": "transform:src:out2",
        ...     },
        ... )
    """

    # Inherited from ComposedSinkElement:
    # internal_elements: list[Element]
    # internal_links: dict[str, str]

    def __post_init__(self) -> None:
        # Validate before parent init
        _validate_ts_elements(self.internal_elements, "TSComposedSinkElement")
        super().__post_init__()


class TSCompose(Compose):
    """Fluent builder for creating composed time-series elements.

    Similar to sgn.compose.Compose but creates TS-specific composed
    elements with validation. All elements added to the composition
    must be TS elements (TSSource, TSTransform, TSSink, or TSComposed*).

    The builder provides a chainable API that mirrors Pipeline's API:
    - insert(*elements) - Add elements without connecting
    - connect(source, sink, link_map) - Add and connect elements

    Example (linear chain):
        >>> composed_source = (
        ...     TSCompose()
        ...     .connect(ts_source, transform1)
        ...     .connect(transform1, transform2)
        ...     .as_source(name="my_composed_ts_source")
        ... )

    Example (non-linear graph):
        >>> composed_source = (
        ...     TSCompose()
        ...     .connect(source1, merge_transform)
        ...     .connect(source2, merge_transform)
        ...     .as_source(name="merged_source")
        ... )

    Example (explicit insert and connect):
        >>> composed = (
        ...     TSCompose()
        ...     .insert(source, t1, t2, t3)
        ...     .connect(source, t1)
        ...     .connect(t1, t2)
        ...     .connect(t1, t3)  # Fan-out
        ...     .as_source()
        ... )

    Example (nested composition):
        >>> inner_transform = (
        ...     TSCompose()
        ...     .connect(filter1, filter2)
        ...     .as_transform(name="filter_chain")
        ... )
        >>>
        >>> outer_source = (
        ...     TSCompose()
        ...     .connect(ts_source, inner_transform)
        ...     .as_source(name="filtered_source")
        ... )
    """

    def as_source(
        self,
        name: str = "",
        also_expose_source_pads: list[str] | None = None,
    ) -> TSComposedSourceElement:
        """Finalize the composition as a TSComposedSourceElement.

        The composition must contain at least one TSSource element and
        cannot contain any TSSink elements.

        Args:
            name: Optional name for the composed element
            also_expose_source_pads: Optional list of internal source pad full names
                (format: "element_name:src:pad_name") that should be exposed externally
                even when they are connected to internal sinks. This enables multilink
                patterns where a single source feeds both internal elements and
                external consumers.

        Returns:
            A new TSComposedSourceElement wrapping the composition

        Raises:
            TypeError: If composition doesn't meet TSComposedSourceElement
                requirements (non-TS elements, missing source, contains sink)
        """
        return TSComposedSourceElement(
            name=name,
            internal_elements=self.elements.copy(),
            internal_links=self._build_link_dict(),
            also_expose_source_pads=also_expose_source_pads or [],
        )

    def as_transform(self, name: str = "") -> TSComposedTransformElement:
        """Finalize the composition as a TSComposedTransformElement.

        The composition must contain only TSTransform elements
        (or TSComposedTransformElement for nested composition).

        Args:
            name: Optional name for the composed element

        Returns:
            A new TSComposedTransformElement wrapping the composition

        Raises:
            TypeError: If composition doesn't meet TSComposedTransformElement
                requirements (non-TS elements, contains source or sink)
        """
        return TSComposedTransformElement(
            name=name,
            internal_elements=self.elements.copy(),
            internal_links=self._build_link_dict(),
        )

    def as_sink(self, name: str = "") -> TSComposedSinkElement:
        """Finalize the composition as a TSComposedSinkElement.

        The composition must contain at least one TSSink element and
        cannot contain any TSSource elements.

        Args:
            name: Optional name for the composed element

        Returns:
            A new TSComposedSinkElement wrapping the composition

        Raises:
            TypeError: If composition doesn't meet TSComposedSinkElement
                requirements (non-TS elements, missing sink, contains source)
        """
        return TSComposedSinkElement(
            name=name,
            internal_elements=self.elements.copy(),
            internal_links=self._build_link_dict(),
        )


__all__ = [
    "TSCompose",
    "TSComposedSourceElement",
    "TSComposedTransformElement",
    "TSComposedSinkElement",
]
