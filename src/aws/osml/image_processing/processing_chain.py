#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

"""ProcessingChain — composable array processing pipeline.

This module provides :class:`ProcessingChain`, a lightweight dataclass
container for an ordered sequence of ``ndarray → ndarray`` callables
with output metadata (band count, dtype, input band hints).

Each step is a callable that accepts a CHW ndarray and returns a CHW
ndarray.  Steps are applied in order when the chain is called.  A chain
is itself callable, so it can be used as a step inside another chain.

Typical usage::

    from aws.osml.image_processing.processing_chain import ProcessingChain

    chain = ProcessingChain(
        steps=[my_dra_step, my_tonemap_step],
        output_bands=3,
        output_dtype=np.uint8,
    )
    result = chain(raw_chw_array)
"""

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass
class ProcessingChain:
    """An ordered sequence of array processing operations.

    Each step is a callable that accepts a CHW ndarray and returns
    a CHW ndarray.  Steps are applied in order.

    Attributes:
        steps: Ordered list of processing steps.
        output_bands: Number of bands in the chain's output.
        output_dtype: Pixel dtype of the chain's output.
        input_bands: Optional tuple of source band indices the chain
            expects as input (read-time hint for consumers).
    """

    steps: List[Callable[[NDArray], NDArray]]
    output_bands: int
    output_dtype: np.dtype = field(default_factory=lambda: np.dtype(np.uint8))
    input_bands: Optional[Tuple[int, ...]] = None

    def __call__(self, image: NDArray) -> NDArray:
        """Apply all steps in order and return the final result.

        When the step list is empty the input array is returned
        unchanged.

        Args:
            image: Input array in CHW format (channels, height, width).

        Returns:
            Processed array in CHW format.
        """
        result = image
        for step in self.steps:
            result = step(result)
        return result


def band_select(bands: Tuple[int, ...]) -> Callable[[NDArray], NDArray]:
    """Create a step that selects and reorders bands from a CHW array.

    Args:
        bands: Tuple of band indices to select from the input array's
            channel dimension.

    Returns:
        A callable step that accepts a CHW ndarray and returns the
        array sliced to the specified band indices.
    """

    def _select(image: NDArray) -> NDArray:
        return image[list(bands), :, :]

    return _select


def compose(first: "ProcessingChain", second: "ProcessingChain") -> "ProcessingChain":
    """Merge two processing chains into a single chain.

    Concatenates step lists (first's steps followed by second's steps).
    Uses second's ``output_bands`` and ``output_dtype`` for the composed
    chain's output metadata, and first's ``input_bands`` for the composed
    chain's input bands.

    When the second chain has ``input_bands`` set, validates that all
    indices are less than the first chain's ``output_bands`` and injects
    a :func:`band_select` step between the two chains' step lists.

    Args:
        first: The chain whose steps are applied first.
        second: The chain whose steps are applied after the first.

    Returns:
        A new ProcessingChain combining both chains.

    Raises:
        ValueError: If the second chain's ``input_bands`` contains any
            index that is >= the first chain's ``output_bands``.
    """
    middle_steps: List[Callable[[NDArray], NDArray]] = []

    if second.input_bands is not None:
        # Validate band indices against first chain's output
        for idx in second.input_bands:
            if idx >= first.output_bands:
                raise ValueError(
                    f"Second chain's input_bands index {idx} is out of range "
                    f"for first chain's output_bands={first.output_bands}"
                )
        # Inject a band selection step between the two chains
        middle_steps.append(band_select(second.input_bands))

    combined_steps = list(first.steps) + middle_steps + list(second.steps)

    return ProcessingChain(
        steps=combined_steps,
        output_bands=second.output_bands,
        output_dtype=second.output_dtype,
        input_bands=first.input_bands,
    )
