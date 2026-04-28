#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

from unittest import TestCase

import numpy as np

from aws.osml.image_processing.processing_chain import ProcessingChain, compose


class TestProcessingChainSingleStep(TestCase):
    """Tests that a single-step chain applies the step correctly."""

    def test_single_step_applied(self):
        """A chain with one step applies that step to the input."""
        image = np.array([[[1, 2], [3, 4]]], dtype=np.uint8)

        def add_ten(arr):
            return arr + 10

        chain = ProcessingChain(steps=[add_ten], output_bands=1, output_dtype=np.dtype(np.uint8))
        result = chain(image)

        expected = image + 10
        np.testing.assert_array_equal(result, expected)

    def test_single_step_preserves_shape(self):
        """A single-step chain preserves the CHW shape of the input."""
        image = np.zeros((3, 8, 8), dtype=np.float32)

        chain = ProcessingChain(steps=[lambda x: x * 2.0], output_bands=3, output_dtype=np.dtype(np.float32))
        result = chain(image)

        self.assertEqual(result.shape, (3, 8, 8))


class TestProcessingChainMultiStep(TestCase):
    """Tests that a multi-step chain with dtype-changing steps works correctly."""

    def test_multi_step_dtype_change(self):
        """Steps that change dtype are applied in sequence."""
        image = np.array([[[100, 200], [50, 150]]], dtype=np.uint16)

        def to_float(arr):
            return arr.astype(np.float32) / 65535.0

        def to_uint8(arr):
            return (arr * 255).astype(np.uint8)

        chain = ProcessingChain(steps=[to_float, to_uint8], output_bands=1, output_dtype=np.dtype(np.uint8))
        result = chain(image)

        # Manually apply steps
        intermediate = image.astype(np.float32) / 65535.0
        expected = (intermediate * 255).astype(np.uint8)
        np.testing.assert_array_equal(result, expected)

    def test_multi_step_order_matters(self):
        """Steps are applied in the order they appear in the list."""
        image = np.array([[[2, 4], [6, 8]]], dtype=np.float64)

        def add_one(arr):
            return arr + 1

        def multiply_two(arr):
            return arr * 2

        # (x + 1) * 2 != (x * 2) + 1
        chain_add_first = ProcessingChain(steps=[add_one, multiply_two], output_bands=1, output_dtype=np.dtype(np.float64))
        chain_mul_first = ProcessingChain(steps=[multiply_two, add_one], output_bands=1, output_dtype=np.dtype(np.float64))

        result_add_first = chain_add_first(image)
        result_mul_first = chain_mul_first(image)

        np.testing.assert_array_equal(result_add_first, (image + 1) * 2)
        np.testing.assert_array_equal(result_mul_first, (image * 2) + 1)
        # Confirm they differ
        self.assertFalse(np.array_equal(result_add_first, result_mul_first))


class TestProcessingChainEmptySteps(TestCase):
    """Tests that an empty step list returns input unchanged."""

    def test_empty_steps_returns_input(self):
        """A chain with no steps returns the input array unchanged."""
        image = np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint8)

        chain = ProcessingChain(steps=[], output_bands=1, output_dtype=np.dtype(np.uint8))
        result = chain(image)

        np.testing.assert_array_equal(result, image)

    def test_empty_steps_same_object(self):
        """An empty chain returns the same array object (no copy)."""
        image = np.zeros((3, 4, 4), dtype=np.float32)

        chain = ProcessingChain(steps=[], output_bands=3, output_dtype=np.dtype(np.float32))
        result = chain(image)

        self.assertIs(result, image)


class TestComposeSimple(TestCase):
    """Tests compose with two simple chains (no input_bands on second)."""

    def test_compose_concatenates_steps(self):
        """compose(A, B) applies A's steps then B's steps."""
        image = np.array([[[1, 2], [3, 4]]], dtype=np.float64)

        def add_one(arr):
            return arr + 1

        def multiply_three(arr):
            return arr * 3

        chain_a = ProcessingChain(steps=[add_one], output_bands=1, output_dtype=np.dtype(np.float64))
        chain_b = ProcessingChain(steps=[multiply_three], output_bands=1, output_dtype=np.dtype(np.float64))

        composed = compose(chain_a, chain_b)
        result = composed(image)

        expected = (image + 1) * 3
        np.testing.assert_array_equal(result, expected)

    def test_compose_uses_second_output_metadata(self):
        """Composed chain uses second chain's output_bands and output_dtype."""
        chain_a = ProcessingChain(steps=[], output_bands=3, output_dtype=np.dtype(np.float32), input_bands=(0, 1, 2))
        chain_b = ProcessingChain(steps=[], output_bands=1, output_dtype=np.dtype(np.uint16))

        composed = compose(chain_a, chain_b)

        self.assertEqual(composed.output_bands, 1)
        self.assertEqual(composed.output_dtype, np.dtype(np.uint16))

    def test_compose_uses_first_input_bands(self):
        """Composed chain uses first chain's input_bands."""
        chain_a = ProcessingChain(steps=[], output_bands=3, output_dtype=np.dtype(np.uint8), input_bands=(2, 1, 0))
        chain_b = ProcessingChain(steps=[], output_bands=3, output_dtype=np.dtype(np.uint8))

        composed = compose(chain_a, chain_b)

        self.assertEqual(composed.input_bands, (2, 1, 0))

    def test_compose_first_input_bands_none(self):
        """When first chain has input_bands=None, composed chain also has None."""
        chain_a = ProcessingChain(steps=[], output_bands=3, output_dtype=np.dtype(np.uint8), input_bands=None)
        chain_b = ProcessingChain(steps=[], output_bands=3, output_dtype=np.dtype(np.uint8))

        composed = compose(chain_a, chain_b)

        self.assertIsNone(composed.input_bands)


class TestComposeIncompatibleBands(TestCase):
    """Tests that compose raises ValueError for incompatible band indices."""

    def test_raises_when_second_input_bands_exceed_first_output(self):
        """ValueError raised when second chain's input_bands has index >= first's output_bands."""
        chain_a = ProcessingChain(steps=[], output_bands=3, output_dtype=np.dtype(np.uint8))
        chain_b = ProcessingChain(steps=[], output_bands=1, output_dtype=np.dtype(np.uint8), input_bands=(5,))

        with self.assertRaises(ValueError):
            compose(chain_a, chain_b)

    def test_raises_when_any_index_out_of_range(self):
        """ValueError raised even if only one index is out of range."""
        chain_a = ProcessingChain(steps=[], output_bands=4, output_dtype=np.dtype(np.uint8))
        chain_b = ProcessingChain(steps=[], output_bands=3, output_dtype=np.dtype(np.uint8), input_bands=(0, 1, 4))

        with self.assertRaises(ValueError):
            compose(chain_a, chain_b)

    def test_no_error_when_indices_valid(self):
        """No error when all second chain's input_bands indices are < first's output_bands."""
        chain_a = ProcessingChain(steps=[], output_bands=5, output_dtype=np.dtype(np.uint8))
        chain_b = ProcessingChain(steps=[], output_bands=3, output_dtype=np.dtype(np.uint8), input_bands=(0, 2, 4))

        # Should not raise
        composed = compose(chain_a, chain_b)
        self.assertIsNotNone(composed)


class TestComposeBandSelectInjection(TestCase):
    """Tests that compose injects a band_select step when second chain has input_bands."""

    def test_band_select_injected(self):
        """When second chain has input_bands, a band_select step is injected between chains."""
        # Create a 4-band image
        image = np.arange(4 * 2 * 2, dtype=np.float32).reshape(4, 2, 2)

        chain_a = ProcessingChain(steps=[], output_bands=4, output_dtype=np.dtype(np.float32))
        chain_b = ProcessingChain(steps=[], output_bands=2, output_dtype=np.dtype(np.float32), input_bands=(1, 3))

        composed = compose(chain_a, chain_b)
        result = composed(image)

        # Should select bands 1 and 3 from the 4-band input
        expected = image[[1, 3], :, :]
        np.testing.assert_array_equal(result, expected)

    def test_no_band_select_when_input_bands_none(self):
        """When second chain has input_bands=None, no band_select step is injected."""
        image = np.ones((3, 2, 2), dtype=np.uint8)

        def add_five(arr):
            return arr + 5

        chain_a = ProcessingChain(steps=[add_five], output_bands=3, output_dtype=np.dtype(np.uint8))
        chain_b = ProcessingChain(steps=[], output_bands=3, output_dtype=np.dtype(np.uint8), input_bands=None)

        composed = compose(chain_a, chain_b)
        result = composed(image)

        # Only add_five should be applied, no band selection
        expected = image + 5
        np.testing.assert_array_equal(result, expected)

    def test_band_select_with_steps_on_both_chains(self):
        """Band select is injected between first's steps and second's steps."""
        # 4-band image
        image = np.ones((4, 2, 2), dtype=np.float64) * 2.0

        def multiply_ten(arr):
            return arr * 10

        def add_one(arr):
            return arr + 1

        chain_a = ProcessingChain(steps=[multiply_ten], output_bands=4, output_dtype=np.dtype(np.float64))
        chain_b = ProcessingChain(steps=[add_one], output_bands=2, output_dtype=np.dtype(np.float64), input_bands=(0, 2))

        composed = compose(chain_a, chain_b)
        result = composed(image)

        # Expected: multiply_ten → band_select(0,2) → add_one
        after_multiply = image * 10  # all 20.0
        after_select = after_multiply[[0, 2], :, :]  # 2 bands of 20.0
        expected = after_select + 1  # all 21.0
        np.testing.assert_array_equal(result, expected)


class TestProcessingChainAsNestedStep(TestCase):
    """Tests that a ProcessingChain can be used as a step inside another chain."""

    def test_nested_chain_as_step(self):
        """A ProcessingChain used as a step in another chain applies its steps."""
        image = np.array([[[1, 2], [3, 4]]], dtype=np.float64)

        inner_chain = ProcessingChain(
            steps=[lambda x: x * 2, lambda x: x + 1],
            output_bands=1,
            output_dtype=np.dtype(np.float64),
        )

        outer_chain = ProcessingChain(
            steps=[inner_chain, lambda x: x * 3],
            output_bands=1,
            output_dtype=np.dtype(np.float64),
        )

        result = outer_chain(image)

        # inner: (x * 2) + 1, then outer multiplies by 3
        expected = ((image * 2) + 1) * 3
        np.testing.assert_array_equal(result, expected)

    def test_nested_empty_chain_as_step(self):
        """An empty ProcessingChain used as a step passes through unchanged."""
        image = np.array([[[5, 10], [15, 20]]], dtype=np.uint8)

        inner_chain = ProcessingChain(steps=[], output_bands=1, output_dtype=np.dtype(np.uint8))

        outer_chain = ProcessingChain(
            steps=[inner_chain, lambda x: x + 1],
            output_bands=1,
            output_dtype=np.dtype(np.uint8),
        )

        result = outer_chain(image)

        expected = image + 1
        np.testing.assert_array_equal(result, expected)
