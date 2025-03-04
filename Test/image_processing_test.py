import pytest
import numpy as np
from image_processing import extract_largest_region


def test_extract_largest_region_correct():
    """
    Test the correct behavior of the extract_largest_region function.

    GIVEN: A 2D binary mask with two regions of a specified label.
    WHEN: The extract_largest_region function is called.
    THEN: The function correctly returns the largest connected region.
    """
    mask = np.array([[1, 1, 0, 0],
                     [1, 1, 0, 0],
                     [1, 0, 1, 1],
                     [0, 0, 1, 1]])

    label_value = 1
    largest_region = extract_largest_region(mask, label_value)

    expected = np.array([[1, 1, 0, 0],
                         [1, 1, 0, 0],
                         [1, 0, 0, 0],
                         [0, 0, 0, 0]])

    # Assert the largest region is correctly extracted
    assert np.array_equal(largest_region, expected), f"Expected largest region {expected}, but got {largest_region}"


def test_extract_largest_region_empty_mask_raises_error():
    """
    Test that the function raises an error when the mask slice is empty.

    GIVEN: An empty 2D mask slice where no regions exist.
    WHEN: The extract_largest_region function is called.
    THEN: The function should raise a ValueError indicating the mask is empty.
    """
    mask_slice = np.zeros((5, 5), dtype=int)  # Empty mask slice
    label_value = 1

    try:
        extract_largest_region(mask_slice, label_value)
    except ValueError as e:
        assert str(e) == "The mask slice is empty, no regions to extract.", "Error message does not match."


def test_extract_largest_region_3d_mask():
    """
    Test that the function raises an error when a 3D mask slice is passed.

    GIVEN: A 3D numpy array (mask) instead of a 2D slice.
    WHEN: The extract_largest_region function is called.
    THEN: The function should raise a ValueError indicating the mask must be 2D.
    """
    mask_slice = np.zeros((5, 5, 5), dtype=int)  # 3D mask slice
    label_value = 1

    try:
        extract_largest_region(mask_slice, label_value)
    except ValueError as e:
        assert str(e) == "The mask slice must be 2D", "Error message does not match."


def test_extract_largest_region_invalid_mask_type():
    """
    Test that the function raises an error when the mask slice is not a numpy array.

    GIVEN: The mask slice is a string.
    WHEN: The extract_largest_region function is called.
    THEN: The function should raise a TypeError indicating the mask slice must be a numpy array.
    """
    mask_slice = "invalid_mask"  # String instead of numpy array
    label_value = 1

    try:
        extract_largest_region(mask_slice, label_value)
    except TypeError as e:
        assert str(e) == "The mask slice must be a numpy array", "Error message does not match."


def test_extract_largest_region_negative_label():
    """
    Test that the function raises an error when the label value is negative.

    GIVEN: A negative label value.
    WHEN: The extract_largest_region function is called.
    THEN: The function should raise a ValueError indicating the label cannot be negative.
    """
    mask_slice = np.array([[1, 1, 0, 0],
                     [1, 1, 0, 0],
                     [1, 0, 1, 1],
                     [0, 0, 1, 1]])  # Valid mask slice
    label_value = -1  # Invalid label value

    try:
        extract_largest_region(mask_slice, label_value)
    except ValueError as e:
        assert str(e) == "Label value cannot be negative", "Error message does not match."


def test_extract_largest_region_label_not_found():
    """
    Test that the function returns None when the label is not found in the mask slice.

    GIVEN: A mask slice with no regions for the specified label.
    WHEN: The extract_largest_region function is called.
    THEN: The function should return None, as the label does not exist in the mask.
    """
    mask_slice = np.array([[2, 2, 0, 0],
                     [2, 2, 0, 0],
                     [2, 0, 3, 3],
                     [0, 0, 3, 3]])  # Valid mask slice
    label_value = 1  # Label not present

    result = extract_largest_region(mask_slice, label_value)

    assert result is None, "The function should return None when the label is not found."


def test_extract_largest_region_found():
    """
    Test that the function correctly extracts the largest region of a given label.

    GIVEN: A mask slice with several regions for the specified label.
    WHEN: The extract_largest_region function is called.
    THEN: The function should return the largest region of the given label.
    """
    mask_slice = np.array([
        [1, 1, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [0, 0, 2, 2, 2],
        [0, 0, 2, 2, 2],
        [0, 0, 0, 0, 0]
    ], dtype=int)  # Two regions: Label 1 and Label 2
    label_value = 1

    result = extract_largest_region(mask_slice, label_value)

    # The largest region should be of label 1
    expected_result = np.array([
        [1, 1, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ], dtype=int)

    assert np.array_equal(result,
                          expected_result), "The function should extract the largest connected region for label 1."

def test_extract_largest_region_swapped_inputs():
    """
    Test that the function raises an error when the inputs are swapped.

    GIVEN: The function is called with an integer as mask_slice and a numpy array as label_value.
    WHEN: The extract_largest_region function is called with incorrect argument order.
    THEN: The function should raise a TypeError indicating the inputs are swapped.
    """
    mask_slice = 1  # Incorrect: should be a numpy array
    label_value = np.array([[0, 1], [1, 0]])  # Incorrect: should be an integer

    try:
        extract_largest_region(mask_slice, label_value)
    except TypeError as e:
        assert str(e) == "Inputs appear to be swapped. Expected mask_slice as a numpy array and label_value as an integer.", \
            "Error message does not match."
