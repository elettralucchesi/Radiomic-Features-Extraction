import numpy as np
from scipy.ndimage import label

def extract_largest_region(mask_slice, label_value):
    """
    Extract the largest connected region of a given label from a binary mask slice.

    :param mask_slice: 2D numpy array representing the mask slice
    :param label_value: Integer label to extract the largest region from
    :return: 2D numpy array containing only the largest connected region of the given label
    """
    # Check if inputs are swapped (i.e., label_value is a numpy array and mask_slice is an integer)
    if isinstance(mask_slice, int) and isinstance(label_value, np.ndarray):
        raise TypeError(
            "Inputs appear to be swapped. Expected mask_slice as a numpy array and label_value as an integer.")

    # Input validation
    if not isinstance(mask_slice, np.ndarray):
        raise TypeError("The mask slice must be a numpy array")

    if mask_slice.ndim != 2:
        raise ValueError("The mask slice must be 2D")

    if np.sum(mask_slice) == 0:
        raise ValueError("The mask slice is empty, no regions to extract.")

    if not isinstance(label_value, int):
        raise TypeError("label_value must be an integer")
    if label_value < 0:
        raise ValueError("Label value cannot be negative")

    # Create a binary mask for the specified label
    region_mask = (mask_slice == label_value)

    # If no region exists for the label, return None
    if np.sum(region_mask) == 0:
        return None

    # Label the connected components in the binary mask
    labeled_region, num_labels = label(region_mask)

    largest_region = None
    largest_area = 0

    # Iterate through all connected components, ignoring the background (0)
    for region_id in range(1, num_labels + 1):
        region = (labeled_region == region_id).astype(mask_slice.dtype) * label_value
        region_area = np.sum(region > 0)

        # Update the largest region if the current one is bigger
        if region_area > largest_area:
            largest_area = region_area
            largest_region = region

    return largest_region
