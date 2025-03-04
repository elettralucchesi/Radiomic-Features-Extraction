import glob
import os
import re
import SimpleITK as sitk

# Extract file and mask path
def get_path_images_masks(path):
    """
    Extracts image and mask file paths from a given directory.

    :param path: Path to the directory containing .nii image and mask files
    :return: A tuple containing two lists:
             - The first list contains paths to the image files (files without 'seg' in the name)
             - The second list contains paths to the mask files (files with 'seg' in the name)
    """
    if not isinstance(path, str):
        raise TypeError("Path must be a string")

    files = glob.glob(os.path.join(path, '*.nii'))

    if not files:
        raise ValueError("The directory is empty or contains no .nii files")

    img = [f for f in files if not f.endswith('seg.nii')]
    mask = [f for f in files if f.endswith('seg.nii')]

    if len(img) != len(mask):
        raise ValueError("The number of image files does not match the number of mask files")

    return img, mask


# Read image and mask
def read_image_and_mask(image_path, mask_path):
    """
    Read an image and its corresponding mask using SimpleITK.

    :param image_path: Path to the image file
    :param mask_path: Path to the mask file
    :return: Tuple containing the image and mask as SimpleITK images
    """
    img = sitk.ReadImage(image_path)
    mask = sitk.ReadImage(mask_path)

    print("Image size:", img.GetSize())
    print("Mask size:", mask.GetSize())
    return img, mask



def extract_id(path):
    """
    Extract the patient ID from the file name.

    :param path: Path to the image file.
    :return: Patient ID as an integer if found and correctly formatted, else None.
    :raises TypeError: If the input is not a string.
    """
    if path is None or not isinstance(path, str):
        raise TypeError("Path must be a string")

    matches = re.findall(r'\bPR(\d+)', path)  # Find all occurrences of "PR<number>"

    if matches:
        if len(matches) > 1:
            print(f"Multiple patient IDs found in '{path}'. The first occurrence ('PR{matches[0]}') will be used.")
        return int(matches[0])  # Use the first valid match

    if "PR" in path:  # If "PR" exists but format is incorrect
        print(
            f"Invalid patient ID format in file name '{path}'. Expected 'PR<number>', e.g., 'PR2'. The ID will be automatically assigned."
        )
    else:
        print(
            f"No valid patient ID found in file name '{path}'. Expected format: 'PR<number>', e.g., 'PR2'. The ID will be automatically assigned."
        )

    return None


# Assign a new patient ID by finding the first available ID
def new_patient_id(patients_id):
    """
    Assign a new patient ID by finding the first available ID (avoiding duplicates).

    :param patients_id: Set of existing patient IDs to check for uniqueness
    :return: A new, unique patient ID
    """
    new_id = 1
    while new_id in patients_id:
        new_id += 1
    return new_id



