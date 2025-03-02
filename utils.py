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
    files = glob.glob(os.path.join(path, '*.nii'))
    img = [f for f in files if not f.endswith('seg.nii')]
    mask = [f for f in files if f.endswith('seg.nii')]
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
    return img, mask


# Extract patient ID from file name
def extract_id(path):
    """
    Extract the patient ID from the file name.

    :param path: Path to the image file
    :return: Patient ID if found, else None
    """
    match = re.search(r'PR(\d+)', path)
    if match:
        return int(match.group(1))
    else:
        print('No patient ID found in the file name.')
        return None

# Assign a new patient ID by finding the first available ID
def assign_new_patient_id(patients_id):
    """
    Assign a new patient ID by finding the first available ID (avoiding duplicates).

    :param patients_id: Set of existing patient IDs to check for uniqueness
    :return: A new, unique patient ID
    """
    new_id = 1
    while new_id in patients_id:
        new_id += 1
    return new_id


