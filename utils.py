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



def extract_id(path):
    """
    Extract the patient ID from the file name

    :param path: Path to the image file.
    :return: Patient ID as an integer if found and correctly formatted, else None.
    :raises TypeError: If the input is not a string.
    """
    if path is None or not isinstance(path, str):
        raise TypeError("Path must be a string")

    filename = os.path.basename(path)
    matches = re.findall(r'\bPR(\d+)', filename)  # Find all occurrences of "PR<number>"

    if matches:
        if len(matches) > 1:
            print(f"Multiple patient IDs found in '{filename}'. The first occurrence ('PR{matches[0]}') will be used.")
        return int(matches[0])  # Use the first valid match

    if "PR" in filename:  # If "PR" exists but format is incorrect
        print(
            f"Invalid patient ID format in file name '{filename}'. Expected 'PR<number>', e.g., 'PR2'. The ID will be automatically assigned."
        )
        return None
    else:
        print(
            f"No valid patient ID found in file name '{filename}'. Expected format: 'PR<number>', e.g., 'PR2'. The ID will be automatically assigned."
        )

    return None


# Assign a new patient ID by finding the first available ID
def new_patient_id(patients_id):
    """
    Assign a new patient ID by finding the first available ID (avoiding duplicates).

    :param patients_id: Set of existing patient IDs to check for uniqueness
    :return: A new, unique patient ID
    """

    if not isinstance(patients_id, set):
        raise TypeError("patients_id must be a set")

    if any(not isinstance(i, int) for i in patients_id):
        raise ValueError("All patient IDs must be integers")

    if any(i < 0 for i in patients_id):
        raise ValueError("Patient IDs cannot be negative")

    new_id = 1
    while new_id in patients_id:
        new_id += 1
    return new_id


def assign_patient_ids(images_path):
    """
    Assigns patient IDs based on image file paths, creating new IDs if not found.

    :param images_path: A list of file paths to the image files
    :return: A list of patient IDs extracted or assigned for each image path
    """
    if not isinstance(images_path, list):
        raise TypeError("images_path must be a list")

    if not images_path:
        raise ValueError("The list of image paths cannot be empty")

    patient_ids = set()
    for im_path in images_path:
        patient_id = extract_id(im_path)
        if patient_id is None:
            patient_id = new_patient_id(patient_ids)
            print(f"Patient ID not found, automatically assigning new ID, for {im_path} id {patient_id}")
        patient_ids.add(patient_id)

    return patient_ids





