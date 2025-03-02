import glob
import os
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