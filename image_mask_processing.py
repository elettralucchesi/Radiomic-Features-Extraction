import glob
import os


# Extract file and mask path
def get_path_images_masks(path):
    files = glob.glob(os.path.join(path, '*.nii'))
    img = [f for f in files if not f.endswith('seg.nii')]
    mask = [f for f in files if f.endswith('seg.nii')]
    return img, mask