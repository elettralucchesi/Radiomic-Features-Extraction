import pytest
import SimpleITK as sitk
from utils import get_path_images_masks, read_image_and_mask


@pytest.fixture
def setup_test_files(tmp_path):
    """
    Creates a temporary directory with test image and mask files.

    GIVEN: A temporary directory.
    WHEN: Dummy .nii files (images and masks) are created in the directory.
    THEN: The function returns the temporary directory with image and mask file names.
    """
    img_files = ["image1.nii", "image2.nii"]
    mask_files = ["image1_seg.nii", "image2_seg.nii"]

    for file in img_files + mask_files:
        (tmp_path / file).write_text("test")  # Create dummy files

    return tmp_path, img_files, mask_files


def test_get_path_images_masks_images(setup_test_files):
    """
    Test if images are correctly separated into a list.

    GIVEN: A temporary directory with valid .nii image files.
    WHEN: The get_path_images_masks function is called on the directory.
    THEN: The function returns a list of image file paths.
    """
    test_dir, img_files, _ = setup_test_files
    img, _ = get_path_images_masks(str(test_dir))

    expected_img = [str(test_dir / f) for f in img_files]

    assert sorted(img) == sorted(expected_img), f"Expected images: {expected_img}, but got: {img}"


def test_get_path_images_masks_masks(setup_test_files):
    """
    Test if masks are correctly separated into a list.

    GIVEN: A temporary directory with valid .nii mask files.
    WHEN: The get_path_images_masks function is called on the directory.
    THEN: The function returns a list of mask file paths.
    """
    test_dir, _, mask_files = setup_test_files
    _, mask = get_path_images_masks(str(test_dir))

    expected_mask = [str(test_dir / f) for f in mask_files]

    assert sorted(mask) == sorted(expected_mask), f"Expected masks: {expected_mask}, but got: {mask}"


def test_empty_directory(tmp_path):
    """
    Test behavior with an empty directory.

    GIVEN: An empty temporary directory.
    WHEN: The get_path_images_masks function is called.
    THEN: The function returns empty lists for both images and masks.
    """
    img, mask = get_path_images_masks(str(tmp_path))

    assert img == [], f"Expected no image files, but got: {img}"


def test_empty_directory_masks(tmp_path):
    """
    Test behavior with an empty directory for masks.

    GIVEN: An empty temporary directory.
    WHEN: The get_path_images_masks function is called.
    THEN: The function returns an empty list for masks.
    """
    img, mask = get_path_images_masks(str(tmp_path))

    assert mask == [], f"Expected no mask files, but got: {mask}"


def test_missing_masks(tmp_path):
    """
    Test behavior when only images are present, without masks.

    GIVEN: A temporary directory with image files but no corresponding masks.
    WHEN: The get_path_images_masks function is called.
    THEN: The function returns a list with image paths and an empty list for masks.
    """
    img_files = ["image3.nii", "image4.nii"]

    for file in img_files:
        (tmp_path / file).write_text("test")

    img, mask = get_path_images_masks(str(tmp_path))
    expected_img = [str(tmp_path / f) for f in img_files]

    assert sorted(img) == sorted(expected_img), f"Expected images: {expected_img}, but got: {img}"


def test_missing_masks_empty_masks(tmp_path):
    """
    Test behavior when no masks are present for images.

    GIVEN: A temporary directory with images but no corresponding masks.
    WHEN: The get_path_images_masks function is called.
    THEN: The function returns an empty list for masks.
    """
    img_files = ["image3.nii", "image4.nii"]

    for file in img_files:
        (tmp_path / file).write_text("test")

    img, mask = get_path_images_masks(str(tmp_path))

    assert mask == [], f"Expected no masks, but got: {mask}"


def test_multiple_masks_for_single_image(tmp_path):
    """
    Test behavior when there are multiple masks for a single image.

    GIVEN: A temporary directory with an image and multiple corresponding masks.
    WHEN: The get_path_images_masks function is called.
    THEN: The test checks that there is a problem when more than one mask exists for the same image.
    """
    img_file = "image1.nii"
    mask_files = ["image1_seg.nii", "image1_seg_2.nii"]  # Two masks for the same image

    (tmp_path / img_file).write_text("test")  # Create the image
    for mask in mask_files:
        (tmp_path / mask).write_text("test")  # Create the two masks

    # Call get_path_images_masks and check if multiple masks are found for the same image
    img, mask = get_path_images_masks(str(tmp_path))

    # Find out if there are multiple masks for any image
    mask_dict = {}
    for m in mask:
        image_name = m.split("_seg")[0]  # Assuming "seg" in mask names for matching with image
        if image_name not in mask_dict:
            mask_dict[image_name] = []
        mask_dict[image_name].append(m)

    # Assert that there's more than one mask for any image
    for image, masks in mask_dict.items():
        assert len(masks) == 1, f"Error: More than one segmentation mask found for image: {image}"

def test_read_image_and_mask_file_not_found():
    """
    Test that the function raises an error when the image or mask file does not exist.

    GIVEN: Non-existent image or mask paths.
    WHEN: The read_image_and_mask function is called with the non-existent paths.
    THEN: A FileNotFoundError is raised.
    """
    non_existent_image = "non_existent_image.nii"
    non_existent_mask = "non_existent_mask.nii"

    with pytest.raises(FileNotFoundError):
        read_image_and_mask(non_existent_image, non_existent_mask)

def test_read_image_and_mask_empty_image(tmp_path):
    """
    Test that the function raises an error when the image file is empty.

    GIVEN: An empty image file and a valid mask file.
    WHEN: The read_image_and_mask function is called with the empty image.
    THEN: An error is raised due to the empty image.
    """
    empty_image = tmp_path / "empty_image.nii"
    valid_mask = tmp_path / "valid_mask_seg.nii"

    # Create an empty image file
    empty_image.write_text("")  # Empty file

    # Create a valid mask file
    (valid_mask).write_text("test")

    with pytest.raises(RuntimeError):
        read_image_and_mask(str(empty_image), str(valid_mask))

def test_read_image_and_mask_valid(tmp_path):
    """
    Test that the function works when both the image and mask files are valid.

    GIVEN: A valid image file and a valid mask file.
    WHEN: The read_image_and_mask function is called with these files.
    THEN: The function should return SimpleITK images for both the image and the mask.
    """
    valid_image = tmp_path / "valid_image.nii"
    valid_mask = tmp_path / "valid_mask_seg.nii"

    # Create a valid image file
    (valid_image).write_text("test")

    # Create a valid mask file
    (valid_mask).write_text("test")

    img, mask = read_image_and_mask(str(valid_image), str(valid_mask))

    assert isinstance(img, sitk.Image), "Expected SimpleITK image for the image file."
    assert isinstance(mask, sitk.Image), "Expected SimpleITK image for the mask file."


def test_read_image_and_mask_swapped_inputs(tmp_path):
    """
    Test that the function raises an error if the image and mask inputs are swapped.

    GIVEN: An image file passed as the mask and a mask file passed as the image.
    WHEN: The read_image_and_mask function is called with the swapped inputs.
    THEN: A ValueError is raised indicating that the files are mismatched.
    """
    image_file = tmp_path / "image_file.nii"
    mask_file = tmp_path / "mask_file_seg.nii"

    # Create an image and a mask file
    (image_file).write_text("test_image")
    (mask_file).write_text("test_mask")

    # Swap inputs: pass the image as the mask and the mask as the image
    with pytest.raises(ValueError, match="The image and mask files seem to be swapped."):
        read_image_and_mask(str(mask_file), str(image_file))


def test_read_image_and_mask_invalid_format(tmp_path):
    """
    Test that the function raises an error when the input files are not in .nii format.

    GIVEN: A .txt file passed as the image or mask.
    WHEN: The read_image_and_mask function is called with a .txt file.
    THEN: A ValueError is raised indicating that the file format is invalid.
    """
    image_file = tmp_path / "image_file.txt"  # Wrong format (.txt instead of .nii)
    mask_file = tmp_path / "mask_file_seg.txt"  # Wrong format for mask

    # Create the invalid image and mask files
    (image_file).write_text("test_image")
    (mask_file).write_text("test_mask")

    # Now let's check if the function raises a ValueError when a .txt file is passed
    with pytest.raises(ValueError, match="Both files must be in .nii format"):
        read_image_and_mask(str(image_file), str(mask_file))

