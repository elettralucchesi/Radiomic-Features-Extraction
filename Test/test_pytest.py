import pytest
from image_mask_processing import get_path_images_masks


# Creates a temporary directory with test image and mask files
@pytest.fixture
def setup_test_files(tmp_path):
    img_files = ["image1.nii", "image2.nii"]
    mask_files = ["image1_seg.nii", "image2_seg.nii"]

    for file in img_files + mask_files:
        (tmp_path / file).write_text("test")  # Create dummy files

    return tmp_path, img_files, mask_files


# Test 1 : Tests if images and masks are correctly separated
def test_get_path_images_masks(setup_test_files):
    test_dir, img_files, mask_files = setup_test_files
    img, mask = get_path_images_masks(str(test_dir))

    expected_img = [str(test_dir / f) for f in img_files]
    expected_mask = [str(test_dir / f) for f in mask_files]

    assert sorted(img) == sorted(expected_img), f"Error: Images files are not correctly separated. Expected: {expected_img}, but got: {img}"
    assert sorted(mask) == sorted(expected_mask), f"Error: Mask files are not correctly separated. Expected: {expected_mask}, but got: {mask}"


# Test 2 : Tests behavior with an empty directory
def test_empty_directory(tmp_path):
    img, mask = get_path_images_masks(str(tmp_path))
    assert img == []
    assert mask == []


# Test 3 : Tests behavior when only images are present, without masks
def test_missing_masks(tmp_path):
    img_files = ["image3.nii", "image4.nii"]

    for file in img_files:
        (tmp_path / file).write_text("test")

    img, mask = get_path_images_masks(str(tmp_path))
    expected_img = [str(tmp_path / f) for f in img_files]

    assert sorted(img) == sorted(expected_img)
    assert mask == []
