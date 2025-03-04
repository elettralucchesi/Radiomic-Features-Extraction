import pytest
import SimpleITK as sitk
from utils import get_path_images_masks, read_image_and_mask, extract_id, new_patient_id, assign_patient_ids


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

def test_get_path_images_masks_invalid_int():
    """
    Test if passing an integer as path raises TypeError.

    GIVEN: An invalid path type (integer).
    WHEN: The get_path_images_masks function is called with an integer.
    THEN: The function raises a TypeError with the appropriate error message.
    """
    with pytest.raises(TypeError, match="Path must be a string") as exc_info:
        get_path_images_masks(123)  # Pass an integer
    assert str(exc_info.value) == "Path must be a string", f"Expected error message 'Path must be a string', but got: {str(exc_info.value)}"


def test_get_path_images_masks_invalid_none():
    """
    Test if passing None as path raises TypeError.

    GIVEN: An invalid path type (None).
    WHEN: The get_path_images_masks function is called with None.
    THEN: The function raises a TypeError with the appropriate error message.
    """
    with pytest.raises(TypeError, match="Path must be a string") as exc_info:
        get_path_images_masks(None)  # Pass None
    assert str(exc_info.value) == "Path must be a string", f"Expected error message 'Path must be a string', but got: {str(exc_info.value)}"


def test_get_path_images_masks_empty_directory(tmp_path):
    """
    Test if passing an empty directory raises ValueError.

    GIVEN: An empty directory.
    WHEN: The get_path_images_masks function is called on the directory.
    THEN: The function raises a ValueError with the appropriate error message.
    """
    with pytest.raises(ValueError, match="The directory is empty or contains no .nii files") as exc_info:
        get_path_images_masks(str(tmp_path))  # Pass an empty directory
    assert str(exc_info.value) == "The directory is empty or contains no .nii files", f"Expected error message 'The directory is empty or contains no .nii files', but got: {str(exc_info.value)}"


def test_get_path_images_masks_no_nii_files(tmp_path):
    """
    Test if a directory without .nii files raises ValueError.

    GIVEN: A directory with files, but none with .nii extension.
    WHEN: The get_path_images_masks function is called on the directory.
    THEN: The function raises a ValueError with the appropriate error message.
    """
    # Create files with incorrect extensions
    non_nii_files = ["file1.txt", "file2.csv", "file3.jpg"]

    for file in non_nii_files:
        (tmp_path / file).write_text("test")  # Create non-.nii files

    with pytest.raises(ValueError, match="The directory is empty or contains no .nii files") as exc_info:
        get_path_images_masks(str(tmp_path))  # Pass the directory without .nii files

    assert str(
        exc_info.value) == "The directory is empty or contains no .nii files", f"Expected error message 'The directory is empty or contains no .nii files', but got: {str(exc_info.value)}"



def test_get_path_images_masks_mismatched_files(tmp_path):
    """
    Test if a directory with mismatched image and mask files raises ValueError.

    GIVEN: A directory containing an unequal number of image and mask files.
    WHEN: The get_path_images_masks function is called.
    THEN: The function raises a ValueError with the appropriate error message.
    """
    img_files = ["patient1.nii", "patient2.nii"]
    mask_files = ["patient1_seg.nii"]  # Only one mask file for two image files

    for file in img_files + mask_files:
        (tmp_path / file).write_text("test")  # Create both images and masks

    with pytest.raises(ValueError, match="The number of image files does not match the number of mask files") as exc_info:
        get_path_images_masks(str(tmp_path))

    assert str(exc_info.value) == "The number of image files does not match the number of mask files", \
        f"Expected error message 'The number of image files does not match the number of mask files', but got: {str(exc_info.value)}"


def test_get_path_images_masks_multiple_masks_for_one_image(tmp_path):
    """
    Test if a directory with more mask files than image files raises ValueError.

    GIVEN: A directory containing more mask files than image files.
    WHEN: The get_path_images_masks function is called.
    THEN: The function raises a ValueError with the appropriate error message.
    """
    img_files = ["patient1.nii"]  # One image file
    mask_files = ["patient1_seg.nii", "patient2_seg.nii"]  # Two mask files

    for file in img_files + mask_files:
        (tmp_path / file).write_text("test")  # Create one image and multiple masks

    with pytest.raises(ValueError, match="The number of image files does not match the number of mask files") as exc_info:
        get_path_images_masks(str(tmp_path))

    assert str(exc_info.value) == "The number of image files does not match the number of mask files", \
        f"Expected error message 'The number of image files does not match the number of mask files', but got: {str(exc_info.value)}"


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



def test_extract_id_valid():
    """
    Test that the function correctly extracts the patient ID from a valid file path.

    GIVEN: A file path with a patient ID.
    WHEN: The extract_id function is called.
    THEN: The function returns the correct patient ID.
    """
    valid_path = "/path/to/PR12345_image.nii"
    result = extract_id(valid_path)
    assert result == 12345, f"Expected patient ID: 12345, but got: {result}"

def test_extract_id_invalid_format_with_pr():
    """
    GIVEN: A filename with an incorrectly formatted patient ID containing 'PR'.
    WHEN: The extract_id function is called.
    THEN: The function prints an error message and returns None.
    """
    result = extract_id('path/to/PR_2_image.nii')
    assert result is None, f"Expected None, but got {result}"

def test_extract_id_invalid_number_before_pr():
    """
    GIVEN: A filename where the number precedes 'PR' (e.g., '2PR').
    WHEN: The extract_id function is called.
    THEN: The function prints an error message and returns None.
    """
    result = extract_id('path/to/2PR_image.nii')
    assert result is None, f"Expected None, but got {result}"

def test_extract_id_no_pr():
    """
    GIVEN: A filename without a patient ID or 'PR' prefix.
    WHEN: The extract_id function is called.
    THEN: The function prints an error message and returns None.
    """
    result = extract_id('path/to/image_without_id.nii')
    assert result is None, f"Expected None, but got {result}"


def test_extract_id_multiple_pr_but_wrong_format():
    """
    GIVEN: A filename containing multiple 'PR' but in an incorrect format.
    WHEN: The extract_id function is called.
    THEN: The function prints an error message and returns None.
    """
    result = extract_id('path/to/PRabc_PR123X_image.nii')
    assert result is None, f"Expected None, but got {result}"


def test_extract_id_no_pr_prefix():
    """
    GIVEN: A filename where the ID is present but lacks the 'PR' prefix.
    WHEN: The extract_id function is called.
    THEN: The function prints an error message and returns None.
    """
    result = extract_id('path/to/12345_image.nii')
    assert result is None, f"Expected None, but got {result}"

def test_extract_id_non_string_path():
    """
    GIVEN: A non-string input as path.
    WHEN: The extract_id function is called.
    THEN: The function raises a TypeError.
    """
    with pytest.raises(TypeError, match="Path must be a string"):
        extract_id(12345)

def test_extract_id_none_input():
    """
    GIVEN: A None input.
    WHEN: The extract_id function is called.
    THEN: The function raises a TypeError.
    """
    with pytest.raises(TypeError, match="Path must be a string"):
        extract_id(None)

def test_extract_id_multiple_valid_pr():
    """
    GIVEN: A filename with multiple valid 'PR<number>' occurrences.
    WHEN: The extract_id function is called.
    THEN: The function returns only the first valid patient ID.
    """
    result = extract_id('path/to/PR12_PR34_image.nii')
    assert result == 12, f"Expected 12, but got {result}"


def test_new_patient_id_empty_set():
    """
    Test that the function returns 1 when the set of patient IDs is empty.

    GIVEN: An empty set of patient IDs.
    WHEN: The new_patient_id function is called.
    THEN: The function returns 1.
    """
    result = new_patient_id(set())
    assert result == 1, f"Expected new ID: 1, but got: {result}"


def test_new_patient_id_sequential():
    """
    Test that the function correctly assigns the next available patient ID when IDs are sequential.

    GIVEN: A set of patient IDs {1, 2, 3, 4, 5}.
    WHEN: The new_patient_id function is called.
    THEN: The function returns 6.
    """
    existing_ids = {1, 2, 3, 4, 5}
    result = new_patient_id(existing_ids)
    assert result == 6, f"Expected new ID: 6, but got: {result}"

def test_new_patient_id_missing_numbers():
    """
    Test that the function assigns the lowest available patient ID when there are missing numbers.

    GIVEN: A set of patient IDs {2, 3, 5}.
    WHEN: The new_patient_id function is called.
    THEN: The function returns 1 as the first available ID.
    """
    existing_ids = {1, 2, 3, 5}
    result = new_patient_id(existing_ids)
    assert result == 4, f"Expected new ID: 4, but got: {result}"


def test_new_patient_id_large_numbers():
    """
    Test that the function correctly assigns 1 if the existing IDs are all large numbers.

    GIVEN: A set of patient IDs {100, 101, 102}.
    WHEN: The new_patient_id function is called.
    THEN: The function returns 1.
    """
    existing_ids = {100, 101, 102}
    result = new_patient_id(existing_ids)
    assert result == 1, f"Expected new ID: 1, but got: {result}"

def test_new_patient_id_invalid_type_list():
    """
    Test that the function raises a TypeError if the input is a list instead of a set.

    GIVEN: A list of patient IDs instead of a set.
    WHEN: The new_patient_id function is called.
    THEN: The function raises a TypeError.
    """
    with pytest.raises(TypeError, match="patients_id must be a set"):
        new_patient_id([1, 2, 3])  # List instead of set


def test_new_patient_id_invalid_type_string():
    """
    Test that the function raises a TypeError if the input is a string instead of a set.

    GIVEN: A string instead of a set.
    WHEN: The new_patient_id function is called.
    THEN: The function raises a TypeError.
    """
    with pytest.raises(TypeError, match="patients_id must be a set"):
        new_patient_id("123")  # String instead of set

def test_new_patient_id_non_integer_values_string():
    """
    Test that the function raises a ValueError if the set contains a string instead of integers.

    GIVEN: A set containing a string value.
    WHEN: The new_patient_id function is called.
    THEN: The function raises a ValueError.
    """
    with pytest.raises(ValueError, match="All patient IDs must be integers"):
        new_patient_id({1, 2, "three"})  # Contains a string

def test_new_patient_id_non_integer_values_float():
    """
    Test that the function raises a ValueError if the set contains a float instead of integers.

    GIVEN: A set containing a float value.
    WHEN: The new_patient_id function is called.
    THEN: The function raises a ValueError.
    """
    with pytest.raises(ValueError, match="All patient IDs must be integers"):
        new_patient_id({1, 2, 3.5})  # Contains a float

def test_new_patient_id_negative_values():
    """
    Test that the function raises a ValueError when the set contains negative patient IDs.

    GIVEN: A set containing negative patient IDs.
    WHEN: The new_patient_id function is called.
    THEN: The function raises a ValueError with an appropriate message.
    """
    existing_ids = {1, -2, 3, -1}
    with pytest.raises(ValueError, match="Patient IDs cannot be negative"):
        new_patient_id(existing_ids)


def test_assign_patient_ids_type_error():
    """
    Test that the function raises a TypeError when images_path is not a list.

    GIVEN: A non-list input for images_path.
    WHEN: The assign_patient_ids function is called.
    THEN: The function raises a TypeError with the appropriate message.
    """
    invalid_input = "not_a_list"  # A string, not a list
    with pytest.raises(TypeError, match="images_path must be a list"):
        assign_patient_ids(invalid_input)


def test_assign_patient_ids_empty_list_error():
    """
    Test that the function raises an error when an empty list is passed as images_path.

    GIVEN: An empty list of image file paths.
    WHEN: The assign_patient_ids function is called.
    THEN: The function raises a ValueError with an appropriate message.
    """
    images_path = []

    with pytest.raises(ValueError, match="The list of image paths cannot be empty"):
        assign_patient_ids(images_path)


def test_assign_patient_ids_existing_ids():
    """
    Test that the function extracts existing patient IDs correctly.

    GIVEN: A list of image file paths containing patient IDs in the format 'PR<number>'.
    WHEN: The assign_patient_ids function is called.
    THEN: The function extracts the correct patient IDs for each image path.
    """
    images_path = [
        "../Radiomic_Features_Extraction/data/PR2/PR2_T2W_TSE_AX.nii",
        "../Radiomic_Features_Extraction/data/PR3/PR3_T2W_TSE_AX.nii",
    ]

    patient_ids = assign_patient_ids(images_path)

    # Assert the correct patient IDs are extracted
    assert patient_ids == {2, 3}, f" Expected patient IDs {2, 3}, but got {patient_ids} "


def test_assign_patient_ids_no_existing_ids():
    """
    Test that the function assigns new patient IDs when no existing IDs are found in the file names.

    GIVEN: A list of image file paths without valid patient IDs.
    WHEN: The assign_patient_ids function is called.
    THEN: The function assigns new patient IDs to each image path.
    """
    images_path = [
        "../Radiomic_Features_Extraction/data/image1_T2W_TSE_AX.nii",
        "../Radiomic_Features_Extraction/data/image2_T2W_TSE_AX.nii",
    ]

    patient_ids = assign_patient_ids(images_path)

    # Assert new patient IDs are assigned correctly
    assert patient_ids == {1, 2}, f" Expected new patient IDs {1, 2}, but got {patient_ids} "


def test_assign_patient_ids_mixed_existing_and_new_ids():
    """
    Test that the function handles a mix of image paths with existing and new patient IDs.

    GIVEN: A list of image file paths, some with existing patient IDs and others without.
    WHEN: The assign_patient_ids function is called.
    THEN: The function correctly extracts existing IDs and assigns new IDs where necessary.
    """
    images_path = [
        "../Radiomic_Features_Extraction/data/PR2/PR2_T2W_TSE_AX.nii",  # Existing ID
        "../Radiomic_Features_Extraction/data/PR3/PR3_T2W_TSE_AX.nii",  # Existing ID
        "../Radiomic_Features_Extraction/data/image4_T2W_TSE_AX.nii",  # New ID
    ]

    patient_ids = assign_patient_ids(images_path)

    # Assert the correct patient IDs are assigned and extracted
    assert patient_ids == {2, 3, 1}, f" Expected patient IDs {2, 3, 1}, but got {patient_ids} "


def test_assign_patient_ids_invalid_id_format():
    """
    Test that the function handles invalid patient ID formats in image paths.

    GIVEN: A list of image file paths with invalid or missing patient IDs.
    WHEN: The assign_patient_ids function is called.
    THEN: The function assigns new patient IDs to each image path.
    """
    images_path = [
        "../Radiomic_Features_Extraction/data/PR_invalid/PR_T2W_TSE_AX.nii",  # Invalid ID format
        "../Radiomic_Features_Extraction/data/PR_invalid/PR_T2W_TSE_AX.nii",  # Invalid ID format
    ]

    patient_ids = assign_patient_ids(images_path)

    # Assert new patient IDs are assigned correctly
    assert patient_ids == {1, 2}, f" Expected new patient IDs {1, 2}, but got {patient_ids} "
