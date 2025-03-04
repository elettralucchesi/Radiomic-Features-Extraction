import logging
import pytest
from radiomics import featureextractor
from radiomics_2d_3d_extractors import *

def test_get_extractor_invalid_type():
    """
    GIVEN a non-string yaml_path
    WHEN get_extractor is called
    THEN it should raise a TypeError.
    """
    with pytest.raises(TypeError, match="yaml_path must be a string."):
        get_extractor(123)

def test_get_extractor_empty_string():
    """
    GIVEN an empty string as yaml_path
    WHEN get_extractor is called
    THEN it should raise a ValueError.
    """
    with pytest.raises(ValueError, match="yaml_path cannot be empty."):
        get_extractor("")

def test_get_extractor_file_not_found():
    """
    GIVEN a non-existent file path
    WHEN get_extractor is called
    THEN it should raise a FileNotFoundError.
    """
    with pytest.raises(FileNotFoundError, match="The file 'non_existent.yaml' does not exist."):
        get_extractor("non_existent.yaml")


import SimpleITK as sitk
import numpy as np
from unittest.mock import Mock
import pytest
import numpy as np
import SimpleITK as sitk
from unittest.mock import Mock
import pytest
import numpy as np
import SimpleITK as sitk
from unittest.mock import Mock

def test_radiomic_extractor_3D_empty_labels():
    """
    GIVEN a patient_dict_3D with an empty mask (no labels)
    WHEN radiomic_extractor_3D is called with these inputs
    THEN it should raise a ValueError indicating no labels in the mask
    """
    # Mock data for the patient (SimpleITK Image objects)
    img_1 = sitk.GetImageFromArray(np.random.rand(10, 10, 10))
    mask_1 = sitk.GetImageFromArray(np.zeros((10, 10, 10)))  # Empty mask (no labels)

    # Mock patient_dict_3D with SimpleITK images
    patient_dict_3D = {
        "123": [{"ImageVolume": img_1, "MaskVolume": mask_1}],
    }

    # Mock extractor object
    extractor = Mock()

    # Call the function under test and check for the raised error
    with pytest.raises(ValueError, match="No labels found in mask for patient 123"):
        radiomic_extractor_3D(patient_dict_3D, extractor)