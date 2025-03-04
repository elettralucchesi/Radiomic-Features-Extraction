import os
import numpy as np
import SimpleITK as sitk
import logging
from radiomics import featureextractor


def get_extractor(yaml_path):
    """
    Creates a RadiomicsFeatureExtractor with a specified configuration file.

    Args:
        yaml_path (str): Path to the YAML file containing configuration parameters.

    Returns:
        extractor: Configured RadiomicsFeatureExtractor object.
    Raises:
        TypeError: If yaml_path is not a string.
        FileNotFoundError: If yaml_path does not exist.
        ValueError: If yaml_path is empty.
    """

    if not isinstance(yaml_path, str):
        raise TypeError("yaml_path must be a string.")

    if not yaml_path:
        raise ValueError("yaml_path cannot be empty.")

    if not os.path.isfile(yaml_path):
        raise FileNotFoundError(f"The file '{yaml_path}' does not exist.")

    extractor = featureextractor.RadiomicsFeatureExtractor(yaml_path)

    # Configure logging for Pyradiomics
    logger = logging.getLogger('radiomics')  # Check log messages given by pyradiomics
    logger.setLevel(logging.ERROR)

    return extractor

def radiomic_extractor_3D(patient_dict_3D, extractor):
    """
    Extracts radiomic features from 3D medical images.

    Args:
        patient_dict_3D (dict): Dictionary containing patient 3D images and masks.
        extractor: Configured RadiomicsFeatureExtractor object.

    Returns:
        dict: Extracted features for each patient and label.
    """
    all_features = {}

    for pr_id, patient_data in patient_dict_3D.items():
        patient_volume = patient_data[0]
        img = patient_volume["ImageVolume"]
        mask = patient_volume["MaskVolume"]
        # Convert SimpleITK Image to NumPy array for processing
        mask_array = sitk.GetArrayFromImage(mask)

        # Get unique labels, excluding 0 (background label)
        labels = np.unique(mask_array)[1:]
        # labels = np.unique(mask)[1:]

        if len(labels) == 0:
            raise ValueError(f"No labels found in mask for patient {pr_id}")

        for lbl in labels:
            try:
                features = extractor.execute(img, mask, label=int(lbl))
                features = {"MaskLabel": lbl, "PatientID": pr_id, **features}
                all_features[f"PR{pr_id} - {lbl:d}"] = features
            except Exception as e:
                logging.error(f"[Invalid Feature] for patient PR{pr_id}, label {lbl}: {e}")

    return all_features


