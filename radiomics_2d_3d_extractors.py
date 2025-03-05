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
        labels = np.unique(mask_array)
        labels = labels[labels != 0]

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


def radiomic_extractor_2D(patient_dict_2D, extractor):
    """
    Extracts radiomic features from 2D medical image slices.

    Args:
        patient_dict_2D (dict): Dictionary containing patient 2D slices.
        extractor: Configured RadiomicsFeatureExtractor object.

    Returns:
        dict: Extracted features for each patient slice and label.
    """
    all_features_2D = {}

    for patient_id, patient_slices in patient_dict_2D.items():
        for slice_data in patient_slices:
            lbl = slice_data["Label"]
            index = slice_data["SliceIndex"]

            if lbl == 0:
                raise ValueError(f"No labels found in mask for patient {patient_id}")

            try:
                img_slice = slice_data["ImageSlice"]
                mask_slice = slice_data["MaskSlice"]

                features = extractor.execute(img_slice, mask_slice, label=int(lbl))

                print(f"Features extracted: {features}")

                features = {"MaskLabel": lbl, "SliceIndex": index, "PatientID": patient_id, **features}
                key = f"{patient_id}-{index}-{lbl}"
                # Debug log to verify key creation
                print(f"Generated key: {key}")  # Verifica la creazione della chiave
                # Log per verificare la creazione della chiave e l'aggiornamento del dizionario
                print(f"Features: {features}")  # Verifica che le feature siano correttamente estratte

                all_features_2D[key] = features
                # Debug log to check if the key is being added
                logging.debug(f"Added features for {key}: {features}")
            except Exception as e:
                logging.error(f"[Invalid Feature] for patient {patient_id}, Slice {index}, Label {lbl}: {e}")
    return all_features_2D


