import os
import numpy as np
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


