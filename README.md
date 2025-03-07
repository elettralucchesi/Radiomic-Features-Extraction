# Radiomic Analysis of 3D MRI Images: Feature Extraction from Lesion Volumes and Individual Slices
---
## Overview
This project enables the extraction of radiomic features from segmented 3D MRI images. The extraction can be performed both on:
- **3D segmented lesion volumes**
- **2D slices of the segmented volumes**

The extracted features are saved in CSV format for each segmented lesion of each patient (for 3D) or for each segmented lesion in individual slices (for 2D).

___
## List of contents
- [Features](#features)
- [Installation](#installation) 
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [Results](#results) 
- [License](license)

___
## Features
- Accepts **3D MRI** images in **NIfTI (.nii)** format.
- Requires an **image** and a **segmentation mask**, where the mask contains the segmented lesion.
- Uses a **configuration file (.yaml)** to specify which radiomic features to extract.
- Outputs results in structured **CSV datasets** stored in the `output_files/` directory.
- Supports **both 3D and 2D feature extraction**:
-- **3D**: Extracts features from the full segmented lesion volume.
-- **2D**: Extracts features from individual slices of the segmented lesion volume.

## Installation
### Prerequisites
Ensure you have **Python 3.11** installed.
### Clone the Repository
Open a _terminal_ and run the following commands:
```shell
git clone https://github.com/elettralucchesi/Radiomic_Features_Extraction.git
cd Radiomic_Features_Extraction
```
## Install Dependecies 
---
Install the required packages using:
```shell
pip install -r requirements.txt
```
## Usage
### Input Data Format
- The input **MRI images** and **segmentation masks** must be **3D NIfTI (.nii)** files.
- Masks should include `seg` in the filename.
- It is recommended to organize the data in a directory structure like:
```
📂 data/
├── 📂 Patient_1/
│   ├── PR1.nii          # MRI Image
│   ├── PR1_seg.nii      # Segmentation Mask
├── 📂 Patient_2/
│   ├── PR2.nii          # MRI Image
│   ├── PR2_mask.nii     # Segmentation
```
- A **YAML configuration file** for feature extraction should be provided in the `data/` folder (e.g., `pyradiomics_config.yaml`).

### Configuration
Edit the `config.ini` file to specify:
```shell
[data]
data_path = ./data/*      # Path to patient folders
output_path = ./output_files/  # Directory to save extracted features .csv file
mode = 3D                 # Extraction mode: '3D' or '2D'
radiomic_config_file = ./data/pyradiomics_config.yaml  # YAML file for feature selection
```
### Run the Feature Extraction
Execute the main script:
```bash
python main.py
```
### Project Structure
```
Radiomic_Features_Extraction/
├── 📂 data/            # Input MRI images and masks
├── 📂 output files/    # Extracted features in CSV format
├── 📂 tests            #  Unit tests
├── config.ini          #  Configuration file
├── requirements.txt    # Dependencies
├── README.md           # Documentation
├── utils.py                # Helper functions
├── radiomics_2d_3d_extractors.py # Feature extraction for 3D and 2D
├── image_processing.py     # Image loading and preprocessing
├── main.py             # Runs the full  extraction
```
### Testing
Unit tests are available in the tests/ directory. To run them:
```bash
pytest test/
```
### Results
The extracted radiomic features are stored in `output_files/` as CSV files. Each file contains features for each segmented lesion:
- **3D Mode**: One row per segmented lesion.
- **2D Mode**: One row per segmented lesion per slice.

### License
This project is released under the **MIT License**.
