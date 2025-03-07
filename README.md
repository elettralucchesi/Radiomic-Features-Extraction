# Radiomic Analysis of 3D MRI Images: Feature Extraction from Lesion Volumes and Individual Slices
---
## Overview
This project enables the extraction of radiomic features from segmented 3D MRI images. The extraction can be performed both on:
- **3D segmentes lesion volumes**
- **2D slices of the segmentes volumes**

The extracted features are saved in CSV format for each segmented lesion of each patient (for 3D) or for each segmented lesion in individual slices (for 2D).

__
## List of contents
- Fetures
- Installation
- Usage
- Project Structure
- Testing 
- Results 
- License

__
## Features
- Accepts 3D MRI images in **NIfTI (.nii)** format.
- Requires an image and a segmentation mask, where the mask contains the segmented lesion.
- Uses a configuration file (.yaml) to specify which radiomic features to extract.
- Outputs results in structured CSV datasets stored in the output_files/ directory.
- Supports both 3D and 2D feature extraction:
-- 3D: Extracts features from the full segmented lesion volume.
-- 2D: Extracts features from individual slices of the segmented lesion volume.

## Installation
### Prerequisites
Ensure you have **Python 3.11** installed.
### Clone the Repository
From _terminal_ move to the desired folder and clone this repository using the following command:

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
- The input MRI images and segmentation masks must be **3D NIfTI (.nii)** files.
- Masks should include seg in the filename.
- It is recommended to organize the data in a directory structure like:
data/
├── Patient_1/
│   ├── PR1.nii       # MRI Image
│   ├── PR1_mask.nii  # Segmentation Mask
├── Patient_2/
│   ├── PR2.nii
│   ├── PR2_mask.nii

- A **YAML configuration** file for feature extraction should be provided in the data/ folder (e.g., pyradiomics_config.yaml).

## Project Structure

## Configuration
Edit the config.ini file to specify:
```shell
[data]
data_path = ./data/*      # Path to patient folders
output_path = ./output_files/  # Directory to save extracted features
mode = 3D                 # Extraction mode: '3D' or '2D'
config_file = ./data/pyradiomics_config.yaml  # YAML file for feature selection
```
### Run the Feature Extraction

### Project Structure

### Testing

### Results
The extracted radiomic features are stored in output_files/ as CSV files. Each file contains features for each segmented lesion:

- **3D Mode**: One row per segmented lesion.
- **2D Mode**: One row per segmented lesion per slice.

### License
