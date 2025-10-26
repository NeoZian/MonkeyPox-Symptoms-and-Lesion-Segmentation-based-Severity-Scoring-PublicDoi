# A Hybrid Framework for Mpox Diagnosis and Severity Assessment via Symptom Classification and Skin Lesion Segmentation

## Overview

This repository hosts the **complete implementation** of a hybrid multimodal framework designed for the **automated diagnosis and severity assessment of Monkeypox (Mpox)** through the integration of symptom-based clinical data and image-based skin lesion analysis.

The framework combines:

- **Traditional machine learning algorithms** for structured symptom data classification  
- **Deep learning architectures** (notably **U-Net**) for skin lesion segmentation  

Thereby providing a **comprehensive computational pipeline** for Mpox diagnosis and clinical triage.

All algorithms, datasets, and scripts have been meticulously curated and annotated to ensure **full reproducibility**, **scalability**, and **cross-platform compatibility**.

---

## Reproducibility

This repository is designed for **full experimental reproducibility**.  
The accompanying datasets and code scripts are provided with explicit configuration details to facilitate model retraining, fine-tuning, or deployment.

### Repository Structure
root-directory
┣ classification/        → All code files related to image-based classification
┣ data/                  → Scripts for dataset handling, preprocessing, and augmentation
┣ segmentation/          → Code for lesion segmentation (U-Net), testing, and evaluation
┣ symptomBased/          → Code for symptom-based classification using structured clinical data
┣ testcase/              → Integration scripts for multimodal testing, severity scoring, and hybrid inference
┣ README.md              → Current documentation file
text---

## Dataset Information

### Symptom-Based Classification Dataset

For symptom-based classification, we utilized the **Monkey-Pox PATIENTS Dataset**, available on Kaggle [[36]](https://doi.org/10.34740/KAGGLE/DSV/4271503).  

This **synthetic, publicly available dataset** (CC0: Public Domain) is structured to simulate clinical symptom data for **binary classification tasks**—distinguishing monkeypox-positive from monkeypox-negative cases.

It includes features such as:
- Symptom presence
- Comorbidities
- Clinical indicators

These were further refined to suit the present research objectives.

> **Reference**:  
> [36] Muhamad Ahmed. (2022). *Monkey-Pox PATIENTS Dataset*. [Data set]. Kaggle. https://doi.org/10.34740/KAGGLE/DSV/4271503

---

### Image-Based Classification Dataset

As per the licensing declaration, the image dataset was originally curated from diverse open-access online repositories and categorized into **four classes**:

- Monkeypox
- Chickenpox
- Measles
- Normal skin

For **binary classification**, the non-monkeypox categories were consolidated into a single **`NonMonkeypox`** class.

After systematic augmentation to enhance generalization, the dataset comprised:

- **818 Monkeypox images**  
- **832 NonMonkeypox images**

Our final augmented dataset has been made **publicly accessible on Kaggle** [[37]](https://www.kaggle.com/datasets/jewelmd/mpox-classification-augmneted-images).

> **Reference**:  
> [37] Jewel Md. (2025). *Mpox Classification Augmented Images*. [Mpox Classification Augmneted images]. Kaggle. https://www.kaggle.com/datasets/jewelmd/mpox-classification-augmneted-images

---

### Segmentation Dataset

For lesion segmentation, a **customized dataset** derived primarily from the **Monkeypox Skin Images Dataset (MSID)** was developed.  
The **U-Net architecture** was trained on this dataset to accurately delineate lesion boundaries.

All image sources adhered to **open-access or research-licensed usage conditions**.

> **References**:  
> [38] Jewel Md. (2025). *Mpox Lesion Segmentation Dataset (Suitable for U-Net)*. [Mpox Lesion Segmentation suitable for Unet]. Kaggle. https://www.kaggle.com/datasets/jewelmd/mpox-lesion-segmentation-suitable-for-unet  
> Bala, Diponkor; Hossain, Md Shamim (2023). *Monkeypox Skin Images Dataset (MSID)*, Mendeley Data, V6, doi: [10.17632/r9bfpnvyxr.6](https://doi.org/10.17632/r9bfpnvyxr.6)

---

## Code Information

All scripts were developed and tested on **Ubuntu 24.04**, but they are **fully portable** across operating systems (Windows/macOS/Linux) provided the required dependencies are installed.

Each script begins with a list of import statements, allowing users to identify any missing libraries during execution.

---

## Requirements

The following dependencies are essential (partial list):

```python
# Deep Learning Libraries
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy

# Image Processing & Visualization
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Data Handling
import pandas as pd
import numpy as np
import os
import glob as gb

# Machine Learning Models
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
```

Install all dependencies easily:

bashpip install -r requirements.txt

Usage Instructions

Download the codebase and datasets from the links provided above.
Install dependencies:
bashpip install -r requirements.txt

Train models by navigating to the appropriate directory and running the scripts:

/classification → Image classification
/segmentation → Lesion segmentation
/symptomBased → Clinical symptom classification


Hybrid inference and severity scoring:

Use scripts in /testcase/


Pre-trained weights (optional) can be replaced with your own trained models.


Methodology
All methodological details—including:

Data curation
Preprocessing
Augmentation
Model architectures
Training protocols
Performance evaluation

—are elaborated in Sections 3–4 of the accompanying research manuscript.
Graphical workflows and algorithmic schematics are provided therein for comprehensive understanding.

License & Contribution Guidelines

All datasets referenced herein are governed by their respective open-access licenses.

Monkey-Pox PATIENTS Dataset: CC0: Public Domain
MSID dataset: Follows Mendeley’s open research data terms




Users must ensure compliance with dataset-specific license requirements before redistribution or modification.

Contributions to improve or extend the framework are welcome via pull requests.
Please ensure proper citation of both this repository and the original data sources when publishing derivative works.

References
[36] Muhamad Ahmed. (2022). Monkey-Pox PATIENTS Dataset. Kaggle.
[37] Jewel Md. (2025). Mpox Classification Augmented Images. Kaggle.
[38] Jewel Md. (2025). Mpox Lesion Segmentation Dataset (Suitable for U-Net). Kaggle.
Bala, Diponkor; Hossain, Md Shamim (2023). Monkeypox Skin Images Dataset (MSID), Mendeley Data, V6, doi: 10.17632/r9bfpnvyxr.6.