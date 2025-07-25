# Breast Cancer Classification with SVM from Scratch

This repository contains a Python implementation of a Support Vector Machine (SVM) classifier built from scratch to predict whether breast cancer is benign or malignant using the Breast Cancer Wisconsin (Diagnostic) Data Set. The project includes data preprocessing, model training, evaluation, and visualization of the decision boundary using Principal Component Analysis (PCA).

## Overview

The project utilizes the Breast Cancer Wisconsin (Diagnostic) Data Set, which contains features computed from digitized images of fine needle aspirate (FNA) of breast masses. These features describe characteristics of cell nuclei, such as radius, texture, perimeter, and area, among others. A linear SVM is implemented using gradient descent to optimize the hinge loss with L2 regularization, trained to classify tumors as malignant (M) or benign (B). The decision boundary is visualized in a 2D PCA projection.

## Dataset

### Source
- **UCI Machine Learning Repository**: [https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)
- **UW CS FTP Server**: `ftp ftp.cs.wisc.edu`, `cd math-prog/cpo-dataset/machine-learn/WDBC/`

### Description
Features are derived from a digitized image of an FNA of a breast mass, describing cell nuclei characteristics. The dataset includes:
- **Total Instances**: 569
- **Class Distribution**: 357 benign, 212 malignant
- **Missing Values**: None
- **Feature Details**: 30 real-valued features computed as the mean, standard error, and "worst" (mean of the three largest values) of ten real-valued features:
  1. **Radius**: Mean of distances from center to points on the perimeter
  2. **Texture**: Standard deviation of gray-scale values
  3. **Perimeter**
  4. **Area**
  5. **Smoothness**: Local variation in radius lengths
  6. **Compactness**: (Perimeter² / Area - 1.0)
  7. **Concavity**: Severity of concave portions of the contour
  8. **Concave Points**: Number of concave portions of the contour
  9. **Symmetry**
  10. **Fractal Dimension**: "Coastline approximation" - 1

All feature values are recorded with four significant digits.

### Citation
[K. P. Bennett and O. L. Mangasarian: "Robust Linear Programming Discrimination of Two Linearly Inseparable Sets", Optimization Methods and Software 1, 1992, 23-34].

## Features

- **Custom SVM Implementation**: A linear SVM built from scratch using Python with gradient descent.
- **Data Preprocessing**: Feature scaling, train-test splitting, and handling of categorical labels.
- **Visualization**: 2D PCA projection with decision boundary plotting.
- **Evaluation**: Accuracy and detailed classification report (precision, recall, F1-score).

## Files

- `breast-cancer.csv`: The Breast Cancer Wisconsin (Diagnostic) Data Set containing 569 instances with 31 attributes (ID, diagnosis, and 30 features).
- `main.py`: The main script that loads the dataset, trains the SVM, evaluates it, and generates visualizations.
- `outcome.png`: A screenshot of the terminal output showing accuracy (e.g., 0.4474) and classification report.
- `SVM Decision Boundary (2D PCA Projection).png`: An image file displaying the SVM decision boundary in a 2D PCA space.
- `svm_model.py`: A separate module containing the `LinearSVM` class implementation.

## Prerequisites

- Python 3.13 (64-bit)
- Required libraries:
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `matplotlib`

Install the dependencies using:
```bash
pip install pandas numpy scikit-learn matplotlib
