# research26July

# Breast Cancer Classification with SVM from Scratch

This repository contains a Python implementation of a Support Vector Machine (SVM) classifier built from scratch to classify breast cancer diagnoses (malignant or benign) using the breast cancer dataset. The project includes data preprocessing, model training, evaluation, and visualization of the decision boundary using PCA.

## Overview

The project uses the breast cancer dataset, which includes various features such as radius, texture, perimeter, and area, to predict whether a tumor is malignant (M) or benign (B). A linear SVM is implemented using gradient descent to optimize the hinge loss with L2 regularization. The code also includes visualization of the decision boundary in a 2D PCA projection.

## Features

- **Custom SVM Implementation**: A linear SVM built from scratch using Python.
- **Data Preprocessing**: Feature scaling and train-test splitting.
- **Visualization**: 2D PCA projection with decision boundary plotting.
- **Evaluation**: Accuracy and classification report metrics.

## Files

- `breast-cancer.csv`: The dataset containing breast cancer features and diagnosis labels.
- `main.py`: The main script that loads the data, trains the SVM, evaluates it, and generates visualizations.
- `outcome.png`: A screenshot of the terminal output showing accuracy and classification report.
- `SVM Decision Boundary (2D PCA Projection)`: An image file displaying the SVM decision boundary in a 2D PCA space.
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
