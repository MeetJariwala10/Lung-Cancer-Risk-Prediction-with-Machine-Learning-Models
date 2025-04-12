# Lung-Cancer-Risk-Prediction-with-Machine-Learning-Models

This repository implements machine learning models for lung cancer risk prediction inspired by the paper:

**Dritsas, E.; Trigka, M. (2022). Lung Cancer Risk Prediction with Machine Learning Models. _Big Data and Cognitive Computing, 6_(4), 139.**  
DOI: [10.3390/bdcc6040139](https://doi.org/10.3390/bdcc6040139)

The paper demonstrates a comparative analysis of several classifiers (e.g., Naive Bayes, SVM, Random Forest, Rotation Forest, etc.) on a publicly available dataset and highlights the superior performance of the Rotation Forest classifier in terms of accuracy, precision, recall, F-Measure, and AUC.

---

## Overview

This project implements lung cancer risk prediction models using machine learning techniques. The key features of this repository include:

- **Data Preprocessing**: Balancing the dataset using SMOTE.
- **Feature Analysis**: Evaluating feature importance using methods like gain ratio and random forest.
- **Modeling**: Training a variety of classification models such as Naive Bayes, Bayesian network, logistic regression, SVM, Random Forest, and Rotation Forest.
- **Evaluation**: Assessing models with metrics including accuracy, precision, recall, F-Measure, and AUC via 10-fold cross-validation in the Weka environment.

The project is implemented in a Jupyter Notebook ([ML_MINI_PROJECT.ipynb](ML_MINI_PROJECT.ipynb)) that contains the code and experiments.

---

## Prerequisites and Installation

To run the code in this repository, please ensure you have the following:

- **Python 3.x** installed.
- Required Python libraries such as:
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `imblearn` (for SMOTE)
  - `matplotlib` or `seaborn` (for plotting)
  
You can install these dependencies using pip:

```bash
pip install numpy pandas scikit-learn imbalanced-learn matplotlib seaborn

