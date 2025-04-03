# SVM-Based Diabetes Prediction

## Overview
This project builds a **binary classification model** using **Support Vector Machines (SVM)** to predict whether a person is diabetic or not based on five diagnostic features from the **diabetes.csv** dataset.

## Features Used
The model is trained using the following features:
- **Glucose**
- **Blood Pressure**
- **Insulin**
- **BMI (Body Mass Index)**
- **Age**

## Implementation Steps

1. **Load and Preprocess Data**
   - Extract relevant features and target variable
   - Implement a custom `train_test_split` function to split the dataset

2. **Train an SVM Model**
   - Train an SVM classifier with a **linear kernel**

3. **Find Hyperplane & Support Vectors**
   - Extract the equation of the hyperplane
   - Identify support vectors

4. **Implement Confusion Matrix Manually**
   - Compute **True Positives (TP)**, **False Negatives (FN)**, **False Positives (FP)**, and **True Negatives (TN)**
   - Calculate accuracy

5. **Hyperparameter Optimization with GridSearchCV**
   - Tune **C (regularization parameter)** and **kernel type**
   - Find the best parameters for improved accuracy

6. **Re-evaluate Performance After Tuning**
   - Train the optimized model
   - Recalculate confusion matrix and accuracy

7. **Visualize Decision Boundaries**
   - Generate **2D scatter plots** for every possible feature pair
   - Draw the SVM **hyperplane and margin lines**

8. **Make a Prediction**
   - Predict diabetes status for a new sample: `Glucose=100, Blood Pressure=90, Insulin=0.5, BMI=55, Age=63`

## Results
- **Best Hyperparameters Found:** _(from GridSearchCV)_
- **Final Model Accuracy:** _(calculated after tuning)_
- **Prediction for Sample Input:** _Diabetic / Not Diabetic_

## Requirements
- Python 3.12.7
- Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `sklearn`

## How to Run
1. Open the Jupyter Notebook or Python script.
2. Load the `diabetes.csv` dataset.
3. Run all the cells to train and evaluate the model.
4. Check the final accuracy and sample prediction.

## Author
Developed With **ABDALLAH BABEKER** 🚀