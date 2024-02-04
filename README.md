# Data-Mining-HW2
This project is based on creating a model with Keras and TenserFlow libraries to cluster the given dataset by their label. The evaluation of the model is checked by graphs and calculated accuracy. After creating an accurate model with an acceptable accuracy, the model used for a bigger dataset. 

Given tasks are:
-	Finding the best number of nodes in a layer
-	Finding the best number of hidden layers
-	Add the suitable activation function
-	Learn how to work with TenserFlow and Keras
-	Change the learning rate 
-	Add the suitable loss function

Tech stack: 
Programming language: Python
Libraries used in this project: Pandas, Numpy, Matplotlib, Sklearn, TenserFlow, Keras


Section 1: 
# Diabetes Prediction using XGBoost

## Overview
This repository contains a Jupyter Notebook (DM_Project_9731084.ipynb) that demonstrates the process of building a diabetes prediction model using XGBoost, a popular machine learning algorithm. The notebook covers various steps, including data preprocessing, model creation, parameter tuning, and evaluation.

## Table of Contents
1. **Introduction**
2. **Preprocessing**
   - Handling Missing Data
   - Renaming Columns
   - Normalization
   - One-Hot Encoding
3. **Model Building**
   - Train-Test Split
   - XGBoost Classifier
4. **Model Evaluation**
   - Hyperparameter Tuning
5. **Conclusion**

## Introduction
The primary goal of this project is to predict diabetes using a machine learning model. The dataset (`diabetes.csv`) is preprocessed and used to train an XGBoost classifier. The model's performance is evaluated, and hyperparameter tuning is performed to improve predictive capabilities.

## Preprocessing
### Handling Missing Data
Null values are identified and replaced with mode or relevant statistical measures. Rows with unknown values are removed for data integrity.

### Renaming Columns
Column names are checked for spaces and replaced with underscores for ease of use.

### Normalization
BMI is binned into categories, and two health columns are normalized using min-max scaling.

### One-Hot Encoding
Categorical features are one-hot encoded for compatibility with machine learning algorithms.

## Model Building
### Train-Test Split
The dataset is split into training and testing sets for model learning and validation.

### XGBoost Classifier
An XGBoost classifier is employed for diabetes prediction, and the model is trained on the training set.

## Model Evaluation
The model's accuracy, recall, and precision are evaluated on both training and testing sets. Confusion matrices are generated to visualize performance.

## Hyperparameter Tuning
The notebook explores hyperparameter tuning using a grid search approach to optimize the XGBoost model's performance. The best parameters are identified, and the model is retrained with these parameters.

## Conclusion
According to the drawn charts, the learning rate is the only parameter with a significant impact. Other parameters have horizontal plots, indicating minimal influence on the model's performance.

Feel free to explore the notebook for a detailed walkthrough of each step and the corresponding code.
