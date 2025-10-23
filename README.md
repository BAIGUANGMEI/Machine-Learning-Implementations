# Machine Learning Implementations

This repository contains implementations of various machine learning algorithms from scratch, along with comparisons to their `sklearn` counterparts.

## Contents
- Linear Regression
- Logistic Regression
- Decision Tree Classifier (CART)

## Linear Regression
The `LinearRegression` class implements the ordinary least squares method for linear regression. It includes methods for fitting the model to data and making predictions.

Using the dataset from Kaggle [Dataset Link](https://www.kaggle.com/datasets/andonians/random-linear-regression/data), we preprocess the data and train both our custom linear regression model and `sklearn`'s implementation for comparison.

|Method|Accuracy|
|------|---|
|Custom Linear Regression|98.8801%|
|Sklearn Linear Regression|98.8801%|

## Logistic Regression

The `CustomLogisticRegression` class implements logistic regression using gradient descent. It supports fitting to data, making predictions, and evaluating accuracy.

Using the dataset from Kaggle [Dataset Link](https://www.kaggle.com/datasets/kaushiksuresh147/customer-segmentation/data), we preprocess the data and train both our custom logistic regression model and `sklearn`'s implementation for comparison.

|Method|Accuracy|
|------|--------|
|Custom Logistic Regression|85.85%|
|Sklearn Logistic Regression|86.08%|


## Decision Tree Classifier (CART)
The `DecisionTreeCART` class implements the CART algorithm for decision trees. It supports fitting to data, making predictions, and evaluating accuracy.

Using the dataset from Kaggle [Dataset Link](https://www.kaggle.com/datasets/kaushiksuresh147/customer-segmentation/data), we preprocess the data and train both our custom decision tree and `sklearn`'s implementation for comparison.

|Method|Accuracy|
|------|--------|
|Custom CART|34.07%|
|Sklearn CART|34.11%|
|Random Forest (Sklearn)|33.73%|