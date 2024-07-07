# MNIST Classification
![](https://www.researchgate.net/publication/306056875/figure/fig1/AS:393921575309346@1470929630835/Example-images-from-the-MNIST-dataset.png)

This project involves classifying handwritten digits from the MNIST dataset using various machine learning techniques. The goal is to train and evaluate models to accurately recognize digits from 0 to 9.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Exploring and Preparing the Dataset](#exploring-and-preparing-the-dataset)
- [Training a Binary Classifier](#training-a-binary-classifier)
- [Performance Measurement](#performance-measurement)
- [Precision-Recall Tradeoff](#precision-recall-tradeoff)
- [The ROC Curve](#the-roc-curve)
- [Random Forest Classifier](#random-forest-classifier)
- [Conclusion](#conclusion)

## Project Overview
The MNIST database is a widely used dataset for handwritten digit classification. It contains 60,000 training images and 10,000 testing images of digits, each image being a 28x28 pixel grayscale image. 
This project demonstrates the process of loading the data, exploring it, and applying machine learning models to classify the digits.

## Dataset
The dataset can be downloaded from [OpenML](https://www.openml.org/d/554).

## Installation
To run this project, you need the following libraries:
- `numpy`
- `matplotlib`
- `scikit-learn`
- `seaborn`

You can install them using pip:
```bash
pip install numpy matplotlib scikit-learn seaborn
```

## Exploring and Preparing the Dataset
First, we load the MNIST dataset and explore its structure:
```python
from sklearn.datasets import fetch_openml

# Load the dataset
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist["data"], mnist["target"]
```
We then split the data into training and test sets and shuffle the training set:
```python
import numpy as np
from sklearn.model_selection import train_test_split

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.142857, random_state=42)

# Shuffle the training set
shuffle_index = np.random.permutation(len(X_train))
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]
```

## Training a Binary Classifier
We create a target vector for binary classification (detecting the digit '5'):
```python
y_train_5 = (y_train == '5')
y_test_5 = (y_test == '5')
```
We then train an SGD classifier:
```python
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)
```

## Performance Measurement
We measure the performance using cross-validation:
```python
from sklearn.model_selection import cross_val_score

cross_val_scores = cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")
print(cross_val_scores)
```

## Precision-Recall Tradeoff
We explore the precision-recall tradeoff by adjusting the decision threshold:
```python
from sklearn.metrics import precision_recall_curve

y_scores = sgd_clf.decision_function(X_train)
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)

# Plot precision and recall as functions of the threshold
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
plt.xlabel("Threshold")
plt.legend(loc="best")
plt.grid(True)
plt.show()
```

## The ROC Curve
We plot the ROC curve and calculate the ROC AUC score:
```python
from sklearn.metrics import roc_curve, roc_auc_score

fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)
roc_auc = roc_auc_score(y_train_5, y_scores)

plt.plot(fpr, tpr, linewidth=2)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

print("ROC AUC Score:", roc_auc)
```

## Random Forest Classifier
We train a RandomForestClassifier and compare its performance to the SGDClassifier:
```python
from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(random_state=42)
forest_clf.fit(X_train, y_train_5)
y_probas_forest = forest_clf.predict_proba(X_train)[:, 1]

fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5, y_probas_forest)
roc_auc_forest = roc_auc_score(y_train_5, y_probas_forest)

plt.plot(fpr, tpr, 'b:', label='SGD')
plt.plot(fpr_forest, tpr_forest, 'g-', linewidth=2, label='Random Forest')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='best')
plt.show()

print("ROC AUC Score for Random Forest:", roc_auc_forest)
```

## Conclusion
This project demonstrates the steps involved in training and evaluating classifiers on the MNIST dataset. 
By experimenting with different classifiers and tuning their parameters, we can improve the performance and achieve better accuracy in digit classification.
