[![Python application](https://github.com/adamingas/ordinalgbt/actions/workflows/python-app.yml/badge.svg)](https://github.com/adamingas/ordinalgbt/actions/workflows/python-app.yml)
# OrdinalGBT
## Introduction
OrdinalGBT, which stands for Ordinal gradient boosted trees, is a Python package that implements an ordinal regression loss function using the lightGBM framework. Ordinal regression is a type of regression analysis used for predicting an ordinal variable, i.e. a variable that can be sorted in some order. LightGBM is a gradient boosting framework that uses tree-based learning algorithms and is designed to be distributed and efficient.

## Installation
You can install OrdinalGBT using pip:

```shell
pip install ordinalgbt
```

## Usage 

Here are a few examples on how to use the `LGBMOrdinal` class:

1. **Fitting the model**

```python
from ordinalgbt.lgb import LGBMOrdinal
import numpy as np

# Create the model
model = LGBMOrdinal()

# Generate some data
X = np.random.rand(100, 10)
y = np.random.randint(0, 3, 100)

# Fit the model
model.fit(X, y)
```

2. **Predicting with the model**

After fitting the model, you can use it to make predictions:

```python
# Generate some new data
X_new = np.random.rand(10, 10)

# Use the model to make predictions
# the .predict method returns the class prediction rather than raw score or
# probabilities
y_pred = model.predict(X_new)

print(y_pred)
```

3. **Predicting probabilities with the model**

The `predict_proba` method can be used to get the probabilities of each class:

```python
# Use the model to predict probabilities
y_proba = model.predict_proba(X_new)

print(y_proba)
```