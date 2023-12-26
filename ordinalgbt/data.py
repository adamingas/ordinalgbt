"""
Script to generate data for experiments
"""
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression


def simplest_case(n_samples):
    feature = np.random.uniform(low=-10,high=10,size = n_samples)
    label = np.select(
        condlist=[
            np.bitwise_and(feature>0 , feature<4),
            feature>4
        ],
        choicelist=[
            1,
            2
        ],
        default=0
    )
    return pd.DataFrame({"feature":feature,"outcome":label})

# @forge.compose( forge.insert(forge.arg('n_classes'),index = 0),
# forge.copy(make_regression))
def make_ordinal_classification(n_classes=3,quantiles = None,**kwargs):
    if not  ( n_classes or quantiles ):
        raise TypeError("Please supply one of n_classes and quantiles")
    X,y = make_regression(**kwargs)
    if quantiles is None:
        quantiles = np.linspace(0,1,n_classes+1,endpoint = True)

    y = pd.cut(y,np.quantile(y,quantiles),labels=list(range(n_classes)),
               include_lowest=True).astype(int)
    return X, y
