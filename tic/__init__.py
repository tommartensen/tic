'''Top-level package for TIC.'''

__author__ = '''Tom Martensen'''
__email__ = 'mail@tommartensen.de'
__version__ = '0.1.0'

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


def load_test_data():
    '''
    Loads the breast cancer test dataset from sklearn and prepares it for the
    examples.
    '''
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name=data.target_names[1])
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    dataset = dict(zip(
        ['X_train', 'X_test', 'y_train', 'y_test'],
        [X_train, X_test, y_train, y_test]
    ))
    return {
        'target_names': data.target_names,
        'feature_names': data.feature_names,
        'dataset': dataset
    }
