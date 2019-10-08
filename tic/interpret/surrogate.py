from typing import Dict, Optional

# import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from tic.interpret.feature_importance import explain_global as direct_fi


def explain_global(
    clf: BaseEstimator,
    surrogate_type: str,
    X_train: pd.DataFrame,
    num_features: int,
    surrogate_kwargs: Optional[Dict] = {},
    feature_importances_kwargs: Optional[Dict] = {}
):
    '''
    Creates a surrogate model that mimics the behavior of the original model.

    Args:
        clf : Fitted classifier from sklearn
        surrogate_type: Type of surrogate model, linear or tree
        X_train: DataFrame with the original training data
        num_features: how many feature importances should be returned
        surrogate_kwargs: Keyword args passed during surrogate initialization
        feature_importances_kwargs: Kwargs passed for feature importance method

    Returns:
        Surrogate model with its feature importances
    '''
    if surrogate_type == 'linear':
        surrogate_clf = LogisticRegression(**surrogate_kwargs)
    elif surrogate_type == 'tree':
        surrogate_clf = DecisionTreeClassifier(**surrogate_kwargs)
    else:
        raise AssertionError('surrogate_type must be linear or tree')

    # Get feature importances for surrogate model that contains all features
    y_pred = clf.predict(X_train)
    surrogate_clf.fit(X_train, y_pred)
    feature_importances = direct_fi(
        clf=surrogate_clf,
        feature_names=X_train.columns,
        num_features=num_features,
        **feature_importances_kwargs
    )

    # Get feature importances for surrogate model with only the `num_features`
    # most important features from the last step
    most_relevant_features = X_train[[x[0] for x in feature_importances]]
    surrogate_clf.fit(most_relevant_features, y_pred)
    feature_importances = direct_fi(
        clf=surrogate_clf,
        feature_names=most_relevant_features.columns,
        **feature_importances_kwargs
    )

    return {
        'surrogate': surrogate_clf,
        'feature_importances': feature_importances,
    }
