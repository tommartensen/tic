from typing import List, Optional

import numpy as np
from sklearn.base import BaseEstimator


def extract_from_classifier(
    clf: BaseEstimator,
    feature_names: List,
    absolute_values: Optional[bool] = True,
    sort: Optional[bool] = True,
    limit: Optional[int] = None
):
    '''
    Uses built-in methods from the classifier object to extract feature
    importances in form of Gini coefficients (for tree based models) and
    coefficients (for linear models).

    Args:
        clf: fitted classifier object as input
        feature_names: feature names as fed to the fitting function of clf
        absolute_values: whether absolute values should be returned (exp)
        sort: whether the importances should be sorted
        limit: how many importances should be returned

    Returns:
        list of feature importances with the given configuration
    '''
    if hasattr(clf, 'feature_importances_'):
        feature_importances = clf.feature_importances_
    elif hasattr(clf, 'coef_'):
        feature_importances = clf.coef_[0]
        if absolute_values:
            feature_importances = np.exp(feature_importances)
    else:
        raise NotImplementedError('''
            Classifier does not support direct feature extraction
        ''')

    feature_importances = zip(feature_names, feature_importances)
    if sort:
        feature_importances = sorted(
            feature_importances,
            key=lambda x: abs(x[1]),
            reverse=True
        )
    else:
        feature_importances = list(feature_importances)

    if limit:
        return feature_importances[:limit]
    return feature_importances
