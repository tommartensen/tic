from typing import Dict, List, Optional

import shap
import pandas as pd
from sklearn.base import BaseEstimator


def _create_explainer(
    clf: BaseEstimator,
    X_train: pd.DataFrame,
    **kwargs
):
    '''
    Creates kernel explainer from SHAP.
    This is the most complete explainer in terms of model type coverage.
    '''
    return shap.KernelExplainer(
        model=clf.predict_proba,
        data=X_train,
        **kwargs
    )


def explain_local(
    clf: BaseEstimator,
    X_train: pd.DataFrame,
    instance: pd.Series,
    class_names: List,
    sample_size: Optional[int] = 100,
    explainer_kwargs: Optional[Dict] = {},
    explanation_kwargs: Optional[Dict] = {}
):
    '''
    Creates an explainer and explains the given instance using SHAP.

    Args:
        clf : Fitted classifier from sklearn
        X_train: data that was used to train the classifier
        instance: instance to explain
        class_names: names of class labels
        sample_size: how many data points are used to create the SHAP values
        explainer_kwargs: Keyword args passed during explainer initialization
        explanation_kwargs: Keyword args passed for explanation

    Returns:
        Enriched SHAP explanation including figure
    '''
    explainer = _create_explainer(
        clf=clf,
        X_train=X_train,
        **explainer_kwargs
    )
    shap_values = explainer.shap_values(instance, nsamples=sample_size)
    figure = shap.force_plot(
        base_value=explainer.expected_value[0],
        shap_values=shap_values[0],
        features=instance,
        out_names=class_names,
        matplotlib=True,
        show=False,
        **explanation_kwargs
    )
    return {
        'explainer': explainer,
        'shap_values': shap_values,
        'figure': figure
    }


def explain_global(
    clf: BaseEstimator,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    class_names: List,
    sample_size: Optional[int] = 100,
    explainer_kwargs: Optional[Dict] = {},
    explanation_kwargs: Optional[Dict] = {}
):
    '''
    Creates an explainer and explanations for a given dataset using SHAP.

    Args:
        clf : Fitted classifier from sklearn
        X_train: data that was used to train the classifier
        X_test: data that should be explained
        class_names: names of class labels
        sample_size: how many data points are used to create the SHAP values
        explainer_kwargs: Keyword args passed during explainer initialization
        explanation_kwargs: Keyword args passed for explanation

    Returns:
        Enriched SHAP explanation including interactive figure
    '''
    explainer = _create_explainer(
        clf=clf,
        X_train=X_train,
        **explainer_kwargs
    )
    shap_values = explainer.shap_values(X_test, nsamples=sample_size)
    figure = shap.force_plot(
        base_value=explainer.expected_value[0],
        shap_values=shap_values[0],
        features=X_test,
        out_names=class_names,
        show=False,
        **explanation_kwargs
    )
    return {
        'explainer': explainer,
        'shap_values': shap_values,
        'figure': figure
    }
