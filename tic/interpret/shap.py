from typing import Dict, Optional

import shap
import pandas as pd
from sklearn.base import BaseEstimator


def _create_explainer(
    clf: BaseEstimator,
    X_train: pd.DataFrame,
    **kwargs
):
    return shap.KernelExplainer(
        model=clf.predict_proba,
        data=X_train,
        **kwargs
    )


def explain_local(
    clf: BaseEstimator,
    X_train: pd.DataFrame,
    instance: pd.Series,
    sample_size: Optional[int] = 100,
    explainer_kwargs: Optional[Dict] = {},
    explanation_kwargs: Optional[Dict] = {}
):
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
        matplotlib=True,
        show=False
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
    sample_size: Optional[int] = 100,
    explainer_kwargs: Optional[Dict] = {},
    explanation_kwargs: Optional[Dict] = {}
):
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
        show=False
    )
    return {
        'explainer': explainer,
        'shap_values': shap_values,
        'figure': figure
    }
