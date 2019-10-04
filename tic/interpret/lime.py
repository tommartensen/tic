from typing import Dict, List, Optional

from lime import lime_tabular, submodular_pick
from lime.explanation import Explanation
from sklearn.base import BaseEstimator


def _result(explanation: Explanation):
    '''Enrich an Explanation with feature importances and a figure.'''
    return {
        'explanation': explanation,
        'feature_importances': explanation.as_list(),
        'figure': explanation.as_pyplot_figure()
    }


def _create_explainer(**kwargs):
    '''Create the tabular explainer from LIME'''
    return lime_tabular.LimeTabularExplainer(
        discretize_continuous=False,
        **kwargs
    )


def explain_instance(
    clf: BaseEstimator,
    instance: List,
    training_data: List,
    feature_names: List,
    class_names: List,
    explainer_kwargs: Optional[Dict] = {},
    explanation_kwargs: Optional[Dict] = {}
):
    '''
    Creates an explainer and explains the given instance.

    Args:
        clf : Fitted classifier from sklearn
        instance: instance to explain
        training_data: data that was used to train the classifier
        feature_names: name of features of dataset
        class_names: names of class labels
        explainer_kwargs: Keyword args passed during explainer initialization
        explanation_kwargs: Keyword args passed for explanation

    Returns:
        Enriched explanation including figure
    '''
    explainer = _create_explainer(
        training_data=training_data,
        feature_names=feature_names,
        class_names=class_names,
        **explainer_kwargs
    )

    explanation = explainer.explain_instance(
        data_row=instance,
        predict_fn=clf.predict_proba,
        **explanation_kwargs
    )
    return _result(explanation)


def explain_dataset(
    clf: BaseEstimator,
    training_data: List,
    feature_names: List,
    class_names: List,
    explainer_kwargs: Optional[Dict] = {},
    explanation_kwargs: Optional[Dict] = {}
):
    '''
    Creates an explainer and creates a submodular pick.
    Returns the explanation with the highest coverage.

    Args:
        clf : Fitted classifier from sklearn
        training_data: data that was used to train the classifier
        feature_names: name of features of dataset
        class_names: names of class labels
        explainer_kwargs: Keyword args passed during explainer initialization
        explanation_kwargs: Keyword args passed for submodular pick

    Returns:
        Enriched explanation with highest coverage including figure
    '''
    explainer = _create_explainer(
        training_data=training_data,
        feature_names=feature_names,
        class_names=class_names,
        **explainer_kwargs
    )

    sp_obj = submodular_pick.SubmodularPick(
        explainer=explainer,
        data=list(training_data.values),
        predict_fn=clf.predict_proba,
        num_exps_desired=1,
        **explanation_kwargs
    )

    return _result(sp_obj.sp_explanations[0])
