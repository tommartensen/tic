# Usage

All examples are detailed in the iPython notebooks in the repository under `examples/`.
This document should give a conclusive guide to new users.

## Feature Importances

The user can calculate local and global feature importances with proven algorithms from a unified interface.
Prerequisites to use are a fitted sklearn classifier and the training data.
In some cases, also the training labels are required.
As an example dataset, TIC comes with the breast cancer dataset from sklearn.

```python
>>> test_data = load_test_data()
>>> class_names = test_data['target_names']
>>> feature_names = test_data['feature_names']
>>> training_data = test_data['dataset']['X_train']
>>> instance = test_data['dataset']['X_test'].sample(1).values[0]
>>> clf = pickle.load(open('.classifiers/logistic_regression.clf', 'rb'))
```

Then, call the `tic` method to retrieve the explanation, feature importances and a figure.
In this case, the explanation is based on [LIME](https://github.com/marcotcr/lime).

```python
>>> from tic.interpret import lime
>>> lime.explain_local(
...     clf=clf,
...     instance=instance,
...     training_data=training_data,
...     feature_names=feature_names,
...     class_names=class_names,
...     explanation_kwargs={'num_features': 5}
... )
{'explanation': <lime.explanation.Explanation at 0x7fad7f620978>,
 'feature_importances': [('mean perimeter', -0.22634431587962064),
  ('mean area', 0.22087904592306984),
  ('worst area', -0.16743486284244824),
  ('area error', -0.13807183263161452),
  ('mean radius', 0.08172324856930932)],
 'figure': <Figure size 432x288 with 1 Axes>}
```

As another example, global surrogate models can be trained and explained for any classifier.
Notice that not all interpretability models support both local and global explanations.

```python
>>> clf_mlp = pickle.load(
...     open('.classifiers/multi_layer_perceptron.clf', 'rb')
... )
>>> from tic.interpret import surrogate
>>> surrogate.explain_global(
...     clf=clf_mlp,
...     num_features=5,
...     X_train=X_train,
...     surrogate_type='tree',
... )
{'surrogate': DecisionTreeClassifier(...),
 'feature_importances': [('worst radius', 0.7830391127502633),
  ('area error', 0.07139301639577515),
  ('symmetry error', 0.05386874344979909),
  ('worst concave points', 0.053428792945424415),
  ('mean compactness', 0.03827033445873795)]}
```
