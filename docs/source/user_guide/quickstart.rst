Quickstart
==========

This guide walks through a complete classification example: train a model,
wrap it with CELIA, define constraints, generate counterfactuals, and inspect
the results.

1. Prepare your data and model
------------------------------

.. code-block:: python

   import pandas as pd
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.model_selection import train_test_split

   # Load your tabular dataset (features + target)
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.2, random_state=42,
   )

   # Train a classifier
   model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
   model.fit(X_train, y_train)

2. Wrap with CELIA
------------------

CELIA needs two wrappers: one for the model and one for the data.

.. code-block:: python

   from celia import Data, SklearnModel

   # Wrap the trained model
   sklearn_model = SklearnModel(model)

   # Wrap the training data with metadata
   data = Data(
       data=X_train,
       targets=y_train,
       target_name="approved",
       continuous_column_names=["age", "annual_income", "credit_score"],
       immutable_column_names=["age"],                    # age cannot change
       feasible_values={"credit_score": (300, 850)},      # realistic bounds
   )

The :class:`~celia.data.Data` object stores feature types, immutability
constraints, feasible value ranges, and other metadata that explainers use to
produce actionable counterfactuals.

3. Generate counterfactuals
---------------------------

Every explainer follows the same two-step interface: **construct**, then call
:meth:`~celia.explainers.BaseExplainer.generate_counterfactuals`.

.. code-block:: python

   from celia import DiceClassifierExplainer, NNCEClassifierExplainer

   sample = X_test.iloc[[0]]

   # DiCE — multiple diverse counterfactuals
   dice = DiceClassifierExplainer(model=sklearn_model, data=data, method="random")
   dice_cfs = dice.generate_counterfactuals(sample, total_CFs=3)

   # NNCE — nearest real training examples with a different prediction
   nnce = NNCEClassifierExplainer(model=sklearn_model, data=data)
   nnce_cfs = nnce.generate_counterfactuals(sample, n_counterfactuals=3)

4. Inspect results
------------------

Each result is a :class:`~celia.counterfactuals.Counterfactual` object with
properties for the original instance, the counterfactual(s), and their
predictions.

.. code-block:: python

   cf = dice_cfs[0]

   cf.original_instance           # the input row
   cf.counterfactuals              # generated alternative(s)
   cf.original_prediction          # model prediction on the original
   cf.counterfactual_prediction    # model prediction on the counterfactual(s)

Use :attr:`~celia.counterfactuals.Counterfactual.highlighted_counterfactuals`
to see only the features that changed (unchanged values appear as ``"-"``):

.. code-block:: python

   cf.highlighted_counterfactuals

Next steps
----------

- :doc:`core_concepts` — understand models, data, and explainers in depth.
- :doc:`methods_overview` — compare all 10 methods and choose the right one.
- :doc:`../api/index` — full API reference.
