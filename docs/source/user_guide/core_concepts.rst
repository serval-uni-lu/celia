Core Concepts
=============

CELIA is built around three components: **Model wrappers**, **Data**, and
**Explainers**. Understanding how they fit together is the key to using the
library effectively.

Model wrappers
--------------

CELIA does not train models — it explains them. You bring a trained model and
wrap it so that every explainer can call ``predict`` and ``predict_proba``
without knowing the backend.

.. code-block:: python

   from celia import SklearnModel, TorchModel

   # scikit-learn (or any estimator with .predict / .predict_proba)
   sklearn_model = SklearnModel(clf)

   # PyTorch (any nn.Module)
   torch_model = TorchModel(net)

Both wrappers inherit from :class:`~celia.model.BaseModel`. If you need to
support a custom backend, subclass ``BaseModel`` and implement ``predict``
and ``predict_proba``.

Data
----

The :class:`~celia.data.Data` object wraps your training data together with
metadata that explainers use to generate realistic counterfactuals:

.. code-block:: python

   from celia import Data

   data = Data(
       data=X_train,
       targets=y_train,
       target_name="income",
       continuous_column_names=["age", "hours_per_week"],
       categorical_column_names=["occupation", "education"],
       immutable_column_names=["age"],
       feasible_values={"hours_per_week": (1, 80)},
       monotonic_increasing_column_names=["education_num"],
       correlated_features=[("education_num", "education", 1.0)],
   )

**Constraints at a glance:**

.. list-table::
   :header-rows: 1
   :widths: 25 55

   * - Constraint
     - Effect
   * - ``immutable_column_names``
     - Features that the explainer must **never change** (e.g., age, race).
   * - ``feasible_values``
     - A ``{feature: (min, max)}`` dict bounding each feature to a realistic
       range (e.g., credit score between 300 and 850).
   * - ``monotonic_increasing_column_names``
     - Features that may only **increase** in the counterfactual (e.g.,
       education level).
   * - ``correlated_features``
     - Triples of ``(cause, effect, delta)`` that link feature changes
       together.

Not every explainer supports every constraint — see :doc:`methods_overview`
for a compatibility matrix.

Explainers
----------

All explainers follow the same two-step pattern:

1. **Construct** with a model, data, and any method-specific options.
2. **Call** :meth:`~celia.explainers.BaseExplainer.generate_counterfactuals`
   with the sample(s) to explain.

.. code-block:: python

   from celia import DiceClassifierExplainer

   explainer = DiceClassifierExplainer(
       model=sklearn_model,
       data=data,
       method="random",
   )
   cfs = explainer.generate_counterfactuals(sample, total_CFs=3)

Under the hood, CELIA uses a **template-method** design:

- The public ``generate_counterfactuals()`` validates inputs (sample shape,
  data types, target range for regression).
- It then delegates to the abstract ``_generate_counterfactuals()`` which
  each method implements.

This means validation is consistent across all methods — you get the same
:class:`~celia.errors.ConfigurationError` messages regardless of which
explainer you choose.

Classification vs. regression
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

CELIA distinguishes between the two task types at the explainer level:

- :class:`~celia.explainers.ClassifierExplainer` — the counterfactual should
  flip the predicted class.
- :class:`~celia.explainers.RegressorExplainer` — the counterfactual should
  push the prediction into a user-specified ``target_range``.

.. code-block:: python

   from celia import DiceRegressorExplainer

   reg_explainer = DiceRegressorExplainer(model=reg_model, data=data)
   cfs = reg_explainer.generate_counterfactuals(
       sample,
       target_range=(50_000, 80_000),
   )

The Counterfactual object
-------------------------

Every call to ``generate_counterfactuals`` returns one or more
:class:`~celia.counterfactuals.Counterfactual` objects. Each one bundles:

- :attr:`~celia.counterfactuals.Counterfactual.original_instance` — the input
  row.
- :attr:`~celia.counterfactuals.Counterfactual.counterfactuals` — the
  generated alternative(s) as a DataFrame.
- :attr:`~celia.counterfactuals.Counterfactual.highlighted_counterfactuals` —
  same shape, but unchanged values are replaced with ``"-"`` for easy
  visual comparison.
- :attr:`~celia.counterfactuals.Counterfactual.original_prediction` and
  :attr:`~celia.counterfactuals.Counterfactual.counterfactual_prediction` —
  model outputs before and after.

Error handling
--------------

CELIA raises specific exceptions when something goes wrong:

- :class:`~celia.errors.ConfigurationError` — invalid user input. Includes
  ``param``, ``hint``, and ``config`` attributes for programmatic handling.
- :class:`~celia.errors.NoCounterfactualsFoundError` — the algorithm ran but
  found no valid counterfactuals.
- :class:`~celia.errors.InstancesAreWithinRangeError` — (regression only) all
  provided samples already fall within the requested ``target_range``.

All exceptions inherit from :class:`~celia.errors.CELIAError`, so you can
catch the entire hierarchy with a single ``except CELIAError``.
