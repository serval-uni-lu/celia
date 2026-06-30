Loan Approval: Explaining a Denied Application
================================================

Alice applies for a loan and gets denied. She wants to know:
*"What would I need to change to get approved?"*

In this tutorial we train a classifier on a synthetic loan dataset, wrap
it with CELIA, and generate counterfactual explanations with three
different methods — each offering a different perspective on what Alice
could do.

1. Create the dataset
---------------------

We generate a synthetic loan-approval dataset with six numeric features.
Approval depends on a weighted score with noise, producing a realistic
binary classification problem.

.. code-block:: python

   import numpy as np
   import pandas as pd
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.model_selection import train_test_split

   rng = np.random.default_rng(42)
   n = 1000

   X = pd.DataFrame({
       "age": rng.integers(21, 66, size=n),
       "annual_income": rng.uniform(20, 150, size=n).round(1),
       "credit_score": rng.integers(300, 851, size=n),
       "employment_years": rng.integers(0, 41, size=n),
       "debt_to_income_ratio": rng.uniform(0.0, 1.0, size=n).round(3),
       "num_credit_lines": rng.integers(0, 21, size=n),
   })

   score = (
       0.01 * (X["credit_score"] - 300)
       + 0.005 * X["annual_income"]
       + 0.02 * X["employment_years"]
       - 1.5 * X["debt_to_income_ratio"]
       + 0.01 * (X["age"] - 21)
       + 0.05 * X["num_credit_lines"]
       + rng.normal(0, 0.5, size=n)
   )
   y = pd.Series((score > 3.5).astype(int), name="approved")

   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.2, stratify=y, random_state=42,
   )

The dataset has a roughly balanced 50/50 approval rate and features that
range from personal attributes (``age``) to financial indicators
(``credit_score``, ``debt_to_income_ratio``).

2. Train a classifier
---------------------

We train a Random Forest — a strong baseline that also provides
``predict_proba``, which several explainers require.

.. code-block:: python

   clf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
   clf.fit(X_train, y_train)
   print(f"Test accuracy: {clf.score(X_test, y_test):.0%}")

::

   Test accuracy: 94%

3. Define real-world constraints
--------------------------------

Before generating explanations, we declare what changes are
**realistic**.  These constraints ensure the suggestions CELIA produces
are actionable — not just mathematically valid.

.. code-block:: python

   # Age cannot be changed by a loan officer's advice
   immutable_features = ["age"]

   # Every feature is continuous (numeric)
   continuous_features = list(X.columns)

   # Realistic bounds for each feature
   feasible_values = {
       "age": (21, 65),
       "annual_income": (20.0, 150.0),
       "credit_score": (300, 850),
       "employment_years": (0, 40),
       "debt_to_income_ratio": (0.0, 1.0),
       "num_credit_lines": (0, 20),
   }

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Constraint
     - Purpose
   * - ``immutable_features``
     - Alice's ``age`` is a fact — the model should never suggest "be younger."
   * - ``feasible_values``
     - A credit score can't exceed 850, income can't be negative, etc.

4. Wrap data and model with CELIA
---------------------------------

CELIA needs two wrappers:

- :class:`~celia.data.Data` — bundles the training data with all the
  metadata (feature types, immutability, feasible ranges) so every
  explainer reads the same constraints.
- :class:`~celia.model.SklearnModel` — gives explainers a uniform
  ``predict`` / ``predict_proba`` interface regardless of the backend.

.. code-block:: python

   from celia import Data, SklearnModel

   data = Data(
       data=X_train,
       targets=y_train,
       target_name="approved",
       continuous_column_names=continuous_features,
       immutable_column_names=immutable_features,
       feasible_values=feasible_values,
   )

   sklearn_model = SklearnModel(clf)

From this point on, every explainer works with the same ``data`` and
``sklearn_model`` objects — the unified API is the core value of CELIA.

5. Meet Alice
-------------

Let's find a test-set applicant whose loan was **denied**.

.. code-block:: python

   denied_mask = clf.predict(X_test) == 0
   alice = X_test[denied_mask].iloc[[0]]
   print(alice.to_string())

::

        age  annual_income  credit_score  employment_years  debt_to_income_ratio  num_credit_lines
   585   62          116.8           419                25                 0.316                 0

Alice has a decent income and long employment history, but a low credit
score and no open credit lines. The model gives her only an 8.9 %
approval probability. What can she change?

6. Generate counterfactual explanations
---------------------------------------

We ask three methods the same question: *"What would Alice need to
change to get approved?"* Each method approaches the problem differently.

6.1 DiCE — diverse suggestions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

DiCE generates **multiple diverse** counterfactuals by optimising a loss
that balances proximity to Alice's profile with diversity among the
suggestions.

.. code-block:: python

   from celia import DiceClassifierExplainer

   dice = DiceClassifierExplainer(
       model=sklearn_model, data=data, method="random",
   )
   dice_cfs = dice.generate_counterfactuals(alice, total_CFs=3)

   dice_cfs[0].highlighted_counterfactuals

::

     age  annual_income  credit_score  employment_years  debt_to_income_ratio  num_credit_lines  Prediction
   0   -          116.8           644                 -                 0.316                 -     0.0 → 1
   1   -          116.8           744                 -                 0.316                11     0.0 → 1
   2   -          116.8           738                 -                 0.316                 -     0.0 → 1

The ``highlighted_counterfactuals`` view replaces unchanged values with
``"-"`` so the changes jump out. Notice that ``age`` is always ``"-"`` —
the immutable constraint was respected. DiCE offers Alice three
different paths: all involve raising her credit score, and one also
suggests opening more credit lines.

6.2 Growing Spheres — the nearest boundary
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Growing Spheres expands a hyperspherical shell around Alice until it
finds the **closest** point where the model's prediction flips.

.. code-block:: python

   from celia import GrowingSpheresClassifierExplainer

   gsg = GrowingSpheresClassifierExplainer(
       model=sklearn_model, data=data, n_samples=1000, seed=42,
   )
   gsg_cf = gsg.generate_counterfactuals(alice)

   gsg_cf.highlighted_counterfactuals

::

       age  annual_income  credit_score  employment_years  debt_to_income_ratio  num_credit_lines  Prediction
   585   -     116.624647    440.913015         40.992394              -1.18375         15.058944       0 → 1

Growing Spheres found the decision boundary, but produced a
``debt_to_income_ratio`` of **-1.18** — clearly impossible. This
highlights an important point: **Growing Spheres does not enforce
feasible value ranges.** Always check the
:doc:`methods overview </user_guide/methods_overview>` to see which
constraints each method supports.

6.3 NNCE — real approved applicants
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

NNCE takes a fundamentally different approach: instead of synthesising
counterfactuals, it searches the **actual training data** for approved
applicants most similar to Alice.

.. code-block:: python

   from celia import NNCEClassifierExplainer

   nnce = NNCEClassifierExplainer(model=sklearn_model, data=data)
   nnce_cf = nnce.generate_counterfactuals(alice, n_counterfactuals=3)

   nnce_cf.highlighted_counterfactuals

::

     age  annual_income  credit_score  employment_years  debt_to_income_ratio  num_credit_lines  Prediction
   0   -           32.6           584                29                 0.999                15       0 → 1
   1   -          113.8           624                 1                 0.310                 -       0 → 1
   2   -          106.3           652                10                 0.956                19       0 → 1

These are **real people** from the training set — existence proofs that
someone with a similar profile to Alice got approved. Because they are
real data points, the values are always realistic.

7. Comparing the methods
-------------------------

Each method told a slightly different story, but a common theme emerged:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Method
     - What it suggested
     - Strength
   * - **DiCE**
     - Raise credit score (644–744); optionally open credit lines.
     - Multiple diverse options; respects immutable + feasible constraints.
   * - **Growing Spheres**
     - Nearest boundary crossing.
     - Shows the minimal change — but may violate feasible ranges.
   * - **NNCE**
     - Real approved applicants with higher credit scores and more credit
       lines.
     - Results are always realistic because they are real data points.

All three methods respected the immutable constraint on ``age``.
DiCE and NNCE produced values within feasible ranges; Growing Spheres
did not — a known limitation of that method.

Key takeaways
-------------

1. **CELIA's unified API** — Wrapping the model and data once with
   :class:`~celia.data.Data` and :class:`~celia.model.SklearnModel` lets
   you swap explainers with a single line change.

2. **Constraints matter** — Declaring immutable features and feasible
   ranges is what makes counterfactuals *actionable*, not just
   mathematically valid. Check the
   :doc:`constraint support table </user_guide/methods_overview>` to
   pick a method that enforces the constraints you care about.

3. **Multiple perspectives** — No single method is best in all
   situations. Running two or three methods gives a richer picture.

4. **Explanations describe the model, not reality** —
   "Raise your credit score to 720" tells you what the *model* would
   predict, not that doing so guarantees real-world approval.

.. seealso::

   - :doc:`/user_guide/core_concepts` — deeper dive into models, data,
     and explainers.
   - :doc:`/user_guide/methods_overview` — full comparison of all 10
     methods.
