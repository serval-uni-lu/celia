CELIA — Counterfactual Explanations for Tabular Data
=====================================================

**CELIA** is a Python library for generating counterfactual explanations for
machine-learning models trained on tabular data. It provides a unified API
across multiple state-of-the-art methods, supporting both scikit-learn and
PyTorch model backends.

Counterfactual explanations answer the question *"What would need to change
in this input for the model to predict a different outcome?"* — making them
one of the most intuitive forms of explainability for end users.

Key features
------------

- **10 explanation methods** spanning optimization, instance-based, generative,
  heuristic, and stochastic approaches.
- **Unified interface** — every method follows the same
  ``Explainer → generate_counterfactuals → Counterfactual`` workflow.
- **Constraint-aware** — declare immutable features, feasible value ranges,
  monotonicity, and feature correlations.
- **Backend-agnostic** — wrap any scikit-learn estimator or PyTorch module
  with a single adapter class.

Implemented methods
-------------------

.. list-table::
   :header-rows: 1
   :widths: 20 35 20

   * - Method
     - Reference
     - Approach
   * - DiCE
     - Mothilal et al., 2020
     - Optimization
   * - Growing Spheres
     - Laugel et al., 2018
     - Optimization
   * - NNCE
     - Nearest Neighbor
     - Instance
   * - GRACE
     - Le et al., 2020
     - Heuristic
   * - OCEAN
     - Parmentier et al., 2021
     - Optimization
   * - CounterGAN
     - Nemirovsky et al., 2022
     - Generative
   * - CLEAR
     - White & d'Avila Garcez, 2020
     - Heuristic
   * - C-CHVAE
     - Pawelczyk et al., 2020
     - Generative
   * - BugDoc
     - Lourenco et al., 2020
     - Heuristic
   * - FastAR
     - Verma et al., 2020
     - Stochastic

.. toctree::
   :hidden:
   :maxdepth: 2

   user_guide/index
   api/index
