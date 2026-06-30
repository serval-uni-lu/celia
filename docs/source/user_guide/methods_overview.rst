Methods Overview
================

CELIA ships with 10 counterfactual explanation methods. This page summarises
their approach, backend support, and constraint compatibility to help you
choose the right one for your use case.

At a glance
-----------

.. list-table::
   :header-rows: 1
   :widths: 15 12 12 45

   * - Method
     - Task
     - Approach
     - Description
   * - :class:`~celia.explainers.dice.DiceClassifierExplainer` /
       :class:`~celia.explainers.dice.DiceRegressorExplainer`
     - Clf / Reg
     - Optimization
     - Generates diverse counterfactuals by optimizing a loss that balances
       proximity, diversity, and feasibility.
   * - :class:`~celia.explainers.growing_spheres.GrowingSpheresClassifierExplainer`
     - Clf
     - Optimization
     - Expands a hyperspherical shell around the input until a candidate that
       crosses the decision boundary is found.
   * - :class:`~celia.explainers.nnce.NNCEClassifierExplainer` /
       :class:`~celia.explainers.nnce.NNCERegressorExplainer`
     - Clf / Reg
     - Instance
     - Finds the nearest real training example(s) whose prediction differs
       from the query.
   * - :class:`~celia.explainers.grace.GRACEClassifierExplainer`
     - Clf
     - Heuristic
     - Generates concise contrastive samples for neural-network classifiers
       by iteratively perturbing features within feasible bounds.
   * - :class:`~celia.explainers.ocean.OCEANClassifierExplainer`
     - Clf
     - Optimization
     - Formulates counterfactual search as a Mixed-Integer Program over
       tree-ensemble classifiers. Requires a Gurobi license.
   * - :class:`~celia.explainers.countergan.CounterGANClassifierExplainer`
     - Clf
     - Generative
     - Trains a GAN to produce counterfactuals that flip the classifier's
       prediction to a target class.
   * - :class:`~celia.explainers.clear.CLEARClassifierExplainer`
     - Clf
     - Heuristic
     - Builds a local surrogate regression model around each instance and
       solves it analytically for a minimal perturbation.
   * - :class:`~celia.explainers.cchvae.CCHVAEClassifierExplainer`
     - Clf
     - Generative
     - Trains a conditional heterogeneous VAE and searches latent-space
       hyperspheres for counterfactuals.
   * - :class:`~celia.explainers.bugdoc.BugDocClassifierExplainer` /
       :class:`~celia.explainers.bugdoc.BugDocRegressorExplainer`
     - Clf / Reg
     - Heuristic
     - Uses decision-tree debugging to find logical-rule counterfactuals.
   * - :class:`~celia.explainers.fastar.FastARClassifierExplainer`
     - Clf
     - Stochastic
     - Trains a PPO reinforcement-learning policy that learns to perturb
       instances until the prediction flips.

Backend support
---------------

.. list-table::
   :header-rows: 1
   :widths: 25 15 15

   * - Method
     - sklearn
     - torch
   * - DiCE
     - Yes
     - Yes
   * - Growing Spheres
     - Yes
     - Yes
   * - NNCE
     - Yes
     - Yes
   * - GRACE
     - --
     - **Only**
   * - OCEAN
     - **Only**
     - --
   * - CounterGAN
     - --
     - **Only**
   * - CLEAR
     - Yes
     - Yes
   * - C-CHVAE
     - Yes
     - Yes
   * - BugDoc
     - Yes
     - Yes
   * - FastAR
     - Yes
     - Yes

Constraint support
------------------

.. list-table::
   :header-rows: 1
   :widths: 25 15 15 15 15

   * - Method
     - Immutable
     - Feasible ranges
     - Monotonic Increase
     - Correlated Features
   * - DiCE
     - Yes
     - Yes
     - --
     - --
   * - Growing Spheres
     - Yes
     - --
     - --
     - --
   * - NNCE
     - Yes
     - --
     - --
     - --
   * - GRACE
     - --
     - Yes
     - --
     - --
   * - OCEAN
     - Yes
     - Yes
     - --
     - --
   * - CounterGAN
     - Yes
     - --
     - --
     - --
   * - CLEAR
     - --
     - --
     - --
     - --
   * - C-CHVAE
     - Yes
     - --
     - --
     - --
   * - BugDoc
     - Yes
     - --
     - --
     - --
   * - FastAR
     - Yes
     - --
     - Yes
     - Yes

Optional extras
---------------

Most methods install with the base package. Three require optional extras:

- **CounterGAN** and **C-CHVAE** need ``uv sync --extra torch``
- **OCEAN** needs ``uv sync --extra ocean`` (plus a Gurobi license)
- **FastAR** needs ``uv sync --extra stochastic``

See :doc:`installation` for details.
