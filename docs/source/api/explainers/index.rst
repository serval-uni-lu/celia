Explainers
==========

This section documents all counterfactual explanation methods available in
CELIA.  Methods are grouped by their algorithmic approach.

Base classes
------------

The abstract classes that all explainers inherit from.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Class
     - Description
   * - :class:`~celia.explainers.BaseExplainer`
     - Abstract root class defining the ``generate_counterfactuals`` interface.
   * - :class:`~celia.explainers.ClassifierExplainer`
     - Base for methods that flip a classifier's predicted class.
   * - :class:`~celia.explainers.RegressorExplainer`
     - Base for methods that push a regressor's output into a target range.

.. toctree::
   :hidden:

   celia.explainers.base

Optimization-based
------------------

Counterfactual explainers based on optimization strategies defines a loss function that accounts for desired properties
and adopts existing optimization algorithms to minimize it.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Class
     - Description
   * - :class:`~celia.explainers.dice.DiceClassifierExplainer` /
       :class:`~celia.explainers.dice.DiceRegressorExplainer`
     - Diverse counterfactuals via a proximity + diversity loss.
   * - :class:`~celia.explainers.growing_spheres.GrowingSpheresClassifierExplainer`
     - Expanding hyperspherical shell search for the nearest boundary crossing.
   * - :class:`~celia.explainers.ocean.OCEANClassifierExplainer`
     - Mixed-Integer Program over tree ensembles (requires Gurobi).

.. toctree::
   :hidden:

   celia.explainers.dice
   celia.explainers.gsg
   celia.explainers.ocean

Instance-based
--------------

Instance-based methods return real training examples whose prediction
differs from the query, rather than generating synthetic points.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Class
     - Description
   * - :class:`~celia.explainers.nnce.NNCEClassifierExplainer` /
       :class:`~celia.explainers.nnce.NNCERegressorExplainer`
     - Nearest-neighbour search for training points with a different prediction.

.. toctree::
   :hidden:

   celia.explainers.nnce

Heuristic-based
---------------

Counterfactual explainers based on heuristic search strategies aim at finding counterfactuals through local and
heuristic choices that at each iteration minimize a certain cost function.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Class
     - Description
   * - :class:`~celia.explainers.grace.GRACEClassifierExplainer`
     - Gradient-based iterative perturbation of the most salient features.
   * - :class:`~celia.explainers.clear.CLEARClassifierExplainer`
     - Local surrogate regression solved analytically for minimal perturbation.
   * - :class:`~celia.explainers.bugdoc.BugDocClassifierExplainer` /
       :class:`~celia.explainers.bugdoc.BugDocRegressorExplainer`
     - Decision-tree debugging for logical-rule counterfactuals.

.. toctree::
   :hidden:

   celia.explainers.grace
   celia.explainers.clear
   celia.explainers.bugdoc

Generative
----------

Generative methods train a deep generative model (GAN or VAE) on the
background data and sample counterfactuals from the learned distribution.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Class
     - Description
   * - :class:`~celia.explainers.countergan.CounterGANClassifierExplainer`
     - Residual GAN trained to flip predictions to a target class.
   * - :class:`~celia.explainers.cchvae.CCHVAEClassifierExplainer`
     - Conditional heterogeneous VAE with latent-space hypersphere search.

.. toctree::
   :hidden:

   celia.explainers.countergan
   celia.explainers.cchvae

Stochastic
----------

Stochastic methods use reinforcement learning to learn an amortised recourse
policy that can generate counterfactuals cheaply at inference time.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Class
     - Description
   * - :class:`~celia.explainers.fastar.FastARClassifierExplainer`
     - PPO-based amortised recourse policy with constraint support.

.. toctree::
   :hidden:

   celia.explainers.fastar
