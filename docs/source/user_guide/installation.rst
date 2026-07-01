Installation
============

CELIA requires **Python 3.12** or later.

From GitHub with ``uv`` (recommended)
--------------------------------------

.. code-block:: bash

   # Create and activate a virtual environment
   uv venv .venv
   source .venv/bin/activate     # macOS / Linux

   # Install the package from GitHub
   uv pip install git+https://github.com/serval-uni-lu/celia.git

From source with ``uv``
-------------------------

.. code-block:: bash

   git clone https://github.com/serval-uni-lu/celia.git
   cd celia

   uv venv .venv
   source .venv/bin/activate

   # Install with the locked dependency set
   uv sync

From GitHub with ``pip``
-------------------------

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate

   pip install git+https://github.com/serval-uni-lu/celia.git

Optional extras
---------------

Some methods require additional dependencies. Install extras as needed:

.. list-table::
   :header-rows: 1
   :widths: 15 40 30

   * - Extra
     - Enables
     - Install command
   * - ``torch``
     - CounterGAN, C-CHVAE (+ PyTorch backend)
     - ``uv sync --extra torch``
   * - ``ocean``
     - OCEAN (requires Gurobi license)
     - ``uv sync --extra ocean``
   * - ``stochastic``
     - FastAR
     - ``uv sync --extra stochastic``
   * - ``all``
     - All of the above
     - ``uv sync --extra all``

Development install
-------------------

To install with linting, testing, and documentation tools:

.. code-block:: bash

   git clone https://github.com/serval-uni-lu/celia.git
   cd celia

   uv venv .venv
   source .venv/bin/activate

   uv sync --group dev
