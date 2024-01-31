.. _kfac:

Kronecker-Factored Approximate Curvature
========================================

The ``bayesian_lora`` package includes a method for calculating an approximate
Fisher information matrix (or GGN) using Kronecker-factored approximate
curvature. Further, these Kronecker factors can themselves be approximated as
low-rank which is particularly useful for LLMs, where the factors may be
:math:`4096 \times 4096` for each layer in a transformer.

.. autofunction:: bayesian_lora.main.calculate_kronecker_factors

Internal Functions
------------------

The above is the main way to use the K-FAC functionality from this library.
It calls a number of internal functions, which we document here for re-use and
completeness.

.. autofunction:: bayesian_lora.main.register_hooks

.. autofunction:: bayesian_lora.main.remove_hooks

.. autofunction:: bayesian_lora.main.save_input_hook

.. autofunction:: bayesian_lora.main.save_output_grad_hook
