.. _kfac:

K-FAC Methods
=============

The :mod:`bayesian_lora.kfac` module provides functions for calculating
an approximate Fisher information matrix (or GGN) using Kronecker-factored
approximate curvature.

Recall that K-FAC first finds a block-diagonal approximation to the full Fisher
/ GGN. If we had a simple 4-layer network, then this would be:

.. figure:: _static/block_diagonal.svg
           :align: center
           :width: 70%
           :alt: Block-diagonal approximation

Eeach of these blocks (:math:`\mathbf{G}_{\ell \ell}`) are further
approximated as the product of two Kronecker factors, one corresponding to the
input *activations*, :math:`\mathbf{A}_{\ell-1}`, and another to the *output
gradients*, :math:`\mathbf{S}_{\ell}`. That is, for a particular layer /
``nn.Module`` indexed by :math:`\ell`, we approximate its block of the full
Fisher as

.. math::
    :label: kfacblock

    \mathbf{G}_{\ell\ell} \approx \mathbf{A}_{\ell-1} \otimes \mathbf{S}_{\ell}.

These factors (curvature information around the network's current parameters)
are calculated over some dataset :math:`\mathcal{D}`, and this is what the
:func:`bayesian_lora.calculate_kronecker_factors` function below calculates.

Rather than using numerical indices :math:`\ell \in \{1, 2, \ldots, L\}`, we use
the ``nn.Module``'s name to identify the different blocks, and return the
factors in dictionaries of type ``dict[str, t.Tensor]``.

Full-Rank K-FAC
---------------

The simplest variant is a *full-rank* Kronecker factorisation, meaning that we
store the :math:`\mathbf{A}` and :math:`\mathbf{S}` matrices exactly.

.. autofunction:: bayesian_lora.calculate_kronecker_factors

Notice how these Kronecker factors can themselves be approximated as low-rank
which is particularly useful for LLMs, where the factors may be :math:`4096
\times 4096` for each layer in a transformer.

Internal Functions
------------------

The above is the main way to use the K-FAC functionality from this library.
It calls a number of internal functions, which we document here for re-use and
completeness.

.. autofunction:: bayesian_lora.kfac.register_hooks

.. autofunction:: bayesian_lora.kfac.remove_hooks

.. autofunction:: bayesian_lora.kfac.save_input_hook

.. autofunction:: bayesian_lora.kfac.save_output_grad_hook
