.. _bayesian_lora:

Bayesian Lora
=============

This file contains the main methods relating to the `Bayesian Low-Rank
Adaptation for Large Language Models
<https://openreview.net/forum?id=FJiUyzOF1m>`_ paper. Namely, calculating the
model evidence for tuning prior and network hyperparameters and calculating the
posterior precision for making (linearised) predictions.

Model Evidence
--------------

The model evidence, or marginal likelihood, is a scalar value that indicates the
evidence provided by the data for a particular model. A model with a higher
marginal likelihood is considered more supported by the data under the given
prior.

.. autofunction:: bayesian_lora.main.model_evidence

Posterior Precision
-------------------

.. autofunction:: bayesian_lora.main.precision

