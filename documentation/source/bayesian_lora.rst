.. _bayesian_lora:

Bayesian Lora
=============

This file contains the main methods relating to the `Bayesian Low-Rank
Adaptation for Large Language Models
<https://openreview.net/forum?id=FJiUyzOF1m>`_ paper. Namely, calculating the
model evidence for tuning prior and network hyperparameters and calculating the
posterior predictive parameters for making (linearised) predictions.

Model Evidence
--------------

The model evidence, or marginal likelihood, is a scalar value that indicates the
evidence provided by the data for a particular model. A model with a higher
marginal likelihood is considered more supported by the data under the given
prior.

.. autofunction:: bayesian_lora.main.model_evidence


Posterior Predictive
--------------------

This involves two steps, calculating the mean and the variance.

For the first, we invoke the (admittedly, awkwardly named) ``jacobian_mean``
function, which returns the Jacobian, and the mean, respectively.

.. autofunction:: bayesian_lora.main.jacobian_mean

As you can see, there are two ways of calling this function, which determine how
we'll handle the outputs from the wrapped network call.

1. **Directly, with parameters** Here, we assume that a model is either a
   sequence-to-sequence model or not (defaults to ``False``), and that we may
   optionally want to pick out some specific logits from the model's full
   vocabulary:


   .. code-block:: py

       jacobian, f_mu = jacobian_mean(
           model, batch_inputs, target_ids=dset.target_ids, is_s2s=False
       )

2. **Custom output callback** Here, we allow the user to provide a callback
   function, taking in the result of the model's ``forward`` call, and returning
   the logits of interest, with arbitrary post-processing in between.

   .. code-block:: py

       def default_output_callback(outputs: ModelOutput) -> Tensor:
          logits = outputs.logits if cfg.llm.is_s2s else outputs.logits[:, -1]
          target_logits = logits[:, dset.target_ids]
          return target_logits

       jacobian, f_mu = jacobian_mean(
           model, batch_inputs, output_callback=output_callback
       )

For the second step, we calculate the output logits' covariance matrix.

.. autofunction:: bayesian_lora.main.variance
