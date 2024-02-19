# Copyright (C) 2023-24 Maxime Robeyns <dev@maximerobeyns.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Bayesian low-rank adaptation.
"""

import logging
import torch as t
import torch.nn as nn

from torch import Tensor
from typing import Callable
from jaxtyping import Float
from torch.func import jacrev, functional_call
from transformers import BatchEncoding
from transformers.modeling_outputs import ModelOutput

from .kfac import stable_cholesky, KFAC_t, activation_t, outgrad_t

__all__ = ["model_evidence", "variance", "cholesky_decompose_small_factors"]


def calc_M(
    activations: activation_t,
    output_grads: outgrad_t,
    n_lora: int,
    n_kfac: int,
    s2: t.Tensor,
    return_LB: bool = False,
) -> t.Tensor | tuple[
    Float[Tensor, "n_lora_x_n_kfac n_lora_x_n_kfac"],
    tuple[Float[Tensor, "n_lora n_lora"], Float[Tensor, "d n_kfac"]] | None,
]:
    """
    Calculates the `M` matrix in Eq. 32 of https://openreview.net/forum?id=FJiUyzOF1m

    Most conventional uses of this library should not need to call this
    function 'externally'.

    Args:
        activations: matrix of uncentred input activation covariances
        output_grads: matrix of uncentred output gradient covariances
        n_lora: LoRA rank
        n_kfac: low rank to use with Kronecker factors
        s2: prior variance
        return_LB: whether to return the `L` and `B` matrices; where
        - `L` is the e.g. Cholesky factorisation of small Kronecker factor with
          shape (n_lora, n_lora), and
        - `B` is the low-rank factorization of the large Kronecker factor with
          shape (d, n_kfac)

    Returns:
        The `M` matrix, and optionally the `L` and `B` matrices too.
    """
    if activations.shape[-2:] == (n_lora, n_lora):
        L, B = (activations, output_grads)
    else:
        B, L = activations, output_grads
    assert L.shape[-2:] == (n_lora, n_lora)
    assert B.shape[-1:] == (n_kfac,)

    M_size = n_lora * n_kfac
    I = t.eye(M_size, device=L.device, dtype=L.dtype)
    M = I + s2 * t.kron(B.mT @ B, L.mT @ L)
    assert M.shape == (M_size, M_size)

    if return_LB:
        return M, (L, B)
    return M


def cholesky_decompose_small_factors(
    factors: KFAC_t, lr_threshold: int, device: str, dtype: t.dtype
) -> KFAC_t:
    """
    Compute the Cholesky factors for the full-rank (smaller) Kronecker
    factors

    Args:
        factors (dict[str, tuple[t.Tensor, t.Tensor]]): the Kronecker factors
        lr_threshold: the threshold beyond which a Kronecker factor is
            considered large and a low-rank approximation is applied.
        device: device to use
        dtype: datatype to store factors in on disk
    Returns:
        Kronecker factors, with small factors Cholesky decomposed.
    """
    for name, (A, S) in factors.items():
        if A.size(0) < lr_threshold:
            A = stable_cholesky(A.to(dtype=t.float64))
        if S.size(0) < lr_threshold:
            S = stable_cholesky(S.to(dtype=t.float64))
        factors[name] = (A.to(device, dtype), S.to(device, dtype))
    return factors


def model_evidence(
    model: nn.Module,
    LL: t.Tensor,
    factors: KFAC_t,
    n_lora: int,
    n_kfac: int,
    s2: Float[Tensor, "1"],
) -> Float[Tensor, "1"]:
    """
    Use this function to calculate the marginal likelihood / model evidence;
    for instance to tune the value of s2 (prior variance).

    Args:
        model: your model
        LL: the log likelihood on a dataset of interest
        factors: dictionary of Kronecker factors
        n_lora: LoRA rank
        n_kfac: K-FAC rank
        s2: prior variance

    Returns:
        model evidence
    """
    logdet = t.tensor(0.0)
    d = 1

    for (A, S) in factors.values():
        d = max(A.shape + S.shape)

        M = calc_M(A, S, n_lora, n_kfac, s2)
        assert isinstance(M, t.Tensor)
        M = M.to(dtype=t.float64)
        _, slogdet = t.slogdet(M)
        logdet = logdet.to(dtype=A.dtype) + slogdet.to(dtype=A.dtype)
    logdet += -n_lora * d * t.log(s2)

    map_norms = 0.0
    # TODO: is this a reliable way of identifying the LoRA parameters?
    lora_params = {
        k: v
        for k, v in dict(model.named_parameters()).items()
        if "lora" in k.lower() and v.requires_grad
    }
    for param in lora_params.values():
        map_norms += t.linalg.norm(param)
    model_evidence = LL + 1 / s2 * map_norms + 0.5 * logdet
    return model_evidence


def default_output_callback(outputs: ModelOutput) -> Tensor:
    """Post process model outputs.

    This function will be passed the results of model(**batch_inputs), and
    should return the relevant logits. For multiple-choice tasks, this is
    the class logits, but for full next-token prediction, this would just
    be all the logits.
    """
    # Get the last token for CausalLM
    logits = outputs.logits if cfg.llm.is_s2s else outputs.logits[:, -1]
    # Select the logits corresponding to our target classes
    target_logits = logits[:, dset.target_ids]
    return target_logits


def jacobian_mean(
    model: nn.Module,
    batch_inputs: BatchEncoding,
    target_ids: Tensor | None = None,
    is_s2s: bool = False,
    output_callback: Callable[[ModelOutput], Tensor] | None = None,
) -> tuple[dict[str, Tensor], Tensor]:
    """Calculates the Jacobian and logit means

    Args:
        model: the LoRA LLM from which to make predictions
        batch_inputs: the batch inputs, exactly as you would pass them into
            your model with ``model(**inputs)``.
        target_ids: selects specific model outputs. Leave this as None if
            either a) you wish to consider all model outputs or b) you are
            providing an output_callback to post-process the model output.
        is_s2s: whether this is an s2s model. Can omit if providing an
            output_callback
        output_callback: a function that takes the results of
            ``model(**batch_inputs)`` and returns the logits of interest
    Returns:
        The Jacobian (a dictionary of module keys and Jacobian Tensors) and the
        logit mean predictions.
    """

    if output_callback is None:

        def ocb(outputs: ModelOutput) -> Tensor:
            logits = outputs.logits if cfg.llm.is_s2s else outputs.logits[:, -1]
            if target_ids is not None:
                logits = logits[:, target_ids]
            return logits

        output_callback = ocb

    def f(
        model: nn.Module, lora_params: dict[str, Tensor], batch_inputs: BatchEncoding
    ):
        outputs = functional_call(model, lora_params, args=(), kwargs=batch_inputs)
        target_logits = output_callback(outputs)
        return target_logits, target_logits

    # Get the LoRA parameters
    # TODO: ensure that these are the same LoRA adapters as applied to the
    # modules targeted in ``calculate_kronecker_factors``.
    lora_params = {
        k: v for k, v in dict(model.named_parameters()).items() if v.requires_grad
    }
    # Sanity check
    for k in lora_params.keys():
        assert "lora" in k.lower()

    # Calculate the Jacobian of each LoRA layer (and mean predictions)
    jacobian, f_mu = jacrev(f, argnums=1, has_aux=True)(
        model, lora_params, batch_inputs
    )
    return jacobian, f_mu


def variance(
    inputs,
    jacobian,
    factors: KFAC_t,
    s2: t.Tensor,
    n_logits: int,
    n_lora: int,
    n_kfac: int,
    device: str,
):
    """
    Calculates the variance matrix for performing (linearised) prediction.

    Args:
        inputs (dict): tokenized batch of inputs (returned from a HF Tokenizer)
        jacobian (dict): a dictionary of first derivatives for each of the
            target module's parameters
        factors: dictionary of Kronecker factors
        s2: prior variance (scalar valued tensor)
        n_logits: the number of  logits to predict (e.g. the number of classes
            in your Categorical likelihood)
        n_lora: rank used in the LoRA adapters
        n_kfac: rank used for the low-rank approximation of large Kronekcer
            factors
        device: device on which to accumulate the variance matrix
    """
    jac_keys = jacobian.keys()

    batch_size = inputs.input_ids.size(0)

    # initialise a matrix to accumulate the result
    var_matrix = t.zeros((batch_size, n_logits, n_logits), device=device)

    # Iterate over the layers; `k` is the layer name / key, `A` is the input
    # activations and `S` are the output gradients.
    for k, (A, S) in factors.items():
        # Jacobian term
        # TODO: make this less brittle ----------------------------------------
        # g_key = "base_model.model." + k + ".weight"
        # g_key = k + ".weight"
        g_key = None
        for jac_key in jac_keys:
            if k in jac_key:
                g_key = jac_key
                break
        assert (
            g_key is not None
        ), f"Could not find weight corresponding to kronecker factor {k}"
        # ---------------------------------------------------------------------

        G = jacobian.get(g_key).squeeze()
        # Ensure that G is [batch, n_logits, d, n_lora] sized at all times
        if G.shape[-1] != n_lora:
            G = G.mT
        assert G.shape[-1] == n_lora

        # Flatten the last 2 dimensions; giving [batch, n_logits, d * n_lora]
        G_vec = G.flatten(-2)
        term_1 = s2 * G_vec @ G_vec.mT
        assert term_1.shape == (batch_size, n_logits, n_logits)

        M, LB = calc_M(A, S, n_lora, n_kfac, s2, return_LB=True)
        assert LB is not None
        L, B = LB
        M_size = n_kfac * n_lora
        assert M.shape == (M_size, M_size)
        M = M.to(dtype=t.float64)

        B_expanded = B.mT[None, None, :]  # [1, 1, n_kfc, d]
        L_expanded = L[None, None, :]  # [1, 1, n_lora, n_lora]
        BGL = B_expanded @ G @ L_expanded
        BGL_vec = BGL.flatten(-2).to(dtype=t.float64)  # [batch, n_logits, M_size]
        term_2 = s2.pow(2.0) * BGL_vec @ t.linalg.inv(M) @ BGL_vec.mT
        assert term_2.shape == (batch_size, n_logits, n_logits)

        var_matrix += term_1 - term_2.to(var_matrix.dtype)

        logging.debug(f"After layer {k}, variance is {var_matrix}")
    return var_matrix
