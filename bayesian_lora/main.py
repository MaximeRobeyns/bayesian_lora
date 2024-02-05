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

from bayesian_lora.kfac import stable_cholesky, KFAC_t

__all__ = ["model_evidence", "precision", "cholesky_decompose_small_factors"]


def calc_M(
    activations: t.Tensor,
    output_grads: t.Tensor,
    n_lora: int,
    n_kfac: int,
    s2: t.Tensor,
    return_LB: bool = False,
) -> t.Tensor | tuple[t.Tensor, tuple[t.Tensor, t.Tensor]]:
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
    s2: t.Tensor,
) -> t.Tensor:
    """
    Use this function to calculate the marginal likelihood / model evidence;
    for instance to tune the value of s2 (prior variance).

    Args:
        model: your model
        LL: the log likelihood on a dataset of interest
        factors: dictionary of Kronecker factors
        n_lora: LoRA rank
        n_kfac: rank to use in low-rank approximation of large Kronecker factors
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
        k: v for k, v in dict(model.named_parameters()).items() if v.requires_grad
    }
    for param in lora_params.values():
        map_norms += t.linalg.norm(param)
    model_evidence = LL + 1 / s2 * map_norms + 0.5 * logdet
    return model_evidence


def precision(
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
    Calculates the precision matrix for performing (linearised) prediction.

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
        device: device on which to accumulate the precision matrix
    """

    batch_size = inputs.input_ids.size(0)

    # initialise a matrix to accumulate the result
    precision = t.zeros((batch_size, n_logits, n_logits), device=device)

    # Iterate over the layers; `k` is the layer name / key, `A` is the input
    # activations and `S` are the output gradients.
    for k, (A, S) in factors.items():
        # Jacobian term
        g_key = k + ".weight"
        G = jacobian.get(g_key).squeeze()
        # Ensure that G is [batch, n_logits, d, n_lora] sized at all times
        if G.shape[-1] != n_lora:
            G = G.mT
        assert G.shape[-1] == n_lora

        # Flatten the last 2 dimensions; giving [batch, n_logits, d * n_lora]
        G_vec = G.flatten(-2)
        term_1 = s2 * G_vec @ G_vec.mT
        assert term_1.shape == (batch_size, n_logits, n_logits)

        M, (L, B) = calc_M(A, S, n_lora, n_kfac, s2, return_LB=True)
        M_size = n_kfac * n_lora
        assert M.shape == (M_size, M_size)
        M = M.to(dtype=t.float64)

        B_expanded = B.mT[None, None, :]  # [1, 1, n_kfc, d]
        L_expanded = L[None, None, :]  # [1, 1, n_lora, n_lora]
        BGL = B_expanded @ G @ L_expanded
        BGL_vec = BGL.flatten(-2).to(dtype=t.float64)  # [batch, n_logits, M_size]
        term_2 = s2.pow(2.0) * BGL_vec @ t.linalg.inv(M) @ BGL_vec.mT
        assert term_2.shape == (batch_size, n_logits, n_logits)

        precision += term_1 - term_2.to(precision.dtype)

        logging.debug(f"After layer {k}, precision is {precision}")
    return precision
