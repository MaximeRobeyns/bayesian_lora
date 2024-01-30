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
Bayesian low-rank adaptation, using K-FAC.

Copy this file into your project to hack on this library.
See example.py for example usage.

Structure:
    1. utility functions
    2. K-FAC functions
    3. Bayesian LoRA
"""

import sys
import logging
import torch as t
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from typing import Any, Callable, Optional
from contextlib import contextmanager
from torch.utils.data import DataLoader

# Utility functions ===========================================================


def stabilise(K: t.Tensor, mult_eps: float, abs_eps: float) -> t.Tensor:
    """Multiply and add the stabilisation terms `mult_eps` and `abs_eps`"""
    eye = t.eye(K.shape[-1], dtype=K.dtype, device=K.device)
    return K * (1.0 + mult_eps * eye) + abs_eps * eye


def stable_cholesky(
    K, mult_eps: float = 1e-8, abs_eps: float = 1e-8, max_tries: int = 1000
) -> t.Tensor:
    # NOTE: this loop rarely runs for more than 2 tries
    # TODO: add stabilisation terms on a nonlinear schedule (i)
    for i in range(max_tries):
        try:
            L = t.linalg.cholesky(stabilise(K, i * mult_eps, i * abs_eps))
            return L
        except t._C._LinAlgError:
            logging.debug(f"Chokesky decomposition failed ({i})")
            continue
    raise ValueError(f"Could not calculate Cholesky decomposition of {K}")


# K-FAC Section ===============================================================

# We add hooks to the nn.Module (e.g. AutoModelForCausalLM, PeftModel, etc) to
# keep track of each layer's input activations and output gradients.
# The following context managers let you enable / disable them without removing
# them.

_hooks_enabled: bool = True
_input_hooks_disabled: bool = False


@contextmanager
def hooks_disabled():
    """
    Allows the hooks for both the activations and output gradients to be
    temporarily disabled within a context.

    Example:
    >>> with hooks_disabled():
    >>>     output = model(**inputs)
    >>>     output.loss.backward()
    """
    global _hooks_enabled
    orig_state = _hooks_enabled
    _hooks_enabled = False
    try:
        yield
    finally:
        _hooks_enabled = orig_state


@contextmanager
def disable_input_hooks():
    """
    Disables just the input activation hooks but keeps the output gradient
    hooks. Useful when calculating a 'pullback' metric.

    Example:
    >>> with disable_input_hooks():
    >>>     loss.backward()
    """
    global _input_hooks_disabled
    orig_state = _input_hooks_disabled
    _input_hooks_disabled = True
    try:
        yield
    finally:
        _input_hooks_disabled = orig_state


def save_input_hook(
    module_name: str,
    activations: dict[str, t.Tensor],
    has_bias: bool = False,
    n_kfac: int = 10,
    use_lr: bool = False,
    svd_dtype: t.dtype = t.float64,
):
    """A closure which returns a new hook to capture a layer's input
    activations.

    Args:
        module_name: name used as a key for the 'activations' dictionary. While
            modules themselves can be hashed, this makes the Kronecker factors
            more portable.
        activations: mapping from layer / module name to input activation
            Kronecker factor
        has_bias: does this layer have a bias?
        n_kfac: the rank we use if we're using a low rank appproximation to
            this Kronecker factor
        use_lr: are we using a low-rank approximation to this Kronecker factor?
        svd_dtype: dtype to cast tensors to for SVD calculations
    """

    def input_hook(_module: nn.Module, pos_args: tuple[t.Tensor]) -> None:
        if not _hooks_enabled or _input_hooks_disabled:
            return
        # Select the first positional argument given to this layer (the input
        # activation), then the last token in the token sequence [:, -1]. `a`
        # should be a [batch, layer_in_dims] tensor.
        a = pos_args[0].clone().detach()[:, -1]
        if has_bias:
            a = t.hstack((a, t.ones_like(a[:, :1])))
        assert a.dim() == 2
        if not use_lr:
            # No LR; just do the outer product of activations for all elements
            # in the batch, then sum along the batch dimension:
            A = (a[..., None] @ a[:, None]).sum(0)
            if module_name not in activations.keys():
                activations[module_name] = A
            else:
                activations[module_name] += A
        else:
            if module_name not in activations.keys():
                # Initialise a correctly sized matrix of 0s
                activations[module_name] = t.zeros(
                    a.size(-1), n_kfac, device=a.device, dtype=svd_dtype
                )
            a = a.to(dtype=svd_dtype)
            # Compute an incremental SVD with a straightforward procedure
            A_prime = t.hstack((activations[module_name], a.T))
            U, S, _ = t.linalg.svd(A_prime, full_matrices=False)
            activations[module_name] = U[:, :n_kfac] @ t.diag(S[:n_kfac])

    return input_hook


def save_output_grad_hook(
    module_name: str,
    output_grads: dict[str, t.Tensor],
    dtype: t.dtype = t.float32,
    n_kfac: int = 10,
    use_lr: bool = False,
    svd_dtype: t.dtype = t.float64,
):
    """A closure which returns a new hook to capture a layer's output
    gradients.

    Args:
        module_name: name used as a key for the 'output_grads' dictionary.
            While modules themselves can be hashed, this makes the Kronecker
            factors more portable.
        output_grads: mapping from layer / module name to the output gradient
            Kronecker factor.
        n_kfac: the rank we use if we're using a low rank appproximation to
            this Kronecker factor
        use_lr: are we using a low-rank approximation to this Kronecker factor?
        svd_dtype: dtype to cast tensors to for SVD calculations
    """

    def output_grad_hook(_module: nn.Module, _, out_pos_grad: tuple[t.Tensor]) -> None:
        if not _hooks_enabled:
            return

        # Select the gradient of the first positional output of this layer,
        # then the last token in the token sequence [:, -1]. `s` should be a
        # [batch, layer_out_dims] tensor.
        # s = out_pos_grad[0].detach()[:, -1]
        s = out_pos_grad[0][:, -1].to(dtype=dtype)
        if not use_lr:
            # No LR; just do the outer product of the output gradients for all
            # elements in the batch, then sum along the batch dimension:
            S = (s[..., None] @ s[:, None]).sum(0)
            if module_name not in output_grads.keys():
                output_grads[module_name] = S
            else:
                output_grads[module_name] += S
        else:
            if module_name not in output_grads.keys():
                # Initialise a correctly sized matrix of 0s
                output_grads[module_name] = t.zeros(
                    s.size(-1), n_kfac, device=s.device, dtype=s.dtype
                )
            s = s.to(dtype=svd_dtype)
            # Compute an incremental SVD with a straightforward procedure
            S_prime = t.hstack((output_grads[module_name], s.T))
            U, S, _ = t.linalg.svd(S_prime, full_matrices=False)
            output_grads[module_name] = U[:, :n_kfac] @ t.diag(S[:n_kfac])

    return output_grad_hook


def register_hooks(
    model: nn.Module,
    activations: dict[str, t.Tensor],
    output_grads: dict[str, t.Tensor],
    target_module_keywords: list[str],
    n_kfac: int = 10,
    lr_threshold: int = 100,
) -> tuple[list, dict]:
    """Registers the activation and output gradient hooks.


    Args:
        model: the `nn.Module` on which to attach the hooks
        activations: dictionary in which to store the parameter activations
        output_grads: dictionary in which to store the output gradients
        target_module_keywords: a list of the network modules to include in the
            GGN. Note, only nn.Linear layers are currently supported.
        n_kfac: the rank we use to approximate large Kronecker factors
        lr_threshold: threshold beyond which to consider a layer's input to be
            wide (to decide whether to approximate a Kronecker factor as low
            rank). LoRA layers with a wide input (e.g. LoRA-A) will have a
            low-rank approximation of their activation Kronecker factor, A,
            while LoRA layers with a narrow input (e.g. LoRA-B) will have a
            low-rank approximation of their output-gradient Kronecker factor,
            S.

    Returns:
        - a list of hooks (for later removal),
        - a map indicating whether a layer has a wide input
    """
    hooks = []
    has_wide_input: dict[str, bool] = dict()
    for name, module in model.named_modules():
        if any([kw in name for kw in target_module_keywords]) and (
            isinstance(module, nn.Linear)
        ):
            logging.info(f"Registering hook for module {name}")
            if name in activations.keys() or name in output_grads.keys():
                raise Exception(f"Module of same name {name} already registered")
            has_bias = hasattr(module, "bias") and module.bias is not None
            has_wide_input[name] = module.in_features > lr_threshold
            fwd_hook = module.register_forward_pre_hook(
                save_input_hook(
                    name, activations, has_bias, n_kfac, use_lr=has_wide_input[name]
                )
            )
            bwd_hook = module.register_full_backward_hook(
                save_output_grad_hook(
                    name, output_grads, n_kfac=n_kfac, use_lr=not has_wide_input[name]
                )
            )
            hooks.extend((fwd_hook, bwd_hook))
    return hooks, has_wide_input


def remove_hooks(hooks: list):
    while len(hooks):
        hooks.pop().remove()


def calculate_kronecker_factors(
    model: nn.Module,
    forward_call: Callable[[nn.Module, Any], t.Tensor],
    loader: DataLoader,
    n_kfac: int,
    lr_threshold: int,
    device: str,
    dtype: Optional[t.dtype] = None,
    target_module_keywords: list[str] = ["lora"],
    use_tqdm: bool = False,
) -> tuple[dict[str, t.Tensor], dict[str, t.Tensor]]:
    """
    Calculate the Kronecer factors, (A, S) for the likelihood, used to
    approximate the GGN / Fisher.

    Args:
        model: the model with LoRA adapters, for which we are calculating the
            Kronecker factors
        forward_call: A function which accepts a batch from the provided data
            loader, and returns the logits from model's predictive
            distribution
        loader: a data loader for the dataset with which to calculate the
            curvature / Kronecker factors
        n_kfac: rank to use for the low-rank approximatino of large Kronecker
            factors
        lr_threshold: the threshold beyond which a Kronecker factor is
            considered large
        device: device to use
        dtype: datatype to store factors in on disk. If omitted, same dtype as
            current model parameters is used.
        target_module_keywords: a list of keywords which identify the network
            modules whose parameters we want to include in the Hessian
            calculation
        use_tqdm: whether to show progress with TQDM

    Warning:
        This function has only been implemented for nn.Linear. Models
        implemented using Conv1D (e.g. GPT2) will sadly not work for now.

    Returns:
        1. A dictionary of activation factors
        2. A dictionary of output gradient factors
        TODO: merge (A, S) into the same dictionary?
    """
    model = model.train()

    activations, output_grads = dict(), dict()
    hooks, has_wide_input = register_hooks(
        model, activations, output_grads, target_module_keywords, n_kfac, lr_threshold
    )
    if dtype is None or not isinstance(dtype, t.dtype):
        for p in model.parameters():
            if p.dtype.is_floating_point:
                dtype = p.dtype

    for batch in tqdm(loader, disable=not use_tqdm, file=sys.stdout):
        model.zero_grad()
        logits = forward_call(model, batch)
        assert logits.dim() == 2

        with t.no_grad():
            sampled_ys = t.multinomial(logits.softmax(-1), 1).view(-1)

        # TODO: support other model distributions
        pullback_loss = F.cross_entropy(logits, sampled_ys)

        with disable_input_hooks():
            pullback_loss.backward()

        t.cuda.empty_cache()

    # Compute the Cholesky factors for the full-rank (smaller) Kronecker
    # factors
    for name, A in activations.items():
        # Activations of LoRA layers with wide inputs are low rank
        is_low_rank = has_wide_input[name]
        if is_low_rank:
            activations[name] = A.to(device, dtype)
        else:
            L = stable_cholesky(A.to(dtype=t.float64))
            activations[name] = L.to(device, dtype)

    for name, S in output_grads.items():
        # Output gradients of LoRA layers with wide outputs are low rank
        is_low_rank = not has_wide_input[name]
        if is_low_rank:
            output_grads[name] = S.to(device, dtype)
        else:
            L = stable_cholesky(S.to(dtype=t.float64))
            output_grads[name] = L.to(device, dtype)

    remove_hooks(hooks)

    return activations, output_grads


# Functions for Bayesian LoRA =================================================


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


def model_evidence(
    model: nn.Module,
    LL: t.Tensor,
    activations: dict[str, t.Tensor],
    output_grads: dict[str, t.Tensor],
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
        activations: dictionary of the 'activation' Kronecker factors
        output_grads: dictionary of the 'output gradient' Kronecker factors
        n_lora: LoRA rank
        n_kfac: rank to use in low-rank approximation of large Kronecker factors
        s2: prior variance

    Returns:
        model evidence
    """
    logdet = 0.0
    d = 1

    for layer_name, A in activations.items():
        S = output_grads[layer_name]
        d = max(A.shape + S.shape)

        M = calc_M(A, S, n_lora, n_kfac, s2).to(dtype=t.float64)
        _, slogdet = t.slogdet(M)
        logdet += slogdet.to(dtype=A.dtype)
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
    activations: dict[str, t.Tensor],
    output_grads: dict[str, t.Tensor],
    s2: t.Tensor,
    n_logits: int,
    n_lora: int,
    n_kfac: int,
    device: str,
):
    """
    Calculates the precision matrix for linearized prediction.
    """

    batch_size = inputs.input_ids.size(0)

    # initialise a matrix to accumulate the result
    precision = t.zeros((batch_size, n_logits, n_logits), device=device)

    # Iterate over the layers; `k` is the layer name / key, `Act` is the input
    # activations.
    for k, A in activations.items():
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

        S = output_grads[k]
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
