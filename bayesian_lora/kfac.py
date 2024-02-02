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
Kronecker-factored approximate curvature methods.
"""

import sys
import logging
import torch as t
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from torch import Tensor
from typing import Any, Callable, Optional
from jaxtyping import Float
from contextlib import contextmanager
from torch.linalg import LinAlgError
from torch.utils.data import DataLoader
from torch.utils.hooks import RemovableHandle


__all__ = [
    "stable_cholesky",
    "calculate_kronecker_factors",
    "calculate_full_kronecker_factors",
]


# Utility functions ===========================================================


def stabilise(
    K: Float[Tensor, "... d d"], mult_eps: float, abs_eps: float
) -> Float[Tensor, "... d d"]:
    """Multiply and add the stabilisation terms `mult_eps` and `abs_eps`"""
    eye = t.eye(K.shape[-1], dtype=K.dtype, device=K.device)
    return K * (1.0 + mult_eps * eye) + abs_eps * eye


def stable_cholesky(
    K: Float[Tensor, "... d d"],
    mult_eps: float = 1e-8,
    abs_eps: float = 1e-8,
    max_tries: int = 1000,
) -> Float[Tensor, "... d d"]:
    for i in range(max_tries):
        try:
            scaled_mult_eps = mult_eps * (1.1**i)
            scaled_abs_eps = abs_eps * (1.1**i)
            # L = t.linalg.cholesky(stabilise(K, i * mult_eps, i * abs_eps))
            L = t.linalg.cholesky(stabilise(K, scaled_mult_eps, scaled_abs_eps))
            return L
        except LinAlgError:
            logging.debug(f"Chokesky decomposition failed ({i})")
            continue
    raise ValueError(f"Could not calculate Cholesky decomposition of {K}")


def incremental_svd(
    A: Float[Tensor, "d r"],
    a: Float[Tensor, "batch d"],
    dtype: t.dtype = t.float64,
    n_kfac: Optional[int] = None,
) -> Float[Tensor, "d n_kfac"]:
    """Calculate a low-rank estimate of a big [d, d] tensor, without
    materialising this full matrix.

    Args:
        A: The accumulated low-rank factor
        a: a new batch of points
        dtype: the datatype to use for the svd
        n_kfac: (optional) specify the rank of the resulting factor. If
            omitted, we use `r` from the `A` argument.
            You may choose to set this higher than the final rank during the
            accumulation.
    """
    if n_kfac is None:
        n_kfac = A.size(-1)
    a = a.to(dtype=dtype)
    A_prime = t.hstack((A, a.T))  # [d, r+batch]
    U, S, _ = t.linalg.svd(A_prime, full_matrices=False)
    return U[:, :n_kfac] @ t.diag(S[:n_kfac])


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
    n_kfac: Optional[int] = 10,
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
        if not use_lr or n_kfac is None:
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
            activations[module_name] = incremental_svd(
                activations[module_name], a, svd_dtype, n_kfac
            )

    return input_hook


def save_output_grad_hook(
    module_name: str,
    output_grads: dict[str, t.Tensor],
    dtype: t.dtype = t.float32,
    n_kfac: Optional[int] = 10,
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
        # TODO: detach or not here?
        # s = out_pos_grad[0].detach()[:, -1]
        s = out_pos_grad[0][:, -1].to(dtype=dtype)
        if not use_lr or n_kfac is None:
            # No LR; just do the outer product of the output gradients for all
            # elements in the batch, then sum along the batch dimension:
            S = (s[..., None] @ s[:, None]).sum(0)
            if module_name not in output_grads.keys():
                output_grads[module_name] = S
            else:
                output_grads[module_name] += S
        else:
            # Never reach this branch if n_kfac is None
            if module_name not in output_grads.keys():
                # Initialise a correctly sized matrix of 0s
                output_grads[module_name] = t.zeros(
                    s.size(-1), n_kfac, device=s.device, dtype=s.dtype
                )
            s = s.to(dtype=svd_dtype)
            output_grads[module_name] = incremental_svd(
                output_grads[module_name], s, svd_dtype, n_kfac
            )

    return output_grad_hook


def register_hooks(
    model: nn.Module,
    activations: dict[str, t.Tensor],
    output_grads: dict[str, t.Tensor],
    target_module_keywords: list[str],
    n_kfac: Optional[int] = 10,
    lr_threshold: int = 100,
    exclude_bias: bool = False,
) -> tuple[list, dict]:
    """Registers the activation and output gradient hooks.


    Args:
        model: the `nn.Module` on which to attach the hooks
        activations: dictionary in which to store the parameter activations
        output_grads: dictionary in which to store the output gradients
        target_module_keywords: a list of the network modules to include in the
            GGN. Note, only nn.Linear layers are currently supported.
        n_kfac: the rank we use to approximate large Kronecker factors. If set
            to None, we treat all factors as full rank (turns off the lr
            approximation).
        lr_threshold: threshold beyond which to consider a layer's input to be
            wide (to decide whether to approximate a Kronecker factor as low
            rank). LoRA layers with a wide input (e.g. LoRA-A) will have a
            low-rank approximation of their activation Kronecker factor, A,
            while LoRA layers with a narrow input (e.g. LoRA-B) will have a
            low-rank approximation of their output-gradient Kronecker factor,
            S.
        exclude_bias: whether to ignore bias terms (just consider the weights)

    Returns:
        - a list of hooks (for later removal),
        - a map indicating whether a layer has a wide input
    """
    hooks: list[RemovableHandle] = []
    has_wide_input: dict[str, bool] = dict()
    for name, module in model.named_modules():
        if any([kw in name for kw in target_module_keywords]) and (
            isinstance(module, nn.Linear)
        ):
            logging.debug(f"Registering hook for module {name}")
            if name in activations.keys() or name in output_grads.keys():
                raise Exception(f"Module of same name {name} already registered")
            has_bias = hasattr(module, "bias") and module.bias is not None
            if exclude_bias:
                has_bias = False
            if n_kfac is None:
                has_wide_input[name] = True
            else:
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


def remove_hooks(hooks: list) -> None:
    """Remove the hooks from the module.

    Args:
        hooks: list of hooks, returned from `register_hooks`
    """
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
    exclude_bias: bool = False,
    use_tqdm: bool = False,
) -> dict[str, tuple[t.Tensor, t.Tensor]]:
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
        exclude_bias: whether to ignore bias terms
        use_tqdm: whether to show progress with TQDM

    Warning:
        This function has only been implemented for nn.Linear. Models
        implemented using Conv1D (e.g. GPT2) will sadly not work for now.

    Returns:
        A dictionary containing the Kronecker factors; keyed by module name,
        containing a tuple (A, S) with the activation factor (A) as the first
        element, and the output gradient factor (S) as the second element.
    """
    model = model.train()

    activations: dict[str, t.Tensor] = dict()
    output_grads: dict[str, t.Tensor] = dict()

    # activations, output_grads = dict(), dict()
    hooks, has_wide_input = register_hooks(
        model,
        activations,
        output_grads,
        target_module_keywords,
        n_kfac,
        lr_threshold,
        exclude_bias=exclude_bias,
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

    factors = {
        k: (A, S) for (k, A), (_, S) in zip(activations.items(), output_grads.items())
    }

    return factors


def calculate_full_kronecker_factors(
    model: nn.Module,
    forward_call: Callable[[nn.Module, Any], t.Tensor],
    loader: DataLoader,
    device: str,
    dtype: Optional[t.dtype] = None,
    target_module_keywords: list[str] = ["lora"],
    exclude_bias: bool = False,
    use_tqdm: bool = False,
) -> tuple[dict[str, t.Tensor], dict[str, t.Tensor]]:
    """Full-rank Kronecker factor calculation

    Args:
        model: the model with LoRA adapters, for which we are calculating the
            Kronecker factors
        forward_call: A function which accepts a batch from the provided data
            loader, and returns the logits from model's predictive
            distribution
        loader: a data loader for the dataset with which to calculate the
            curvature / Kronecker factors
        device: device to use
        dtype: datatype to store factors in on disk. If omitted, same dtype as
            current model parameters is used.
        target_module_keywords: a list of keywords which identify the network
            modules whose parameters we want to include in the Hessian
            calculation
        exclude_bias: whether to ignore bias terms
        use_tqdm: whether to show progress with TQDM

    Returns:
        A dictionary containing the Kronecker factors; keyed by module name,
        containing a tuple (A, S) with the activation factor (A) as the first
        element, and the output gradient factor (S) as the second element.
    """
    # TODO: wrap `calculate_kronecker_factors`; passing in appropriate arguments

    model = model.train()

    activations, output_grads = dict(), dict()
    hooks, _ = register_hooks(
        model,
        activations,
        output_grads,
        target_module_keywords,
        n_kfac=None,
        exclude_bias=exclude_bias,
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

        pullback_loss = F.cross_entropy(logits, sampled_ys)

        with disable_input_hooks():
            pullback_loss.backward()

        t.cuda.empty_cache()

    activations = {n: A.to(device, dtype) for n, A in activations.items()}
    output_grads = {n: S.to(device, dtype) for n, S in output_grads.items()}

    remove_hooks(hooks)

    return activations, output_grads