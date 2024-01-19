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
        a = pos_args[0].detach()[:, -1]
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
        s = out_pos_grad[0].detach()[:, -1]
        if not use_lr:
            # No LR; just do the outer product of the output gradients for all
            # elements in the batch, then sum along the batch dimension:
            S = (s[..., None] @ s[:, None]).sum(-1)
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
    n_kfac: int = 10,
    lr_threshold: int = 100,
) -> tuple[list, dict]:
    """Registers the activation and output gradient hooks.


    Args:
        model: the `nn.Module` on which to attach the hooks
        activations: dictionary in which to store the parameter activations
        output_grads: dictionary in which to store the output gradients
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
        if "lora" in name.lower() and (isinstance(module, nn.Linear)):
            logging.debug(f"Registering hook for module {name}")
            has_bias = hasattr(module, "bias") and module.bias is not None
            has_wide_input[name] = module.in_features > lr_threshold
            fwd_hook = module.register_forward_pre_hook(
                save_input_hook(
                    name, activations, has_bias, use_lr=has_wide_input[name]
                )
            )
            bwd_hook = module.register_full_backward_hook(
                save_output_grad_hook(
                    name, output_grads, n_kfac, not has_wide_input[name]
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
    use_tqdm: bool = False,
) -> tuple[dict[str, t.Tensor], dict[str, t.Tensor]]:
    """
    Calculate the Kronecer factors, (A, S) for the likelyhood, used to
    approximate the Gauss-Newton Hessian.

    Args:
        model: the model with LoRA adapters for which to calculate the
            Kronecker factors
        forward_call: A function which accepts a batch from the provided data
            loader, and returns the resulting logits from model's predictive
            distribution.
        loader: a data loader on which to calculate the Kronecker factors
        n_kfac: rank to use for the low-rank approximatino of large Kronecker
            factor's
        lr_threshold: the threshold beyond which a Kronecker factor is
            considered large
        device: device to use
        dtype: datatype to use
        use_tqdm: whether to show progress with TQDM

    Returns:
        1. A dictionary of activation factors
        2. A dictionary of output gradient factors

        TODO: merge these into the same dictionary?
    """
    model = model.train()

    activations, output_grads = dict(), dict()
    hooks, has_wide_input = register_hooks(
        model, activations, output_grads, n_kfac, lr_threshold
    )

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