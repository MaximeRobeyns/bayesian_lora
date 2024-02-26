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
from typing import Any, Callable
from jaxtyping import Float
from contextlib import contextmanager
from torch.linalg import LinAlgError
from torch.utils.data import DataLoader
from torch.utils.hooks import RemovableHandle


__all__ = [
    "stable_cholesky",
    "calculate_kronecker_factors",
    "activation_t",
    "outgrad_t",
    "KFAC_t",
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
    n_kfac: int | None = None,
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

# Datatype for Kronecker factors. l_in, l_out refers to the number of input and
# output features of layer l in the network, respectively. Note, if the layer
# has a bias, then l_in will in fact bt l_in + 1.
activation_t = Float[Tensor, "l_in l_in_or_n_kfac"]
outgrad_t = Float[Tensor, "l_out l_out_or_n_kfac"]
KFAC_t = dict[str, tuple[activation_t, outgrad_t]]

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
    activations: dict[str, tuple[activation_t, bool]],
    n_kfac: int | None,
    lr_threshold: int,
    has_bias: bool = False,
    svd_dtype: t.dtype = t.float64,
):
    """A closure which returns a new hook to capture a layer's input
    activations.

    Args:
        module_name: name used as a key for the 'activations' dictionary. While
            modules themselves can be hashed, this makes the Kronecker factors
            more portable.
        activations: mapping from layer / module name to input activation
            Kronecker factor, and a flag indicating whether it is low-rank
        n_kfac: the rank we use if we're using a low rank appproximation to
            this Kronecker factor
        lr_threshold: if the side length `l_in+1` exceeds this threshold, and
            n_kfac is not none, treat the factor as low-rank
        has_bias: does this layer have a bias?
        svd_dtype: dtype to cast tensors to for SVD calculations
    """

    def input_hook(_module: nn.Module, pos_args: tuple[t.Tensor]) -> None:
        if not _hooks_enabled or _input_hooks_disabled:
            return
        # Select the first positional argument given to this layer (the input
        # activation), then the last token in the token sequence [:, -1]. `a`
        # should be a [batch, l_in] tensor.
        a: Float[Tensor, "batch l_in"] = pos_args[0].detach().clone()[:, -1]
        if has_bias:
            a = t.hstack((a, t.ones_like(a[:, :1])))
        assert a.dim() == 2
        if a.size(-1) < lr_threshold or n_kfac is None:
            # We're not using a low-rank approximation for this factor; just do
            # the outer product of the activations for all the elements in the
            # batch, then sum along batch dim:
            A = (a[..., None] @ a[:, None]).sum(0)
            if module_name not in activations.keys():
                activations[module_name] = A, False
            else:
                A_tmp = activations[module_name][0]
                activations[module_name] = A_tmp + A, False
        else:
            if module_name not in activations.keys():
                # Initialise a correctly sized matrix of 0s
                activations[module_name] = (
                    t.zeros(a.size(-1), n_kfac, device=a.device, dtype=svd_dtype),
                    True,
                )
            A = incremental_svd(activations[module_name][0], a, svd_dtype, n_kfac)
            activations[module_name] = A, True

    return input_hook


def save_output_grad_hook(
    module_name: str,
    output_grads: dict[str, tuple[outgrad_t, bool]],
    n_kfac: int | None,
    lr_threshold: int,
    svd_dtype: t.dtype = t.float64,
):
    """A closure which returns a new hook to capture a layer's output
    gradients.

    Args:
        module_name: name used as a key for the 'output_grads' dictionary.
            While modules themselves can be hashed, this makes the Kronecker
            factors more portable.
        output_grads: mapping from layer / module name to the output gradient
            Kronecker factor, and a flag indicating whether it is low-rank.
        n_kfac: the rank we use if we're using a low rank appproximation to
            this Kronecker factor
        lr_threshold: if the side length `l_in+1` exceeds this threshold, and
            n_kfac is not none, treat the factor as low-rank
        svd_dtype: dtype to cast tensors to for SVD calculations
    """

    def output_grad_hook(_module: nn.Module, _, out_pos_grad: tuple[Tensor]) -> None:
        if not _hooks_enabled:
            return

        # Select the gradient of the first positional output of this layer,
        # then the last token in the token sequence [:, -1]. `s` should be a
        # [batch, l_out] tensor.
        s: Float[Tensor, "batch l_out"] = out_pos_grad[0].detach().clone()[:, -1]
        if s.size(-1) < lr_threshold or n_kfac is None:
            # We're not using a low-rank approximation for this factor; just do
            # the outer product of the output gradients for all elements in the
            # batch, then sum along the batch dimension; giving an [l_out,
            # l_out] tensor.
            S = (s[..., None] @ s[:, None]).sum(0)
            if module_name not in output_grads.keys():
                output_grads[module_name] = S, False
            else:
                S_tmp = output_grads[module_name][0]
                output_grads[module_name] = S_tmp + S, False
        else:
            # Never reach this branch if n_kfac is None
            if module_name not in output_grads.keys():
                # Initialise a correctly sized matrix of 0s
                output_grads[module_name] = (
                    t.zeros(s.size(-1), n_kfac, device=s.device, dtype=s.dtype),
                    True,
                )
            S = incremental_svd(output_grads[module_name][0], s, svd_dtype, n_kfac)
            output_grads[module_name] = S, True

    return output_grad_hook


def register_hooks(
    model: nn.Module,
    activations: dict[str, tuple[activation_t, bool]],
    output_grads: dict[str, tuple[outgrad_t, bool]],
    target_module_keywords: list[str],
    n_kfac: int | None = 10,
    lr_threshold: int = 100,
    exclude_bias: bool = False,
) -> list[RemovableHandle]:
    """Registers the activation and output gradient hooks.

    Args:
        model: the ``nn.Module`` on which to attach the hooks (usually the full
            model)
        activations: dictionary in which to store the parameter activations and
            flag indicating whether this factor is low-rank.
            The side length is ``l_in`` (i.e. equal to the number of input
            features in layer ``l``), or ``l_in + 1`` if there is a bias. The
            last dimension is ``n_kfac`` if ``l_in >= lr_threshold``.
        output_grads: dictionary in which to store the output gradients and a
            flag indicating whether this factor is low-rank. The side length
            ``l_out`` is equal to the number of output features of layer ``l``
            (regardless of the presence of a bias; unlike the activations). The
            last dimension is ``n_kfac`` if ``l_out >= lr_threshold``
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
    """
    hooks: list[RemovableHandle] = []
    for name, module in model.named_modules():
        if any([kw in name for kw in target_module_keywords]) and (
            isinstance(module, nn.Linear)
        ):
            logging.debug(f"Registering hook for module {name}")
            if name in activations.keys() or name in output_grads.keys():
                raise Exception(f"Module of same name {name} already registered")
            has_bias = hasattr(module, "bias") and module.bias is not None
            if exclude_bias:
                # NOTE: this is a hack that should be removed
                has_bias = False
            fwd_hook = module.register_forward_pre_hook(
                save_input_hook(name, activations, n_kfac, lr_threshold, has_bias)
            )
            bwd_hook = module.register_full_backward_hook(
                save_output_grad_hook(name, output_grads, n_kfac, lr_threshold)
            )
            hooks.extend((fwd_hook, bwd_hook))

    return hooks


def remove_hooks(hooks: list[RemovableHandle]) -> None:
    """Remove the hooks from the module.

    Args:
        hooks: list of hooks, returned from `register_hooks`
    """
    while len(hooks):
        hooks.pop().remove()


def calculate_kronecker_factors(
    model: nn.Module,
    forward_call: Callable[[nn.Module, Any], Float[Tensor, "batch n_classes"]],
    loader: DataLoader,
    n_kfac: int | None = None,
    lr_threshold: int = 512,
    target_module_keywords: list[str] = [""],
    exclude_bias: bool = False,
    use_tqdm: bool = False,
) -> KFAC_t:
    """
    Calculate the Kronecer factors, (A, S) for the likelihood, used to
    approximate the GGN / Fisher.

    Args:
        model: the model for which we are calculating the Kronecker factors.
            Note that it needn't have LoRA adapters.
        forward_call: A function which accepts a batch from the provided data
            loader, and returns the parameters of the model's predictive
            distribution, as a ``Tensor``. Usually this contains the logits
            over each class label.
        loader: a data loader for the dataset with which to calculate the
            curvature / Kronecker factors.
        n_kfac: an optional integer rank to use for a low-rank approximation of
            large Kronecker factors. If this is ``None``, then no low-rank
            approximations are used.
        lr_threshold: the threshold beyond which the side length of a Kronecker
            factor is considered large and a low-rank approximation is applied.
        target_module_keywords: a list of keywords which identify the network
            modules whose parameters we want to include in the Hessian
            calculation. This is particularly useful when working with LoRA
            adapters. By deafult, this is ``[""]``; targetting every module.
        exclude_bias: whether to ignore bias terms (NOTE: this is a hack and
            should not be used)
        use_tqdm: whether to show progress with ``tqdm``.

    Warning:
        This function has only been implemented for nn.Linear. Models
        implemented using Conv1D (e.g. GPT2) will sadly not work for now.

    Warning:
        Your data loader should not have a partial final batch, since this will
        result in an incorrect expectation. You can drop the final batch with
        `drop_last=True` in a standard PyTorch DataLoader.

    Examples:

        Full-rank Kronecker factor calculation.

        >>> factors = calculate_kronecker_factors(
        >>>     model, fwd_call, loader
        >>> )

        Low-rank Kronecker factors on LoRA adaptors with inputs

        >>> factors = calculate_kronecker_factors(
        >>>     model, fwd_call, loader, n_kfac=10,
        >>>     lr_threshold=512, target_module_keywords=["lora"],
        >>> )


    Returns:
        A dictionary containing the Kronecker factors; keyed by module name,
        containing a tuple (A, S) with the activation factor (A) as the first
        element, and the output gradient factor (S) as the second element.
    """
    model = model.train()

    activations: dict[str, tuple[t.Tensor, bool]] = dict()
    output_grads: dict[str, tuple[t.Tensor, bool]] = dict()

    hooks = register_hooks(
        model,
        activations,
        output_grads,
        target_module_keywords,
        n_kfac,
        lr_threshold,
        exclude_bias=exclude_bias,
    )

    for batch in tqdm(loader, disable=not use_tqdm, file=sys.stdout):
        model.zero_grad()
        logits = forward_call(model, batch)
        assert logits.dim() == 2

        with t.no_grad():
            sampled_ys = t.multinomial(logits.softmax(-1), 1).view(-1)

        # TODO: support other model distributions
        # We use the mean reduction here to keep the magnitude of the output
        # gradients invariant to the batch size used when calculating the
        # Kronecker factors.
        pullback_loss = F.cross_entropy(logits, sampled_ys, reduction="mean")

        with disable_input_hooks():
            pullback_loss.backward()

        t.cuda.empty_cache()

    remove_hooks(hooks)
    factors: KFAC_t = dict()
    for k, (A, A_lr) in activations.items():
        S, S_lr = output_grads[k]
        # Average only the non low-rank factors.
        if not S_lr:
            S = S / len(loader)
        if not A_lr:
            A = A / len(loader)
        factors[k] = A, S

    return factors
