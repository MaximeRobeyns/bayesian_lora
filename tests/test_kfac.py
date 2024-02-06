"""
K-FAC tests
"""

import pytest
import torch as t
import torch.nn as nn

from torch import Tensor
from typing import Any
from jaxtyping import Float
from torch.linalg import LinAlgError
from torch.utils.data import TensorDataset, DataLoader

from bayesian_lora.kfac import (
    stable_cholesky,
    incremental_svd,
    calculate_kronecker_factors,
)


def test_ill_conditioned_matrix():
    """Test with an ill-conditioned matrix."""
    # Create an ill-conditioned matrix
    ill_cond_matrix = t.rand((10, 10))
    ill_cond_matrix = ill_cond_matrix @ ill_cond_matrix.T  # Make it symmetric
    ill_cond_matrix[0, 0] = 1e-8  # Introduce ill-conditioning

    # Verify that the matrix really is ill-conditioned
    with pytest.raises(LinAlgError):
        L = t.linalg.cholesky(ill_cond_matrix)

    # Test if stable Cholesky decomposition succeeds
    L = stable_cholesky(ill_cond_matrix)
    assert isinstance(L, t.Tensor)
    assert not t.isnan(L).any()


def test_well_conditioned_matrix():
    """Test with a well-conditioned matrix."""
    well_cond_matrix = t.rand((10, 10))
    well_cond_matrix = well_cond_matrix @ well_cond_matrix.T + t.eye(10)

    # Test if Cholesky decomposition succeeds
    L = stable_cholesky(well_cond_matrix)
    assert isinstance(L, t.Tensor)
    assert not t.isnan(L).any()


def test_non_square_matrix():
    """Test with a non-square matrix."""
    non_square_matrix = t.rand((10, 9))

    with pytest.raises(Exception):
        stable_cholesky(non_square_matrix)


def test_zero_matrix():
    """Test with a zero matrix."""
    zero_matrix = t.zeros((10, 10))

    L = stable_cholesky(zero_matrix)
    assert isinstance(L, t.Tensor)
    assert not t.isnan(L).any()


def test_incremental_svd():
    d, n_kfac, batch = 1024, 10, 16
    A = t.randn(d, n_kfac)
    a = t.randn(batch, d)
    B = incremental_svd(A, a)
    assert A.shape == B.shape
    assert not t.isnan(B).any()


class _TestingModel(nn.Module):
    def __init__(self, features: list[int], bias: bool = False):
        super().__init__()
        self.net = nn.Sequential()
        for i, (j, k) in enumerate(zip(features[:-1], features[1:])):
            self.net.add_module(name=f"FC{i}", module=nn.Linear(j, k, bias=bias))
            if i < len(features) - 2:
                self.net.add_module(name=f"A{i}", module=nn.ReLU())
                self.net.add_module(name=f"LN{i}", module=nn.LayerNorm(k))
            else:
                self.net.add_module(name=f"SM{i}", module=nn.Softmax(-1))

    def forward(self, x: Float[Tensor, "b n"]) -> Float[Tensor, "b m"]:
        return self.net(x).softmax(-1)


def fwd_call(model: nn.Module, batch: Any) -> Float[Tensor, "batch out_params"]:
    xs, _ = batch
    logits = model(xs)
    logits = logits[:, -1]  # emulate selecting the last token
    return logits


def test_full_rank_kfac():
    N, S, bs = 100, 8, 16
    features = [10, 20, 5]
    tmp_model = _TestingModel(features)
    xs, ys = t.randn(N, S, features[0]), t.randn(N, S, features[-1])
    loader = DataLoader(TensorDataset(xs, ys), batch_size=bs)

    # Sanity check test setup
    for b in loader:
        xs, ys = b
        assert xs.shape == (bs, S, features[0])
        assert ys.shape == (bs, S, features[-1])
        out = fwd_call(tmp_model, b)
        assert out.shape == (bs, features[-1])
        break

    factors = calculate_kronecker_factors(
        tmp_model, fwd_call, loader, target_module_keywords=["FC"]
    )

    assert factors is not None
    assert len(factors) == len(features) - 1
    for i, (k, (A, S)) in enumerate(factors.items()):
        n, m = features[i], features[i + 1]
        assert A.shape == (n, n), f"Unexpected shape for {k}:A"
        assert S.shape == (m, m), f"Unexpected shape for {k}:S"


def test_low_rank_kfac():
    N, S, bs = 100, 8, 16
    n_kfac, lr_threshold = 4, 128
    features = [256, 256, 10]
    tmp_model = _TestingModel(features)
    xs, ys = t.randn(N, S, features[0]), t.randn(N, S, features[-1])
    loader = DataLoader(TensorDataset(xs, ys), batch_size=bs)

    factors = calculate_kronecker_factors(
        tmp_model,
        fwd_call,
        loader,
        n_kfac=n_kfac,
        lr_threshold=128,
        target_module_keywords=["FC"],
    )

    assert factors is not None
    assert len(factors) == len(features) - 1
    for i, (k, (A, S)) in enumerate(factors.items()):
        n, m = features[i], features[i + 1]
        if n < lr_threshold:
            assert A.shape == (n, n), f"Unexpected shape for {k}:A"
        else:
            assert A.shape == (n, n_kfac), f"Unexpected shape for {k}:A"
        if m < lr_threshold:
            assert S.shape == (m, m), f"Unexpected shape for {k}:S"
        else:
            assert S.shape == (m, n_kfac), f"Unexpected shape for {k}:S"


def test_low_rank_kfac_lora_like():
    """
    LoRA-like alternating feature shapes
    """
    N, S, bs = 100, 8, 16
    n_kfac, lr_threshold = 4, 128
    features = [256, 32, 256, 32, 512]
    tmp_model = _TestingModel(features)
    xs, ys = t.randn(N, S, features[0]), t.randn(N, S, features[-1])
    loader = DataLoader(TensorDataset(xs, ys), batch_size=bs)

    factors = calculate_kronecker_factors(
        tmp_model,
        fwd_call,
        loader,
        n_kfac=n_kfac,
        lr_threshold=128,
        target_module_keywords=["FC"],
    )

    assert factors is not None
    assert len(factors) == len(features) - 1
    for i, (k, (A, S)) in enumerate(factors.items()):
        n, m = features[i], features[i + 1]
        if n < lr_threshold:
            assert A.shape == (n, n), f"Unexpected shape for {k}:A"
        else:
            assert A.shape == (n, n_kfac), f"Unexpected shape for {k}:A"
        if m < lr_threshold:
            assert S.shape == (m, m), f"Unexpected shape for {k}:S"
        else:
            assert S.shape == (m, n_kfac), f"Unexpected shape for {k}:S"
