"""
K-FAC tests
"""

import pytest
import torch as t

from torch.linalg import LinAlgError

from bayesian_lora.kfac import stable_cholesky, incremental_svd


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