"""
Simple example unit test.
"""

import pytest


def test_example():
    assert 2 + 2 == 4


@pytest.mark.slow
def test_example_slow():
    for i in range(int(1e8)):
        assert i == i
