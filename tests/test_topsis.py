import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from src.optimization.topsis import TOPSIS

def test_topsis_invalid_weights():
    # Weights must sum to 1.0
    weights = np.array([0.4, 0.4, 0.4])
    topsis = TOPSIS(
        weights=weights,
        benefit_criteria=[False, False, False],
        criteria_names=["c1", "c2", "c3"]
    )
    toy_matrix = np.array([[10, 20, 30], [20, 30, 40]])
    with pytest.raises(AssertionError):
        topsis.fit(toy_matrix)

def test_topsis_fit_dimensions():
    weights = np.array([0.5, 0.2, 0.3])
    topsis = TOPSIS(
        weights=weights,
        benefit_criteria=[False, False, False],
        criteria_names=["c1", "c2", "c3"]
    )
    toy_matrix = np.array([
        [10, 20, 30],
        [15, 25, 35],
        [20, 15, 25]
    ])
    topsis.fit(toy_matrix)
    
    assert topsis.is_fitted is True
    assert topsis.closeness.shape == (3,)
    assert topsis.ranks.shape == (3,)

def test_topsis_manual_math_verification():
    # Weights: w = [0.5, 0.5]
    # Matrix:
    # Sol 0: [2, 8]
    # Sol 1: [8, 2]
    # Both are cost criteria (minimize)
    weights = np.array([0.5, 0.5])
    topsis = TOPSIS(
        weights=weights,
        benefit_criteria=[False, False],
        criteria_names=["c1", "c2"]
    )
    toy_matrix = np.array([
        [2.0, 8.0],
        [8.0, 2.0]
    ])
    topsis.fit(toy_matrix)
    
    # Check normalization: norm = sqrt(2^2 + 8^2) = sqrt(68) = 8.2462
    # Normalized:
    # Sol 0: [2/8.2462, 8/8.2462] = [0.2425, 0.9701]
    # Sol 1: [0.9701, 0.2425]
    # Weighted:
    # Sol 0: [0.1213, 0.4851]
    # Sol 1: [0.4851, 0.1213]
    # A+ (min) = [0.1213, 0.1213]
    # A- (max) = [0.4851, 0.4851]
    # Distance to A+ (best):
    # Sol 0: sqrt((0.1213-0.1213)^2 + (0.4851-0.1213)^2) = 0.3638
    # Sol 1: sqrt((0.4851-0.1213)^2 + (0.1213-0.1213)^2) = 0.3638
    # Distance to A- (worst):
    # Sol 0: sqrt((0.1213-0.4851)^2 + (0.4851-0.4851)^2) = 0.3638
    # Sol 1: sqrt((0.4851-0.4851)^2 + (0.1213-0.4851)^2) = 0.3638
    # Closeness: C = D- / (D+ + D-) = 0.3638 / 0.7276 = 0.5
    # Since they are perfectly symmetric, closeness should be exactly equal
    assert abs(topsis.closeness[0] - 0.5) < 1e-4
    assert abs(topsis.closeness[1] - 0.5) < 1e-4

def test_topsis_rank_ordering():
    # Weights: w = [0.5, 0.5], cost criteria
    # Matrix:
    # Sol 0: [2.0, 2.0] (Very good/low cost)
    # Sol 1: [8.0, 8.0] (Very bad/high cost)
    weights = np.array([0.5, 0.5])
    topsis = TOPSIS(
        weights=weights,
        benefit_criteria=[False, False],
        criteria_names=["c1", "c2"]
    )
    toy_matrix = np.array([
        [2.0, 2.0],
        [8.0, 8.0]
    ])
    topsis.fit(toy_matrix)
    
    # Sol 0 should be rank 1, Sol 1 should be rank 2
    assert topsis.ranks[0] == 1
    assert topsis.ranks[1] == 2
    assert topsis.closeness[0] > topsis.closeness[1]

def test_topsis_latex_generation():
    weights = np.array([0.5, 0.5])
    topsis = TOPSIS(
        weights=weights,
        benefit_criteria=[False, False],
        criteria_names=["c1", "c2"]
    )
    toy_matrix = np.array([
        [2.0, 2.0],
        [8.0, 8.0]
    ])
    topsis.fit(toy_matrix)
    latex_out = topsis.to_latex()
    
    assert "begin{table}" in latex_out
    assert "toprule" in latex_out
    assert "bottomrule" in latex_out
    assert "Rank" in latex_out
