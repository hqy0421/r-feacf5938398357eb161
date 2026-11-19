# -*- coding: utf-8 -*-
# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Tests for joint (ellipsoidal) multi-output conformal prediction."""

import numpy as np
import pytest

from deel.puncc.api.joint import EllipsoidalSet
from deel.puncc.api.joint import JointCalibrator
from deel.puncc.api.joint import JointCP
from deel.puncc.api.joint import JointEllipsoid
from deel.puncc.api.joint import column_space_coordinates
from deel.puncc.api.joint import column_space_norm
from deel.puncc.api.joint import column_space_norm_builder
from deel.puncc.api.joint import linear_functional_bounds
from deel.puncc.api.utils import quantile

N_OUTPUTS = 4
N_COMPONENTS = 2
L4 = np.array(
    [
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [0.5, -1.0],
    ]
)


def _target(X):
    x = np.asarray(X, dtype=float).reshape(len(X), -1)[:, 0]
    return np.column_stack((x, 2 * x, -x, x**2))


class _FixedPredictor:
    """Fixed predictor exposing only a ``predict`` method."""

    def predict(self, X):
        return _target(X)


class _FailingPredictor:
    def predict(self, X):
        raise ValueError("boom")


def _make_dataset(n, seed=0, noise=1.0):
    rng = np.random.RandomState(seed)
    X = rng.uniform(-1.0, 1.0, size=(n, 1))
    # Noise lives exactly in the column space of L4, so scores stay finite.
    y = _target(X) + rng.randn(n, N_COMPONENTS) @ L4.T * noise
    return X, y


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Scores ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def test_column_space_norm_matches_coordinates_norm():
    rng = np.random.RandomState(0)
    z = rng.randn(20, N_COMPONENTS)
    residuals = z @ L4.T  # exactly in the column space
    y_pred = np.zeros_like(residuals)

    scores = column_space_norm(y_pred, residuals, L4)
    coords, outside = column_space_coordinates(y_pred, residuals, L4)

    assert not np.any(outside)
    assert np.allclose(scores, np.linalg.norm(z, axis=-1))
    assert np.allclose(coords, z)


def test_column_space_norm_outside_space_is_infinite():
    residual = np.array([[0.0, 0.0, 0.0, 1.0]])  # not in span(L4)
    scores = column_space_norm(np.zeros_like(residual), residual, L4)
    assert np.isinf(scores[0])


def test_column_space_norm_builder_matches_function():
    rng = np.random.RandomState(1)
    y_pred = rng.randn(10, N_OUTPUTS)
    y_true = _target(np.zeros((10, 1))) + rng.randn(10, N_COMPONENTS) @ L4.T

    score_func = column_space_norm_builder(L4)
    assert np.allclose(
        score_func(y_pred, y_true), column_space_norm(y_pred, y_true, L4)
    )


def test_column_space_builder_rejects_invalid_mapping():
    with pytest.raises(ValueError, match="full column rank"):
        column_space_norm_builder(np.array([[1.0, 2.0], [2.0, 4.0]]))


def test_column_space_functions_validate_shapes_and_values():
    good = np.zeros((3, N_OUTPUTS))
    with pytest.raises(ValueError, match="same shape"):
        column_space_norm(good, np.zeros((2, N_OUTPUTS)), L4)
    with pytest.raises(ValueError, match="same shape"):
        column_space_coordinates(good, np.zeros((2, N_OUTPUTS)), L4)
    with pytest.raises(ValueError, match="finite"):
        column_space_norm(good, np.full((3, N_OUTPUTS), np.nan), L4)
    with pytest.raises(ValueError, match="real-valued"):
        column_space_norm(good, np.zeros((3, N_OUTPUTS), dtype=complex), L4)
    with pytest.raises(ValueError, match="per sample"):
        column_space_norm(good, np.zeros((3, 3)), L4)
    with pytest.raises(ValueError, match="float array"):
        column_space_norm(good, "not an array", L4)
    with pytest.raises(ValueError, match="converted"):
        column_space_norm(good, np.array([["a"] * N_OUTPUTS] * 3), L4)

    # A single sample of length d and a scalar for 1-D outputs are accepted.
    assert column_space_norm(np.zeros(N_OUTPUTS), np.zeros(N_OUTPUTS), L4).shape == (1,)
    one_d = JointCalibrator(np.eye(1))
    assert one_d.as_outputs(3.0).shape == (1, 1)
    assert one_d.as_outputs(np.array([1.0, 2.0])).shape == (2, 1)
    assert column_space_norm(0.0, 1.0, np.eye(1)).shape == (1,)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Mapping validation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


@pytest.mark.parametrize(
    "bad_L",
    [
        np.zeros((4,)),  # not 2-D
        np.zeros((3, 4)),  # more columns than rows
        np.array([[1.0, 2.0], [2.0, 4.0]]),  # rank deficient
        np.zeros((4, 0)),  # no columns
        np.array([[1.0, np.nan], [0.0, 1.0]]),  # non finite
    ],
)
def test_invalid_mapping_raises(bad_L):
    with pytest.raises(ValueError):
        JointCalibrator(bad_L)


def test_low_dimensional_column_space_is_supported():
    L1 = np.ones((N_OUTPUTS, 1))  # k = 1 < d
    calib = JointCalibrator(L1)
    rng = np.random.RandomState(0)
    z = rng.randn(50, 1)
    y_true = z @ L1.T
    y_pred = np.zeros_like(y_true)
    calib.fit(y_true=y_true, y_pred=y_pred)
    assert np.all(np.isfinite(calib.get_nonconformity_scores()))
    assert calib.compute_radius(0.1) > 0


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Radius ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def test_radius_uses_native_corrected_quantile():
    rng = np.random.RandomState(2)
    y_pred = rng.randn(100, 1)
    y_true = y_pred + rng.randn(100, 1)
    calib = JointCalibrator(np.eye(1))
    calib.fit(y_true=y_true, y_pred=y_pred)

    scores = np.abs(y_true[:, 0] - y_pred[:, 0])
    expected = quantile(np.concatenate([scores, np.array([np.inf])]), 1 - 0.1)

    assert calib.compute_radius(0.1) == pytest.approx(expected)


def test_radius_too_small_alpha_raises():
    calib = JointCalibrator(np.eye(1))
    calib.fit(y_true=np.zeros((5, 1)), y_pred=np.zeros((5, 1)))
    with pytest.raises(ValueError):
        calib.compute_radius(0.001)


def test_radius_before_fit_raises():
    with pytest.raises(RuntimeError):
        JointCalibrator(L4).compute_radius(0.1)


def test_radius_becomes_infinite_with_out_of_space_residual():
    N = 60
    X, y = _make_dataset(N, seed=3)
    y[:2] += np.array([0.0, 0.0, 0.0, 1.0])  # leaves the column space

    cp = JointCP(_FixedPredictor(), L4)
    cp.fit(X, y)
    assert np.isinf(cp.get_nonconformity_scores()).sum() == 2

    # The corrected quantile selects the ceil((1-alpha)(n+1))-th smallest
    # score. Two +inf scores are reached for alpha <= 0.03 with n=60, while
    # alpha = 0.1 selects a finite score in between.
    assert np.isinf(cp.compute_radius(0.03))
    assert np.isfinite(cp.compute_radius(0.1))
    # Even a single out-of-space score can be selected, but only for very
    # small alpha; below 1/(n+1) the usual "alpha too small" error applies.
    single = _make_dataset(N, seed=3)[1]
    single[0] += np.array([0.0, 0.0, 0.0, 1.0])
    cp_single = JointCP(_FixedPredictor(), L4)
    cp_single.fit(X, single)
    assert np.isinf(cp_single.get_nonconformity_scores()).sum() == 1
    assert np.isinf(cp_single.compute_radius(0.02))
    assert np.isfinite(cp_single.compute_radius(0.1))
    with pytest.raises(ValueError, match="Alpha is too small"):
        cp_single.compute_radius(0.01)


def test_radius_infinite_requires_enough_out_of_space_scores():
    N = 60
    X, y = _make_dataset(N, seed=3)
    # 6 <= floor(alpha * (n+1)) out-of-space scores push +inf into the
    # selected order statistic for alpha = 0.1 (floor(6.1) = 6).
    y[:6] += np.array([0.0, 0.0, 0.0, 1.0])

    cp = JointCP(_FixedPredictor(), L4)
    cp.fit(X, y)
    assert np.isinf(cp.get_nonconformity_scores()).sum() == 6
    assert np.isinf(cp.compute_radius(0.1))


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Bounds ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def test_linear_functional_bounds_are_tight():
    rng = np.random.RandomState(4)
    center = rng.randn(N_OUTPUTS)
    radius = 1.3
    pred_set = JointEllipsoid(center, L4, radius)

    a = np.array([1.0, -2.0, 0.5, 1.0])
    lower, upper = pred_set.linear_bounds(a, 0.25)

    samples = pred_set.sample(200000, random_state=0)
    values = samples @ a + 0.25

    assert values.max() <= upper
    assert values.min() >= lower
    assert upper - values.max() < 0.05 * (upper - lower)
    assert values.min() - lower < 0.05 * (upper - lower)


def test_linear_functional_bounds_closed_form():
    center = np.array([1.0, -1.0, 2.0, 0.5])
    radius = 2.0
    A = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0]])
    b = np.array([1.0, -2.0])

    lower, upper = linear_functional_bounds(center, L4, radius, A, b)
    expected_half_width = radius * np.linalg.norm(A @ L4, axis=1)
    expected_mid = A @ center + b

    assert np.allclose(lower, expected_mid - expected_half_width)
    assert np.allclose(upper, expected_mid + expected_half_width)


def test_bounds_are_not_a_box():
    """Joint bounds of a functional beat the box obtained by summing widths."""
    center = np.zeros(N_OUTPUTS)
    pred_set = JointEllipsoid(center, L4, 1.0)
    a = np.ones(N_OUTPUTS)

    lower, upper = pred_set.linear_bounds(a, 0.0)

    # Per-coordinate box assembled from marginal bounds.
    marginal_lo, marginal_hi = pred_set.linear_bounds()
    box_width = np.sum(marginal_hi - marginal_lo)
    joint_width = upper - lower

    assert joint_width <= box_width
    # The joint width is the exact norm of a.L, not the sum of |components|.
    assert joint_width == pytest.approx(2 * np.linalg.norm(L4.T @ a))


def test_bounds_of_constant_query_are_finite_with_infinite_radius():
    pred_set = JointEllipsoid(np.zeros(N_OUTPUTS), L4, np.inf)
    lower, upper = pred_set.linear_bounds(np.zeros(N_OUTPUTS), 3.25)
    assert lower == pytest.approx(3.25)
    assert upper == pytest.approx(3.25)

    lower, upper = pred_set.linear_bounds(np.ones(N_OUTPUTS), 0.0)
    assert lower == -np.inf
    assert upper == np.inf


def test_bounds_batch_of_centers():
    centers = np.array([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
    pred_set = JointEllipsoid(centers, L4, 0.5)
    a = np.array([1.0, -1.0, 0.0, 0.0])

    lower, upper = pred_set.linear_bounds(a, 0.0)
    assert lower.shape == (2,)
    assert np.allclose(lower, centers @ a - 0.5 * np.linalg.norm(L4.T @ a))
    assert np.allclose(upper, centers @ a + 0.5 * np.linalg.norm(L4.T @ a))


def test_bounds_dimension_errors():
    pred_set = JointEllipsoid(np.zeros(N_OUTPUTS), L4, 1.0)
    with pytest.raises(ValueError):
        pred_set.linear_bounds(np.ones(3))
    with pytest.raises(ValueError):
        pred_set.linear_bounds(np.ones(N_OUTPUTS), np.ones(2))
    with pytest.raises(ValueError, match="last dimension"):
        pred_set.linear_bounds(np.ones((2, 3)))
    with pytest.raises(ValueError, match="at least one row"):
        pred_set.linear_bounds(np.zeros((0, N_OUTPUTS)))
    with pytest.raises(ValueError, match="2-D matrix"):
        pred_set.linear_bounds(np.zeros((2, 2, N_OUTPUTS)))
    with pytest.raises(ValueError, match="to match `A`"):
        pred_set.linear_bounds(np.eye(N_OUTPUTS), np.ones(3))
    with pytest.raises(ValueError, match="single ellipsoid center"):
        linear_functional_bounds(np.zeros((2, N_OUTPUTS)), L4, 1.0)
    with pytest.raises(ValueError, match="non-negative"):
        linear_functional_bounds(np.zeros(N_OUTPUTS), L4, -1.0)


def test_bounds_single_query_returns_scalars():
    lower, upper = linear_functional_bounds(
        np.zeros(N_OUTPUTS), L4, 1.0, np.ones(N_OUTPUTS), 2.0
    )
    assert isinstance(lower, float) and isinstance(upper, float)

    pred_set = JointEllipsoid(np.zeros(N_OUTPUTS), L4, 1.0)
    lower, upper = pred_set.linear_bounds(np.ones(N_OUTPUTS))
    assert isinstance(lower, float) and isinstance(upper, float)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Membership ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def test_contains_in_and_out():
    center = np.array([0.5, -0.5, 1.0, 0.0])
    radius = 1.0
    pred_set = JointEllipsoid(center, L4, radius)

    inside = center + L4 @ np.array([0.5, 0.5])  # norm ~0.707 < 1
    boundary = center + L4 @ np.array([0.0, 1.0])  # norm exactly 1
    outside = center + L4 @ np.array([1.0, 1.0])  # norm ~1.414 > 1
    off_space = center + np.array([0.0, 0.0, 0.0, 0.5])  # leaves the space

    assert pred_set.contains(inside)
    assert pred_set.contains(boundary)
    assert not pred_set.contains(outside)
    assert not pred_set.contains(off_space)


def test_contains_vectorised():
    centers = np.tile(_target(np.zeros((1, 1))), (5, 1))
    pred_set = JointEllipsoid(centers, L4, 1.0)
    rng = np.random.RandomState(5)

    inside = centers + 0.1 * rng.randn(5, N_COMPONENTS) @ L4.T
    outside = centers + 10.0 * rng.randn(5, N_COMPONENTS) @ L4.T
    assert pred_set.contains(inside).all()
    assert not pred_set.contains(outside).any()


def test_contains_all_when_radius_is_infinite():
    pred_set = JointEllipsoid(np.zeros(N_OUTPUTS), L4, np.inf)
    arbitrary = np.array([1e9, -1e9, 3.0, -7.0])
    assert pred_set.contains(arbitrary)
    assert pred_set.is_whole_space


def test_in_operator_single_point_only():
    pred_set = JointEllipsoid(np.zeros(N_OUTPUTS), L4, 1.0)
    assert np.zeros(N_OUTPUTS) in pred_set
    with pytest.raises(ValueError):
        _ = np.zeros((2, N_OUTPUTS)) in pred_set


def test_sample_is_inside_set():
    pred_set = JointEllipsoid(np.ones(N_OUTPUTS), L4, 0.75)
    samples = pred_set.sample(500, random_state=5)
    assert pred_set.contains(samples).all()
    assert pred_set.scores(samples).max() <= 0.75 + 1e-9


def test_sample_rejects_batch_and_infinite_radius():
    batch = JointEllipsoid(np.zeros((2, N_OUTPUTS)), L4, 1.0)
    with pytest.raises(ValueError, match="single prediction set"):
        batch.sample(10)

    whole = JointEllipsoid(np.zeros(N_OUTPUTS), L4, np.inf)
    with pytest.raises(ValueError, match="whole output space"):
        whole.sample(10)

    single = JointEllipsoid(np.zeros(N_OUTPUTS), L4, 1.0)
    with pytest.raises(ValueError, match="positive integer"):
        single.sample(0)


def test_joint_ellipsoid_requires_center_mapping_and_radius():
    with pytest.raises(ValueError, match="center"):
        JointEllipsoid(None, L4, 1.0)
    with pytest.raises(ValueError, match="mapping"):
        JointEllipsoid(np.zeros(N_OUTPUTS), None, 1.0)
    with pytest.raises(ValueError, match="radius"):
        JointEllipsoid(np.zeros(N_OUTPUTS), L4, None)


def test_joint_ellipsoid_radius_and_center_properties():
    center = np.array([1.0, 2.0, 3.0, 4.0])
    pred_set = JointEllipsoid(center, L4, 2.5)
    assert pred_set.q == 2.5
    assert pred_set.radius == 2.5
    assert np.allclose(pred_set.y_pred, center)
    assert not pred_set.is_whole_space
    assert "JointEllipsoid" in repr(pred_set)


def test_batch_broadcasts_single_candidate_and_rejects_mismatch():
    centers = np.tile(np.array([1.0, 2.0, 3.0, 4.0]), (3, 1))
    pred_set = JointEllipsoid(centers, L4, 1.0)
    # A single candidate is broadcast to the batch: one boolean per set.
    assert pred_set.contains(centers[0]).all()
    with pytest.raises(ValueError, match="one sample per prediction set"):
        pred_set.contains(np.zeros((2, N_OUTPUTS)))


def test_calibrator_shape_mismatch_and_aliases_equal():
    calib = JointCalibrator(L4)
    with pytest.raises(ValueError, match="same shape"):
        calib.fit(y_true=np.zeros((5, N_OUTPUTS)), y_pred=np.zeros((4, N_OUTPUTS)))
    with pytest.raises(ValueError, match="y_true"):
        calib._joint_score(np.zeros((5, N_OUTPUTS)), np.zeros((4, N_OUTPUTS)))

    # Equal values provided through both names are accepted.
    assert JointCalibrator(L4, mapping=np.array(L4)).n_outputs == N_OUTPUTS
    assert JointEllipsoid(np.zeros(N_OUTPUTS), L4, 1.0, q=1.0).radius == 1.0


def test_calibrator_calibrate_before_fit_raises():
    with pytest.raises(RuntimeError, match="Run `fit`"):
        JointCalibrator(L4).calibrate(alpha=0.1, y_pred=np.zeros(N_OUTPUTS))


def test_check_alpha_validation():
    with pytest.raises(ValueError, match="real scalar"):
        JointCalibrator.check_alpha("bad")
    with pytest.raises(ValueError, match="scalar"):
        JointCalibrator.check_alpha(np.array([0.1, 0.2]))
    with pytest.raises(ValueError, match="NaN"):
        JointCalibrator.check_alpha(np.nan)
    assert JointCalibrator.check_alpha(0.1) == pytest.approx(0.1)


def test_joint_cp_compute_radius_before_fit_raises():
    with pytest.raises(RuntimeError, match="before computing a radius"):
        JointCP(_FixedPredictor(), L4).compute_radius(0.1)


def test_joint_cp_predict_none_features_raises():
    X_calib, y_calib = _make_dataset(20, seed=19)
    cp = JointCP(_FixedPredictor(), L4)
    cp.fit(X_calib, y_calib, alpha=0.2)
    with pytest.raises(ValueError, match="Input features"):
        cp.predict(None)


def test_calibrator_calibrate_returns_joint_set():
    rng = np.random.RandomState(21)
    z = rng.randn(200, N_COMPONENTS)
    y_pred = rng.randn(200, N_OUTPUTS)
    y_true = y_pred + z @ L4.T

    calib = JointCalibrator(L4)
    calib.fit(y_true=y_true, y_pred=y_pred)
    pred_set = calib.calibrate(alpha=0.1, y_pred=np.zeros(N_OUTPUTS))

    assert isinstance(pred_set, JointEllipsoid)
    assert pred_set.radius == calib.compute_radius(0.1)
    assert np.allclose(pred_set.center, np.zeros(N_OUTPUTS))
    # The mapping and the radius are preserved.
    assert np.allclose(pred_set.mapping, L4)


def test_calibrator_properties_and_aliases():
    calib = JointCalibrator(mapping=L4)
    assert calib.n_outputs == N_OUTPUTS
    assert calib.n_components == N_COMPONENTS
    assert np.array_equal(calib.L, calib.mapping)
    # The alias is validated too.
    bad_alias = np.array([[1.0, 1.0], [1.0, 1.0]])
    with pytest.raises(ValueError, match="full column rank"):
        JointCalibrator(mapping=bad_alias)

    with pytest.raises(ValueError, match="aliases"):
        JointCalibrator(L4, mapping=np.eye(N_OUTPUTS))
    with pytest.raises(ValueError, match="must be provided"):
        JointCalibrator()


def test_calibrator_empty_calibration_set_raises():
    calib = JointCalibrator(L4)
    with pytest.raises(ValueError, match="At least one calibration sample"):
        calib.fit(y_true=np.zeros((0, N_OUTPUTS)), y_pred=np.zeros((0, N_OUTPUTS)))


def test_joint_ellipsoid_alias_conflicts_raise():
    with pytest.raises(ValueError, match="aliases"):
        JointEllipsoid(np.zeros(N_OUTPUTS), L4, 1.0, q=2.0)
    with pytest.raises(ValueError, match="must be provided"):
        JointEllipsoid(np.zeros(N_OUTPUTS), L4)
    JointEllipsoid(center=np.zeros(N_OUTPUTS), mapping=L4, radius=1.0)
    JointEllipsoid(y_pred=np.zeros(N_OUTPUTS), L=L4, q=1.0)


def test_joint_ellipsoid_rejects_invalid_radius():
    with pytest.raises(ValueError, match="non-negative"):
        JointEllipsoid(np.zeros(N_OUTPUTS), L4, -1.0)
    with pytest.raises(ValueError, match="NaN"):
        JointEllipsoid(np.zeros(N_OUTPUTS), L4, np.nan)
    with pytest.raises(ValueError, match="real scalar"):
        JointEllipsoid(np.zeros(N_OUTPUTS), L4, "not a number")
    with pytest.raises(ValueError, match="scalar"):
        JointEllipsoid(np.zeros(N_OUTPUTS), L4, np.ones(2))


def test_outputs_validation_rejects_bad_ndim():
    pred_set = JointEllipsoid(np.zeros(N_OUTPUTS), L4, 1.0)
    # 1-D input whose length is neither d nor 1.
    with pytest.raises(ValueError, match="incompatible"):
        pred_set.contains(np.zeros(3))
    # More than two dimensions.
    with pytest.raises(ValueError, match="1-D or 2-D"):
        pred_set.contains(np.zeros((2, 2, N_OUTPUTS)))


def test_coordinates_query():
    center = np.array([1.0, 2.0, 3.0, 4.0])
    pred_set = JointEllipsoid(center, L4, 1.0)
    z = np.array([0.2, -0.4])
    coords = pred_set.coordinates(center + L4 @ z)
    assert np.allclose(coords[0], z)


def test_scores_are_scalars_for_single_candidate():
    pred_set = JointEllipsoid(np.zeros(N_OUTPUTS), L4, 1.0)
    score = pred_set.scores(np.zeros(N_OUTPUTS))
    assert isinstance(score, float)
    assert score == 0.0


def test_mapping_preserved_in_calibrator_and_pred_set():
    calib = JointCalibrator(L4)
    rng = np.random.RandomState(22)
    y_pred = rng.randn(50, N_OUTPUTS)
    y_true = y_pred + rng.randn(50, N_COMPONENTS) @ L4.T
    calib.fit(y_true=y_true, y_pred=y_pred)
    pred_set = calib.calibrate(alpha=0.2, y_pred=y_pred[:1])
    assert np.array_equal(pred_set.L, L4)
    assert pred_set.n_components == N_COMPONENTS


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ JointCP end-to-end ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def test_joint_cp_generates_real_predictions_and_joint_set():
    X_calib, y_calib = _make_dataset(500, seed=6)
    cp = JointCP(_FixedPredictor(), L4)
    cp.fit(X_calib, y_calib)

    X_test = np.linspace(-0.9, 0.9, 7).reshape(-1, 1)
    y_pred, pred_set = cp.predict(X_test, alpha=0.1)

    # Real point predictions (the model is really called).
    assert np.allclose(y_pred, _target(X_test))
    assert isinstance(pred_set, JointEllipsoid)
    # The radius and the mapping are preserved.
    assert pred_set.radius == cp.radius
    assert np.allclose(pred_set.mapping, L4)
    assert pred_set.n_sets == len(X_test)


def test_joint_cp_joint_coverage():
    X_calib, y_calib = _make_dataset(1000, seed=7)
    cp = JointCP(_FixedPredictor(), L4)
    cp.fit(X_calib, y_calib)

    rng = np.random.RandomState(7)
    X_test = rng.uniform(-1.0, 1.0, size=(4000, 1))
    y_test = _target(X_test) + rng.randn(4000, N_COMPONENTS) @ L4.T

    _, pred_set = cp.predict(X_test, alpha=0.1)
    coverage = pred_set.contains(y_test).mean()

    # Marginal coverage at least 1 - alpha (up to finite-sample fluctuation).
    assert coverage >= 0.88


def test_joint_cp_per_output_bounds_are_consistent_with_membership():
    X_calib, y_calib = _make_dataset(300, seed=8)
    cp = JointCP(_FixedPredictor(), L4)
    cp.fit(X_calib, y_calib, alpha=0.1)

    rng = np.random.RandomState(8)
    X_test = rng.uniform(-1.0, 1.0, size=(20, 1))
    _, pred_set = cp.predict(X_test)

    lower, upper = pred_set.linear_bounds()
    inside = pred_set.contains(_target(X_test))

    # Members of the joint set must satisfy every marginal bound.
    assert np.all(_target(X_test)[inside] >= lower[inside] - 1e-9)
    assert np.all(_target(X_test)[inside] <= upper[inside] + 1e-9)


def test_joint_cp_retrained_predictor_is_supported():
    from sklearn.linear_model import LinearRegression

    rng = np.random.RandomState(9)
    X = rng.uniform(-1, 1, size=(300, 3))
    coef = rng.randn(3, N_COMPONENTS)
    y = X @ coef @ L4.T + 0.1 * rng.randn(300, N_COMPONENTS) @ L4.T

    cp = JointCP(LinearRegression(), L4, train=True)
    cp.fit(X[:200], y[:200], X_fit=X[:150], y_fit=y[:150])

    y_pred, pred_set = cp.predict(X[200:], alpha=0.2)
    assert y_pred.shape == (100, N_OUTPUTS)
    assert pred_set.contains(y[200:]).mean() >= 0.7


def test_joint_cp_predict_before_fit_raises():
    cp = JointCP(_FixedPredictor(), L4)
    with pytest.raises(RuntimeError):
        cp.predict(np.zeros((2, 1)))


def test_joint_cp_predict_without_alpha_raises():
    X_calib, y_calib = _make_dataset(50, seed=10)
    cp = JointCP(_FixedPredictor(), L4)
    cp.fit(X_calib, y_calib)
    with pytest.raises(ValueError, match="No significance level"):
        cp.predict(X_calib[:1])


def test_joint_cp_requires_calibration_set():
    cp = JointCP(_FixedPredictor(), L4)
    with pytest.raises(ValueError, match="calibration set"):
        cp.fit(None, None)


def test_joint_cp_wrong_output_dimension_raises():
    X_calib, _ = _make_dataset(50, seed=11)
    cp = JointCP(_FixedPredictor(), L4)
    with pytest.raises(ValueError, match="output"):
        cp.fit(X_calib, np.zeros((50, 3)))


def test_joint_cp_sample_mismatch_raises():
    X_calib, y_calib = _make_dataset(50, seed=12)
    cp = JointCP(_FixedPredictor(), L4)
    with pytest.raises(ValueError, match="same number of samples"):
        cp.fit(X_calib, y_calib[:10])


def test_joint_cp_model_failure_raises():
    X_calib, y_calib = _make_dataset(20, seed=13)
    cp = JointCP(_FailingPredictor(), L4)
    with pytest.raises(RuntimeError, match="failed"):
        cp.fit(X_calib, y_calib)


def test_joint_cp_rejects_predictor_without_predict():
    with pytest.raises(RuntimeError, match="predict"):
        JointCP(object(), L4)


def test_joint_cp_train_without_fit_set_raises():
    from sklearn.linear_model import LinearRegression

    X_calib, y_calib = _make_dataset(30, seed=14)
    cp = JointCP(LinearRegression(), L4, train=True)
    with pytest.raises(RuntimeError, match="fit dataset"):
        cp.fit(X_calib, y_calib)


def test_joint_cp_get_scores_before_fit_raises():
    with pytest.raises(RuntimeError):
        JointCP(_FixedPredictor(), L4).get_nonconformity_scores()


def test_joint_cp_accepts_base_predictor_as_is():
    from deel.puncc.api.prediction import BasePredictor

    predictor = BasePredictor(_FixedPredictor())
    cp = JointCP(predictor, L4)
    assert cp.predictor is predictor


def test_joint_cp_rejects_missing_predictor_or_mapping():
    with pytest.raises(ValueError, match="predictor"):
        JointCP(None, L4)
    with pytest.raises(ValueError, match="must be provided"):
        JointCP(_FixedPredictor())
    with pytest.raises(ValueError, match="aliases"):
        JointCP(_FixedPredictor(), L4, mapping=np.eye(N_OUTPUTS))


def test_joint_cp_mapping_and_flag_properties():
    cp = JointCP(_FixedPredictor(), L4)
    assert np.array_equal(cp.mapping, L4)
    assert cp.n_outputs == N_OUTPUTS
    assert cp.is_fitted is False
    assert cp.radius is None


class _NonePredictor:
    def predict(self, X):
        return None


def test_joint_cp_predictor_returning_none_raises():
    from deel.puncc.api.prediction import BasePredictor

    class _NoneBasePredictor(BasePredictor):
        def predict(self, X, **kwargs):
            return None

    cp = JointCP(_NoneBasePredictor(_NonePredictor()), L4)
    with pytest.raises(RuntimeError, match="returned None"):
        cp.fit(np.zeros((3, 1)), np.zeros((3, N_OUTPUTS)))


def test_joint_cp_fit_without_features_raises():
    _, y_calib = _make_dataset(20, seed=16)
    cp = JointCP(_FixedPredictor(), L4)
    with pytest.raises(ValueError, match="calibration set"):
        cp.fit(None, y_calib)


def test_joint_cp_radius_stored_at_fit_time():
    X_calib, y_calib = _make_dataset(100, seed=17)
    cp = JointCP(_FixedPredictor(), L4)
    cp.fit(X_calib, y_calib, alpha=0.2)
    assert cp.radius == pytest.approx(cp.compute_radius(0.2))

    _, pred_set = cp.predict(X_calib[:4])  # alpha reused from fit
    assert pred_set.radius == cp.radius


def test_joint_cp_get_scores_after_fit():
    X_calib, y_calib = _make_dataset(30, seed=18)
    cp = JointCP(_FixedPredictor(), L4)
    cp.fit(X_calib, y_calib)
    scores = cp.get_nonconformity_scores()
    assert scores.shape == (30,)
    assert np.all(np.isfinite(scores))


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Backward compatibility ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def test_aliases_are_exposed():
    assert EllipsoidalSet is JointEllipsoid
    from deel.puncc.joint import MultiOutputJointCP

    assert MultiOutputJointCP is JointCP


def test_existing_regression_api_still_importable():
    from deel.puncc.regression import CQR
    from deel.puncc.regression import LocallyAdaptiveCP
    from deel.puncc.regression import SplitCP

    assert SplitCP is not None
    assert LocallyAdaptiveCP is not None
    assert CQR is not None


def test_splitcp_still_returns_intervals():
    from sklearn.ensemble import RandomForestRegressor

    rng = np.random.RandomState(15)
    X = rng.uniform(0, 10, size=(200, 2))
    y = X[:, 0] + rng.randn(200)
    from deel.puncc.api.prediction import BasePredictor
    from deel.puncc.regression import SplitCP

    cp = SplitCP(BasePredictor(RandomForestRegressor(random_state=0)))
    cp.fit(
        X_fit=X[:100],
        y_fit=y[:100],
        X_calib=X[100:150],
        y_calib=y[100:150],
    )
    y_pred, y_lo, y_hi = cp.predict(X[150:], alpha=0.2)
    assert y_pred.shape == (50,)
    assert np.all(y_lo <= y_hi)
