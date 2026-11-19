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
"""
Joint (multivariate) conformal prediction for multi-output regression, with
exact linear-functional queries on the resulting prediction sets.

Given a fixed point predictor :math:`\\hat{f}`, an independent calibration set
:math:`(X_{\\text{calib}}, y_{\\text{calib}})` and a matrix
:math:`L \\in \\mathbb{R}^{d \\times k}` with full column rank, the
nonconformity score of a sample is the norm of the coordinates of the residual
:math:`y - \\hat{f}(x)` expressed in the column space of :math:`L`:

.. math::

    s(x, y) =
    \\begin{cases}
        \\|z\\|_2 & \\text{if } y - \\hat{f}(x) = L z,\\\\
        +\\infty & \\text{otherwise.}
    \\end{cases}

The radius :math:`q` is the native finite-sample corrected conformal quantile
of these scores (the :math:`\\lceil (1-\\alpha)(n+1) \\rceil`-th order statistic
of the scores augmented with :math:`+\\infty`, obtained through
:func:`deel.puncc.api.utils.quantile`). The prediction set attached to a new
input :math:`x` is the ellipsoid

.. math::

    C(x) = \\{\\hat{f}(x) + L z \\;:\\; \\|z\\|_2 \\leq q\\}.

Unlike per-coordinate intervals, this set captures the joint (correlated)
uncertainty of the outputs. For any linear map :math:`A \\in
\\mathbb{R}^{m \\times d}` and offset :math:`b \\in \\mathbb{R}^m`, the bounds of
:math:`A y + b` over :math:`C(x)` are given in closed form and are tight:

.. math::

    A \\hat{f}(x) + b \\pm q \\, \\|L^{\\top} A^{\\top}\\|_2,

where the :math:`\\ell_2` norm is taken on the rows of :math:`A L`. Each bound
is attained on the boundary of the ellipsoid. When :math:`q = +\\infty` the
prediction set is the whole output space :math:`\\mathbb{R}^d`; bounds remain
finite for constant queries (:math:`A = 0`).

.. note::

    The implementation is NumPy-only. Use :class:`JointCP` for the calibrated
    conformal predictor, :class:`JointCalibrator` to only compute scores and
    radius, and :class:`JointEllipsoid` to query an already-built prediction
    set.

Example::

    import numpy as np
    from deel.puncc.api.joint import JointCP

    rng = np.random.RandomState(0)
    k = 2

    def f(X):
        x = X[:, 0]
        return np.column_stack((x, 2 * x, -x, x ** 2))

    L = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, -1.0]])
    X_calib = rng.uniform(-1, 1, size=(500, 1))
    y_calib = f(X_calib) + rng.randn(500, k) @ L.T

    class FixedPredictor:
        def predict(self, X):
            return f(X)

    cp = JointCP(FixedPredictor(), L)
    cp.fit(X_calib, y_calib)

    X_test = rng.uniform(-1, 1, size=(3, 1))
    y_pred, pred_set = cp.predict(X_test, alpha=0.1)

    # Exact joint bounds of  y_1 - y_2 + 0.5  (not a box!)
    y_lo, y_hi = pred_set.linear_bounds(np.array([1.0, -1.0, 0.0, 0.0]), 0.5)

    # Membership queries against the joint set
    y_test = f(X_test)
    is_in = pred_set.contains(y_test)
"""

import logging
from typing import Any
from typing import Callable
from typing import Iterable
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np

from deel.puncc.api.calibration import BaseCalibrator
from deel.puncc.api.prediction import BasePredictor

logger = logging.getLogger(__name__)

#: Default relative tolerance used to decide whether a residual belongs to the
#: column space of the mapping :math:`L`. It is scaled by the residual norm
#: (with an absolute floor of one) so that residuals produced by exact
#: float arithmetic are correctly identified as in-space.
DEFAULT_SUBSPACE_TOL = 1e-8

#: Numerical slack used by membership queries so that points lying exactly on
#: the boundary of the ellipsoid are not rejected because of roundoff.
_BOUNDARY_RTOL = 1e-12


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Validation helpers ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def _resolve_alias(
    primary: Optional[Any], alias: Optional[Any], primary_name: str, alias_name: str
) -> Optional[Any]:
    """Resolve an argument provided either by name or by alias.

    :param Optional[Any] primary: value of the primary argument.
    :param Optional[Any] alias: value of the alias argument.
    :param str primary_name: primary argument name, for error messages.
    :param str alias_name: alias argument name, for error messages.

    :returns: the resolved value (``None`` when both are ``None``).
    :rtype: Optional[Any]

    :raises ValueError: both arguments were provided with different values.
    """
    if primary is None:
        return alias
    if alias is None:
        return primary
    try:
        same = bool(np.array_equal(np.asarray(primary), np.asarray(alias)))
    except Exception:  # pragma: no cover - exotic inputs
        same = False
    if not same:
        raise ValueError(
            f"Arguments `{primary_name}` and `{alias_name}` are aliases and "
            "cannot be provided with different values."
        )
    return primary


def _as_float_array(values: Any, name: str) -> np.ndarray:
    """Convert ``values`` to a finite, real float NumPy array.

    :param Any values: input values.
    :param str name: argument name used in error messages.

    :returns: converted array.
    :rtype: np.ndarray

    :raises ValueError: conversion failed or the array contains non-finite
        values.
    """
    try:
        arr = np.asarray(values)
        if np.iscomplexobj(arr):
            raise ValueError(
                f"Argument `{name}` must be real-valued, got complex values."
            )
        arr = arr.astype(float, copy=False)
    except (TypeError, ValueError) as err:
        raise ValueError(
            f"Argument `{name}` could not be converted to a float array: {err}"
        ) from err

    if not np.all(np.isfinite(arr)):
        raise ValueError(f"Argument `{name}` must contain only finite values.")
    return np.asarray(arr)


class _MappingGeometry:
    """Pre-computed geometry of the linear mapping :math:`L`.

    The mapping must have shape :math:`(d, k)` and full column rank, with
    :math:`1 \\leq k \\leq d`. A reduced QR decomposition :math:`L = Q R` is
    stored to compute coordinates and membership tests in a numerically stable
    way.

    :param Iterable L: linear mapping with shape ``(n_outputs, n_components)``.

    :raises ValueError: the mapping is not a 2-D array with full column rank.
    """

    __slots__ = ("L", "Q", "R", "n_outputs", "n_components")

    def __init__(self, L: Iterable):
        arr = _as_float_array(L, "L")

        if arr.ndim != 2:
            raise ValueError(
                "Argument `L` must be a 2-D array of shape "
                "(n_outputs, n_components). "
                f"Got an array with {arr.ndim} dimension(s)."
            )

        n_outputs, n_components = arr.shape
        if n_outputs == 0 or n_components == 0:
            raise ValueError(
                "Argument `L` must have at least one row and one column. "
                f"Got shape {arr.shape}."
            )
        if n_components > n_outputs:
            raise ValueError(
                "Argument `L` must have full column rank, which is impossible "
                f"for shape {arr.shape} with more columns than rows."
            )

        rank = int(np.linalg.matrix_rank(arr))
        if rank != n_components:
            raise ValueError(
                "Argument `L` must have full column rank: expected rank "
                f"{n_components}, got rank {rank} for shape {arr.shape}."
            )

        Q, R = np.linalg.qr(arr, mode="reduced")

        self.L = np.array(arr, copy=True)
        self.Q = Q
        self.R = R
        self.n_outputs = int(n_outputs)
        self.n_components = int(n_components)

    def as_outputs(self, values: Any, name: str) -> np.ndarray:
        """Validate and reshape values to an ``(n, n_outputs)`` matrix.

        A 1-D input is understood as a single sample when its length equals
        :math:`d`; a scalar is understood as a single one-dimensional output.

        :param Any values: vector or matrix of outputs.
        :param str name: argument name used in error messages.

        :returns: 2-D array of outputs.
        :rtype: np.ndarray

        :raises ValueError: inconsistent dimensions.
        """
        arr = _as_float_array(values, name)
        d = self.n_outputs

        if arr.ndim == 0:
            arr = arr.reshape(1, 1)
        elif arr.ndim == 1:
            if d == 1:
                arr = arr.reshape(-1, 1)
            elif arr.shape[0] == d:
                arr = arr.reshape(1, d)
            else:
                raise ValueError(
                    f"Argument `{name}` has shape {arr.shape}, which is "
                    f"incompatible with outputs of dimension {d}."
                )
        elif arr.ndim != 2:
            raise ValueError(
                f"Argument `{name}` must be 1-D or 2-D. "
                f"Got an array with {arr.ndim} dimensions."
            )

        if arr.shape[1] != d:
            raise ValueError(
                f"Argument `{name}` must have {d} output(s) per sample, got "
                f"{arr.shape[1]} for shape {arr.shape}."
            )

        return arr

    def coordinates(
        self, residuals: np.ndarray, tol: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Express residuals as coordinates in the column space of ``L``.

        :param np.ndarray residuals: residuals with shape ``(n, d)``.
        :param float tol: relative tolerance of the column-space test.

        :returns: couple ``(z, outside)`` where ``z`` has shape ``(n, k)`` and
            ``outside`` is a boolean mask of residuals that do not belong to
            the column space of ``L``.
        :rtype: Tuple[np.ndarray, np.ndarray]
        """
        residuals = self.as_outputs(residuals, "residuals")

        # L = Q R with Q orthonormal, so z solves  R z = Q^T r.
        projected = residuals @ self.Q
        z = np.linalg.solve(self.R, projected.T).T

        residual_norms = np.linalg.norm(residuals, axis=-1)
        orthogonal = residuals - projected @ self.Q.T
        orthogonal_norms = np.linalg.norm(orthogonal, axis=-1)

        threshold = tol * np.maximum(1.0, residual_norms)
        outside = orthogonal_norms > threshold
        return z, outside

    def scores(self, residuals: np.ndarray, tol: float) -> np.ndarray:
        """Joint nonconformity scores of residuals.

        The score is the Euclidean norm of the coordinates in the column space
        of ``L``, and ``+inf`` for residuals outside of that space.

        :param np.ndarray residuals: residuals with shape ``(n, d)``.
        :param float tol: relative tolerance of the column-space test.

        :returns: nonconformity scores with shape ``(n,)``.
        :rtype: np.ndarray
        """
        z, outside = self.coordinates(residuals, tol)
        scores = np.array(np.linalg.norm(z, axis=-1), dtype=float, copy=True)
        scores[outside] = np.inf
        return scores

    def functional_norms(self, A: np.ndarray) -> np.ndarray:
        """Operator norms :math:`\\|L^{\\top} a_i\\|_2` of the rows of ``A``.

        :param np.ndarray A: matrix with shape ``(m, d)`` of linear
            functionals (one per row).

        :returns: array of norms with shape ``(m,)``.
        :rtype: np.ndarray
        """
        return np.abs(np.linalg.norm(A @ self.L, axis=-1))


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Core functions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def column_space_coordinates(
    y_pred: Iterable,
    y_true: Iterable,
    L: Iterable,
    *,
    tol: float = DEFAULT_SUBSPACE_TOL,
) -> Tuple[np.ndarray, np.ndarray]:
    """Coordinates of the residuals in the column space of ``L``.

    :param Iterable y_pred: predicted outputs with shape ``(n, d)`` (or
        ``(d,)`` for a single sample).
    :param Iterable y_true: true outputs with the same shape as ``y_pred``.
    :param Iterable L: mapping with shape ``(d, k)`` and full column rank.
    :param float tol: relative tolerance of the column-space test.

    :returns: couple ``(z, outside)`` where ``z`` contains the coordinates of
        the residuals in the column space of ``L`` and ``outside`` flags the
        residuals that do not belong to that space.
    :rtype: Tuple[np.ndarray, np.ndarray]

    :raises ValueError: inconsistent dimensions or invalid mapping.
    """
    geometry = _MappingGeometry(L)
    pred = geometry.as_outputs(y_pred, "y_pred")
    true = geometry.as_outputs(y_true, "y_true")
    if pred.shape != true.shape:
        raise ValueError(
            "Arguments `y_pred` and `y_true` must have the same shape. "
            f"Got {pred.shape} and {true.shape}."
        )
    return geometry.coordinates(true - pred, tol)


def column_space_norm(
    y_pred: Iterable,
    y_true: Iterable,
    L: Iterable,
    *,
    tol: float = DEFAULT_SUBSPACE_TOL,
) -> np.ndarray:
    """Joint nonconformity score: column-space norm of the residuals.

    Residuals outside the column space of :math:`L` receive a score of
    :math:`+\\infty`, since the prediction set is anchored on that space.

    :param Iterable y_pred: predicted outputs.
    :param Iterable y_true: true outputs.
    :param Iterable L: mapping with shape ``(d, k)`` and full column rank.
    :param float tol: relative tolerance of the column-space test.

    :returns: nonconformity scores with shape ``(n,)``.
    :rtype: np.ndarray

    :raises ValueError: inconsistent dimensions or invalid mapping.
    """
    geometry = _MappingGeometry(L)
    pred = geometry.as_outputs(y_pred, "y_pred")
    true = geometry.as_outputs(y_true, "y_true")
    if pred.shape != true.shape:
        raise ValueError(
            "Arguments `y_pred` and `y_true` must have the same shape. "
            f"Got {pred.shape} and {true.shape}."
        )
    return geometry.scores(true - pred, tol)


def column_space_norm_builder(
    L: Iterable, *, tol: float = DEFAULT_SUBSPACE_TOL
) -> Callable:
    """Build a nonconformity score function bound to the mapping ``L``.

    The returned callable follows the ``(y_pred, y_true)`` signature expected
    by :class:`deel.puncc.api.calibration.BaseCalibrator`.

    :param Iterable L: mapping with shape ``(d, k)`` and full column rank.
    :param float tol: relative tolerance of the column-space test.

    :returns: nonconformity score function.
    :rtype: Callable

    :raises ValueError: invalid mapping.
    """
    _MappingGeometry(L)  # fail fast on an invalid mapping

    def _score_function(y_pred: Iterable, y_true: Iterable) -> np.ndarray:
        return column_space_norm(y_pred, y_true, L, tol=tol)

    return _score_function


def _validate_radius(radius: Any) -> float:
    """Validate and convert a radius value.

    :param Any radius: radius value.

    :returns: radius as a float (possibly ``+inf``).
    :rtype: float

    :raises ValueError: non-scalar, negative or NaN radius.
    """
    try:
        radius_arr = np.asarray(radius, dtype=float)
    except (TypeError, ValueError) as err:
        raise ValueError(
            f"Argument `radius` must be a real scalar, got {radius!r}."
        ) from err
    if radius_arr.size != 1:
        raise ValueError(
            "Argument `radius` must be a scalar, got an array of shape "
            f"{radius_arr.shape}."
        )
    value = float(radius_arr.reshape(-1)[0])
    if np.isnan(value):
        raise ValueError("Argument `radius` must not be NaN.")
    if value < 0:
        raise ValueError(f"Argument `radius` must be non-negative, got {value}.")
    return value


def _prepare_linear_map(
    A: Optional[Iterable], n_outputs: int
) -> Tuple[np.ndarray, bool]:
    """Validate ``A`` and reshape it to ``(m, d)``.

    :param Optional[Iterable] A: linear map.
    :param int n_outputs: output dimension ``d``.

    :returns: couple ``(A_2d, single_query)`` where ``single_query`` is True
        when ``A`` was provided as a single functional.
    :rtype: Tuple[np.ndarray, bool]

    :raises ValueError: inconsistent dimensions.
    """
    if A is None:
        return np.eye(n_outputs), False

    A_arr = _as_float_array(A, "A")
    if A_arr.ndim == 1:
        if A_arr.shape[0] != n_outputs:
            raise ValueError(
                "Argument `A` must have one coefficient per output "
                f"({n_outputs}), got {A_arr.shape[0]}."
            )
        return A_arr.reshape(1, n_outputs), True
    if A_arr.ndim == 2:
        if A_arr.shape[1] != n_outputs:
            raise ValueError(
                "Argument `A` must have one coefficient per output "
                f"({n_outputs}) in its last dimension, got shape "
                f"{A_arr.shape}."
            )
        if A_arr.shape[0] == 0:
            raise ValueError("Argument `A` must have at least one row.")
        return A_arr, False
    raise ValueError(
        "Argument `A` must be a 1-D vector or a 2-D matrix, got an array "
        f"with {A_arr.ndim} dimensions."
    )


def _prepare_offset(b: Optional[Iterable], m: int) -> np.ndarray:
    """Validate and broadcast the offset ``b`` to shape ``(m,)``.

    :param Optional[Iterable] b: offset.
    :param int m: number of queried functionals.

    :returns: broadcast offset.
    :rtype: np.ndarray

    :raises ValueError: inconsistent dimensions.
    """
    if b is None:
        return np.zeros(m)
    b_arr = _as_float_array(b, "b")
    if b_arr.ndim == 0:
        return np.full(m, float(b_arr))
    b_arr = b_arr.reshape(-1)
    if b_arr.shape[0] != m:
        raise ValueError(
            f"Argument `b` must have {m} element(s) to match `A`, got "
            f"{b_arr.shape[0]}."
        )
    return b_arr


def _ellipsoid_widths(norms: np.ndarray, radius: float) -> np.ndarray:
    """Half-widths :math:`q \\|L^{\\top} a_i\\|_2`, safe with ``q = +inf``.

    The product ``inf * 0`` is replaced by zero so that constant queries
    (whose functional has a zero norm) keep finite bounds even when the
    prediction set is the whole output space.

    :param np.ndarray norms: operator norms with shape ``(m,)``.
    :param float radius: radius :math:`q`.

    :returns: half-widths with shape ``(m,)``.
    :rtype: np.ndarray
    """
    if np.isinf(radius):
        widths = np.zeros_like(norms)
        widths[norms > 0] = np.inf
        return widths
    return radius * norms


def linear_functional_bounds(
    center: Iterable,
    L: Iterable,
    radius: float,
    A: Optional[Iterable] = None,
    b: Optional[Iterable] = None,
    *,
    tol: float = DEFAULT_SUBSPACE_TOL,
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """Exact (tight) bounds of :math:`A y + b` over the ellipsoid.

    The ellipsoid is :math:`\\{c + L z : \\|z\\|_2 \\leq q\\}`. For each row
    :math:`a_i` of :math:`A`, the supremum of :math:`a_i^{\\top} y + b_i` is
    :math:`a_i^{\\top} c + b_i + q \\|L^{\\top} a_i\\|_2`, and the infimum is
    the same quantity with a minus sign. These bounds are attained (tight) and
    are **not** obtained by assembling per-coordinate intervals.

    :param Iterable center: ellipsoid center :math:`c` with shape ``(d,)``.
    :param Iterable L: mapping with shape ``(d, k)`` and full column rank.
    :param float radius: radius :math:`q \\in [0, +\\infty]`.
    :param Iterable A: linear map with shape ``(m, d)`` or ``(d,)``. Defaults
        to the identity, i.e. per-output bounds.
    :param Iterable b: offset with shape ``(m,)`` or a scalar. Defaults to 0.
    :param float tol: kept for signature consistency with the class-based API;
        bounds are exact and do not involve a tolerance.

    :returns: couple ``(lower, upper)``. Scalars are returned for a single
        functional, arrays otherwise.
    :rtype: Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]

    :raises ValueError: inconsistent dimensions or invalid arguments.
    """
    del tol  # bounds are exact; no tolerance is involved

    geometry = _MappingGeometry(L)
    center_arr = geometry.as_outputs(center, "center")
    if center_arr.shape[0] != 1:
        raise ValueError(
            "Argument `center` must contain a single ellipsoid center. "
            f"Got shape {center_arr.shape}. Use `JointEllipsoid` for batches."
        )
    center_arr = center_arr[0]

    radius = _validate_radius(radius)
    A_arr, single_query = _prepare_linear_map(A, geometry.n_outputs)
    b_arr = _prepare_offset(b, A_arr.shape[0])

    mid = A_arr @ center_arr + b_arr
    widths = _ellipsoid_widths(geometry.functional_norms(A_arr), radius)

    lower = mid - widths
    upper = mid + widths

    if single_query:
        return lower.item(), upper.item()
    return lower, upper


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Prediction set ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class JointEllipsoid:
    """Joint prediction set of the form :math:`c + \\{L z : \\|z\\|_2 \\leq q\\}`.

    The object keeps both the radius :math:`q` and the mapping :math:`L`
    (``q = +\\infty`` encodes the whole output space). It supports exact
    membership tests and tight bounds for linear metrics :math:`A y + b`.

    :param Iterable center: point prediction(s). Either a vector of shape
        ``(d,)`` for a single set or a matrix of shape ``(n, d)`` for a batch
        of ``n`` sets sharing the same mapping and radius.
    :param Iterable mapping: mapping :math:`L` with shape ``(d, k)`` and full
        column rank.
    :param float radius: radius :math:`q \\in [0, +\\infty]`.
    :param float tol: relative tolerance of the column-space test.

    :raises ValueError: invalid center, mapping or radius.
    """

    def __init__(
        self,
        center: Optional[Iterable] = None,
        mapping: Optional[Iterable] = None,
        radius: Optional[float] = None,
        *,
        L: Optional[Iterable] = None,
        y_pred: Optional[Iterable] = None,
        q: Optional[float] = None,
        tol: float = DEFAULT_SUBSPACE_TOL,
    ):
        center = _resolve_alias(center, y_pred, "center", "y_pred")
        mapping = _resolve_alias(mapping, L, "mapping", "L")
        radius = _resolve_alias(radius, q, "radius", "q")

        if center is None:
            raise ValueError(
                "Argument `center` (or its alias `y_pred`) must be provided."
            )
        if mapping is None:
            raise ValueError("Argument `mapping` (or its alias `L`) must be provided.")
        if radius is None:
            raise ValueError("Argument `radius` (or its alias `q`) must be provided.")

        geometry = _MappingGeometry(mapping)
        center_arr = geometry.as_outputs(center, "center")
        self._geometry = geometry
        self.center = center_arr[0].copy() if center_arr.shape[0] == 1 else center_arr
        self.radius = _validate_radius(radius)
        self.tol = float(tol)

    # ~~~~~~~~~~~~ Accessors ~~~~~~~~~~~~

    @property
    def mapping(self) -> np.ndarray:
        """Mapping :math:`L` of the ellipsoid."""
        return self._geometry.L

    @property
    def L(self) -> np.ndarray:
        """Mapping :math:`L` of the ellipsoid (alias of ``mapping``)."""
        return self._geometry.L

    @property
    def q(self) -> float:
        """Radius :math:`q` of the ellipsoid (alias of ``radius``)."""
        return self.radius

    @property
    def y_pred(self) -> np.ndarray:
        """Point prediction(s) (alias of ``center``)."""
        return self.center

    @property
    def n_outputs(self) -> int:
        """Dimension :math:`d` of the output space."""
        return self._geometry.n_outputs

    @property
    def n_components(self) -> int:
        """Dimension :math:`k` of the (possibly lower-dimensional) space."""
        return self._geometry.n_components

    @property
    def n_sets(self) -> int:
        """Number of sets held by the object (1 for a single center)."""
        return 1 if self.center.ndim == 1 else self.center.shape[0]

    @property
    def is_whole_space(self) -> bool:
        """Whether the set covers the whole output space (``q = +inf``)."""
        return np.isinf(self.radius)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"{self.__class__.__name__}(n_sets={self.n_sets}, "
            f"n_outputs={self.n_outputs}, n_components={self.n_components}, "
            f"radius={self.radius})"
        )

    # ~~~~~~~~~~~~ Core queries ~~~~~~~~~~~~

    def _aligned_residuals(self, values: Iterable) -> np.ndarray:
        """Residuals between candidate outputs and the set center(s).

        :param Iterable values: candidate outputs.

        :returns: residuals with shape ``(n, d)``.
        :rtype: np.ndarray

        :raises ValueError: inconsistent number of samples.
        """
        targets = self._geometry.as_outputs(values, "y")

        if self.center.ndim == 1:
            return targets - self.center

        if targets.shape[0] == 1 and self.center.shape[0] != 1:
            targets = np.broadcast_to(targets, self.center.shape)
        if targets.shape[0] != self.center.shape[0]:
            raise ValueError(
                "Argument `y` must provide one sample per prediction set. "
                f"Got {targets.shape[0]} sample(s) for {self.n_sets} set(s)."
            )
        return targets - self.center

    def scores(self, y: Iterable) -> Union[float, np.ndarray]:
        """Joint nonconformity score of candidate outputs.

        The score is :math:`+\\infty` for candidates outside the column space
        of :math:`L`, and the norm of their coordinates otherwise.

        :param Iterable y: candidate outputs.

        :returns: scores. A scalar is returned for a single candidate and a
            single set; an array of shape ``(n,)`` otherwise.
        :rtype: Union[float, np.ndarray]

        :raises ValueError: inconsistent dimensions.
        """
        scores = self._geometry.scores(self._aligned_residuals(y), self.tol)
        if scores.shape[0] == 1:
            return float(scores[0])
        return scores

    #: Alias of :meth:`scores`.
    score = scores

    def coordinates(self, y: Iterable) -> np.ndarray:
        """Coordinates of candidate outputs in the column space of ``L``.

        :param Iterable y: candidate outputs.

        :returns: coordinates with shape ``(n, k)``.
        :rtype: np.ndarray

        :raises ValueError: inconsistent dimensions.
        """
        z, _ = self._geometry.coordinates(self._aligned_residuals(y), self.tol)
        return z

    def contains(self, y: Iterable) -> Union[bool, np.ndarray]:
        """Test the joint membership of candidate outputs.

        A candidate belongs to the set if its residual lies in the column space
        of :math:`L` and the norm of its coordinates is at most :math:`q`.
        When :math:`q = +\\infty`, every candidate belongs to the set because
        the set is the whole output space.

        :param Iterable y: candidate outputs.

        :returns: boolean for a single candidate, boolean array otherwise.
        :rtype: Union[bool, np.ndarray]

        :raises ValueError: inconsistent dimensions.
        """
        scores = self._geometry.scores(self._aligned_residuals(y), self.tol)
        if np.isinf(self.radius):
            inside = np.ones(scores.shape, dtype=bool)
        else:
            threshold = self.radius * (1.0 + _BOUNDARY_RTOL) + _BOUNDARY_RTOL
            inside = scores <= threshold
        if inside.shape[0] == 1:
            return bool(inside[0])
        return inside

    def __contains__(self, y: Iterable) -> bool:
        result = self.contains(y)
        if isinstance(result, np.ndarray):
            raise ValueError(
                "Membership of several samples is ambiguous; call "
                "`contains`, which returns one boolean per sample."
            )
        return result

    def linear_bounds(
        self,
        A: Optional[Iterable] = None,
        b: Optional[Iterable] = None,
    ) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        """Exact (tight) bounds of :math:`A y + b` over the joint set.

        For each row :math:`a_i` of :math:`A`, the bounds are
        :math:`a_i^{\\top} c + b_i \\pm q \\|L^{\\top} a_i\\|_2`, where
        :math:`c` is the set center. These are the exact extrema over the
        ellipsoid and are attained; they are **not** obtained by assembling
        marginal per-coordinate intervals. Constant queries (:math:`a_i = 0`)
        always yield the finite bound :math:`b_i`, even when
        :math:`q = +\\infty`.

        :param Iterable A: linear map with shape ``(m, d)`` or a single
            functional with shape ``(d,)``. Defaults to the identity.
        :param Iterable b: offset with shape ``(m,)`` or a scalar. Defaults
            to 0.

        :returns: couple ``(lower, upper)``. Scalars are returned for a single
            functional and a single center; arrays otherwise (with a leading
            sample axis for batches of centers).
        :rtype: Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]

        :raises ValueError: inconsistent dimensions.
        """
        A_arr, single_query = _prepare_linear_map(A, self.n_outputs)
        b_arr = _prepare_offset(b, A_arr.shape[0])
        widths = _ellipsoid_widths(self._geometry.functional_norms(A_arr), self.radius)

        if self.center.ndim == 1:
            mid = A_arr @ self.center + b_arr
            lower = mid - widths
            upper = mid + widths
            if single_query:
                return lower.item(), upper.item()
            return lower, upper

        mid = self.center @ A_arr.T + b_arr  # (n_sets, m)
        lower = mid - widths
        upper = mid + widths
        if single_query:
            return lower[:, 0], upper[:, 0]
        return lower, upper

    #: Alias of :meth:`linear_bounds`.
    bounds = linear_bounds

    def sample(
        self, n_samples: int = 1, random_state: Optional[int] = None
    ) -> np.ndarray:
        """Draw points uniformly from the (single) joint set.

        :param int n_samples: number of points to draw.
        :param Optional[int] random_state: seed of the random generator.

        :returns: points with shape ``(n_samples, d)``.
        :rtype: np.ndarray

        :raises ValueError: the object holds a batch of sets or has an
            infinite radius.
        """
        if self.center.ndim != 1:
            raise ValueError("`sample` is only defined for a single prediction set.")
        if self.is_whole_space:
            raise ValueError(
                "Cannot sample from the whole output space (radius is inf)."
            )
        if int(n_samples) < 1:
            raise ValueError(
                f"`n_samples` must be a positive integer, got {n_samples}."
            )

        rng = np.random.RandomState(random_state)
        n_samples = int(n_samples)
        directions = rng.randn(n_samples, self.n_components)
        norms = np.linalg.norm(directions, axis=1)
        zero = norms == 0
        while np.any(zero):  # pragma: no cover - probability zero
            directions[zero] = rng.randn(int(zero.sum()), self.n_components)
            norms = np.linalg.norm(directions, axis=1)
            zero = norms == 0
        directions /= norms[:, np.newaxis]
        exponent = 1.0 / self.n_components
        scales = self.radius * rng.uniform(size=n_samples) ** exponent
        z = directions * scales[:, np.newaxis]
        return self.center + z @ self.L.T


#: Aliases of :class:`JointEllipsoid`.
EllipsoidalSet = JointEllipsoid
Ellipsoid = JointEllipsoid
JointPredictionSet = JointEllipsoid
JointSet = JointEllipsoid


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Calibration ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class JointCalibrator(BaseCalibrator):
    """Calibrator for joint (ellipsoidal) conformal prediction.

    It computes joint nonconformity scores (column-space norm of the
    residuals, ``+inf`` outside the column space) and, at calibration time,
    returns a :class:`JointEllipsoid` whose radius is the native finite-sample
    corrected conformal quantile.

    :param Iterable L: mapping :math:`L` with shape ``(d, k)`` and full column
        rank.
    :param float tol: relative tolerance of the column-space test.

    :raises ValueError: invalid mapping.
    """

    def __init__(
        self,
        L: Optional[Iterable] = None,
        *,
        mapping: Optional[Iterable] = None,
        tol: float = DEFAULT_SUBSPACE_TOL,
    ):
        L = _resolve_alias(L, mapping, "L", "mapping")
        if L is None:
            raise ValueError("Argument `L` (or its alias `mapping`) must be provided.")
        self._geometry = _MappingGeometry(L)
        self.tol = float(tol)
        super().__init__(
            nonconf_score_func=self._joint_score,
            pred_set_func=self._joint_set,
        )

    @property
    def L(self) -> np.ndarray:
        """Mapping :math:`L` (full column rank)."""
        return self._geometry.L

    @property
    def mapping(self) -> np.ndarray:
        """Mapping :math:`L` (alias of ``L``)."""
        return self._geometry.L

    @property
    def n_outputs(self) -> int:
        """Dimension :math:`d` of the output space."""
        return self._geometry.n_outputs

    @property
    def n_components(self) -> int:
        """Dimension :math:`k` of the (possibly lower-dimensional) space."""
        return self._geometry.n_components

    def as_outputs(self, values: Any, name: str = "y") -> np.ndarray:
        """Validate/reshape outputs to an ``(n, d)`` matrix.

        :param Any values: vector or matrix of outputs.
        :param str name: argument name used in error messages.

        :returns: 2-D array of outputs.
        :rtype: np.ndarray

        :raises ValueError: inconsistent dimensions.
        """
        return self._geometry.as_outputs(values, name)

    def _joint_score(self, y_pred, y_true) -> np.ndarray:
        pred = self._geometry.as_outputs(y_pred, "y_pred")
        true = self._geometry.as_outputs(y_true, "y_true")
        if pred.shape != true.shape:
            raise ValueError(
                "Arguments `y_pred` and `y_true` must have the same shape. "
                f"Got {pred.shape} and {true.shape}."
            )
        return self._geometry.scores(true - pred, self.tol)

    def _joint_set(self, y_pred, scores_quantile) -> JointEllipsoid:
        return JointEllipsoid(
            self._geometry.as_outputs(y_pred, "y_pred"),
            self._geometry.L,
            scores_quantile,
            tol=self.tol,
        )

    def fit(self, *, y_true: Iterable, y_pred: Iterable, **kwargs) -> None:
        """Compute and store the joint nonconformity scores.

        :param Iterable y_true: true outputs with shape ``(n, d)``.
        :param Iterable y_pred: predicted outputs with shape ``(n, d)``.

        :raises ValueError: inconsistent dimensions or empty calibration set.
        """
        del kwargs
        pred = self._geometry.as_outputs(y_pred, "y_pred")
        true = self._geometry.as_outputs(y_true, "y_true")
        if pred.shape != true.shape:
            raise ValueError(
                "Arguments `y_pred` and `y_true` must have the same shape. "
                f"Got {pred.shape} and {true.shape}."
            )
        if pred.shape[0] == 0:
            raise ValueError(
                "At least one calibration sample is required. Got 0 samples."
            )
        super().fit(y_true=true, y_pred=pred)

    def calibrate(
        self,
        *,
        alpha: float,
        y_pred: Iterable,
        X: Optional[Iterable] = None,
        weights: Optional[Iterable] = None,
        correction: Optional[Callable] = None,
    ) -> JointEllipsoid:
        """Build the joint prediction set for new predictions.

        :param float alpha: significance level (max miscoverage target).
        :param Iterable y_pred: predicted outputs with shape ``(n, d)`` or
            ``(d,)``.
        :param Optional[Iterable] X: unused, kept for interface compatibility.
        :param Optional[Iterable] weights: unused.
        :param Optional[Callable] correction: unused. The score is already a
            joint score, so no multiple-testing correction is applied.

        :returns: joint prediction set.
        :rtype: JointEllipsoid

        :raises RuntimeError: called before :meth:`fit`.
        :raises ValueError: inconsistent dimensions or insufficient
            calibration samples for the requested ``alpha``.
        """
        del X, weights, correction
        if self._residuals is None:
            raise RuntimeError("Run `fit` method before calling `calibrate`.")
        radius = self.compute_radius(alpha)
        return self._joint_set(y_pred, radius)

    def compute_radius(self, alpha: float) -> float:
        """Native corrected conformal radius for ``alpha``.

        The radius is the :math:`(1-\\alpha)`-quantile of the calibration
        scores augmented with :math:`+\\infty`, computed with the library's
        finite-sample-corrected empirical quantile.

        :param float alpha: significance level.

        :returns: radius :math:`q \\in [0, +\\infty]`.
        :rtype: float

        :raises RuntimeError: called before :meth:`fit`.
        :raises ValueError: invalid ``alpha`` or insufficient calibration
            samples.
        """
        if self._residuals is None:
            raise RuntimeError("Run `fit` method before computing the radius.")
        alpha = self.check_alpha(alpha)
        radius = self.compute_quantile(alpha=alpha)
        value = float(np.asarray(radius).reshape(-1)[0])
        if np.isnan(value) or value < 0:
            raise RuntimeError(
                "Invalid conformal radius computed from the calibration "
                f"scores: {value}."
            )
        return value

    #: Alias of :meth:`compute_radius`.
    radius = compute_radius

    @staticmethod
    def check_alpha(alpha: Any) -> float:
        """Validate a scalar significance level.

        :param Any alpha: significance level.

        :returns: ``alpha`` as a float.
        :rtype: float

        :raises ValueError: non-scalar or NaN ``alpha``.
        """
        try:
            alpha_arr = np.asarray(alpha, dtype=float)
        except (TypeError, ValueError) as err:
            raise ValueError(
                f"Argument `alpha` must be a real scalar, got {alpha!r}."
            ) from err
        if alpha_arr.size != 1:
            raise ValueError(
                "Argument `alpha` must be a scalar, got an array of shape "
                f"{alpha_arr.shape}."
            )
        value = float(alpha_arr.reshape(-1)[0])
        if np.isnan(value):
            raise ValueError("Argument `alpha` must not be NaN.")
        return value

    #: Backward-compatible private alias.
    _check_alpha = check_alpha


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Conformal predictor ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class JointCP:
    """Joint split conformal prediction for multi-output regression.

    The underlying predictor is fixed: this class does not train it unless
    ``train=True`` and a fit set is provided. Nonconformity scores are computed
    on an independent calibration set and are the column-space norms of the
    residuals; the resulting prediction set is the ellipsoid
    :math:`\\hat{f}(x) + L \\{z : \\|z\\|_2 \\leq q\\}`, where :math:`q` is the
    native finite-sample corrected conformal quantile.

    :param predictor: fixed prediction model, either a
        :class:`~deel.puncc.api.prediction.BasePredictor` or any object
        exposing a ``predict`` method.
    :param Iterable L: mapping :math:`L` with shape ``(d, k)`` and full column
        rank. Low-dimensional column spaces (:math:`k < d`) are supported.
    :param bool train: if ``True``, the underlying model is trained on
        ``(X_fit, y_fit)`` when :meth:`fit` is called. Defaults to ``False``
        (the predictor is considered fixed and already trained).
    :param float tol: relative tolerance of the column-space test.

    :raises ValueError: ``predictor`` is None or ``L`` is invalid.
    :raises RuntimeError: the predictor has no ``predict`` method.

    .. warning::

        With a low-dimensional column space (:math:`k < d`), calibration
        residuals that are not exactly representable as :math:`L z` receive an
        infinite score, which typically leads to :math:`q = +\\infty` and thus
        to a prediction set equal to the whole output space. This behaviour is
        the faithful consequence of the scoring rule and keeps the coverage
        guarantee valid.

    Example::

        import numpy as np
        from deel.puncc.api.joint import JointCP

        L = np.array([[1.0], [1.0]])
        cp = JointCP(model, L)
        cp.fit(X_calib, y_calib)
        y_pred, pred_set = cp.predict(X_test, alpha=0.1)
        lower, upper = pred_set.linear_bounds(np.array([1.0, -1.0]), 0.0)
    """

    def __init__(
        self,
        predictor: Any,
        L: Optional[Iterable] = None,
        *,
        mapping: Optional[Iterable] = None,
        train: bool = False,
        tol: float = DEFAULT_SUBSPACE_TOL,
    ):
        if predictor is None:
            raise ValueError("Argument `predictor` must not be None.")

        L = _resolve_alias(L, mapping, "L", "mapping")
        if L is None:
            raise ValueError("Argument `L` (or its alias `mapping`) must be provided.")

        if isinstance(predictor, BasePredictor):
            self.predictor = predictor
        elif hasattr(predictor, "predict"):
            self.predictor = BasePredictor(predictor, is_trained=not train)
        else:
            raise RuntimeError(
                "Provided predictor has no predict method. "
                "Use a BasePredictor or an object implementing `predict`."
            )

        self.calibrator = JointCalibrator(L, tol=tol)
        self.L = self.calibrator.L
        self.train = bool(train)
        self.tol = float(tol)

        self.scores_ = None
        self.radius_ = None
        self.alpha_ = None
        self._fitted = False

    # ~~~~~~~~~~~~ Accessors ~~~~~~~~~~~~

    @property
    def mapping(self) -> np.ndarray:
        """Mapping :math:`L` used by the calibrated predictor."""
        return self.L

    @property
    def radius(self) -> Optional[float]:
        """Last computed conformal radius, or ``None``."""
        return self.radius_

    @property
    def is_fitted(self) -> bool:
        """Whether calibration scores are available."""
        return self._fitted

    @property
    def n_outputs(self) -> int:
        """Dimension :math:`d` of the output space."""
        return self.calibrator.n_outputs

    # ~~~~~~~~~~~~ Prediction helper ~~~~~~~~~~~~

    def _predict(self, X: Iterable) -> np.ndarray:
        """Call the fixed predictor and validate output dimensions.

        :param Iterable X: input features.

        :returns: predictions with shape ``(n, d)``.
        :rtype: np.ndarray

        :raises ValueError: missing inputs or inconsistent output dimensions.
        :raises RuntimeError: the underlying model failed.
        """
        if X is None:
            raise ValueError(
                "Input features `X` must be provided to compute predictions."
            )

        try:
            raw = self.predictor.predict(X)
        except Exception as err:
            raise RuntimeError(
                "The underlying predictor failed to compute predictions."
            ) from err

        if raw is None:
            raise RuntimeError(
                "The underlying predictor returned None instead of predictions."
            )

        return self.calibrator.as_outputs(raw, "y_pred")

    # ~~~~~~~~~~~~ Fit / predict ~~~~~~~~~~~~

    def fit(
        self,
        X_calib: Optional[Iterable] = None,
        y_calib: Optional[Iterable] = None,
        *,
        X_fit: Optional[Iterable] = None,
        y_fit: Optional[Iterable] = None,
        alpha: Optional[float] = None,
        **kwargs: Optional[dict],
    ) -> "JointCP":
        """Calibrate the joint conformal predictor.

        The underlying model is not retrained, unless ``train=True`` was set at
        construction time and ``(X_fit, y_fit)`` are provided. Scores are
        computed on the independent calibration set ``(X_calib, y_calib)``; the
        model is really called on ``X_calib``.

        :param Iterable X_calib: features of the calibration set.
        :param Iterable y_calib: outputs of the calibration set, shape
            ``(n, d)``.
        :param Optional[Iterable] X_fit: features used to train the model when
            ``train=True``.
        :param Optional[Iterable] y_fit: outputs used to train the model when
            ``train=True``.
        :param Optional[float] alpha: if provided, the conformal radius is
            computed and stored at fit time.
        :param dict kwargs: keyword arguments forwarded to the model ``fit``.

        :returns: the fitted predictor.
        :rtype: JointCP

        :raises ValueError: missing or inconsistent calibration data.
        :raises RuntimeError: training requested without a fit set, or model
            failure.
        """
        if X_calib is None or y_calib is None:
            raise ValueError(
                "An independent calibration set is required: both `X_calib` "
                "and `y_calib` must be provided."
            )

        y_calib_arr = self.calibrator.as_outputs(y_calib, "y_calib")

        if self.train:
            if X_fit is None or y_fit is None:
                raise RuntimeError(
                    "`train` is set to True but no fit dataset was provided. "
                    "Pass `X_fit` and `y_fit` or construct JointCP with "
                    "`train=False`."
                )
            self.predictor.fit(X_fit, y_fit, **kwargs)

        y_pred_calib = self._predict(X_calib)

        if y_pred_calib.shape[0] != y_calib_arr.shape[0]:
            raise ValueError(
                "`X_calib` and `y_calib` must contain the same number of "
                f"samples. Got {y_pred_calib.shape[0]} prediction(s) for "
                f"{y_calib_arr.shape[0]} label(s)."
            )

        self.calibrator.fit(y_true=y_calib_arr, y_pred=y_pred_calib)
        self.scores_ = self.calibrator.get_nonconformity_scores()
        self._fitted = True

        if alpha is not None:
            self.radius_ = self.calibrator.compute_radius(alpha)
            self.alpha_ = self.calibrator.check_alpha(alpha)

        return self

    def compute_radius(self, alpha: float) -> float:
        """Compute and store the corrected conformal radius for ``alpha``.

        :param float alpha: significance level.

        :returns: radius :math:`q`.
        :rtype: float

        :raises RuntimeError: :meth:`fit` was not called.
        :raises ValueError: insufficient calibration samples or invalid
            ``alpha``.
        """
        if not self._fitted:
            raise RuntimeError(
                "JointCP must be calibrated (call `fit`) before computing a radius."
            )
        self.radius_ = self.calibrator.compute_radius(alpha)
        self.alpha_ = self.calibrator.check_alpha(alpha)
        return self.radius_

    def predict(
        self, X: Iterable, alpha: Optional[float] = None
    ) -> Tuple[np.ndarray, JointEllipsoid]:
        """Compute point predictions and their joint prediction sets.

        The model is really called on ``X``; the returned set is the true
        ellipsoid :math:`\\hat{f}(x) + L\\{z : \\|z\\|_2 \\leq q\\}` (it is not
        a per-coordinate box).

        :param Iterable X: features of new samples.
        :param Optional[float] alpha: significance level. If omitted, the
            radius stored during :meth:`fit` or a previous call is used.

        :returns: couple ``(y_pred, pred_set)``.
        :rtype: Tuple[np.ndarray, JointEllipsoid]

        :raises RuntimeError: :meth:`fit` was not called.
        :raises ValueError: no ``alpha`` available or inconsistent inputs.
        """
        if not self._fitted:
            raise RuntimeError("`fit` must be called before `predict`.")

        if alpha is None:
            if self.radius_ is None:
                raise ValueError(
                    "No significance level available. Pass `alpha` to "
                    "`predict` or provide it to `fit`."
                )
            radius = self.radius_
        else:
            radius = self.compute_radius(alpha)

        y_pred = self._predict(X)
        pred_set = JointEllipsoid(y_pred, self.L, radius, tol=self.tol)
        return y_pred, pred_set

    def get_nonconformity_scores(self) -> np.ndarray:
        """Return the calibration joint scores.

        :returns: scores with shape ``(n_calib,)``.
        :rtype: np.ndarray

        :raises RuntimeError: :meth:`fit` was not called.
        """
        if not self._fitted:
            raise RuntimeError(
                "`fit` must be called before accessing nonconformity scores."
            )
        return self.scores_


#: Aliases of :class:`JointCP`.
JointSplitCP = JointCP
MultiOutputJointCP = JointCP
MultiOutputCP = JointCP
