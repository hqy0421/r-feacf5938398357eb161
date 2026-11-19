.. _joint:

🧭 Joint (multi-output) regression
===================================


The wrappers on the :doc:`regression <regression>` page produce one interval
per output, and their uncertainty is described by assembling per-dimension
intervals. When the outputs are correlated this is both loose and wrong: the
uncertainty of a derived metric (e.g. a weighted sum of outputs) cannot be
recovered by summing the individual interval widths.

This page presents joint, ellipsoidal conformal prediction for multi-output
regression. Given a fixed predictor, an independent calibration set and a
matrix :math:`L` with full column rank, the joint prediction set for a new
input :math:`x` is

.. math::

    C(x) = \{\hat{y}(x) + L z \;:\; \|z\|_2 \leq q\},

where the radius :math:`q` is obtained with puncc's native corrected
quantile. The nonconformity score of a calibration sample is the norm of the
coordinates of the residual :math:`y - \hat{y}` in the column space of
:math:`L`; a residual outside that space scores :math:`+\infty`, so the
assumption that the residual follows the chosen linear model is itself
calibrated. Low-dimensional column spaces, including
:math:`\mathrm{rank}(L) = 1`, are supported.

For any linear query :math:`A y + b`, the prediction set yields **tight exact**
bounds (attained on the boundary of the ellipsoid)

.. math::

    (A \hat{y} + b)_i \pm q \, \|L^{\top} a_i\|_2,

computed from the joint set rather than from per-dimension interval boxes.
Membership of a candidate output in :math:`C(x)` can be checked directly.
When :math:`q = +\infty` the joint set is the whole output space: membership
always holds and non-constant queries return :math:`\pm\infty`, while
constant queries (a zero row in :math:`A`) still return finite bounds.

.. autoclass:: deel.puncc.joint.JointCP

.. autoclass:: deel.puncc.joint.JointCalibrator

.. autoclass:: deel.puncc.joint.JointEllipsoid
    :members:

.. autofunction:: deel.puncc.joint.linear_functional_bounds

.. autofunction:: deel.puncc.joint.column_space_norm

.. autofunction:: deel.puncc.joint.column_space_norm_builder

.. autofunction:: deel.puncc.joint.column_space_coordinates
