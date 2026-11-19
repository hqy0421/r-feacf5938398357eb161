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
Public entry point for joint (multivariate) conformal prediction for
multi-output regression.

This module re-exports the implementation of
:mod:`deel.puncc.api.joint` so that it can be imported from the package root,
e.g. ``from deel.puncc.joint import JointCP``.
"""

from deel.puncc.api.joint import DEFAULT_SUBSPACE_TOL
from deel.puncc.api.joint import Ellipsoid
from deel.puncc.api.joint import EllipsoidalSet
from deel.puncc.api.joint import JointCalibrator
from deel.puncc.api.joint import JointCP
from deel.puncc.api.joint import JointEllipsoid
from deel.puncc.api.joint import JointPredictionSet
from deel.puncc.api.joint import JointSet
from deel.puncc.api.joint import JointSplitCP
from deel.puncc.api.joint import MultiOutputCP
from deel.puncc.api.joint import MultiOutputJointCP
from deel.puncc.api.joint import column_space_coordinates
from deel.puncc.api.joint import column_space_norm
from deel.puncc.api.joint import column_space_norm_builder
from deel.puncc.api.joint import linear_functional_bounds

__all__ = [
    "DEFAULT_SUBSPACE_TOL",
    "Ellipsoid",
    "EllipsoidalSet",
    "JointCalibrator",
    "JointCP",
    "JointEllipsoid",
    "JointPredictionSet",
    "JointSet",
    "JointSplitCP",
    "MultiOutputCP",
    "MultiOutputJointCP",
    "column_space_coordinates",
    "column_space_norm",
    "column_space_norm_builder",
    "linear_functional_bounds",
]
