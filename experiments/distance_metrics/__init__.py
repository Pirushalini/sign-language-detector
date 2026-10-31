"""Alternative distance metrics for template matching.

Implements and evaluates four distance formulations for the 132-dimensional
hand landmark feature space:
  1. Euclidean Distance
  2. Cosine Distance
  3. Standardized Euclidean Distance (variance-normalized)
  4. Mahalanobis Distance (covariance-regularized)
"""
from .metrics import (
    euclidean_distance,
    cosine_distance,
    standardized_euclidean_distance,
    mahalanobis_distance,
    fit_feature_variance,
    fit_covariance,
)
from .matcher import DistanceTemplateMatcher

__all__ = [
    "euclidean_distance",
    "cosine_distance",
    "standardized_euclidean_distance",
    "mahalanobis_distance",
    "fit_feature_variance",
    "fit_covariance",
    "DistanceTemplateMatcher",
]
