"""
BaseSampler — abstract base class for all point sampling strategies.
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseSampler(ABC):
    """
    Abstract base class for point sampling.

    Every subclass must implement `sample`, which takes an (N x 2) array of
    coordinates and returns the indices of the n_keep points to retain.
    """

    @abstractmethod
    def sample(self, points: np.ndarray, n_keep: int) -> np.ndarray:
        """
        Select n_keep points from the input array.

        Args:
            points:  (N, 2) array of [lat, lng] coordinates.
            n_keep:  Number of points to keep.

        Returns:
            1-D integer array of length n_keep with the chosen row indices.
        """
        pass

    def _validate(self, points: np.ndarray, n_keep: int) -> bool:
        """Return True if early-exit (n_keep >= N) is needed."""
        return n_keep >= len(points)
