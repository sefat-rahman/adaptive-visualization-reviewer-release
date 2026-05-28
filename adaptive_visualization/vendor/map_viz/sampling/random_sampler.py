"""
RandomSampler — uniform random sampling without replacement.
"""

import numpy as np

from adaptive_visualization.vendor.map_viz.sampling.base import BaseSampler


class RandomSampler(BaseSampler):
    """
    Uniform random sampling.

    Selects n_keep points chosen uniformly at random without replacement.
    Provides a baseline to compare against FPS and NPS strategies.
    """

    def __init__(self, random_seed: int = 42):
        self.rng = np.random.default_rng(random_seed)

    def sample(self, points: np.ndarray, n_keep: int) -> np.ndarray:
        """
        Args:
            points:  (N, 2) array of coordinates.
            n_keep:  How many points to keep.

        Returns:
            Indices of the n_keep selected points.
        """
        if self._validate(points, n_keep):
            return np.arange(len(points))

        return self.rng.choice(len(points), size=n_keep, replace=False)
