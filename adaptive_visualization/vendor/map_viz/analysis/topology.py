"""
TopologicalAnalyzer - computes persistence diagrams and comparison metrics.

Ripser and persim are optional dependencies. If they are unavailable, the
analyzer returns graceful placeholders so the dashboard can keep running.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional

try:
    from ripser import ripser
    RIPSER_AVAILABLE = True
except ImportError:
    RIPSER_AVAILABLE = False

try:
    from persim import bottleneck, wasserstein
    PERSIM_AVAILABLE = True
except ImportError:
    PERSIM_AVAILABLE = False


class TopologicalAnalyzer:
    """
    Computes H0/H1 persistence for a geographic point set and compares the
    original and reduced diagrams with bottleneck and Wasserstein distances.
    """

    def __init__(self, max_points: int = 500, random_seed: int = 42):
        self.max_points = max_points
        self.random_seed = random_seed

    @property
    def available(self) -> bool:
        """True when Ripser is installed."""
        return RIPSER_AVAILABLE

    @property
    def distances_available(self) -> bool:
        """True when persim distance metrics are installed."""
        return PERSIM_AVAILABLE

    def compute(self, df: pd.DataFrame) -> Optional[Dict[str, np.ndarray]]:
        """
        Compute persistence diagrams for the point set.

        Returns:
            Dict with keys `h0` and `h1`, each an (N, 2) ndarray. Returns None
            if Ripser is unavailable.
        """
        if not RIPSER_AVAILABLE:
            return None

        points = df[['Start_Lat', 'Start_Lng']].values.astype(float)
        if len(points) < 2:
            return {
                'h0': np.empty((0, 2), dtype=float),
                'h1': np.empty((0, 2), dtype=float),
            }

        if len(points) > self.max_points:
            rng = np.random.default_rng(self.random_seed)
            idx = rng.choice(len(points), size=self.max_points, replace=False)
            points = points[idx]

        diagrams = ripser(points, maxdim=1)['dgms']

        h0 = diagrams[0] if len(diagrams) > 0 else np.empty((0, 2), dtype=float)
        h1 = diagrams[1] if len(diagrams) > 1 else np.empty((0, 2), dtype=float)

        h0 = h0[np.isfinite(h0[:, 1])] if len(h0) > 0 else h0
        h1 = h1[np.isfinite(h1[:, 1])] if len(h1) > 0 else h1

        return {'h0': h0, 'h1': h1}

    def compare(
        self,
        original_df: pd.DataFrame,
        reduced_df: pd.DataFrame,
    ) -> Optional[Dict]:
        """
        Compare original and reduced persistence diagrams and distances.

        Returns:
            Dict with keys:
            - `original`: {'h0': ndarray, 'h1': ndarray}
            - `reduced`: {'h0': ndarray, 'h1': ndarray}
            - `distances`: comparison metrics or None if persim unavailable
        """
        if not RIPSER_AVAILABLE:
            return None

        original = self.compute(original_df)
        reduced = self.compute(reduced_df)
        if original is None or reduced is None:
            return None

        distances = None
        if PERSIM_AVAILABLE:
            distances = {
                'bottleneck_h0': self._safe_distance(
                    original['h0'], reduced['h0'], bottleneck
                ),
                'bottleneck_h1': self._safe_distance(
                    original['h1'], reduced['h1'], bottleneck
                ),
                'wasserstein_h0': self._safe_distance(
                    original['h0'], reduced['h0'], wasserstein
                ),
                'wasserstein_h1': self._safe_distance(
                    original['h1'], reduced['h1'], wasserstein
                ),
            }

        return {
            'original': original,
            'reduced': reduced,
            'distances': distances,
        }

    @staticmethod
    def _safe_distance(
        dgm1: np.ndarray,
        dgm2: np.ndarray,
        fn,
    ) -> Optional[float]:
        """
        Compute a persistence-diagram distance while handling empty diagrams.
        """
        if dgm1.size == 0 and dgm2.size == 0:
            return 0.0

        try:
            return float(fn(dgm1, dgm2))
        except Exception:
            return None
