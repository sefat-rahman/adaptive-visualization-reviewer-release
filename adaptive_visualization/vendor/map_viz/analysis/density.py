"""
DensityAnalyzer - computes grid-based density comparisons.

Builds comparable 2D density grids for the original and reduced point sets
over a shared latitude/longitude extent, then reports simple distance metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict


class DensityAnalyzer:
    """
    Compares original and reduced data with normalized grid densities.
    """

    def __init__(self, bins_lat: int = 40, bins_lng: int = 40):
        self.bins_lat = bins_lat
        self.bins_lng = bins_lng

    def compare(
        self,
        original_df: pd.DataFrame,
        reduced_df: pd.DataFrame,
    ) -> Dict:
        """
        Compute density grids and L2/Linf distances.
        """
        orig_points = self._extract_points(original_df)
        reduced_points = self._extract_points(reduced_df)

        lat_edges, lng_edges = self._compute_edges(orig_points, reduced_points)
        original_density = self._density_grid(orig_points, lat_edges, lng_edges)
        reduced_density = self._density_grid(reduced_points, lat_edges, lng_edges)
        difference = original_density - reduced_density

        return {
            'original_density': original_density,
            'reduced_density': reduced_density,
            'difference_density': difference,
            'lat_edges': lat_edges,
            'lng_edges': lng_edges,
            'metrics': {
                'l2_norm': float(np.linalg.norm(difference.ravel(), ord=2)),
                'linf_distance': float(np.max(np.abs(difference))),
            },
        }

    @staticmethod
    def _extract_points(df: pd.DataFrame) -> np.ndarray:
        """Extract latitude/longitude points."""
        if df.empty:
            return np.empty((0, 2), dtype=float)

        return df[['Start_Lat', 'Start_Lng']].dropna().to_numpy(dtype=float)

    def _compute_edges(
        self,
        original_points: np.ndarray,
        reduced_points: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute shared histogram edges over the union of both point sets."""
        all_points = np.vstack([
            pts for pts in (original_points, reduced_points) if len(pts) > 0
        ]) if (len(original_points) > 0 or len(reduced_points) > 0) else np.empty((0, 2))

        if len(all_points) == 0:
            lat_edges = np.linspace(0.0, 1.0, self.bins_lat + 1)
            lng_edges = np.linspace(0.0, 1.0, self.bins_lng + 1)
            return lat_edges, lng_edges

        lat_min, lng_min = np.min(all_points, axis=0)
        lat_max, lng_max = np.max(all_points, axis=0)

        lat_span = max(lat_max - lat_min, 1e-6)
        lng_span = max(lng_max - lng_min, 1e-6)

        lat_pad = lat_span * 0.02
        lng_pad = lng_span * 0.02

        lat_edges = np.linspace(lat_min - lat_pad, lat_max + lat_pad, self.bins_lat + 1)
        lng_edges = np.linspace(lng_min - lng_pad, lng_max + lng_pad, self.bins_lng + 1)
        return lat_edges, lng_edges

    @staticmethod
    def _density_grid(
        points: np.ndarray,
        lat_edges: np.ndarray,
        lng_edges: np.ndarray,
    ) -> np.ndarray:
        """Compute a normalized 2D histogram."""
        if len(points) == 0:
            return np.zeros((len(lat_edges) - 1, len(lng_edges) - 1), dtype=float)

        hist, _, _ = np.histogram2d(
            points[:, 0],
            points[:, 1],
            bins=[lat_edges, lng_edges],
        )
        total = np.sum(hist)
        if total <= 0:
            return np.zeros_like(hist, dtype=float)
        return hist / total
