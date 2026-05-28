"""
StatisticalAnalyzer — computes and compares statistical properties of
a point set using latitude/longitude as the two dimensions.
"""

import numpy as np
import pandas as pd
from typing import Dict


class StatisticalAnalyzer:
    """
    Computes five summary statistics for a geographic point set and
    calculates the absolute difference between two sets (original vs sampled).

    Statistics computed:
        mean_lat    — mean latitude
        mean_lng    — mean longitude
        std_lat     — standard deviation of latitude
        std_lng     — standard deviation of longitude
        correlation — Pearson correlation between latitude and longitude
    """

    def compute(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Compute summary statistics for a dataframe.

        Args:
            df: DataFrame with 'Start_Lat' and 'Start_Lng' columns.

        Returns:
            Dictionary of the five statistics.
        """
        lats = df['Start_Lat'].values.astype(float)
        lngs = df['Start_Lng'].values.astype(float)

        if len(lats) < 2:
            return {'mean_lat': float(lats[0]), 'mean_lng': float(lngs[0]),
                    'std_lat': 0.0, 'std_lng': 0.0, 'correlation': 0.0}

        corr_matrix = np.corrcoef(lats, lngs)
        correlation = corr_matrix[0, 1] if not np.isnan(corr_matrix[0, 1]) else 0.0

        return {
            'mean_lat':    float(np.mean(lats)),
            'mean_lng':    float(np.mean(lngs)),
            'std_lat':     float(np.std(lats)),
            'std_lng':     float(np.std(lngs)),
            'correlation': float(correlation),
        }

    def compare(
        self,
        original_df: pd.DataFrame,
        sampled_df: pd.DataFrame,
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute statistics for both datasets and their absolute differences.

        Args:
            original_df: Full (unsampled) dataset.
            sampled_df:  Subsampled dataset.

        Returns:
            Dictionary with keys 'original', 'sampled', 'error' — each
            containing the five statistics (or differences for 'error').
        """
        orig_stats    = self.compute(original_df)
        sampled_stats = self.compute(sampled_df)

        error = {k: abs(orig_stats[k] - sampled_stats[k]) for k in orig_stats}

        return {
            'original': orig_stats,
            'sampled':  sampled_stats,
            'error':    error,
        }
