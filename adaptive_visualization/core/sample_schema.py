"""Shared schema for persisted display/sample point CSVs."""

from __future__ import annotations

import pandas as pd


DISPLAY_SAMPLE_COLUMNS = ("ID", "Start_Lat", "Start_Lng", "State", "County", "City")
REQUIRED_POINT_COLUMNS = ("Start_Lat", "Start_Lng")


def trim_sample_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only columns needed for plotting, navigation, and point identity."""
    missing = [column for column in REQUIRED_POINT_COLUMNS if column not in df.columns]
    if missing:
        raise KeyError(f"Sample data is missing required coordinate columns: {missing}")
    columns = [column for column in DISPLAY_SAMPLE_COLUMNS if column in df.columns]
    return df.loc[:, columns].copy()
