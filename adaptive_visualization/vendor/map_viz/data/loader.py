"""
AccidentDataLoader — loads and filters the US accidents CSV data.
"""

import pandas as pd
from pathlib import Path


class AccidentDataLoader:
    """
    Loads accident data from a CSV file and provides filtering utilities
    for state and city level views.

    All columns from the source CSV are preserved; only rows with missing
    coordinates are dropped.
    """

    # Columns that must exist for the map to work
    COORDINATE_COLS = ['Start_Lat', 'Start_Lng']

    def load(self, csv_path: str) -> pd.DataFrame:
        """
        Load accident data from a CSV file, keeping all columns.

        Args:
            csv_path: Path to the CSV file.

        Returns:
            Full DataFrame with all columns; rows with missing coordinates dropped.
        """
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        df = pd.read_csv(csv_path, low_memory=False)
        df = df.dropna(subset=self.COORDINATE_COLS)
        df = df.reset_index(drop=True)
        return df

    def filter_by_state(self, df: pd.DataFrame, state: str) -> pd.DataFrame:
        """Return only rows belonging to the given state abbreviation (e.g. 'CA')."""
        return df[df['State'] == state].reset_index(drop=True)

    def filter_by_city(self, df: pd.DataFrame, city: str, state: str) -> pd.DataFrame:
        """Return only rows belonging to the given city within a state."""
        mask = (df['City'] == city) & (df['State'] == state)
        return df[mask].reset_index(drop=True)

    def get_states(self, df: pd.DataFrame) -> list:
        """Return sorted list of unique state abbreviations present in df."""
        return sorted(df['State'].dropna().unique().tolist())

    def get_cities(self, df: pd.DataFrame, state: str) -> list:
        """Return sorted list of unique city names within a state."""
        state_df = df[df['State'] == state]
        return sorted(state_df['City'].dropna().unique().tolist())
