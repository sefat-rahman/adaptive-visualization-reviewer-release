"""County-level FPS topology precompute for the OpenLayers app."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from adaptive_visualization.core.fps_order_cache import FPSOrderCache
from adaptive_visualization.core.state_fps_topology_cache import StateFPSTopologyCache


class CountyFPSTopologyCache(StateFPSTopologyCache):
    """Precompute full-point topology for county-level FPS samples."""

    CACHE_VERSION = "county_fps_topology_full_v1"
    CACHE_DIRNAME = "county_fps_topology_cache"
    BASELINE_PERCENTAGES = tuple(range(5, 31, 5))
    BASELINE_PERCENTAGE = 5
    PERCENTAGES = tuple(range(1, 101))

    def get_or_create(
        self,
        *,
        state: str,
        county: str | None = None,
        source_df: pd.DataFrame,
        error_threshold: float,
        metric_name: str = StateFPSTopologyCache.DEFAULT_THRESHOLD_METRIC,
        baseline_percentage: int | None = None,
    ) -> pd.DataFrame:
        return super().get_or_create(
            state=self._coerce_region_id(state, county),
            source_df=source_df,
            error_threshold=error_threshold,
            metric_name=metric_name,
            baseline_percentage=baseline_percentage,
        )

    def get_selection(
        self,
        *,
        state: str,
        county: str | None = None,
        source_df: pd.DataFrame,
        error_threshold: float,
        metric_name: str = StateFPSTopologyCache.DEFAULT_THRESHOLD_METRIC,
        baseline_percentage: int | None = None,
    ) -> dict[str, Any] | None:
        return super().get_selection(
            state=self._coerce_region_id(state, county),
            source_df=source_df,
            error_threshold=error_threshold,
            metric_name=metric_name,
            baseline_percentage=baseline_percentage,
        )

    def get_payload(
        self,
        *,
        state: str,
        county: str | None = None,
        source_df: pd.DataFrame,
        current_percentage: int,
        baseline_percentage: int | None = None,
    ) -> dict[str, Any]:
        return super().get_payload(
            state=self._coerce_region_id(state, county),
            source_df=source_df,
            current_percentage=current_percentage,
            baseline_percentage=baseline_percentage,
        )

    def get_curve(self, *, state: str, county: str | None = None, source_df: pd.DataFrame) -> pd.DataFrame:
        return super().get_curve(state=self._coerce_region_id(state, county), source_df=source_df)

    def ensure_precomputed_until(
        self,
        source_df: pd.DataFrame,
        *,
        state: str,
        county: str | None = None,
        max_percentage: int | None = None,
    ) -> pd.DataFrame:
        return super().ensure_precomputed_until(
            source_df,
            state=self._coerce_region_id(state, county),
            max_percentage=max_percentage,
        )

    def ensure_precomputed(
        self,
        source_df: pd.DataFrame,
        *,
        state: str,
        county: str | None = None,
    ) -> pd.DataFrame:
        return self.ensure_precomputed_until(source_df, state=state, county=county)

    def _load_shared_order(self, *, state: str, source_df: pd.DataFrame) -> np.ndarray | None:
        state_code, county_name = self._split_region_id(state)
        shared_order_path = (
            self.shared_order_dir
            / "county"
            / self._sanitize(state_code)
            / self._sanitize(county_name)
            / "order.npy"
        )
        shared_metadata_path = self.shared_order_dir / "metadata.json"
        if not shared_order_path.exists() or not shared_metadata_path.exists():
            return None

        try:
            metadata = json.loads(shared_metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

        if metadata.get("cache_version") != FPSOrderCache.CACHE_VERSION:
            return None

        source_signature = self._frame_signature(source_df)
        expected_path = str(shared_order_path.relative_to(self.shared_order_dir)).replace("\\", "/")
        expected_state = state_code.strip().upper()
        expected_county = county_name.strip().lower()
        match_found = False
        for info in metadata.get("entries", {}).values():
            if not isinstance(info, dict):
                continue
            if info.get("path") != expected_path:
                continue
            if info.get("zoom") != "county":
                continue
            if str(info.get("state", "")).strip().upper() != expected_state:
                continue
            if str(info.get("county", "")).strip().lower() != expected_county:
                continue
            if info.get("source_signature") != source_signature:
                continue
            if int(info.get("n_points", -1)) != len(source_df):
                continue
            match_found = True
            break

        if not match_found:
            return None

        try:
            shared_order = np.load(shared_order_path)
        except Exception:
            return None
        if shared_order.shape != (len(source_df),):
            return None
        return shared_order.astype(np.int32, copy=False)

    def _state_dir(self, state: str) -> Path:
        state_code, county_name = self._split_region_id(state)
        return self.cache_dir / "counties" / self._sanitize(state_code) / self._sanitize(county_name)

    def _state_key(self, state: str) -> str:
        state_code, county_name = self._split_region_id(state)
        return f"counties/{self._sanitize(state_code)}/{self._sanitize(county_name)}"

    @staticmethod
    def _region_id(state: str, county: str) -> str:
        return f"{state.strip().upper()}::{county.strip()}"

    @classmethod
    def _coerce_region_id(cls, state: str, county: str | None = None) -> str:
        if county is None and "::" in state:
            return state
        return cls._region_id(state, county or "")

    @staticmethod
    def _split_region_id(region_id: str) -> tuple[str, str]:
        if "::" not in region_id:
            return region_id.strip().upper(), ""
        state, county = region_id.split("::", 1)
        return state.strip().upper(), county.strip()


