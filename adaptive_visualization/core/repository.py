"""Data access and analysis reuse for the OpenLayers adaptive visualization."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from adaptive_visualization.core.boundaries import BoundaryService
from adaptive_visualization.core.country_fps_topology_cache import CountryFPSTopologyCache
from adaptive_visualization.core.county_fps_topology_cache import CountyFPSTopologyCache
from adaptive_visualization.core.fps_distance_baseline_cache import FPSDistanceBaselineCache
from adaptive_visualization.core.fps_order_cache import FPSOrderCache
from adaptive_visualization.paths import CACHE_DIR
from adaptive_visualization.core.pixel_country_cache import OpenLayersCountryPixelCache
from adaptive_visualization.core.pixel_county_cache import OpenLayersCountyPixelCache
from adaptive_visualization.core.pixel_state_cache import OpenLayersStatePixelCache
from adaptive_visualization.core.runtime_sampling import RuntimeSamplingCache
from adaptive_visualization.core.state_fps_topology_cache import StateFPSTopologyCache
from adaptive_visualization.vendor.map_viz.analysis.density import DensityAnalyzer
from adaptive_visualization.vendor.map_viz.analysis.statistics import StatisticalAnalyzer
from adaptive_visualization.vendor.map_viz.analysis.topology import TopologicalAnalyzer
from adaptive_visualization.vendor.map_viz.data.loader import AccidentDataLoader


class AdaptiveDataRepository:
    """Cached access to generated datasets plus reusable analysis helpers."""

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        self.loader = AccidentDataLoader()
        self.statistical = StatisticalAnalyzer()
        self.density = DensityAnalyzer()
        self.topological = TopologicalAnalyzer()
        self.boundaries = BoundaryService(CACHE_DIR)
        self.country_pixel_cache = OpenLayersCountryPixelCache(self.data_dir, self.loader)
        self.state_pixel_cache = OpenLayersStatePixelCache(self.data_dir, self.loader)
        self.county_pixel_cache = OpenLayersCountyPixelCache(self.data_dir, self.loader)
        self.fps_order_cache = FPSOrderCache(self.data_dir)
        self.country_fps_topology = CountryFPSTopologyCache(self.data_dir, self.fps_order_cache)
        self.county_fps_topology = CountyFPSTopologyCache(self.data_dir)
        self.state_fps_topology = StateFPSTopologyCache(self.data_dir)
        self.fps_distance_baseline = FPSDistanceBaselineCache(self.data_dir, self.fps_order_cache)
        self.runtime_sampling = RuntimeSamplingCache(self.data_dir, self.loader)

    @lru_cache(maxsize=1)
    def load_original(self) -> pd.DataFrame:
        return self.loader.load(str(self.data_dir / 'original.csv'))

    @lru_cache(maxsize=256)
    def load_state_reference(self, state: str) -> pd.DataFrame:
        return self.loader.filter_by_state(self.load_original(), state)

    def load_openlayers_state_pixel_sample(
        self,
        state: str,
        viewport_spec: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        reference_df = self.load_state_reference(state)
        return self.state_pixel_cache.get_or_create(state, reference_df, viewport_spec=viewport_spec)

    @lru_cache(maxsize=1024)
    def load_county_reference(self, state: str, county: str) -> pd.DataFrame:
        df = self.load_original()
        mask = (df['State'] == state) & (df['County'] == county)
        return df[mask].reset_index(drop=True)

    def load_openlayers_county_pixel_sample(
        self,
        state: str,
        county: str,
        viewport_spec: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        reference_df = self.load_county_reference(state, county)
        return self.county_pixel_cache.get_or_create(state, county, reference_df, viewport_spec=viewport_spec)

    def get_available_states(self) -> list[str]:
        return self.loader.get_states(self.load_original())

    def get_fps_threshold_cache_status(self, *, zoom: str, state: str = '', county: str = '') -> dict[str, Any]:
        """Return saved FPS-threshold topology-cache status without computing missing rows."""
        zoom = (zoom or '').strip().lower()
        if zoom == 'country':
            curve = self.country_fps_topology._load_available_curve()
            complete = self.country_fps_topology._curve_complete(curve) if not curve.empty else False
            exact_count = int(len(self.load_original().dropna(subset=['Start_Lat', 'Start_Lng'])))
            return self._fps_threshold_status_payload(
                zoom='country',
                state='',
                county='',
                curve=curve,
                complete=complete,
                exact_count=exact_count,
                cache_path=self.country_fps_topology.cache_dir / 'country',
                expected_row_count=len(self.country_fps_topology.PERCENTAGES),
            )

        if zoom == 'state':
            state = state.strip().upper()
            reference_df = self.load_state_reference(state) if state else pd.DataFrame()
            curve = self.state_fps_topology._load_available_curve(state) if state else pd.DataFrame()
            complete = self.state_fps_topology._curve_complete(curve) if not curve.empty else False
            exact_count = int(len(reference_df.dropna(subset=['Start_Lat', 'Start_Lng']))) if not reference_df.empty else 0
            return self._fps_threshold_status_payload(
                zoom='state',
                state=state,
                county='',
                curve=curve,
                complete=complete,
                exact_count=exact_count,
                cache_path=self.state_fps_topology._state_dir(state),
                expected_row_count=len(self.state_fps_topology.PERCENTAGES),
            )

        state = state.strip().upper()
        county = county.strip()
        county_exact = pd.DataFrame()
        if state and county:
            county_df = self.load_county_reference(state, county)
            if not county_df.empty:
                boundary = self.boundaries.get_county_boundary(state, county, county_df=county_df)
                county_exact = self.boundaries.filter_points_to_boundary(county_df, boundary).reset_index(drop=True)
        region = self.county_fps_topology._coerce_region_id(state, county) if state and county else ''
        curve = self.county_fps_topology._load_available_curve(region) if region else pd.DataFrame()
        complete = self.county_fps_topology._curve_complete(curve) if not curve.empty else False
        exact_count = int(len(county_exact.dropna(subset=['Start_Lat', 'Start_Lng']))) if not county_exact.empty else 0
        return self._fps_threshold_status_payload(
            zoom='county',
            state=state,
            county=county,
            curve=curve,
            complete=complete,
            exact_count=exact_count,
            cache_path=self.county_fps_topology._state_dir(region) if region else self.county_fps_topology.cache_dir / 'counties',
            expected_row_count=len(self.county_fps_topology.PERCENTAGES),
        )

    def generate_fps_threshold_cache(self, *, zoom: str, state: str = '', county: str = '') -> dict[str, Any]:
        """Generate the full saved FPS-threshold topology cache for one state or county."""
        zoom = (zoom or '').strip().lower()
        if zoom == 'state':
            state = state.strip().upper()
            reference_df = self.load_state_reference(state)
            self.state_fps_topology.ensure_precomputed_until(reference_df, state=state)
            return self.get_fps_threshold_cache_status(zoom=zoom, state=state)
        if zoom == 'county':
            state = state.strip().upper()
            county = county.strip()
            county_df = self.load_county_reference(state, county)
            boundary = self.boundaries.get_county_boundary(state, county, county_df=county_df)
            county_exact = self.boundaries.filter_points_to_boundary(county_df, boundary).reset_index(drop=True)
            self.county_fps_topology.ensure_precomputed_until(county_exact, state=state, county=county)
            return self.get_fps_threshold_cache_status(zoom=zoom, state=state, county=county)
        raise ValueError('FPS-threshold cache generation is only supported for state or county views.')

    def fps_threshold_cache_available(self, *, zoom: str, state: str = '', county: str = '') -> bool:
        status = self.get_fps_threshold_cache_status(zoom=zoom, state=state, county=county)
        return bool(status.get('available'))

    def get_fps_threshold_coverage(self) -> dict[str, Any]:
        """List state/county scopes with saved FPS-threshold topology rows."""
        states = []
        for state in self.get_available_states():
            curve = self.state_fps_topology._load_available_curve(state)
            if not curve.empty:
                states.append(state)

        counties_by_state: dict[str, list[str]] = {}
        counties_root = self.county_fps_topology.cache_dir / 'counties'
        if counties_root.exists():
            for state_dir in counties_root.iterdir():
                if not state_dir.is_dir():
                    continue
                state = state_dir.name.strip().upper()
                for county_dir in state_dir.iterdir():
                    if not county_dir.is_dir() or county_dir.name == 'samples':
                        continue
                    county = county_dir.name.replace('_', ' ')
                    region = self.county_fps_topology._coerce_region_id(state, county)
                    curve = self.county_fps_topology._load_available_curve(region)
                    if curve.empty:
                        continue
                    counties_by_state.setdefault(state, []).append(county)

        return {
            'states': sorted(states),
            'counties': {
                state: sorted(counties)
                for state, counties in sorted(counties_by_state.items())
            },
        }

    def get_display_df(
        self,
        zoom: str,
        method: str,
        state: str = '',
        county: str = '',
        analysis_property: str = 'statistical',
        viewport_spec: dict[str, Any] | None = None,
        retain_percentage: int = 50,
        error_threshold: float = 0.05,
        topology_baseline_percentage: int | str | None = None,
    ) -> pd.DataFrame:
        baseline_df = self._baseline_display_df(
            zoom=zoom,
            method=method,
            state=state,
            county=county,
            analysis_property=analysis_property,
            viewport_spec=viewport_spec,
            retain_percentage=retain_percentage,
            error_threshold=error_threshold,
            topology_baseline_percentage=topology_baseline_percentage,
        )
        return baseline_df

    def get_reference_df(self, zoom: str, state: str = '', county: str = '') -> pd.DataFrame:
        if zoom == 'country':
            return self.load_original()
        if zoom == 'state':
            return self.load_state_reference(state)
        return self.load_county_reference(state, county)

    def get_county_labels(self, state: str, min_count: int = 2, limit: int = 160) -> list[dict[str, Any]]:
        state_df = self.load_state_reference(state)
        if state_df.empty:
            return []

        grouped = (
            state_df.dropna(subset=['County', 'Start_Lat', 'Start_Lng'])
            .groupby('County', as_index=False)
            .agg(
                lat=('Start_Lat', 'mean'),
                lng=('Start_Lng', 'mean'),
                count=('ID', 'count'),
            )
        )
        grouped = grouped[grouped['count'] >= min_count]
        grouped = grouped.sort_values(['count', 'County'], ascending=[False, True]).head(limit)
        return grouped.to_dict(orient='records')

    def get_analysis(
        self,
        analysis_property: str,
        zoom: str,
        method: str,
        state: str = '',
        county: str = '',
        viewport_spec: dict[str, Any] | None = None,
        retain_percentage: int = 50,
        error_threshold: float = 0.05,
        topology_baseline_percentage: int | str | None = None,
    ) -> dict[str, Any]:
        baseline_display_df = self._baseline_display_df(
            zoom=zoom,
            method=method,
            state=state,
            county=county,
            analysis_property=analysis_property,
            viewport_spec=viewport_spec,
            retain_percentage=retain_percentage,
            error_threshold=error_threshold,
            topology_baseline_percentage=topology_baseline_percentage,
        )
        reference_df = self.get_reference_df(zoom=zoom, state=state, county=county)
        display_df, reference_df = self._prepare_analysis_inputs(
            zoom=zoom,
            state=state,
            county=county,
            display_df=baseline_display_df,
            reference_df=reference_df,
        )

        comparison = self._compute_analysis_comparison(
            analysis_property=analysis_property,
            reference_df=reference_df,
            display_df=display_df,
            method=method,
            zoom=zoom,
            state=state,
            county=county,
            viewport_spec=viewport_spec,
            retain_percentage=retain_percentage,
            error_threshold=error_threshold,
            topology_baseline_percentage=topology_baseline_percentage,
        )

        return self._serialize_analysis(analysis_property, comparison)

    def _baseline_display_df(
        self,
        *,
        zoom: str,
        method: str,
        state: str,
        county: str,
        analysis_property: str,
        viewport_spec: dict[str, Any] | None,
        retain_percentage: int,
        error_threshold: float,
        topology_baseline_percentage: int | str | None,
    ) -> pd.DataFrame:
        if zoom == 'country':
            if method == 'all':
                return self.load_original().reset_index(drop=True)
            if method == 'pixel':
                return self.country_pixel_cache.get_or_create(self.load_original(), viewport_spec)
            if method == 'fps_threshold':
                if self._distance_baseline_requested(topology_baseline_percentage):
                    return self._distance_baseline_display_df(
                        zoom=zoom,
                        state=state,
                        county=county,
                        source_df=self.load_original(),
                        baseline_key=self._distance_baseline_key(topology_baseline_percentage),
                        error_threshold=error_threshold,
                    )
                return self.country_fps_topology.get_or_create(
                    source_df=self.load_original(),
                    error_threshold=error_threshold,
                    baseline_percentage=self._numeric_topology_baseline(topology_baseline_percentage),
                )
            if method == 'random':
                return self.runtime_sampling.get_or_create(
                    method=method,
                    retain_percentage=retain_percentage,
                    zoom=zoom,
                    state=state,
                    county=county,
                    source_df=self.load_original(),
                    analysis_property=analysis_property,
                    error_threshold=error_threshold,
                )
            return self.load_original().reset_index(drop=True)

        if zoom == 'state':
            state_reference = self.load_state_reference(state)
            if method == 'all':
                return state_reference.reset_index(drop=True)
            if method == 'pixel':
                return self.load_openlayers_state_pixel_sample(state, viewport_spec=viewport_spec)
            if method == 'fps_threshold':
                if self._state_topology_cache_enabled(state):
                    if self._distance_baseline_requested(topology_baseline_percentage):
                        return self._distance_baseline_display_df(
                            zoom=zoom,
                            state=state,
                            county=county,
                            source_df=state_reference,
                            baseline_key=self._distance_baseline_key(topology_baseline_percentage),
                            error_threshold=error_threshold,
                        )
                    return self.state_fps_topology.get_or_create(
                        state=state,
                        source_df=state_reference,
                        error_threshold=error_threshold,
                        baseline_percentage=self._numeric_topology_baseline(topology_baseline_percentage),
                    )
                return state_reference.reset_index(drop=True)
            if method == 'random':
                return self.runtime_sampling.get_or_create(
                    method=method,
                    retain_percentage=retain_percentage,
                    zoom=zoom,
                    state=state,
                    county=county,
                    source_df=state_reference,
                    analysis_property=analysis_property,
                    error_threshold=error_threshold,
                )
            return state_reference.reset_index(drop=True)

        county_reference = self.load_county_reference(state, county)
        boundary = self.boundaries.get_county_boundary(state, county, county_df=county_reference)
        county_exact = self.boundaries.filter_points_to_boundary(county_reference, boundary).reset_index(drop=True)
        if method == 'all':
            return county_exact
        if method == 'pixel':
            return self.load_openlayers_county_pixel_sample(state, county, viewport_spec=viewport_spec)
        if method == 'fps_threshold':
            if self._county_topology_cache_enabled(state, county):
                if self._distance_baseline_requested(topology_baseline_percentage):
                    return self._distance_baseline_display_df(
                        zoom=zoom,
                        state=state,
                        county=county,
                        source_df=county_exact,
                        baseline_key=self._distance_baseline_key(topology_baseline_percentage),
                        error_threshold=error_threshold,
                    )
                return self.county_fps_topology.get_or_create(
                    state=state,
                    county=county,
                    source_df=county_exact,
                    error_threshold=error_threshold,
                    baseline_percentage=self._numeric_topology_baseline(topology_baseline_percentage),
                )
            return county_exact
        if method == 'random':
            return self.runtime_sampling.get_or_create(
                method=method,
                retain_percentage=retain_percentage,
                zoom=zoom,
                state=state,
                county=county,
                source_df=county_exact,
                analysis_property=analysis_property,
                error_threshold=error_threshold,
            )
        return county_exact

    def _prepare_analysis_inputs(
        self,
        *,
        zoom: str,
        state: str,
        county: str,
        display_df: pd.DataFrame,
        reference_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if zoom != 'county':
            return display_df.reset_index(drop=True), reference_df.reset_index(drop=True)

        boundary = self.boundaries.get_county_boundary(state, county, county_df=reference_df)
        filtered_display = self.boundaries.filter_points_to_boundary(display_df, boundary).reset_index(drop=True)
        filtered_reference = self.boundaries.filter_points_to_boundary(reference_df, boundary).reset_index(drop=True)
        return filtered_display, filtered_reference

    def _compute_analysis_comparison(
        self,
        *,
        analysis_property: str,
        reference_df: pd.DataFrame,
        display_df: pd.DataFrame,
        method: str,
        zoom: str,
        state: str,
        county: str,
        viewport_spec: dict[str, Any] | None,
        retain_percentage: int,
        error_threshold: float,
        topology_baseline_percentage: int | str | None,
    ) -> Any:
        if analysis_property == 'statistical':
            return self.statistical.compare(reference_df, display_df)
        if analysis_property == 'density':
            return self.density.compare(reference_df, display_df)
        if analysis_property == 'topological':
            if zoom == 'country' and method == 'fps_threshold':
                if self._distance_baseline_requested(topology_baseline_percentage):
                    return self._distance_baseline_payload(
                        zoom=zoom,
                        state=state,
                        county=county,
                        source_df=reference_df,
                        baseline_key=self._distance_baseline_key(topology_baseline_percentage),
                        method=method,
                        retain_percentage=retain_percentage,
                        error_threshold=error_threshold,
                    )
                current_percentage = self._country_topology_percentage(
                    reference_df=reference_df,
                    error_threshold=error_threshold,
                    topology_baseline_percentage=topology_baseline_percentage,
                )
                return self.country_fps_topology.get_payload(
                    source_df=reference_df,
                    current_percentage=current_percentage,
                    baseline_percentage=self._numeric_topology_baseline(topology_baseline_percentage),
                )
            if zoom == 'state' and method == 'fps_threshold' and self._state_topology_cache_enabled(state):
                if self._distance_baseline_requested(topology_baseline_percentage):
                    return self._distance_baseline_payload(
                        zoom=zoom,
                        state=state,
                        county=county,
                        source_df=reference_df,
                        baseline_key=self._distance_baseline_key(topology_baseline_percentage),
                        method=method,
                        retain_percentage=retain_percentage,
                        error_threshold=error_threshold,
                    )
                current_percentage = self._state_topology_percentage(
                    reference_df=reference_df,
                    state=state,
                    error_threshold=error_threshold,
                    topology_baseline_percentage=topology_baseline_percentage,
                )
                return self.state_fps_topology.get_payload(
                    state=state,
                    source_df=reference_df,
                    current_percentage=current_percentage,
                    baseline_percentage=self._numeric_topology_baseline(topology_baseline_percentage),
                )
            if zoom == 'county' and method == 'fps_threshold' and self._county_topology_cache_enabled(state, county):
                if self._distance_baseline_requested(topology_baseline_percentage):
                    return self._distance_baseline_payload(
                        zoom=zoom,
                        state=state,
                        county=county,
                        source_df=reference_df,
                        baseline_key=self._distance_baseline_key(topology_baseline_percentage),
                        method=method,
                        retain_percentage=retain_percentage,
                        error_threshold=error_threshold,
                    )
                current_percentage = self._county_topology_percentage(
                    reference_df=reference_df,
                    state=state,
                    county=county,
                    error_threshold=error_threshold,
                    topology_baseline_percentage=topology_baseline_percentage,
                )
                return self.county_fps_topology.get_payload(
                    state=state,
                    county=county,
                    source_df=reference_df,
                    current_percentage=current_percentage,
                    baseline_percentage=self._numeric_topology_baseline(topology_baseline_percentage),
                )
            return self.topological.compare(reference_df, display_df)
        raise ValueError(f'Unsupported analysis property: {analysis_property}')

    def _country_topology_percentage(
        self,
        *,
        reference_df: pd.DataFrame,
        error_threshold: float,
        topology_baseline_percentage: int | str | None,
    ) -> int:
        selection = self.country_fps_topology.get_selection(
            source_df=reference_df,
            error_threshold=error_threshold,
            baseline_percentage=self._numeric_topology_baseline(topology_baseline_percentage),
        )
        if selection is None:
            return 100
        return int(selection['percentage'])

    def _state_topology_percentage(
        self,
        *,
        reference_df: pd.DataFrame,
        state: str,
        error_threshold: float,
        topology_baseline_percentage: int | str | None,
    ) -> int:
        selection = self.state_fps_topology.get_selection(
            state=state,
            source_df=reference_df,
            error_threshold=error_threshold,
            baseline_percentage=self._numeric_topology_baseline(topology_baseline_percentage),
        )
        if selection is None:
            return 100
        return int(selection['percentage'])

    def _county_topology_percentage(
        self,
        *,
        reference_df: pd.DataFrame,
        state: str,
        county: str,
        error_threshold: float,
        topology_baseline_percentage: int | str | None,
    ) -> int:
        selection = self.county_fps_topology.get_selection(
            state=state,
            county=county,
            source_df=reference_df,
            error_threshold=error_threshold,
            baseline_percentage=self._numeric_topology_baseline(topology_baseline_percentage),
        )
        if selection is None:
            return 100
        return int(selection['percentage'])

    def _distance_baseline_display_df(
        self,
        *,
        zoom: str,
        state: str,
        county: str,
        source_df: pd.DataFrame,
        baseline_key: str,
        error_threshold: float,
    ) -> pd.DataFrame:
        valid_df = source_df.dropna(subset=["Start_Lat", "Start_Lng"]).reset_index(drop=True)
        selection = self._distance_baseline_selection(
            zoom=zoom,
            state=state,
            county=county,
            source_df=valid_df,
            baseline_key=baseline_key,
            error_threshold=error_threshold,
        )
        if selection is None or int(selection["percentage"]) >= 100:
            return valid_df
        sample_path = Path(selection["sample_abs_path"])
        if not sample_path.exists():
            return valid_df
        return pd.read_csv(sample_path)

    def _distance_baseline_payload(
        self,
        *,
        zoom: str,
        state: str,
        county: str,
        source_df: pd.DataFrame,
        baseline_key: str,
        method: str,
        retain_percentage: int,
        error_threshold: float,
    ) -> dict[str, Any]:
        baseline_key = self._distance_baseline_key(baseline_key)
        baseline_label = FPSDistanceBaselineCache.baseline_label(baseline_key)
        valid_df = source_df.dropna(subset=["Start_Lat", "Start_Lng"]).reset_index(drop=True)
        selection = self._distance_baseline_selection(
            zoom=zoom,
            state=state,
            county=county,
            source_df=valid_df,
            baseline_key=baseline_key,
            error_threshold=error_threshold if method == "fps_threshold" else 1.0,
        )
        if selection is None:
            return {
                "original": {"h0": [], "h1": []},
                "reduced": {"h0": [], "h1": []},
                "baseline_distances": {metric: 0.0 for metric in FPSDistanceBaselineCache.METRICS},
                "current_distances": {metric: 0.0 for metric in FPSDistanceBaselineCache.METRICS},
                "normalized_distances": {metric: 0.0 for metric in FPSDistanceBaselineCache.METRICS},
                "baseline_key": baseline_key,
                "baseline_label": baseline_label,
                "current_percentage": retain_percentage,
                "baseline_percentage": None,
                "complete": False,
            }

        context = selection["context"]
        baseline_entry = selection["baseline_entry"]
        curve = selection["curve"]
        current_row = selection.get("row")
        if method != "fps_threshold":
            current_row = self._curve_row_for_percentage(curve, retain_percentage, prefer_at_or_below=zoom == "country")
        if current_row is None:
            current_row = self._curve_row_for_percentage(curve, 100, prefer_at_or_below=True)

        base_dir = context["base_dir"]
        cache = context["cache"]
        fallback_original_dir = self.fps_distance_baseline._scope_dir(zoom=zoom, state=state, county=county)
        original_h0_path = base_dir / "original_h0.csv"
        original_h1_path = base_dir / "original_h1.csv"
        if not original_h0_path.exists():
            original_h0_path = fallback_original_dir / "original_h0.csv"
        if not original_h1_path.exists():
            original_h1_path = fallback_original_dir / "original_h1.csv"
        original_h0 = cache._load_diagram_array(original_h0_path)
        original_h1 = cache._load_diagram_array(original_h1_path)
        if selection.get("is_baseline_sample") and method == "fps_threshold":
            reduced_h0 = cache._load_diagram_array(self.fps_distance_baseline.cache_dir / str(baseline_entry["reduced_h0_path"]))
            reduced_h1 = cache._load_diagram_array(self.fps_distance_baseline.cache_dir / str(baseline_entry["reduced_h1_path"]))
            current_distances = FPSDistanceBaselineCache.metric_block(baseline_entry)
            normalized_distances = {
                metric: self._normalized_against_itself(value)
                for metric, value in current_distances.items()
            }
            current_percentage = None
            current_label = str(baseline_entry.get("baseline_label", baseline_label))
        elif current_row is None:
            reduced_h0 = original_h0
            reduced_h1 = original_h1
            current_distances = {metric: 0.0 for metric in FPSDistanceBaselineCache.METRICS}
            normalized_distances = {metric: 0.0 for metric in FPSDistanceBaselineCache.METRICS}
            current_percentage = 100
            current_label = None
        else:
            current_percentage = int(current_row["percentage"])
            if current_percentage >= 100:
                reduced_h0 = original_h0
                reduced_h1 = original_h1
            else:
                reduced_h0 = cache._load_diagram_array(base_dir / f"pct_{current_percentage:02d}_h0.csv")
                reduced_h1 = cache._load_diagram_array(base_dir / f"pct_{current_percentage:02d}_h1.csv")
            current_distances = self._metric_block_from_row(current_row)
            normalized_distances = FPSDistanceBaselineCache.normalized_block(current_row, baseline_entry)
            current_label = None

        country_min_distance = baseline_entry.get("country_min_pixel_distance")
        if country_min_distance is None:
            country_min_distance = baseline_entry.get("country_min_distance")
        source_min_distance = baseline_entry.get("source_min_pixel_distance")
        if source_min_distance is None:
            source_min_distance = baseline_entry.get("baseline_source_min_pixel_distance")
        if source_min_distance is None:
            source_min_distance = country_min_distance
        if source_min_distance is None:
            source_min_distance = 0.0
        applied_min_distance = baseline_entry.get("applied_min_pixel_distance")
        if applied_min_distance is None:
            applied_min_distance = source_min_distance

        return {
            "original": {"h0": original_h0, "h1": original_h1},
            "reduced": {"h0": reduced_h0, "h1": reduced_h1},
            "baseline_distances": FPSDistanceBaselineCache.metric_block(baseline_entry),
            "current_distances": current_distances,
            "normalized_distances": normalized_distances,
            "current_percentage": current_percentage,
            "current_label": current_label,
            "baseline_percentage": None,
            "baseline_key": str(baseline_entry.get("baseline_key", baseline_key)),
            "baseline_label": str(baseline_entry.get("baseline_label", baseline_label)),
            "baseline_n_points": int(baseline_entry.get("n_points", 0)),
            "baseline_equivalent_percentage": float(baseline_entry.get("equivalent_percentage", 0.0)),
            "country_min_distance": None if country_min_distance is None else float(country_min_distance),
            "country_min_pixel_distance": None if country_min_distance is None else float(country_min_distance),
            "source_min_pixel_distance": float(source_min_distance),
            "baseline_source_zoom": baseline_entry.get("baseline_source_zoom"),
            "baseline_source_state": baseline_entry.get("baseline_source_state"),
            "baseline_source_county": baseline_entry.get("baseline_source_county"),
            "baseline_source_percentage": baseline_entry.get("baseline_source_percentage"),
            "applied_min_pixel_distance": float(applied_min_distance),
            "complete": context["cache"]._curve_complete(curve) if not curve.empty else False,
        }

    def _distance_baseline_selection(
        self,
        *,
        zoom: str,
        state: str,
        county: str,
        source_df: pd.DataFrame,
        baseline_key: str,
        error_threshold: float,
    ) -> dict[str, Any] | None:
        baseline_key = self._distance_baseline_key(baseline_key)
        context = self._distance_baseline_context(zoom=zoom, state=state, county=county)
        baseline_entry = self.fps_distance_baseline.get_or_create(
            zoom=zoom,
            state=state,
            county=county,
            source_df=source_df,
            baseline_key=baseline_key,
            baseline_source_df=self._distance_baseline_source_df(baseline_key),
            topology_cache=context["cache"],
            topology_region=context["region"],
            topology_base_dir=context["base_dir"],
        )
        if baseline_entry is None:
            return None
        curve = self._distance_baseline_curve(
            zoom=zoom,
            state=state,
            county=county,
            source_df=source_df,
            context=context,
        )
        chosen = FPSDistanceBaselineCache.select_curve_row(
            curve,
            error_threshold=error_threshold,
            metric_name=FPSDistanceBaselineCache.DEFAULT_THRESHOLD_METRIC,
            baseline_entry=baseline_entry,
        )
        baseline_self_error = self._normalized_against_itself(
            FPSDistanceBaselineCache.metric_block(baseline_entry)[FPSDistanceBaselineCache.DEFAULT_THRESHOLD_METRIC]
        )
        choose_baseline = baseline_self_error <= float(max(0.0, error_threshold))
        if choose_baseline and (
            chosen is None
            or int(baseline_entry.get("n_points", 0)) <= int(chosen.get("n_points", 0))
        ):
            return {
                "percentage": 0,
                "n_points": int(baseline_entry.get("n_points", 0)),
                "raw_error": float(FPSDistanceBaselineCache.metric_block(baseline_entry)[FPSDistanceBaselineCache.DEFAULT_THRESHOLD_METRIC]),
                "normalized_error": float(baseline_self_error),
                "metric_name": FPSDistanceBaselineCache.DEFAULT_THRESHOLD_METRIC,
                "sample_path": str(baseline_entry["sample_path"]),
                "sample_abs_path": self.fps_distance_baseline.sample_path(baseline_entry),
                "baseline_key": str(baseline_entry.get("baseline_key", baseline_key)),
                "baseline_label": str(baseline_entry.get("baseline_label", FPSDistanceBaselineCache.baseline_label(baseline_key))),
                "baseline_n_points": int(baseline_entry.get("n_points", 0)),
                "baseline_equivalent_percentage": float(baseline_entry.get("equivalent_percentage", 0.0)),
                "is_baseline_sample": True,
                "baseline_entry": baseline_entry,
                "context": context,
                "curve": curve,
                "row": None,
            }

        if chosen is None:
            return None
        normalized = FPSDistanceBaselineCache.normalized_block(chosen, baseline_entry)
        sample_path = None if pd.isna(chosen["sample_path"]) else str(chosen["sample_path"])
        return {
            "percentage": int(chosen["percentage"]),
            "n_points": int(chosen["n_points"]),
            "raw_error": float(chosen[FPSDistanceBaselineCache.DEFAULT_THRESHOLD_METRIC]),
            "normalized_error": float(normalized[FPSDistanceBaselineCache.DEFAULT_THRESHOLD_METRIC]),
            "metric_name": FPSDistanceBaselineCache.DEFAULT_THRESHOLD_METRIC,
            "sample_path": sample_path,
            "sample_abs_path": context["cache"].cache_dir / str(sample_path),
            "baseline_key": str(baseline_entry.get("baseline_key", baseline_key)),
            "baseline_label": str(baseline_entry.get("baseline_label", FPSDistanceBaselineCache.baseline_label(baseline_key))),
            "baseline_n_points": int(baseline_entry.get("n_points", 0)),
            "baseline_equivalent_percentage": float(baseline_entry.get("equivalent_percentage", 0.0)),
            "is_baseline_sample": False,
            "baseline_entry": baseline_entry,
            "context": context,
            "curve": curve,
            "row": chosen,
        }

    def _distance_baseline_curve(
        self,
        *,
        zoom: str,
        state: str,
        county: str,
        source_df: pd.DataFrame,
        context: dict[str, Any],
    ) -> pd.DataFrame:
        cache = context["cache"]
        if zoom == "country":
            curve = cache._load_available_curve()
            if curve.empty:
                curve = cache.ensure_precomputed(source_df)
            return curve
        curve = cache._load_available_curve(context["region"])
        if curve.empty:
            if zoom == "state":
                curve = cache.ensure_precomputed(source_df, state=state)
            else:
                curve = cache.ensure_precomputed(source_df, state=state, county=county)
        return curve

    def _distance_baseline_context(self, *, zoom: str, state: str, county: str) -> dict[str, Any]:
        if zoom == "country":
            return {
                "cache": self.country_fps_topology,
                "region": "country",
                "base_dir": self.country_fps_topology.cache_dir / "country",
            }
        if zoom == "state":
            return {
                "cache": self.state_fps_topology,
                "region": state,
                "base_dir": self.state_fps_topology._state_dir(state),
            }
        region = self.county_fps_topology._coerce_region_id(state, county)
        return {
            "cache": self.county_fps_topology,
            "region": region,
            "base_dir": self.county_fps_topology._state_dir(region),
        }

    def _distance_baseline_source_df(self, baseline_key: str) -> pd.DataFrame:
        config = FPSDistanceBaselineCache.baseline_config(baseline_key)
        source_zoom = str(config["source_zoom"])
        source_state = str(config.get("source_state", ""))
        source_county = str(config.get("source_county", ""))
        if source_zoom == "country":
            return self.load_original()
        if source_zoom == "state":
            return self.load_state_reference(source_state)
        county_df = self.load_county_reference(source_state, source_county)
        boundary = self.boundaries.get_county_boundary(source_state, source_county, county_df=county_df)
        return self.boundaries.filter_points_to_boundary(county_df, boundary).reset_index(drop=True)

    @staticmethod
    def _distance_baseline_requested(value: int | str | None) -> bool:
        return FPSDistanceBaselineCache.is_baseline_key(value)

    @staticmethod
    def _distance_baseline_key(value: int | str | None) -> str:
        return FPSDistanceBaselineCache.baseline_key(value)

    @staticmethod
    def _numeric_topology_baseline(value: int | str | None) -> int | None:
        if value is None or FPSDistanceBaselineCache.is_baseline_key(value):
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _metric_block_from_row(row: pd.Series) -> dict[str, float]:
        return {metric: float(row.get(metric, 0.0)) for metric in FPSDistanceBaselineCache.METRICS}

    @staticmethod
    def _normalized_against_itself(value: float) -> float:
        value = float(value)
        return 0.0 if value <= 0.0 else 1.0

    @staticmethod
    def _curve_row_for_percentage(
        curve_df: pd.DataFrame,
        percentage: int,
        *,
        prefer_at_or_below: bool,
    ) -> pd.Series | None:
        if curve_df.empty or "percentage" not in curve_df.columns:
            return None
        requested = int(max(1, min(100, percentage)))
        available = sorted({int(value) for value in curve_df["percentage"].dropna()})
        if not available:
            return None
        if requested in available:
            chosen_percentage = requested
        elif prefer_at_or_below:
            candidates = [value for value in available if value <= requested]
            chosen_percentage = candidates[-1] if candidates else available[0]
        else:
            candidates = [value for value in available if value >= requested]
            chosen_percentage = candidates[0] if candidates else available[-1]
        match = curve_df[curve_df["percentage"] == chosen_percentage]
        if match.empty:
            return None
        return match.iloc[0]

    def _state_topology_cache_enabled(self, state: str) -> bool:
        """Use state FPS topology whenever that state's curve has been saved.

        This keeps uncached states on the lightweight runtime fallback while
        allowing newly precomputed states beyond FL to drive the dashboard.
        Partial curves are accepted so interrupted precompute jobs can still be
        visualized with the checkpoints that already exist.
        """
        if not state or not state.strip():
            return False
        return not self.state_fps_topology._load_available_curve(state.strip().upper()).empty

    def _county_topology_cache_enabled(self, state: str, county: str) -> bool:
        if not state or not state.strip() or not county or not county.strip():
            return False
        region = self.county_fps_topology._coerce_region_id(state.strip().upper(), county.strip())
        return not self.county_fps_topology._load_available_curve(region).empty

    @staticmethod
    def _fps_threshold_status_payload(
        *,
        zoom: str,
        state: str,
        county: str,
        curve: pd.DataFrame,
        complete: bool,
        exact_count: int,
        cache_path: Path,
        expected_row_count: int,
    ) -> dict[str, Any]:
        percentages = []
        if not curve.empty and 'percentage' in curve.columns:
            percentages = sorted({int(value) for value in curve['percentage'].dropna()})
        scope_label = 'country'
        if zoom == 'state':
            scope_label = state or 'state'
        elif zoom == 'county':
            scope_label = f'{county}, {state}' if county and state else 'county'
        return {
            'available': bool(percentages),
            'complete': bool(complete),
            'zoom': zoom,
            'state': state,
            'county': county,
            'scope_label': scope_label,
            'exact_count': int(exact_count),
            'row_count': len(percentages),
            'expected_row_count': int(expected_row_count),
            'percentages': percentages,
            'cache_path': str(cache_path),
        }

    @staticmethod
    def _scope_and_region(zoom: str, state: str, county: str) -> tuple[str, str | None]:
        if zoom == 'country':
            return 'country', None
        if zoom == 'state':
            return 'state', state
        return 'county', f'{state}_{county}'

    def _serialize_analysis(
        self,
        analysis_property: str,
        comparison: Any,
    ) -> dict[str, Any]:
        if comparison is None:
            return {'available': False}

        if analysis_property == 'statistical':
            return {
                'available': True,
                'original': comparison['original'],
                'reduced': comparison['sampled'] if 'sampled' in comparison else comparison['reduced'],
                'error': comparison['error'],
            }

        if analysis_property == 'density':
            return {
                'available': True,
                'original_density': np.asarray(comparison['original_density']).tolist(),
                'reduced_density': np.asarray(comparison['reduced_density']).tolist(),
                'difference_density': np.asarray(comparison['difference_density']).tolist(),
                'lat_edges': np.asarray(comparison['lat_edges']).tolist(),
                'lng_edges': np.asarray(comparison['lng_edges']).tolist(),
                'metrics': comparison['metrics'],
            }

        if analysis_property == 'topological':
            distances = comparison.get('distances') if isinstance(comparison, dict) else None
            current_distances = comparison.get('current_distances') if isinstance(comparison, dict) else None
            baseline_distances = comparison.get('baseline_distances') if isinstance(comparison, dict) else None
            normalized_distances = comparison.get('normalized_distances') if isinstance(comparison, dict) else None
            return {
                'available': True,
                'original': {
                    'h0': np.asarray(comparison['original']['h0']).tolist(),
                    'h1': np.asarray(comparison['original']['h1']).tolist(),
                },
                'reduced': {
                    'h0': np.asarray(comparison['reduced']['h0']).tolist(),
                    'h1': np.asarray(comparison['reduced']['h1']).tolist(),
                },
                'distances': current_distances or distances,
                'baseline_distances': baseline_distances,
                'current_distances': current_distances or distances,
                'normalized_distances': normalized_distances,
                'baseline_percentage': comparison.get('baseline_percentage') if isinstance(comparison, dict) else None,
                'baseline_key': comparison.get('baseline_key') if isinstance(comparison, dict) else None,
                'baseline_label': comparison.get('baseline_label') if isinstance(comparison, dict) else None,
                'baseline_n_points': comparison.get('baseline_n_points') if isinstance(comparison, dict) else None,
                'baseline_equivalent_percentage': comparison.get('baseline_equivalent_percentage') if isinstance(comparison, dict) else None,
                'country_min_distance': comparison.get('country_min_distance') if isinstance(comparison, dict) else None,
                'country_min_pixel_distance': comparison.get('country_min_pixel_distance') if isinstance(comparison, dict) else None,
                'source_min_pixel_distance': comparison.get('source_min_pixel_distance') if isinstance(comparison, dict) else None,
                'baseline_source_zoom': comparison.get('baseline_source_zoom') if isinstance(comparison, dict) else None,
                'baseline_source_state': comparison.get('baseline_source_state') if isinstance(comparison, dict) else None,
                'baseline_source_county': comparison.get('baseline_source_county') if isinstance(comparison, dict) else None,
                'baseline_source_percentage': comparison.get('baseline_source_percentage') if isinstance(comparison, dict) else None,
                'applied_min_pixel_distance': comparison.get('applied_min_pixel_distance') if isinstance(comparison, dict) else None,
                'current_percentage': comparison.get('current_percentage') if isinstance(comparison, dict) else None,
                'current_label': comparison.get('current_label') if isinstance(comparison, dict) else None,
                'complete': comparison.get('complete') if isinstance(comparison, dict) else None,
            }

        raise ValueError(f'Unsupported analysis property: {analysis_property}')


