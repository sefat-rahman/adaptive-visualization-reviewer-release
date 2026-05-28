"""Lazy on-demand cache for OpenLayers county-level pixel occupancy samples."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from adaptive_visualization.core.boundaries import BoundaryService
from adaptive_visualization.paths import CACHE_DIR
from adaptive_visualization.core.sample_schema import trim_sample_columns
from adaptive_visualization.vendor.map_viz.data.loader import AccidentDataLoader
from adaptive_visualization.vendor.map_viz.sampling.pixel_sampler import PixelSampler


class OpenLayersCountyPixelCache:
    """Materialize county-level pixel samples lazily and reuse them across runs."""

    CACHE_VERSION = 'openlayers_county_pixel_exact_circle_v2'
    CACHE_DIRNAME = 'pixel_openlayers_county_cache'

    def __init__(self, data_dir: str | Path, loader: AccidentDataLoader):
        self.data_dir = Path(data_dir)
        self.loader = loader
        self.cache_dir = self.data_dir / self.CACHE_DIRNAME
        self.counties_dir = self.cache_dir / 'counties'
        self.metadata_path = self.cache_dir / 'metadata.json'
        self._memory_cache: dict[str, pd.DataFrame] = {}
        self._metadata: dict[str, Any] | None = None

        self.boundaries = BoundaryService(CACHE_DIR)
        self.sampler = PixelSampler()

    def get_or_create(
        self,
        state: str,
        county: str,
        reference_df: pd.DataFrame,
        viewport_spec: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        state = (state or '').strip().upper()
        county = (county or '').strip()
        if not state or not county:
            return reference_df.reset_index(drop=True)

        normalized = self._normalize_viewport_spec(viewport_spec)
        profile_id = self._profile_id(normalized)
        cache_key = f'{profile_id}__{state}__{county}'

        cached = self._memory_cache.get(cache_key)
        if cached is not None:
            return cached

        self.counties_dir.mkdir(parents=True, exist_ok=True)
        metadata = self._load_metadata()
        cache_path = self._cache_path(profile_id, state, county)

        if cache_path.exists() and self._is_cache_valid(metadata, cache_key, reference_df, normalized, cache_path):
            print(f'[pixel-county-cache] loading cached county sample for {state}/{county} ({profile_id}) from disk')
            cached_df = self.loader.load(str(cache_path))
            self._memory_cache[cache_key] = cached_df
            return cached_df

        print(
            f'[pixel-county-cache] computing county sample for {state}/{county} ({profile_id}) '
            f'from {len(reference_df):,} reference points'
        )
        boundary = self.boundaries.get_county_boundary(state, county, county_df=reference_df)
        bounded_df = self.boundaries.filter_points_to_boundary(reference_df, boundary).reset_index(drop=True)
        sampled_df = self._compute_county_sample(bounded_df, boundary, normalized)
        sampled_df = trim_sample_columns(sampled_df)

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        sampled_df.to_csv(cache_path, index=False)

        metadata.setdefault('counties', {})[cache_key] = {
            'path': cache_path.relative_to(self.data_dir).as_posix(),
            'state': state,
            'county': county,
            'source_records': int(len(reference_df)),
            'bounded_records': int(len(bounded_df)),
            'cached_records': int(len(sampled_df)),
            'viewport': normalized,
            'created_at': datetime.now(timezone.utc).isoformat(),
        }
        self._save_metadata(metadata)
        self._memory_cache[cache_key] = sampled_df
        return sampled_df

    def _compute_county_sample(
        self,
        bounded_df: pd.DataFrame,
        boundary: dict[str, Any] | None,
        viewport_spec: dict[str, Any],
    ) -> pd.DataFrame:
        if bounded_df.empty:
            return bounded_df.reset_index(drop=True)

        coords = bounded_df[['Start_Lat', 'Start_Lng']].to_numpy(dtype=float)
        bounds = self.boundaries.get_feature_bounds(boundary)
        if bounds is not None:
            lon_min, lat_min, lon_max, lat_max = bounds
            selected_idx = self.sampler.sample_county_openlayers(
                coords,
                county_lat_range=[lat_min, lat_max],
                county_lon_range=[lon_min, lon_max],
                n_keep=len(bounded_df),
                viewport_width_px=viewport_spec['width'],
                viewport_height_px=viewport_spec['height'],
                padding_px=tuple(viewport_spec['padding']),
            )
        else:
            lat_min = float(bounded_df['Start_Lat'].min())
            lat_max = float(bounded_df['Start_Lat'].max())
            lon_min = float(bounded_df['Start_Lng'].min())
            lon_max = float(bounded_df['Start_Lng'].max())
            selected_idx = self.sampler.sample_county_openlayers(
                coords,
                county_lat_range=[lat_min, lat_max],
                county_lon_range=[lon_min, lon_max],
                n_keep=len(bounded_df),
                viewport_width_px=viewport_spec['width'],
                viewport_height_px=viewport_spec['height'],
                padding_px=tuple(viewport_spec['padding']),
            )
        return bounded_df.iloc[selected_idx].reset_index(drop=True)

    def _load_metadata(self) -> dict[str, Any]:
        if self._metadata is not None:
            return self._metadata

        metadata: dict[str, Any] | None = None
        if self.metadata_path.exists():
            try:
                metadata = json.loads(self.metadata_path.read_text(encoding='utf-8'))
            except json.JSONDecodeError:
                metadata = None

        if metadata is None or not self._metadata_matches(metadata):
            metadata = self._default_metadata()
            self._memory_cache.clear()

        self._metadata = metadata
        return self._metadata

    def _save_metadata(self, metadata: dict[str, Any]) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        existing: dict[str, Any] | None = None
        if self.metadata_path.exists():
            try:
                existing = json.loads(self.metadata_path.read_text(encoding='utf-8'))
            except json.JSONDecodeError:
                existing = None

        if existing is not None and self._metadata_matches(existing):
            merged = dict(existing)
            merged['counties'] = dict(existing.get('counties', {}))
            merged['counties'].update(metadata.get('counties', {}))
            for key, value in metadata.items():
                if key == 'counties':
                    continue
                merged[key] = value
            metadata = merged

        self.metadata_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        self._metadata = metadata

    def _metadata_matches(self, metadata: dict[str, Any]) -> bool:
        expected = self._metadata_signature()
        for key, value in expected.items():
            if metadata.get(key) != value:
                return False
        return isinstance(metadata.get('counties'), dict)

    def _is_cache_valid(
        self,
        metadata: dict[str, Any],
        cache_key: str,
        reference_df: pd.DataFrame,
        viewport_spec: dict[str, Any],
        cache_path: Path,
    ) -> bool:
        if not self._metadata_matches(metadata):
            return False

        county_info = metadata.get('counties', {}).get(cache_key)
        if not isinstance(county_info, dict):
            return False

        if county_info.get('path') != cache_path.relative_to(self.data_dir).as_posix():
            return False

        if int(county_info.get('source_records', -1)) != int(len(reference_df)):
            return False

        if county_info.get('viewport') != viewport_spec:
            return False

        if not cache_path.exists():
            return False

        return True

    def _default_metadata(self) -> dict[str, Any]:
        metadata = self._metadata_signature()
        metadata['counties'] = {}
        return metadata

    def _metadata_signature(self) -> dict[str, Any]:
        return {
            'cache_version': self.CACHE_VERSION,
            'sampling_mode': 'pixel_occupancy_openlayers_county_exact_circle_fit_extent_dynamic',
            'marker_radius_px': float(self.sampler.COUNTY_MARKER_RADIUS_PX),
            'marker_stroke_width_px': float(self.sampler.COUNTY_MARKER_STROKE_WIDTH_PX),
            'min_center_distance_px': float(self.sampler._county_min_center_distance_px),
            'boundary_filtering': 'points_clipped_to_county_boundary',
            'boundary_source': 'county_boundary_cache_or_atlas',
            'source_original': self._source_signature(),
        }

    def _source_signature(self) -> dict[str, Any]:
        source_path = self.data_dir / 'original.csv'
        if not source_path.exists():
            return {
                'path': 'original.csv',
                'size_bytes': None,
                'mtime_ns': None,
            }

        stat = source_path.stat()
        return {
            'path': 'original.csv',
            'size_bytes': int(stat.st_size),
            'mtime_ns': int(stat.st_mtime_ns),
        }

    def _cache_path(self, profile_id: str, state: str, county: str) -> Path:
        safe_county = self._sanitize(county)
        return self.counties_dir / profile_id / state / f'{safe_county}.csv'

    def _normalize_viewport_spec(self, viewport_spec: dict[str, Any] | None) -> dict[str, Any]:
        if viewport_spec is None:
            viewport_spec = {
                'width': self.sampler.COUNTY_VIEWPORT_WIDTH_PX,
                'height': self.sampler.COUNTY_VIEWPORT_HEIGHT_PX,
                'padding': list(self.sampler.COUNTY_VIEWPORT_PADDING_PX),
            }
        width = max(int(round(float(viewport_spec.get('width', 0) or 0))), 1)
        height = max(int(round(float(viewport_spec.get('height', 0) or 0))), 1)
        padding = viewport_spec.get('padding', list(self.sampler.COUNTY_VIEWPORT_PADDING_PX))
        if len(padding) != 4:
            padding = list(self.sampler.COUNTY_VIEWPORT_PADDING_PX)
        normalized_padding = [max(int(round(float(value or 0))), 0) for value in padding]
        return {
            'width': width,
            'height': height,
            'padding': normalized_padding,
        }

    @staticmethod
    def _sanitize(value: str) -> str:
        safe = value.replace('\\', '_').replace('/', '_').replace(':', '_')
        return safe.replace(' ', '_')

    @staticmethod
    def _profile_id(viewport_spec: dict[str, Any]) -> str:
        padding = '-'.join(str(value) for value in viewport_spec['padding'])
        return f"w{viewport_spec['width']}_h{viewport_spec['height']}_p{padding}"


