"""Lazy on-demand cache for OpenLayers country-level pixel occupancy samples."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from adaptive_visualization.core.sample_schema import trim_sample_columns
from adaptive_visualization.vendor.map_viz.data.loader import AccidentDataLoader
from adaptive_visualization.vendor.map_viz.sampling.pixel_sampler import PixelSampler


class OpenLayersCountryPixelCache:
    """Materialize country-level pixel samples for live OpenLayers viewports."""

    CACHE_VERSION = 'openlayers_country_pixel_dynamic_v1'
    CACHE_DIRNAME = 'pixel_openlayers_country_cache'

    def __init__(self, data_dir: str | Path, loader: AccidentDataLoader):
        self.data_dir = Path(data_dir)
        self.loader = loader
        self.cache_dir = self.data_dir / self.CACHE_DIRNAME
        self.profiles_dir = self.cache_dir / 'profiles'
        self.metadata_path = self.cache_dir / 'metadata.json'
        self._memory_cache: dict[str, pd.DataFrame] = {}
        self._metadata: dict[str, Any] | None = None
        self.sampler = PixelSampler()

    def get_or_create(self, reference_df: pd.DataFrame, viewport_spec: dict[str, Any] | None) -> pd.DataFrame:
        viewport_spec = self._normalize_viewport_spec(viewport_spec)
        profile_id = self._profile_id(viewport_spec)
        if not profile_id:
            return reference_df.reset_index(drop=True)

        cached = self._memory_cache.get(profile_id)
        if cached is not None:
            return cached

        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        metadata = self._load_metadata()
        cache_path = self.profiles_dir / f'{profile_id}.csv'

        if cache_path.exists() and self._is_cache_valid(metadata, profile_id, reference_df, viewport_spec, cache_path):
            print(f'[pixel-country-cache] loading cached country sample for {profile_id} from disk')
            cached_df = self.loader.load(str(cache_path))
            self._memory_cache[profile_id] = cached_df
            return cached_df

        print(
            f'[pixel-country-cache] computing country sample for {profile_id} '
            f'from {len(reference_df):,} reference points'
        )
        sampled_df = self._compute_country_sample(reference_df, viewport_spec)
        sampled_df = trim_sample_columns(sampled_df)
        sampled_df.to_csv(cache_path, index=False)

        metadata.setdefault('profiles', {})[profile_id] = {
            'path': cache_path.relative_to(self.data_dir).as_posix(),
            'source_records': int(len(reference_df)),
            'cached_records': int(len(sampled_df)),
            'viewport': self._normalize_viewport_spec(viewport_spec),
            'created_at': datetime.now(timezone.utc).isoformat(),
        }
        self._save_metadata(metadata)
        self._memory_cache[profile_id] = sampled_df
        return sampled_df

    def _compute_country_sample(self, reference_df: pd.DataFrame, viewport_spec: dict[str, Any]) -> pd.DataFrame:
        if reference_df.empty:
            return reference_df.reset_index(drop=True)

        normalized = self._normalize_viewport_spec(viewport_spec)
        coords = reference_df[['Start_Lat', 'Start_Lng']].to_numpy(dtype=float)
        selected_idx = self.sampler.sample_country_openlayers(
            coords,
            viewport_width_px=normalized['width'],
            viewport_height_px=normalized['height'],
            padding_px=tuple(normalized['padding']),
            n_keep=len(reference_df),
        )
        return reference_df.iloc[selected_idx].reset_index(drop=True)

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
            merged['profiles'] = dict(existing.get('profiles', {}))
            merged['profiles'].update(metadata.get('profiles', {}))
            for key, value in metadata.items():
                if key == 'profiles':
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
        return isinstance(metadata.get('profiles'), dict)

    def _is_cache_valid(
        self,
        metadata: dict[str, Any],
        profile_id: str,
        reference_df: pd.DataFrame,
        viewport_spec: dict[str, Any],
        cache_path: Path,
    ) -> bool:
        if not self._metadata_matches(metadata):
            return False

        profile_info = metadata.get('profiles', {}).get(profile_id)
        if not isinstance(profile_info, dict):
            return False

        if profile_info.get('path') != cache_path.relative_to(self.data_dir).as_posix():
            return False

        if int(profile_info.get('source_records', -1)) != int(len(reference_df)):
            return False

        if profile_info.get('viewport') != self._normalize_viewport_spec(viewport_spec):
            return False

        if not cache_path.exists():
            return False

        return True

    def _default_metadata(self) -> dict[str, Any]:
        metadata = self._metadata_signature()
        metadata['profiles'] = {}
        return metadata

    def _metadata_signature(self) -> dict[str, Any]:
        return {
            'cache_version': self.CACHE_VERSION,
            'sampling_mode': 'pixel_occupancy_openlayers_country_dynamic_fit_extent',
            'marker_radius_px': float(self.sampler.COUNTRY_MARKER_RADIUS_PX),
            'marker_stroke_width_px': float(self.sampler.COUNTRY_MARKER_STROKE_WIDTH_PX),
            'min_center_distance_px': float(self.sampler._country_min_center_distance_px),
            'country_lat_bounds': [float(v) for v in self.sampler.COUNTRY_LAT_BOUNDS],
            'country_lon_bounds': [float(v) for v in self.sampler.COUNTRY_LON_BOUNDS],
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

    @staticmethod
    def _normalize_viewport_spec(viewport_spec: dict[str, Any] | None) -> dict[str, Any]:
        if viewport_spec is None:
            viewport_spec = {
                'width': PixelSampler.COUNTRY_VIEWPORT_WIDTH_PX,
                'height': PixelSampler.COUNTRY_VIEWPORT_HEIGHT_PX,
                'padding': [28, 28, 28, 28],
            }
        width = max(int(round(float(viewport_spec.get('width', 0) or 0))), 1)
        height = max(int(round(float(viewport_spec.get('height', 0) or 0))), 1)
        padding = viewport_spec.get('padding', [28, 28, 28, 28])
        if len(padding) != 4:
            padding = [28, 28, 28, 28]
        normalized_padding = [max(int(round(float(value or 0))), 0) for value in padding]
        return {
            'width': width,
            'height': height,
            'padding': normalized_padding,
        }

    @classmethod
    def _profile_id(cls, viewport_spec: dict[str, Any]) -> str:
        normalized = cls._normalize_viewport_spec(viewport_spec)
        padding = '-'.join(str(value) for value in normalized['padding'])
        return f"w{normalized['width']}_h{normalized['height']}_p{padding}"


