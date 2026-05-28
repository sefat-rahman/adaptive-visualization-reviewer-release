"""Distance-spaced FPS baseline samples for the OpenLayers app."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from gudhi.hera import bottleneck_distance, wasserstein_distance

from adaptive_visualization.core.boundaries import BoundaryService
from adaptive_visualization.core.fps_order_cache import FPSOrderCache
from adaptive_visualization.paths import CACHE_DIR
from adaptive_visualization.core.sample_schema import trim_sample_columns
from adaptive_visualization.core.state_fps_topology_cache import StateFPSTopologyCache
from adaptive_visualization.vendor.map_viz.sampling.pixel_sampler import PixelSampler


class FPSDistanceBaselineCache:
    """Build FPS-order samples with a shared source spacing as a minimum distance."""

    CACHE_VERSION = "fps_distance_baseline_multi_pixel_v1"
    CACHE_DIRNAME = "fps_distance_baseline_cache"
    COUNTRY_1PCT_KEY = "country_1pct_distance"
    COUNTY_5PCT_KEY = "county_5pct_distance"
    COUNTY_10PCT_KEY = "county_10pct_distance"
    COUNTY_20PCT_KEY = "county_20pct_distance"
    DEFAULT_BASELINE_KEY = COUNTRY_1PCT_KEY
    BASELINE_CONFIGS = {
        COUNTRY_1PCT_KEY: {
            "label": "Country 1% distance",
            "source_zoom": "country",
            "source_state": "",
            "source_county": "",
            "source_percentage": 1,
        },
        COUNTY_5PCT_KEY: {
            "label": "County 5% distance",
            "source_zoom": "county",
            "source_state": "NC",
            "source_county": "Mecklenburg",
            "source_percentage": 5,
        },
        COUNTY_10PCT_KEY: {
            "label": "County 10% distance",
            "source_zoom": "county",
            "source_state": "NC",
            "source_county": "Mecklenburg",
            "source_percentage": 10,
        },
        COUNTY_20PCT_KEY: {
            "label": "County 20% distance",
            "source_zoom": "county",
            "source_state": "NC",
            "source_county": "Mecklenburg",
            "source_percentage": 20,
        },
    }
    COUNTRY_PADDING_PX = (28.0, 28.0, 28.0, 28.0)
    METRICS = StateFPSTopologyCache.METRICS
    DEFAULT_THRESHOLD_METRIC = StateFPSTopologyCache.DEFAULT_THRESHOLD_METRIC

    def __init__(self, data_dir: str | Path, fps_order_cache: FPSOrderCache):
        self.data_dir = Path(data_dir)
        self.cache_dir = self.data_dir / self.CACHE_DIRNAME
        self.fps_order_cache = fps_order_cache
        self.shared_order_dir = self.data_dir / FPSOrderCache.CACHE_DIRNAME
        self.boundaries = BoundaryService(CACHE_DIR)
        self.pixel_sampler = PixelSampler()
        self._safe_distance_info: dict[str, dict[str, Any]] = {}
        self._entries: dict[str, dict[str, Any]] = {}

    @classmethod
    def is_baseline_key(cls, value: Any) -> bool:
        return str(value or "").strip().lower() in cls.BASELINE_CONFIGS

    @classmethod
    def baseline_key(cls, value: Any) -> str:
        key = str(value or "").strip().lower()
        if key in cls.BASELINE_CONFIGS:
            return key
        return cls.DEFAULT_BASELINE_KEY

    @classmethod
    def baseline_label(cls, value: Any) -> str:
        return str(cls.BASELINE_CONFIGS[cls.baseline_key(value)]["label"])

    @classmethod
    def baseline_config(cls, value: Any) -> dict[str, Any]:
        return dict(cls.BASELINE_CONFIGS[cls.baseline_key(value)])

    def get_or_create(
        self,
        *,
        zoom: str,
        state: str,
        county: str,
        source_df: pd.DataFrame,
        baseline_key: str,
        baseline_source_df: pd.DataFrame,
        topology_cache: Any,
        topology_region: str,
        topology_base_dir: Path,
    ) -> dict[str, Any] | None:
        valid_df = source_df.dropna(subset=["Start_Lat", "Start_Lng"]).reset_index(drop=True)
        baseline_source_valid = baseline_source_df.dropna(subset=["Start_Lat", "Start_Lng"]).reset_index(drop=True)
        if valid_df.empty or baseline_source_valid.empty:
            return None

        baseline_key = self.baseline_key(baseline_key)
        baseline_config = self.baseline_config(baseline_key)
        safe_info = self.source_safe_distance(baseline_key, baseline_source_valid)
        safe_distance = float(safe_info["min_distance"])
        applied_distance = self._applied_min_pixel_distance(zoom, safe_distance)
        scope_key = f"{baseline_key}::{self._scope_key(zoom=zoom, state=state, county=county)}"
        base_dir = self._scope_dir(zoom=zoom, state=state, county=county)
        entry_path = base_dir / f"{baseline_key}.json"
        source_signature = self._frame_signature(valid_df)

        cached = self._entries.get(scope_key)
        if cached is not None and self._entry_valid(
            cached,
            source_signature=source_signature,
            baseline_source_signature=safe_info["baseline_source_signature"],
            safe_distance=safe_distance,
            applied_distance=applied_distance,
        ):
            return cached

        cached = self._load_entry(entry_path)
        if cached is not None and self._entry_valid(
            cached,
            source_signature=source_signature,
            baseline_source_signature=safe_info["baseline_source_signature"],
            safe_distance=safe_distance,
            applied_distance=applied_distance,
        ):
            self._entries[scope_key] = cached
            return cached

        base_dir.mkdir(parents=True, exist_ok=True)
        samples_dir = base_dir / "samples"
        samples_dir.mkdir(parents=True, exist_ok=True)

        order = self._get_or_compute_order(zoom=zoom, state=state, county=county, source_df=valid_df)
        coords = valid_df[["Start_Lat", "Start_Lng"]].to_numpy(dtype=float)
        if self._scope_matches_source(zoom=zoom, state=state, county=county, config=baseline_config):
            n_keep = StateFPSTopologyCache._n_keep(len(valid_df), int(baseline_config["source_percentage"]))
            selected_indices = order[:n_keep].astype(np.int32, copy=False)
        else:
            pixel_xy = self._pixel_coordinates(zoom=zoom, state=state, county=county, coords=coords)
            selected_indices = self._distance_filtered_order_indices(pixel_xy, order, applied_distance)
        if selected_indices.size == 0 and len(valid_df) > 0:
            selected_indices = np.asarray([int(order[0])], dtype=np.int32)

        sample_path = samples_dir / f"{baseline_key}.csv"
        sampled_df = trim_sample_columns(valid_df.iloc[selected_indices].reset_index(drop=True))
        sampled_df.to_csv(sample_path, index=False)

        reduced_h0_path = base_dir / f"{baseline_key}_h0.csv"
        reduced_h1_path = base_dir / f"{baseline_key}_h1.csv"
        metrics = self._reuse_country_1pct_metrics(
            baseline_key=baseline_key,
            zoom=zoom,
            order=order,
            selected_indices=selected_indices,
            topology_cache=topology_cache,
            topology_base_dir=topology_base_dir,
            reduced_h0_path=reduced_h0_path,
            reduced_h1_path=reduced_h1_path,
        )
        if metrics is None:
            original = self._load_or_compute_original(
                source_df=valid_df,
                topology_cache=topology_cache,
                topology_base_dir=topology_base_dir,
                baseline_dir=base_dir,
            )
            if sampled_df.empty:
                reduced = {"h0": np.empty((0, 2), dtype=float), "h1": np.empty((0, 2), dtype=float)}
            else:
                reduced = topology_cache._compute_diagrams(sampled_df[["Start_Lat", "Start_Lng"]].to_numpy(dtype=float))
            topology_cache._save_diagram_array(reduced_h0_path, reduced["h0"])
            topology_cache._save_diagram_array(reduced_h1_path, reduced["h1"])
            metrics = self._distance_block(original, reduced)

        entry = {
            "cache_version": self.CACHE_VERSION,
            "baseline_key": baseline_key,
            "baseline_label": baseline_config["label"],
            "zoom": zoom,
            "state": state,
            "county": county,
            "topology_region": topology_region,
            "source_signature": source_signature,
            "baseline_source_signature": safe_info["baseline_source_signature"],
            "baseline_source_zoom": baseline_config["source_zoom"],
            "baseline_source_state": baseline_config["source_state"],
            "baseline_source_county": baseline_config["source_county"],
            "baseline_source_percentage": int(baseline_config["source_percentage"]),
            "baseline_source_min_pixel_distance": safe_distance,
            "source_min_pixel_distance": safe_distance,
            "country_min_pixel_distance": safe_distance if baseline_key == self.COUNTRY_1PCT_KEY else None,
            "country_min_distance": safe_distance if baseline_key == self.COUNTRY_1PCT_KEY else None,
            "applied_min_pixel_distance": applied_distance,
            "coordinate_space": "OpenLayers pixel",
            "n_source_points": int(len(valid_df)),
            "n_points": int(len(sampled_df)),
            "equivalent_percentage": self._equivalent_percentage(len(sampled_df), len(valid_df)),
            "sample_path": str(sample_path.relative_to(self.cache_dir)).replace("\\", "/"),
            "reduced_h0_path": str(reduced_h0_path.relative_to(self.cache_dir)).replace("\\", "/"),
            "reduced_h1_path": str(reduced_h1_path.relative_to(self.cache_dir)).replace("\\", "/"),
            "metrics": {metric: float(metrics.get(metric, 0.0)) for metric in self.METRICS},
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        entry_path.write_text(json.dumps(entry, indent=2), encoding="utf-8")
        self._entries[scope_key] = entry
        return entry

    def source_safe_distance(self, baseline_key: str, baseline_source_df: pd.DataFrame) -> dict[str, Any]:
        baseline_key = self.baseline_key(baseline_key)
        config = self.baseline_config(baseline_key)
        source_valid = baseline_source_df.dropna(subset=["Start_Lat", "Start_Lng"]).reset_index(drop=True)
        signature = self._frame_signature(source_valid)
        cached_memory = self._safe_distance_info.get(baseline_key)
        if cached_memory is not None and cached_memory.get("baseline_source_signature") == signature:
            return cached_memory

        safe_distance_path = self.cache_dir / f"{baseline_key}.json"
        cached = self._load_entry(safe_distance_path)
        if (
            cached is not None
            and cached.get("cache_version") == self.CACHE_VERSION
            and cached.get("baseline_key") == baseline_key
            and cached.get("baseline_source_signature") == signature
            and float(cached.get("min_distance", 0.0)) >= 0.0
        ):
            self._safe_distance_info[baseline_key] = cached
            return cached

        order = self._get_or_compute_order(
            zoom=str(config["source_zoom"]),
            state=str(config["source_state"]),
            county=str(config["source_county"]),
            source_df=source_valid,
        )
        source_percentage = int(config["source_percentage"])
        n_keep = StateFPSTopologyCache._n_keep(len(source_valid), source_percentage)
        selected_indices = order[:n_keep]
        coords = source_valid.iloc[selected_indices][["Start_Lat", "Start_Lng"]].to_numpy(dtype=float)
        pixel_xy = self._pixel_coordinates(
            zoom=str(config["source_zoom"]),
            state=str(config["source_state"]),
            county=str(config["source_county"]),
            coords=coords,
        )
        min_distance = self._minimum_pairwise_distance(pixel_xy)
        info = {
            "cache_version": self.CACHE_VERSION,
            "baseline_key": baseline_key,
            "baseline_label": config["label"],
            "baseline_source_signature": signature,
            "baseline_source_zoom": config["source_zoom"],
            "baseline_source_state": config["source_state"],
            "baseline_source_county": config["source_county"],
            "baseline_source_percentage": source_percentage,
            "baseline_source_n_points": int(len(source_valid)),
            "baseline_source_sample_n_points": int(n_keep),
            "min_distance": float(min_distance),
            "coordinate_space": "OpenLayers pixel",
            "viewport": self._viewport_info(
                str(config["source_zoom"]),
                str(config["source_state"]),
                str(config["source_county"]),
            ),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        safe_distance_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
        self._safe_distance_info[baseline_key] = info
        return info

    @classmethod
    def select_curve_row(
        cls,
        curve_df: pd.DataFrame,
        *,
        error_threshold: float,
        metric_name: str,
        baseline_entry: dict[str, Any],
    ) -> pd.Series | None:
        if curve_df.empty:
            return None
        metric_name = cls._metric_name(metric_name)
        baseline_metrics = cls.metric_block(baseline_entry)
        threshold = float(max(0.0, error_threshold))
        ordered = curve_df.sort_values("percentage", ascending=True).reset_index(drop=True)
        ordered = ordered[ordered.apply(lambda row: cls._row_has_metrics(row), axis=1)]
        min_points = int(max(1, baseline_entry.get("n_points", 1)))
        candidates = ordered[ordered["n_points"] >= min_points]
        if candidates.empty:
            candidates = ordered
        match = candidates[
            candidates.apply(
                lambda row: cls._normalized(float(row.get(metric_name, 0.0)), baseline_metrics[metric_name]) <= threshold,
                axis=1,
            )
        ]
        if not match.empty:
            return match.iloc[0]
        if candidates.empty:
            return None
        return candidates.iloc[-1]

    @classmethod
    def metric_block(cls, baseline_entry: dict[str, Any]) -> dict[str, float]:
        metrics = baseline_entry.get("metrics", {})
        return {metric: float(metrics.get(metric, 0.0)) for metric in cls.METRICS}

    @classmethod
    def normalized_block(cls, row: pd.Series, baseline_entry: dict[str, Any]) -> dict[str, float]:
        baseline_metrics = cls.metric_block(baseline_entry)
        return {
            metric: cls._normalized(float(row.get(metric, 0.0)), baseline_metrics[metric])
            for metric in cls.METRICS
        }

    def sample_path(self, baseline_entry: dict[str, Any]) -> Path:
        return self.cache_dir / str(baseline_entry["sample_path"])

    def _load_or_compute_original(
        self,
        *,
        source_df: pd.DataFrame,
        topology_cache: Any,
        topology_base_dir: Path,
        baseline_dir: Path,
    ) -> dict[str, np.ndarray]:
        topology_h0 = topology_base_dir / "original_h0.csv"
        topology_h1 = topology_base_dir / "original_h1.csv"
        if topology_h0.exists() and topology_h1.exists():
            return {
                "h0": topology_cache._load_diagram_array_np(topology_h0),
                "h1": topology_cache._load_diagram_array_np(topology_h1),
            }

        baseline_h0 = baseline_dir / "original_h0.csv"
        baseline_h1 = baseline_dir / "original_h1.csv"
        if baseline_h0.exists() and baseline_h1.exists():
            return {
                "h0": topology_cache._load_diagram_array_np(baseline_h0),
                "h1": topology_cache._load_diagram_array_np(baseline_h1),
            }

        original = topology_cache._compute_diagrams(source_df[["Start_Lat", "Start_Lng"]].to_numpy(dtype=float))
        topology_cache._save_diagram_array(baseline_h0, original["h0"])
        topology_cache._save_diagram_array(baseline_h1, original["h1"])
        return original

    def _reuse_country_1pct_metrics(
        self,
        *,
        baseline_key: str,
        zoom: str,
        order: np.ndarray,
        selected_indices: np.ndarray,
        topology_cache: Any,
        topology_base_dir: Path,
        reduced_h0_path: Path,
        reduced_h1_path: Path,
    ) -> dict[str, float] | None:
        if baseline_key != self.COUNTRY_1PCT_KEY:
            return None
        if zoom != "country":
            return None

        n_keep = StateFPSTopologyCache._n_keep(
            len(order),
            int(self.BASELINE_CONFIGS[self.COUNTRY_1PCT_KEY]["source_percentage"]),
        )
        if selected_indices.shape != (n_keep,) or not np.array_equal(selected_indices, order[:n_keep]):
            return None

        curve = topology_cache._load_available_curve()
        if curve.empty:
            return None
        source_percentage = int(self.BASELINE_CONFIGS[self.COUNTRY_1PCT_KEY]["source_percentage"])
        match = curve[curve["percentage"] == source_percentage]
        if match.empty:
            return None
        row = match.iloc[0]
        if not all(metric in row and pd.notna(row[metric]) for metric in self.METRICS):
            return None

        source_h0_path = topology_base_dir / f"pct_{source_percentage:02d}_h0.csv"
        source_h1_path = topology_base_dir / f"pct_{source_percentage:02d}_h1.csv"
        if source_h0_path.exists() and not reduced_h0_path.exists():
            topology_cache._save_diagram_array(reduced_h0_path, topology_cache._load_diagram_array_np(source_h0_path))
        if source_h1_path.exists() and not reduced_h1_path.exists():
            topology_cache._save_diagram_array(reduced_h1_path, topology_cache._load_diagram_array_np(source_h1_path))

        return {metric: float(row[metric]) for metric in self.METRICS}

    def _get_or_compute_order(
        self,
        *,
        zoom: str,
        state: str,
        county: str,
        source_df: pd.DataFrame,
    ) -> np.ndarray:
        shared_path = self._shared_order_path(zoom=zoom, state=state, county=county)
        if shared_path.exists():
            try:
                order = np.load(shared_path)
                if order.shape == (len(source_df),):
                    return order.astype(np.int32, copy=False)
            except Exception:
                pass

        local_path = self._scope_dir(zoom=zoom, state=state, county=county) / "order.npy"
        if local_path.exists():
            try:
                order = np.load(local_path)
                if order.shape == (len(source_df),):
                    return order.astype(np.int32, copy=False)
            except Exception:
                pass

        order = StateFPSTopologyCache._compute_order(source_df)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(local_path, order)
        return order

    def _pixel_coordinates(self, *, zoom: str, state: str, county: str, coords: np.ndarray) -> np.ndarray:
        if zoom == "country":
            return self.pixel_sampler._to_fitted_state_pixel_coordinates(
                points=coords,
                lat_range=[float(PixelSampler.COUNTRY_LAT_BOUNDS[0]), float(PixelSampler.COUNTRY_LAT_BOUNDS[1])],
                lon_range=[float(PixelSampler.COUNTRY_LON_BOUNDS[0]), float(PixelSampler.COUNTRY_LON_BOUNDS[1])],
                viewport_width_px=float(PixelSampler.COUNTRY_VIEWPORT_WIDTH_PX),
                viewport_height_px=float(PixelSampler.COUNTRY_VIEWPORT_HEIGHT_PX),
                padding_px=self.COUNTRY_PADDING_PX,
            )

        if zoom == "state":
            bounds = self.boundaries.get_state_bounds(state)
            viewport_width = float(PixelSampler.STATE_VIEWPORT_WIDTH_PX)
            viewport_height = float(PixelSampler.STATE_VIEWPORT_HEIGHT_PX)
            padding = PixelSampler.STATE_VIEWPORT_PADDING_PX
        else:
            boundary = self.boundaries.get_county_boundary(state, county, county_df=None)
            bounds = self.boundaries.get_feature_bounds(boundary)
            viewport_width = float(PixelSampler.COUNTY_VIEWPORT_WIDTH_PX)
            viewport_height = float(PixelSampler.COUNTY_VIEWPORT_HEIGHT_PX)
            padding = PixelSampler.COUNTY_VIEWPORT_PADDING_PX

        lon_min, lat_min, lon_max, lat_max = self._bounds_or_point_extent(bounds, coords)
        return self.pixel_sampler._to_fitted_state_pixel_coordinates(
            points=coords,
            lat_range=[float(lat_min), float(lat_max)],
            lon_range=[float(lon_min), float(lon_max)],
            viewport_width_px=viewport_width,
            viewport_height_px=viewport_height,
            padding_px=padding,
        )

    def _viewport_info(self, zoom: str, state: str, county: str) -> dict[str, Any]:
        if zoom == "country":
            return {
                "width": int(PixelSampler.COUNTRY_VIEWPORT_WIDTH_PX),
                "height": int(PixelSampler.COUNTRY_VIEWPORT_HEIGHT_PX),
                "padding": [float(value) for value in self.COUNTRY_PADDING_PX],
            }
        if zoom == "state":
            return {
                "width": int(PixelSampler.STATE_VIEWPORT_WIDTH_PX),
                "height": int(PixelSampler.STATE_VIEWPORT_HEIGHT_PX),
                "padding": [float(value) for value in PixelSampler.STATE_VIEWPORT_PADDING_PX],
            }
        return {
            "width": int(PixelSampler.COUNTY_VIEWPORT_WIDTH_PX),
            "height": int(PixelSampler.COUNTY_VIEWPORT_HEIGHT_PX),
            "padding": [float(value) for value in PixelSampler.COUNTY_VIEWPORT_PADDING_PX],
        }

    @staticmethod
    def _applied_min_pixel_distance(zoom: str, country_min_distance: float) -> float:
        marker_distance_by_zoom = {
            "country": 2.0 * (PixelSampler.COUNTRY_MARKER_RADIUS_PX + (PixelSampler.COUNTRY_MARKER_STROKE_WIDTH_PX / 2.0)),
            "state": 2.0 * (PixelSampler.STATE_MARKER_RADIUS_PX + (PixelSampler.STATE_MARKER_STROKE_WIDTH_PX / 2.0)),
            "county": 2.0 * (PixelSampler.COUNTY_MARKER_RADIUS_PX + (PixelSampler.COUNTY_MARKER_STROKE_WIDTH_PX / 2.0)),
        }
        return float(max(float(country_min_distance), marker_distance_by_zoom.get(zoom, 0.0)))

    @staticmethod
    def _scope_matches_source(*, zoom: str, state: str, county: str, config: dict[str, Any]) -> bool:
        if zoom != str(config["source_zoom"]):
            return False
        if zoom == "country":
            return True
        if str(state or "").strip().upper() != str(config["source_state"] or "").strip().upper():
            return False
        if zoom == "state":
            return True
        return str(county or "").strip().lower() == str(config["source_county"] or "").strip().lower()

    @staticmethod
    def _bounds_or_point_extent(bounds: tuple[float, float, float, float] | None, coords: np.ndarray) -> tuple[float, float, float, float]:
        if bounds is not None:
            return bounds
        if len(coords) == 0:
            return (-1.0, -1.0, 1.0, 1.0)
        lat_min = float(np.min(coords[:, 0]))
        lat_max = float(np.max(coords[:, 0]))
        lon_min = float(np.min(coords[:, 1]))
        lon_max = float(np.max(coords[:, 1]))
        if math.isclose(lat_min, lat_max):
            lat_min -= 0.5
            lat_max += 0.5
        if math.isclose(lon_min, lon_max):
            lon_min -= 0.5
            lon_max += 0.5
        return (lon_min, lat_min, lon_max, lat_max)

    def _shared_order_path(self, *, zoom: str, state: str, county: str) -> Path:
        if zoom == "country":
            return self.shared_order_dir / "country" / "order.npy"
        if zoom == "state":
            return self.shared_order_dir / "state" / self._sanitize(state) / "order.npy"
        return self.shared_order_dir / "county" / self._sanitize(state) / self._sanitize(county) / "order.npy"

    def _scope_dir(self, *, zoom: str, state: str, county: str) -> Path:
        if zoom == "country":
            return self.cache_dir / "country"
        if zoom == "state":
            return self.cache_dir / "states" / self._sanitize(state)
        return self.cache_dir / "counties" / self._sanitize(state) / self._sanitize(county)

    def _scope_key(self, *, zoom: str, state: str, county: str) -> str:
        parts = [zoom]
        if state:
            parts.append(state)
        if county:
            parts.append(county)
        return "/".join(self._sanitize(part) for part in parts)

    def _load_entry(self, path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def _entry_valid(
        self,
        entry: dict[str, Any],
        *,
        source_signature: dict[str, Any],
        baseline_source_signature: dict[str, Any],
        safe_distance: float,
        applied_distance: float,
    ) -> bool:
        if entry.get("cache_version") != self.CACHE_VERSION:
            return False
        if not self.is_baseline_key(entry.get("baseline_key")):
            return False
        if entry.get("source_signature") != source_signature:
            return False
        if entry.get("baseline_source_signature") != baseline_source_signature:
            return False
        entry_distance = entry.get(
            "baseline_source_min_pixel_distance",
            entry.get("source_min_pixel_distance", entry.get("country_min_pixel_distance", entry.get("country_min_distance", -1.0))),
        )
        if not math.isclose(float(entry_distance), float(safe_distance), rel_tol=1e-12, abs_tol=1e-15):
            return False
        if not math.isclose(float(entry.get("applied_min_pixel_distance", entry_distance)), float(applied_distance), rel_tol=1e-12, abs_tol=1e-15):
            return False
        if not all(metric in entry.get("metrics", {}) for metric in self.METRICS):
            return False
        for key in ("sample_path", "reduced_h0_path", "reduced_h1_path"):
            if not (self.cache_dir / str(entry.get(key, ""))).exists():
                return False
        return True

    @staticmethod
    def _minimum_pairwise_distance(coords: np.ndarray) -> float:
        if len(coords) <= 1:
            return 0.0
        min_sq = float("inf")
        for idx in range(len(coords) - 1):
            diff = coords[idx + 1 :] - coords[idx]
            if diff.size == 0:
                continue
            dist_sq = diff[:, 0] ** 2 + diff[:, 1] ** 2
            local_min = float(np.min(dist_sq))
            if local_min < min_sq:
                min_sq = local_min
        if not math.isfinite(min_sq):
            return 0.0
        return float(math.sqrt(max(0.0, min_sq)))

    @staticmethod
    def _distance_filtered_order_indices(coords: np.ndarray, order: np.ndarray, min_distance: float) -> np.ndarray:
        if len(coords) == 0:
            return np.empty(0, dtype=np.int32)
        if min_distance <= 0.0:
            return order.astype(np.int32, copy=False)

        threshold_sq = float(min_distance * min_distance)
        epsilon = max(1e-15, threshold_sq * 1e-12)
        cell_size = float(min_distance)
        selected: list[int] = []
        grid: dict[tuple[int, int], list[int]] = {}

        for raw_idx in order:
            idx = int(raw_idx)
            lat = float(coords[idx, 0])
            lng = float(coords[idx, 1])
            cell = (math.floor(lat / cell_size), math.floor(lng / cell_size))
            too_close = False
            for dx in (-1, 0, 1):
                if too_close:
                    break
                for dy in (-1, 0, 1):
                    for selected_idx in grid.get((cell[0] + dx, cell[1] + dy), []):
                        diff_lat = lat - float(coords[selected_idx, 0])
                        diff_lng = lng - float(coords[selected_idx, 1])
                        if diff_lat * diff_lat + diff_lng * diff_lng < threshold_sq - epsilon:
                            too_close = True
                            break
                    if too_close:
                        break
            if too_close:
                continue
            selected.append(idx)
            grid.setdefault(cell, []).append(idx)

        return np.asarray(selected, dtype=np.int32)

    def _distance_block(self, original: dict[str, np.ndarray], reduced: dict[str, np.ndarray]) -> dict[str, float]:
        return {
            "bottleneck_h0": self._safe_h0_distance(original["h0"], reduced["h0"], self._zero_birth_bottleneck, bottleneck_distance),
            "bottleneck_h1": self._safe_distance(original["h1"], reduced["h1"], bottleneck_distance),
            "wasserstein_h0": self._safe_h0_distance(original["h0"], reduced["h0"], self._zero_birth_wasserstein, self._safe_wasserstein),
            "wasserstein_h1": self._safe_wasserstein(original["h1"], reduced["h1"]),
        }

    @classmethod
    def _safe_h0_distance(cls, dgm1: np.ndarray, dgm2: np.ndarray, fn, fallback_fn) -> float:
        if dgm1.size == 0 and dgm2.size == 0:
            return 0.0
        if cls._zero_birth_diagram(dgm1) and cls._zero_birth_diagram(dgm2):
            return float(fn(dgm1[:, 1], dgm2[:, 1]))
        return cls._safe_distance(dgm1, dgm2, fallback_fn)

    @staticmethod
    def _safe_distance(dgm1: np.ndarray, dgm2: np.ndarray, fn) -> float:
        if dgm1.size == 0 and dgm2.size == 0:
            return 0.0
        return float(fn(dgm1, dgm2))

    @staticmethod
    def _safe_wasserstein(dgm1: np.ndarray, dgm2: np.ndarray) -> float:
        if dgm1.size == 0 and dgm2.size == 0:
            return 0.0
        return float(
            wasserstein_distance(
                dgm1,
                dgm2,
                order=1,
                internal_p=float("inf"),
                delta=0.01,
            )
        )

    @staticmethod
    def _zero_birth_diagram(dgm: np.ndarray) -> bool:
        if dgm.size == 0:
            return True
        return dgm.ndim == 2 and dgm.shape[1] == 2 and bool(np.allclose(dgm[:, 0], 0.0, atol=1e-12))

    @staticmethod
    def _zero_birth_bottleneck(deaths1: np.ndarray, deaths2: np.ndarray) -> float:
        deaths1 = np.sort(np.asarray(deaths1, dtype=float))
        deaths2 = np.sort(np.asarray(deaths2, dtype=float))
        m = len(deaths2)
        previous = np.empty(m + 1, dtype=float)
        previous[0] = 0.0
        for col in range(1, m + 1):
            previous[col] = max(previous[col - 1], deaths2[col - 1] / 2.0)

        for death1 in deaths1:
            current = np.empty(m + 1, dtype=float)
            current[0] = max(previous[0], death1 / 2.0)
            for col, death2 in enumerate(deaths2, start=1):
                match_cost = max(previous[col - 1], abs(death1 - death2))
                drop_left_cost = max(previous[col], death1 / 2.0)
                drop_right_cost = max(current[col - 1], death2 / 2.0)
                current[col] = min(match_cost, drop_left_cost, drop_right_cost)
            previous = current
        return float(previous[m])

    @staticmethod
    def _zero_birth_wasserstein(deaths1: np.ndarray, deaths2: np.ndarray) -> float:
        deaths1 = np.sort(np.asarray(deaths1, dtype=float))
        deaths2 = np.sort(np.asarray(deaths2, dtype=float))
        m = len(deaths2)
        previous = np.empty(m + 1, dtype=float)
        previous[0] = 0.0
        for col in range(1, m + 1):
            previous[col] = previous[col - 1] + deaths2[col - 1] / 2.0

        for death1 in deaths1:
            current = np.empty(m + 1, dtype=float)
            current[0] = previous[0] + death1 / 2.0
            for col, death2 in enumerate(deaths2, start=1):
                match_cost = previous[col - 1] + abs(death1 - death2)
                drop_left_cost = previous[col] + death1 / 2.0
                drop_right_cost = current[col - 1] + death2 / 2.0
                current[col] = min(match_cost, drop_left_cost, drop_right_cost)
            previous = current
        return float(previous[m])

    @staticmethod
    def _frame_signature(df: pd.DataFrame) -> dict[str, Any]:
        if df.empty:
            return {"rows": 0, "first_id": "", "last_id": ""}
        ids = df["ID"].fillna("").astype(str) if "ID" in df.columns else df.index.astype(str)
        return {
            "rows": int(len(df)),
            "first_id": str(ids.iloc[0]),
            "last_id": str(ids.iloc[-1]),
        }

    @staticmethod
    def _equivalent_percentage(n_points: int, n_source_points: int) -> float:
        if n_source_points <= 0:
            return 0.0
        return float((n_points / n_source_points) * 100.0)

    @classmethod
    def _metric_name(cls, metric_name: str) -> str:
        metric = (metric_name or cls.DEFAULT_THRESHOLD_METRIC).strip().lower()
        if metric not in cls.METRICS:
            return cls.DEFAULT_THRESHOLD_METRIC
        return metric

    @classmethod
    def _row_has_metrics(cls, row: pd.Series) -> bool:
        return all(metric in row and pd.notna(row[metric]) for metric in cls.METRICS)

    @staticmethod
    def _normalized(value: float, baseline: float) -> float:
        if baseline <= 0.0:
            return 0.0 if value <= 0.0 else float("inf")
        return float(value / baseline)

    @staticmethod
    def _sanitize(value: str) -> str:
        safe = str(value).replace("\\", "_").replace("/", "_").replace(":", "_")
        return safe.replace(" ", "_")


