"""Runtime cache for the active random-sampling baseline."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from adaptive_visualization.core.sample_schema import trim_sample_columns
from adaptive_visualization.vendor.map_viz.data.loader import AccidentDataLoader
from adaptive_visualization.vendor.map_viz.sampling.random_sampler import RandomSampler


class RuntimeSamplingCache:
    """Compute and persist random samples used by the OpenLayers dashboard."""

    CACHE_VERSION = "runtime_sampling_random_v1"
    CACHE_DIRNAME = "runtime_sampling_cache"

    def __init__(self, data_dir: str | Path, loader: AccidentDataLoader):
        self.data_dir = Path(data_dir)
        self.loader = loader
        self.cache_dir = self.data_dir / self.CACHE_DIRNAME
        self.metadata_path = self.cache_dir / "metadata.json"
        self._cache: dict[str, pd.DataFrame] = {}
        self._metadata: dict[str, Any] | None = None

    def get_or_create(
        self,
        *,
        method: str,
        retain_percentage: int,
        zoom: str,
        state: str,
        county: str,
        source_df: pd.DataFrame,
        analysis_property: str = "statistical",
        error_threshold: float = 0.05,
    ) -> pd.DataFrame:
        del analysis_property, error_threshold
        method = (method or "").strip().lower()
        if method != "random":
            return source_df.reset_index(drop=True)

        retain_percentage = int(max(1, min(100, retain_percentage)))
        entry = self._entry_spec(
            retain_percentage=retain_percentage,
            zoom=zoom,
            state=state,
            county=county,
            source_df=source_df,
        )
        cache_key = entry["cache_key"]
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        cache_path = self.cache_dir / entry["path"]
        metadata = self._load_metadata()
        if cache_path.exists() and self._is_cache_valid(metadata, entry, cache_path):
            print(f"[runtime-sampling-cache] loading cached random sample for {cache_key} from disk")
            cached_df = self.loader.load(str(cache_path))
            self._cache[cache_key] = cached_df
            return cached_df

        sampled_df = trim_sample_columns(
            self._compute_random_sample(
                retain_percentage=retain_percentage,
                zoom=zoom,
                state=state,
                county=county,
                source_df=source_df,
            )
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        sampled_df.to_csv(cache_path, index=False)

        metadata = self._load_metadata()
        metadata.setdefault("entries", {})[cache_key] = {
            "path": entry["path"],
            "method": "random",
            "zoom": zoom,
            "state": state,
            "county": county,
            "retain_percentage": retain_percentage,
            "source_signature": entry["source_signature"],
            "cached_records": int(len(sampled_df)),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self._save_metadata(metadata)
        self._cache[cache_key] = sampled_df
        print(f"[runtime-sampling-cache] saved random sample to {cache_path.relative_to(self.data_dir).as_posix()}")
        return sampled_df

    def _compute_random_sample(
        self,
        *,
        retain_percentage: int,
        zoom: str,
        state: str,
        county: str,
        source_df: pd.DataFrame,
    ) -> pd.DataFrame:
        valid_df = source_df.dropna(subset=["Start_Lat", "Start_Lng"]).reset_index(drop=True)
        if valid_df.empty:
            return valid_df

        n_points = len(valid_df)
        n_keep = min(n_points, max(1, int(round(n_points * (retain_percentage / 100.0)))))
        if n_keep >= n_points:
            return valid_df.reset_index(drop=True)

        coords = valid_df[["Start_Lat", "Start_Lng"]].to_numpy(dtype=float)
        seed = self._stable_seed("random", zoom, state, county, retain_percentage)
        selected_idx = np.asarray(RandomSampler(random_seed=seed).sample(coords, n_keep), dtype=int)
        if selected_idx.size == 0:
            return valid_df.iloc[0:0].reset_index(drop=True)
        return valid_df.iloc[selected_idx].reset_index(drop=True)

    def _entry_spec(
        self,
        *,
        retain_percentage: int,
        zoom: str,
        state: str,
        county: str,
        source_df: pd.DataFrame,
    ) -> dict[str, Any]:
        source_signature = self._frame_signature(source_df)
        parts = [zoom]
        if state:
            parts.append(state)
        if county:
            parts.append(county)
        parts.append(f"random__retain{retain_percentage}.csv")
        path = "/".join(self._sanitize(part) for part in parts)
        cache_key = json.dumps(
            {
                "method": "random",
                "retain_percentage": retain_percentage,
                "zoom": zoom,
                "state": state,
                "county": county,
                "source_signature": source_signature,
            },
            sort_keys=True,
        )
        return {
            "cache_key": cache_key,
            "path": path,
            "source_signature": source_signature,
        }

    def _load_metadata(self) -> dict[str, Any]:
        if self._metadata is not None:
            return self._metadata

        metadata: dict[str, Any] | None = None
        if self.metadata_path.exists():
            try:
                metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                metadata = None

        if metadata is None or not self._metadata_matches(metadata):
            metadata = self._default_metadata()
            self._cache.clear()

        self._metadata = metadata
        return self._metadata

    def _save_metadata(self, metadata: dict[str, Any]) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        self._metadata = metadata

    def _metadata_matches(self, metadata: dict[str, Any]) -> bool:
        expected = self._metadata_signature()
        for key, value in expected.items():
            if metadata.get(key) != value:
                return False
        return isinstance(metadata.get("entries"), dict)

    def _is_cache_valid(self, metadata: dict[str, Any], entry: dict[str, Any], cache_path: Path) -> bool:
        if not self._metadata_matches(metadata):
            return False
        info = metadata.get("entries", {}).get(entry["cache_key"])
        if not isinstance(info, dict):
            return False
        return (
            info.get("path") == entry["path"]
            and info.get("source_signature") == entry["source_signature"]
            and cache_path.exists()
        )

    def _default_metadata(self) -> dict[str, Any]:
        metadata = self._metadata_signature()
        metadata["entries"] = {}
        return metadata

    def _metadata_signature(self) -> dict[str, Any]:
        return {
            "cache_version": self.CACHE_VERSION,
            "persisted_methods": ["random"],
            "source_original": self._source_signature(),
        }

    def _source_signature(self) -> dict[str, Any]:
        source_path = self.data_dir / "original.csv"
        if not source_path.exists():
            return {"path": "original.csv", "size_bytes": None, "mtime_ns": None}
        stat = source_path.stat()
        return {
            "path": "original.csv",
            "size_bytes": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }

    @staticmethod
    def _stable_seed(method: str, zoom: str, state: str, county: str, retain_percentage: int) -> int:
        key = json.dumps(
            {
                "method": method,
                "zoom": zoom,
                "state": state,
                "county": county,
                "retain_percentage": int(retain_percentage),
            },
            sort_keys=True,
        )
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return int(digest[:8], 16)

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
    def _sanitize(value: str) -> str:
        safe = str(value).replace("\\", "_").replace("/", "_").replace(":", "_")
        return safe.replace(" ", "_")


