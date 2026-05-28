"""Deterministic farthest-first prefix order cache for Adaptive OpenLayers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class FPSOrderCache:
    """Persist and resume exact greedy farthest-point traversal orders."""

    CACHE_VERSION = "fps_order_exact_v1"
    CACHE_DIRNAME = "fps_order_cache"
    CHECKPOINT_INTERVAL = 250

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        self.cache_dir = self.data_dir / self.CACHE_DIRNAME
        self.metadata_path = self.cache_dir / "metadata.json"
        self._cache: dict[str, np.ndarray] = {}
        self._metadata: dict[str, Any] | None = None

    def get_order(
        self,
        *,
        zoom: str,
        state: str,
        county: str,
        source_df: pd.DataFrame,
    ) -> np.ndarray:
        valid_df = source_df.dropna(subset=["Start_Lat", "Start_Lng"]).reset_index(drop=True)
        if valid_df.empty:
            return np.empty(0, dtype=int)

        spec = self._entry_spec(zoom=zoom, state=state, county=county, source_df=valid_df)
        cached = self._cache.get(spec["cache_key"])
        if cached is not None:
            return cached

        metadata = self._load_metadata()
        order_path = self.cache_dir / spec["base_path"] / "order.npy"
        if order_path.exists() and self._is_order_valid(metadata, spec, order_path):
            order = np.load(order_path)
            self._cache[spec["cache_key"]] = order
            return order

        order = self._compute_order(spec=spec, source_df=valid_df)
        self._cache[spec["cache_key"]] = order
        return order

    def _compute_order(self, *, spec: dict[str, Any], source_df: pd.DataFrame) -> np.ndarray:
        base_dir = self.cache_dir / spec["base_path"]
        base_dir.mkdir(parents=True, exist_ok=True)

        order_path = base_dir / "order.npy"
        partial_order_path = base_dir / "order.partial.npy"
        selected_path = base_dir / "selected.partial.npy"
        min_distance_path = base_dir / "min_distance_sq.partial.npy"

        coords = source_df[["Start_Lat", "Start_Lng"]].to_numpy(dtype=float)
        n_points = len(coords)
        if n_points <= 1:
            order = np.arange(n_points, dtype=np.int32)
            np.save(order_path, order)
            self._persist_metadata(spec, order_path, n_points)
            return order

        order, selected, min_distance_sq, step = self._restore_partial_state(
            partial_order_path=partial_order_path,
            selected_path=selected_path,
            min_distance_path=min_distance_path,
            n_points=n_points,
        )

        if step <= 0:
            start_idx = self._initial_seed(coords)
            order[0] = start_idx
            selected[start_idx] = True
            min_distance_sq = self._update_min_distances(coords, min_distance_sq, start_idx)
            min_distance_sq[selected] = -np.inf
            step = 1
            self._save_partial_state(
                partial_order_path=partial_order_path,
                selected_path=selected_path,
                min_distance_path=min_distance_path,
                order=order,
                selected=selected,
                min_distance_sq=min_distance_sq,
            )

        while step < n_points:
            next_idx = int(np.argmax(min_distance_sq))
            order[step] = next_idx
            selected[next_idx] = True
            min_distance_sq = self._update_min_distances(coords, min_distance_sq, next_idx)
            min_distance_sq[selected] = -np.inf
            step += 1

            if step % self.CHECKPOINT_INTERVAL == 0 or step == n_points:
                self._save_partial_state(
                    partial_order_path=partial_order_path,
                    selected_path=selected_path,
                    min_distance_path=min_distance_path,
                    order=order,
                    selected=selected,
                    min_distance_sq=min_distance_sq,
                )

        np.save(order_path, order)
        self._cleanup_partial_state(partial_order_path, selected_path, min_distance_path)
        self._persist_metadata(spec, order_path, n_points)
        return order

    def _persist_metadata(self, spec: dict[str, Any], order_path: Path, n_points: int) -> None:
        metadata = self._load_metadata()
        metadata.setdefault("entries", {})[spec["cache_key"]] = {
            "path": str(order_path.relative_to(self.cache_dir)).replace("\\", "/"),
            "zoom": spec["zoom"],
            "state": spec["state"],
            "county": spec["county"],
            "source_signature": spec["source_signature"],
            "n_points": int(n_points),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self._save_metadata(metadata)

    def _restore_partial_state(
        self,
        *,
        partial_order_path: Path,
        selected_path: Path,
        min_distance_path: Path,
        n_points: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        if partial_order_path.exists() and selected_path.exists() and min_distance_path.exists():
            try:
                order = np.load(partial_order_path)
                selected = np.load(selected_path)
                min_distance_sq = np.load(min_distance_path)
                if (
                    order.shape == (n_points,)
                    and selected.shape == (n_points,)
                    and min_distance_sq.shape == (n_points,)
                ):
                    step = int(selected.sum())
                    return (
                        order.astype(np.int32, copy=False),
                        selected.astype(bool, copy=False),
                        min_distance_sq.astype(float, copy=False),
                        step,
                    )
            except Exception:
                pass

        return (
            np.full(n_points, -1, dtype=np.int32),
            np.zeros(n_points, dtype=bool),
            np.full(n_points, np.inf, dtype=float),
            0,
        )

    @staticmethod
    def _save_partial_state(
        *,
        partial_order_path: Path,
        selected_path: Path,
        min_distance_path: Path,
        order: np.ndarray,
        selected: np.ndarray,
        min_distance_sq: np.ndarray,
    ) -> None:
        np.save(partial_order_path, order)
        np.save(selected_path, selected)
        np.save(min_distance_path, min_distance_sq)

    @staticmethod
    def _cleanup_partial_state(*paths: Path) -> None:
        for path in paths:
            if path.exists():
                path.unlink()

    @staticmethod
    def _initial_seed(coords: np.ndarray) -> int:
        centroid = coords.mean(axis=0)
        diff = coords - centroid
        dist_sq = diff[:, 0] ** 2 + diff[:, 1] ** 2
        return int(np.argmax(dist_sq))

    @staticmethod
    def _update_min_distances(coords: np.ndarray, min_distance_sq: np.ndarray, selected_idx: int) -> np.ndarray:
        diff = coords - coords[selected_idx]
        dist_sq = diff[:, 0] ** 2 + diff[:, 1] ** 2
        return np.minimum(min_distance_sq, dist_sq)

    def _entry_spec(
        self,
        *,
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
        base_path = "/".join(self._sanitize(part) for part in parts)
        cache_key = json.dumps(
            {
                "zoom": zoom,
                "state": state,
                "county": county,
                "source_signature": source_signature,
            },
            sort_keys=True,
        )
        return {
            "cache_key": cache_key,
            "base_path": base_path,
            "zoom": zoom,
            "state": state,
            "county": county,
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

        self._metadata = metadata
        return metadata

    def _save_metadata(self, metadata: dict[str, Any]) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        existing: dict[str, Any] | None = None
        if self.metadata_path.exists():
            try:
                existing = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                existing = None

        if existing is not None and self._metadata_matches(existing):
            merged = dict(existing)
            merged["entries"] = dict(existing.get("entries", {}))
            merged["entries"].update(metadata.get("entries", {}))
            for key, value in metadata.items():
                if key == "entries":
                    continue
                merged[key] = value
            metadata = merged

        self.metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        self._metadata = metadata

    def _metadata_matches(self, metadata: dict[str, Any]) -> bool:
        expected = self._metadata_signature()
        for key, value in expected.items():
            if metadata.get(key) != value:
                return False
        return isinstance(metadata.get("entries"), dict)

    def _is_order_valid(self, metadata: dict[str, Any], spec: dict[str, Any], order_path: Path) -> bool:
        if not self._metadata_matches(metadata):
            return False
        info = metadata.get("entries", {}).get(spec["cache_key"])
        if not isinstance(info, dict):
            return False
        if info.get("source_signature") != spec["source_signature"]:
            return False
        if info.get("path") != str(order_path.relative_to(self.cache_dir)).replace("\\", "/"):
            return False
        return order_path.exists()

    def _default_metadata(self) -> dict[str, Any]:
        metadata = self._metadata_signature()
        metadata["entries"] = {}
        return metadata

    def _metadata_signature(self) -> dict[str, Any]:
        return {
            "cache_version": self.CACHE_VERSION,
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
    def _frame_signature(df: pd.DataFrame) -> dict[str, Any]:
        if df.empty:
            return {"rows": 0, "first_id": "", "last_id": ""}
        if "ID" in df.columns:
            ids = df["ID"].fillna("").astype(str)
            return {
                "rows": int(len(df)),
                "first_id": str(ids.iloc[0]),
                "last_id": str(ids.iloc[-1]),
            }
        return {
            "rows": int(len(df)),
            "first_id": str(df.index[0]),
            "last_id": str(df.index[-1]),
        }

    @staticmethod
    def _sanitize(value: str) -> str:
        safe = str(value).replace("\\", "_").replace("/", "_").replace(":", "_")
        return safe.replace(" ", "_")
