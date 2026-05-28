"""State-level FPS topology precompute for the OpenLayers app."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import gudhi
import numpy as np
import pandas as pd
from gudhi.hera import bottleneck_distance, wasserstein_distance

from adaptive_visualization.core.fps_order_cache import FPSOrderCache
from adaptive_visualization.core.sample_schema import trim_sample_columns


class StateFPSTopologyCache:
    """Precompute full-point topology for state-level FPS samples."""

    CACHE_VERSION = "state_fps_topology_dense_to_30_v5"
    CACHE_DIRNAME = "state_fps_topology_cache"
    DIAGRAM_EQUAL_ATOL = 1e-9
    BASELINE_PERCENTAGES = (1, 5, 10)
    BASELINE_PERCENTAGE = 5
    # Keep fine-grained state-level thresholds where the dashboard decisions
    # change quickly, then use even checkpoints for the rest of the curve.
    PERCENTAGES = tuple(range(1, 31)) + tuple(range(32, 101, 2))
    DEFAULT_THRESHOLD_METRIC = "bottleneck_h0"
    METRICS = (
        "bottleneck_h0",
        "bottleneck_h1",
        "wasserstein_h0",
        "wasserstein_h1",
    )

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        self.cache_dir = self.data_dir / self.CACHE_DIRNAME
        self.metadata_path = self.cache_dir / "metadata.json"
        self.shared_order_dir = self.data_dir / FPSOrderCache.CACHE_DIRNAME
        self._metadata: dict[str, Any] | None = None
        self._curves: dict[str, pd.DataFrame] = {}
        self._frames: dict[str, pd.DataFrame] = {}

    def get_or_create(
        self,
        *,
        state: str,
        source_df: pd.DataFrame,
        error_threshold: float,
        metric_name: str = DEFAULT_THRESHOLD_METRIC,
        baseline_percentage: int | None = None,
    ) -> pd.DataFrame:
        valid_df = source_df.dropna(subset=["Start_Lat", "Start_Lng"]).reset_index(drop=True)
        if valid_df.empty:
            return valid_df

        selection = self.get_selection(
            state=state,
            source_df=valid_df,
            error_threshold=error_threshold,
            metric_name=metric_name,
            baseline_percentage=baseline_percentage,
        )
        if selection is None or int(selection["percentage"]) >= 100:
            return valid_df.reset_index(drop=True)

        frame_key = f"{self._state_key(state)}::{int(selection['percentage'])}"
        cached = self._frames.get(frame_key)
        if cached is not None:
            return cached

        sample_path = self.cache_dir / str(selection["sample_path"])
        sampled_df = pd.read_csv(sample_path)
        self._frames[frame_key] = sampled_df
        return sampled_df

    def get_selection(
        self,
        *,
        state: str,
        source_df: pd.DataFrame,
        error_threshold: float,
        metric_name: str = DEFAULT_THRESHOLD_METRIC,
        baseline_percentage: int | None = None,
    ) -> dict[str, Any] | None:
        metric_name = self._metric_name(metric_name)
        valid_df = source_df.dropna(subset=["Start_Lat", "Start_Lng"]).reset_index(drop=True)
        if valid_df.empty:
            return None

        requested_baseline = self._requested_baseline_percentage(baseline_percentage)
        curve = self._load_available_curve(state)
        if curve.empty or not self._curve_has_rows(curve, {requested_baseline}):
            curve = self.ensure_precomputed(valid_df, state=state)
        baseline_row = self._baseline_row(curve, requested_baseline)
        if baseline_row is None:
            return None
        chosen = self._select_curve_row(
            curve,
            error_threshold,
            metric_name=metric_name,
            baseline_row=baseline_row,
        )
        if chosen is None:
            return None

        return {
            "percentage": int(chosen["percentage"]),
            "n_points": int(chosen["n_points"]),
            "raw_error": float(chosen[metric_name]),
            "normalized_error": self._normalized_metric_value(chosen, metric_name, baseline_row),
            "metric_name": metric_name,
            "sample_path": None if pd.isna(chosen["sample_path"]) else str(chosen["sample_path"]),
            "baseline_percentage": int(baseline_row["percentage"]),
        }

    def get_payload(
        self,
        *,
        state: str,
        source_df: pd.DataFrame,
        current_percentage: int,
        baseline_percentage: int | None = None,
    ) -> dict[str, Any]:
        requested_baseline = self._requested_baseline_percentage(baseline_percentage)
        current_percentage = self._configured_percentage_at_or_above(current_percentage)
        curve = self._load_available_curve(state)
        if curve.empty or not self._curve_has_rows(curve, {requested_baseline, current_percentage}):
            curve = self.ensure_precomputed(source_df, state=state)
        if curve.empty:
            return {
                "original": {"h0": [], "h1": []},
                "reduced": {"h0": [], "h1": []},
                "baseline_distances": {metric: 0.0 for metric in self.METRICS},
                "current_distances": {metric: 0.0 for metric in self.METRICS},
                "normalized_distances": {metric: 0.0 for metric in self.METRICS},
                "current_percentage": current_percentage,
                "baseline_percentage": baseline_percentage,
                "complete": False,
            }

        baseline_row = self._baseline_row(curve, requested_baseline)
        if baseline_row is None:
            baseline_row = curve.iloc[0]
        baseline_percentage = int(baseline_row["percentage"])
        current_row = curve[curve["percentage"] == current_percentage]
        if current_row.empty:
            current_row = curve[curve["percentage"] == 100]
        if current_row.empty:
            current_row = curve.iloc[[-1]]
        current_row = current_row.iloc[0]
        baseline_row = curve[curve["percentage"] == baseline_percentage].iloc[0]
        base_dir = self._state_dir(state)

        original_h0 = self._load_diagram_array(base_dir / "original_h0.csv")
        original_h1 = self._load_diagram_array(base_dir / "original_h1.csv")
        if int(current_row["percentage"]) >= 100:
            reduced_h0 = original_h0
            reduced_h1 = original_h1
        else:
            reduced_h0 = self._load_diagram_array(base_dir / f"pct_{int(current_row['percentage']):02d}_h0.csv")
            reduced_h1 = self._load_diagram_array(base_dir / f"pct_{int(current_row['percentage']):02d}_h1.csv")

        return {
            "original": {"h0": original_h0, "h1": original_h1},
            "reduced": {"h0": reduced_h0, "h1": reduced_h1},
            "baseline_distances": self._metric_block(baseline_row),
            "current_distances": self._metric_block(current_row),
            "normalized_distances": self._normalized_block(current_row, baseline_row),
            "current_percentage": int(current_row["percentage"]),
            "baseline_percentage": baseline_percentage,
            "complete": self._curve_complete(curve),
        }

    def get_curve(self, *, state: str, source_df: pd.DataFrame) -> pd.DataFrame:
        return self.ensure_precomputed(source_df, state=state).copy()

    def ensure_precomputed(
        self,
        source_df: pd.DataFrame,
        *,
        state: str,
    ) -> pd.DataFrame:
        return self.ensure_precomputed_until(source_df, state=state)

    def ensure_precomputed_until(
        self,
        source_df: pd.DataFrame,
        *,
        state: str,
        max_percentage: int | None = None,
    ) -> pd.DataFrame:
        valid_df = source_df.dropna(subset=["Start_Lat", "Start_Lng"]).reset_index(drop=True)
        if valid_df.empty:
            return pd.DataFrame()

        signature = self._frame_signature(valid_df)
        state_key = self._state_key(state)
        curve_path = self._state_dir(state) / "curve.csv"
        cached = self._curves.get(state_key)
        if (
            cached is not None
            and self._metadata_matches_signature(state, signature)
            and self._curve_complete(cached, max_percentage=max_percentage)
        ):
            return cached

        metadata = self._load_metadata()
        if curve_path.exists() and self._is_valid(metadata, state, signature, curve_path):
            curve_df = pd.read_csv(curve_path)
            if self._curve_complete(curve_df, max_percentage=max_percentage):
                self._curves[state_key] = curve_df
                return curve_df

        curve_df = self._compute_curve(valid_df, state=state, signature=signature, max_percentage=max_percentage)
        self._curves[state_key] = curve_df
        return curve_df

    def _compute_curve(
        self,
        source_df: pd.DataFrame,
        *,
        state: str,
        signature: dict[str, Any],
        max_percentage: int | None = None,
    ) -> pd.DataFrame:
        base_dir = self._state_dir(state)
        base_dir.mkdir(parents=True, exist_ok=True)
        curve_path = base_dir / "curve.csv"
        partial_curve_path = base_dir / "curve.partial.csv"

        fps_order = self._get_or_compute_order(state=state, source_df=source_df)
        percentages = self._configured_percentages(max_percentage)
        samples_dir = base_dir / "samples"
        samples_dir.mkdir(parents=True, exist_ok=True)

        original_h0_path = base_dir / "original_h0.csv"
        original_h1_path = base_dir / "original_h1.csv"
        if original_h0_path.exists() and original_h1_path.exists():
            original = {
                "h0": self._load_diagram_array_np(original_h0_path),
                "h1": self._load_diagram_array_np(original_h1_path),
            }
        else:
            original_coords = source_df[["Start_Lat", "Start_Lng"]].to_numpy(dtype=float)
            original = self._compute_diagrams(original_coords)
            self._save_diagram_array(original_h0_path, original["h0"])
            self._save_diagram_array(original_h1_path, original["h1"])

        configured_percentages = set(self.PERCENTAGES)
        rows_by_percentage = {
            percentage: row
            for percentage, row in self._load_existing_rows(curve_path, partial_curve_path).items()
            if percentage in configured_percentages
        }
        baseline_metrics = None
        baseline_row = rows_by_percentage.get(self.BASELINE_PERCENTAGE)
        if baseline_row is not None and self._row_has_metrics(baseline_row):
            baseline_metrics = {metric: float(baseline_row[metric]) for metric in self.METRICS}

        for percentage in percentages:
            n_keep = self._n_keep(len(source_df), percentage)
            sample_path = samples_dir / f"fps__pct{percentage:02d}.csv"
            sample_rel_path = str(sample_path.relative_to(self.cache_dir)).replace("\\", "/")
            existing_row = rows_by_percentage.get(percentage)
            reduced_h0_path = base_dir / f"pct_{percentage:02d}_h0.csv"
            reduced_h1_path = base_dir / f"pct_{percentage:02d}_h1.csv"

            if (
                existing_row is not None
                and reduced_h0_path.exists()
                and reduced_h1_path.exists()
                and self._row_has_metrics(existing_row)
            ):
                if not sample_path.exists():
                    self._load_or_create_sample(
                        source_df=source_df,
                        fps_order=fps_order,
                        n_keep=n_keep,
                        sample_path=sample_path,
                    )
                rows_by_percentage[percentage] = self._refresh_existing_row(
                    existing_row,
                    n_points=n_keep,
                    sample_path=sample_rel_path,
                )
                if percentage == self.BASELINE_PERCENTAGE and baseline_metrics is None:
                    baseline_metrics = {metric: float(existing_row[metric]) for metric in self.METRICS}
                continue

            sampled_df = self._load_or_create_sample(
                source_df=source_df,
                fps_order=fps_order,
                n_keep=n_keep,
                sample_path=sample_path,
            )

            if percentage >= 100:
                if not reduced_h0_path.exists():
                    self._save_diagram_array(reduced_h0_path, original["h0"])
                if not reduced_h1_path.exists():
                    self._save_diagram_array(reduced_h1_path, original["h1"])
                metrics = {metric: 0.0 for metric in self.METRICS}
            elif reduced_h0_path.exists() and reduced_h1_path.exists():
                reduced = {
                    "h0": self._load_diagram_array_np(reduced_h0_path),
                    "h1": self._load_diagram_array_np(reduced_h1_path),
                }
                metrics = self._distance_block(original, reduced)
            else:
                sampled_coords = sampled_df[["Start_Lat", "Start_Lng"]].to_numpy(dtype=float)
                reduced = self._compute_diagrams(sampled_coords)
                self._save_diagram_array(reduced_h0_path, reduced["h0"])
                self._save_diagram_array(reduced_h1_path, reduced["h1"])
                metrics = self._distance_block(original, reduced)

            if percentage == self.BASELINE_PERCENTAGE:
                baseline_metrics = dict(metrics)

            row = {
                "percentage": percentage,
                "n_points": int(n_keep),
                "sample_path": sample_rel_path,
            }
            for metric_name, metric_value in metrics.items():
                row[metric_name] = metric_value
            rows_by_percentage[percentage] = row
            self._apply_normalized_metrics(rows_by_percentage, baseline_metrics)
            pd.DataFrame(
                [rows_by_percentage[pct] for pct in sorted(rows_by_percentage)]
            ).to_csv(partial_curve_path, index=False)

        self._apply_normalized_metrics(rows_by_percentage, baseline_metrics)

        curve_df = pd.DataFrame([rows_by_percentage[pct] for pct in sorted(rows_by_percentage)])
        curve_df.to_csv(curve_path, index=False)
        if partial_curve_path.exists():
            partial_curve_path.unlink()

        metadata = self._load_metadata()
        metadata.setdefault("entries", {})[self._state_key(state)] = {
            "path": str(curve_path.relative_to(self.cache_dir)).replace("\\", "/"),
            "state": state,
            "source_signature": signature,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self._save_metadata(metadata)
        return curve_df

    def _get_or_compute_order(self, *, state: str, source_df: pd.DataFrame) -> np.ndarray:
        shared_order = self._load_shared_order(state=state, source_df=source_df)
        if shared_order is not None:
            return shared_order

        base_dir = self._state_dir(state)
        order_path = base_dir / "order.npy"
        if order_path.exists():
            try:
                order = np.load(order_path)
                if order.shape == (len(source_df),):
                    return order.astype(np.int32, copy=False)
            except Exception:
                pass

        order = self._compute_order(source_df)
        base_dir.mkdir(parents=True, exist_ok=True)
        np.save(order_path, order)
        return order

    def _load_shared_order(self, *, state: str, source_df: pd.DataFrame) -> np.ndarray | None:
        shared_order_path = self.shared_order_dir / "state" / self._sanitize(state) / "order.npy"
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
        expected_state = state.strip().upper()
        match_found = False
        for info in metadata.get("entries", {}).values():
            if not isinstance(info, dict):
                continue
            if info.get("path") != expected_path:
                continue
            if info.get("zoom") != "state":
                continue
            if str(info.get("state", "")).strip().upper() != expected_state:
                continue
            if info.get("county", "") != "":
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

    @staticmethod
    def _compute_order(source_df: pd.DataFrame) -> np.ndarray:
        coords = source_df[["Start_Lat", "Start_Lng"]].to_numpy(dtype=float)
        n_points = len(coords)
        if n_points <= 1:
            return np.arange(n_points, dtype=np.int32)

        order = np.full(n_points, -1, dtype=np.int32)
        selected = np.zeros(n_points, dtype=bool)
        min_distance_sq = np.full(n_points, np.inf, dtype=float)

        start_idx = StateFPSTopologyCache._initial_seed(coords)
        order[0] = start_idx
        selected[start_idx] = True
        min_distance_sq = StateFPSTopologyCache._update_min_distances(coords, min_distance_sq, start_idx)
        min_distance_sq[selected] = -np.inf

        for step in range(1, n_points):
            next_idx = int(np.argmax(min_distance_sq))
            order[step] = next_idx
            selected[next_idx] = True
            min_distance_sq = StateFPSTopologyCache._update_min_distances(coords, min_distance_sq, next_idx)
            min_distance_sq[selected] = -np.inf

        return order

    def _compute_diagrams(self, coords: np.ndarray) -> dict[str, np.ndarray]:
        alpha = gudhi.AlphaComplex(points=coords.tolist())
        simplex_tree = alpha.create_simplex_tree()
        persistence = simplex_tree.persistence()

        h0_rows: list[list[float]] = []
        h1_rows: list[list[float]] = []
        for dim, pair in persistence:
            if dim not in (0, 1):
                continue
            birth, death = pair
            if not np.isfinite(death):
                continue
            if dim == 0:
                h0_rows.append([float(birth), float(death)])
            else:
                h1_rows.append([float(birth), float(death)])

        return {
            "h0": np.asarray(h0_rows, dtype=float).reshape(-1, 2) if h0_rows else np.empty((0, 2), dtype=float),
            "h1": np.asarray(h1_rows, dtype=float).reshape(-1, 2) if h1_rows else np.empty((0, 2), dtype=float),
        }

    def _distance_block(self, original: dict[str, np.ndarray], reduced: dict[str, np.ndarray]) -> dict[str, float]:
        return {
            "bottleneck_h0": self._safe_h0_distance(
                original["h0"],
                reduced["h0"],
                self._zero_birth_bottleneck,
                bottleneck_distance,
            ),
            "bottleneck_h1": self._safe_distance(original["h1"], reduced["h1"], bottleneck_distance),
            "wasserstein_h0": self._safe_h0_distance(
                original["h0"],
                reduced["h0"],
                self._zero_birth_wasserstein,
                self._safe_wasserstein,
            ),
            "wasserstein_h1": self._safe_wasserstein(original["h1"], reduced["h1"]),
        }

    @classmethod
    def _safe_h0_distance(cls, dgm1: np.ndarray, dgm2: np.ndarray, fn, fallback_fn) -> float:
        if dgm1.size == 0 and dgm2.size == 0:
            return 0.0
        if cls._diagrams_effectively_equal(dgm1, dgm2):
            return 0.0
        if cls._zero_birth_diagram(dgm1) and cls._zero_birth_diagram(dgm2):
            return float(fn(dgm1[:, 1], dgm2[:, 1]))
        return cls._safe_distance(dgm1, dgm2, fallback_fn)

    @staticmethod
    def _safe_distance(dgm1: np.ndarray, dgm2: np.ndarray, fn) -> float:
        if dgm1.size == 0 and dgm2.size == 0:
            return 0.0
        if StateFPSTopologyCache._diagrams_effectively_equal(dgm1, dgm2):
            return 0.0
        return float(fn(dgm1, dgm2))

    @staticmethod
    def _safe_wasserstein(dgm1: np.ndarray, dgm2: np.ndarray) -> float:
        if dgm1.size == 0 and dgm2.size == 0:
            return 0.0
        if StateFPSTopologyCache._diagrams_effectively_equal(dgm1, dgm2):
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

    @classmethod
    def _diagrams_effectively_equal(cls, dgm1: np.ndarray, dgm2: np.ndarray) -> bool:
        return dgm1.shape == dgm2.shape and bool(
            np.allclose(dgm1, dgm2, rtol=0.0, atol=cls.DIAGRAM_EQUAL_ATOL)
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

    @staticmethod
    def _normalized(value: float, baseline: float) -> float:
        if baseline <= 0.0:
            return 0.0 if value <= 0.0 else float("inf")
        return float(value / baseline)

    @staticmethod
    def _save_diagram_array(path: Path, array: np.ndarray) -> None:
        frame = pd.DataFrame(array, columns=["birth", "death"])
        frame.to_csv(path, index=False)

    @staticmethod
    def _load_diagram_array(filename_or_path: str | Path) -> list[list[float]]:
        path = Path(filename_or_path)
        if not path.exists():
            return []
        frame = pd.read_csv(path)
        if frame.empty:
            return []
        return frame[["birth", "death"]].to_numpy(dtype=float).tolist()

    @staticmethod
    def _load_diagram_array_np(filename_or_path: str | Path) -> np.ndarray:
        path = Path(filename_or_path)
        if not path.exists():
            return np.empty((0, 2), dtype=float)
        frame = pd.read_csv(path)
        if frame.empty:
            return np.empty((0, 2), dtype=float)
        return frame[["birth", "death"]].to_numpy(dtype=float)

    @classmethod
    def _metric_block(cls, row: pd.Series) -> dict[str, float]:
        return {metric: float(row[metric]) for metric in cls.METRICS}

    @classmethod
    def _normalized_block(cls, row: pd.Series, baseline_row: pd.Series) -> dict[str, float]:
        return {
            metric: cls._normalized_metric_value(row, metric, baseline_row)
            for metric in cls.METRICS
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

    def _metadata_matches_signature(self, state: str, signature: dict[str, Any]) -> bool:
        metadata = self._load_metadata()
        info = metadata.get("entries", {}).get(self._state_key(state))
        return isinstance(info, dict) and info.get("source_signature") == signature

    def _is_valid(
        self,
        metadata: dict[str, Any],
        state: str,
        signature: dict[str, Any],
        curve_path: Path,
    ) -> bool:
        if not self._metadata_matches(metadata):
            return False
        info = metadata.get("entries", {}).get(self._state_key(state))
        if not isinstance(info, dict):
            return False
        if info.get("source_signature") != signature:
            return False
        if info.get("path") != str(curve_path.relative_to(self.cache_dir)).replace("\\", "/"):
            return False
        return curve_path.exists()

    def _default_metadata(self) -> dict[str, Any]:
        metadata = self._metadata_signature()
        metadata["entries"] = {}
        return metadata

    def _metadata_signature(self) -> dict[str, Any]:
        return {
            "cache_version": self.CACHE_VERSION,
            "metrics": list(self.METRICS),
            "baseline_percentage": self.BASELINE_PERCENTAGE,
            "baseline_percentages": list(self.BASELINE_PERCENTAGES),
            "percentages": list(self.PERCENTAGES),
            "default_threshold_metric": self.DEFAULT_THRESHOLD_METRIC,
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
        ids = df["ID"].fillna("").astype(str) if "ID" in df.columns else df.index.astype(str)
        return {
            "rows": int(len(df)),
            "first_id": str(ids.iloc[0]),
            "last_id": str(ids.iloc[-1]),
        }

    def _curve_complete(self, curve_df: pd.DataFrame, max_percentage: int | None = None) -> bool:
        expected = set(self.PERCENTAGES)
        if max_percentage is not None:
            max_percentage = self._configured_percentage_at_or_above(max_percentage)
            expected = {percentage for percentage in expected if percentage <= max_percentage}
        found = set(int(value) for value in curve_df.get("percentage", []))
        return expected.issubset(found)

    @staticmethod
    def _load_existing_rows(curve_path: Path, partial_curve_path: Path) -> dict[int, dict[str, Any]]:
        rows: dict[int, dict[str, Any]] = {}
        for path in (curve_path, partial_curve_path):
            if not path.exists():
                continue
            frame = pd.read_csv(path)
            if frame.empty:
                continue
            for _, row in frame.iterrows():
                rows[int(row["percentage"])] = row.to_dict()
        return rows

    @staticmethod
    def _refresh_existing_row(row: dict[str, Any], n_points: int, sample_path: str | None) -> dict[str, Any]:
        refreshed = dict(row)
        refreshed["n_points"] = int(n_points)
        refreshed["sample_path"] = sample_path
        return refreshed

    @staticmethod
    def _n_keep(n_points: int, percentage: int) -> int:
        return min(n_points, max(1, int(round(n_points * (percentage / 100.0)))))

    @staticmethod
    def _load_or_create_sample(
        *,
        source_df: pd.DataFrame,
        fps_order: np.ndarray,
        n_keep: int,
        sample_path: Path,
    ) -> pd.DataFrame:
        if sample_path.exists():
            return pd.read_csv(sample_path)
        sampled_df = trim_sample_columns(source_df.iloc[fps_order[:n_keep]].reset_index(drop=True))
        sampled_df.to_csv(sample_path, index=False)
        return sampled_df

    @classmethod
    def _configured_percentages(cls, max_percentage: int | None = None) -> tuple[int, ...]:
        if max_percentage is None:
            return cls.PERCENTAGES
        max_percentage = cls._configured_percentage_at_or_above(max_percentage)
        return tuple(percentage for percentage in cls.PERCENTAGES if percentage <= max_percentage)

    @classmethod
    def _configured_percentage_at_or_above(cls, percentage: int) -> int:
        requested = int(max(1, min(100, percentage)))
        for configured in cls.PERCENTAGES:
            if configured >= requested:
                return configured
        return 100

    @classmethod
    def _row_has_metrics(cls, row: dict[str, Any]) -> bool:
        return all(metric in row and pd.notna(row[metric]) for metric in cls.METRICS)

    @classmethod
    def _apply_normalized_metrics(
        cls,
        rows_by_percentage: dict[int, dict[str, Any]],
        baseline_metrics: dict[str, float] | None,
    ) -> None:
        for row in rows_by_percentage.values():
            for metric_name in cls.METRICS:
                baseline_value = 0.0 if baseline_metrics is None else float(baseline_metrics.get(metric_name, 0.0))
                metric_value = float(row.get(metric_name, 0.0))
                row[f"normalized_{metric_name}"] = cls._normalized(metric_value, baseline_value)

    def _load_available_curve(self, state: str) -> pd.DataFrame:
        base_dir = self._state_dir(state)
        rows_by_percentage = self._load_existing_rows(
            base_dir / "curve.csv",
            base_dir / "curve.partial.csv",
        )
        if not rows_by_percentage:
            return pd.DataFrame()
        configured_percentages = set(self.PERCENTAGES)
        rows_by_percentage = {
            percentage: row
            for percentage, row in rows_by_percentage.items()
            if percentage in configured_percentages
        }
        if not rows_by_percentage:
            return pd.DataFrame()
        return pd.DataFrame([rows_by_percentage[pct] for pct in sorted(rows_by_percentage)])

    @staticmethod
    def _curve_has_rows(curve_df: pd.DataFrame, percentages: set[int]) -> bool:
        found = set(int(value) for value in curve_df.get("percentage", []))
        return percentages.issubset(found)

    def _select_curve_row(
        self,
        curve_df: pd.DataFrame,
        error_threshold: float,
        *,
        metric_name: str,
        baseline_row: pd.Series,
    ) -> pd.Series | None:
        threshold = float(max(0.0, error_threshold))
        ordered = curve_df.sort_values("percentage", ascending=True).reset_index(drop=True)
        ordered = ordered[ordered["percentage"] >= int(baseline_row["percentage"])]
        match = ordered[
            ordered.apply(
                lambda row: self._normalized_metric_value(row, metric_name, baseline_row) <= threshold,
                axis=1,
            )
        ]
        if not match.empty:
            return match.iloc[0]
        if ordered.empty:
            return None
        return ordered.iloc[-1]

    @classmethod
    def _requested_baseline_percentage(cls, baseline_percentage: int | None) -> int:
        if baseline_percentage is None:
            return cls.BASELINE_PERCENTAGE
        requested = int(max(1, min(100, baseline_percentage)))
        if requested in cls.BASELINE_PERCENTAGES:
            return requested
        return cls.BASELINE_PERCENTAGE

    @classmethod
    def _baseline_row(cls, curve_df: pd.DataFrame, baseline_percentage: int) -> pd.Series | None:
        if curve_df.empty:
            return None
        match = curve_df[curve_df["percentage"] == baseline_percentage]
        if not match.empty:
            return match.iloc[0]
        return None

    @classmethod
    def _normalized_metric_value(cls, row: pd.Series, metric_name: str, baseline_row: pd.Series) -> float:
        return cls._normalized(float(row.get(metric_name, 0.0)), float(baseline_row.get(metric_name, 0.0)))

    @classmethod
    def _metric_name(cls, metric_name: str) -> str:
        metric = (metric_name or cls.DEFAULT_THRESHOLD_METRIC).strip().lower()
        if metric not in cls.METRICS:
            return cls.DEFAULT_THRESHOLD_METRIC
        return metric

    def _state_dir(self, state: str) -> Path:
        return self.cache_dir / "states" / self._sanitize(state)

    def _state_key(self, state: str) -> str:
        return f"states/{self._sanitize(state)}"

    @staticmethod
    def _sanitize(value: str) -> str:
        safe = str(value).replace("\\", "_").replace("/", "_").replace(":", "_")
        return safe.replace(" ", "_")


