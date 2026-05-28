"""Boundary fetching and caching helpers for the OpenLayers adaptive app."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from adaptive_visualization.core.state_metadata import STATE_CODE_TO_FIPS, STATE_CODE_TO_NAME


US_ATLAS_STATES_URL = "https://cdn.jsdelivr.net/npm/us-atlas@3/states-10m.json"
US_ATLAS_NATION_URL = "https://cdn.jsdelivr.net/npm/us-atlas@3/nation-10m.json"
US_ATLAS_COUNTIES_URL = "https://cdn.jsdelivr.net/npm/us-atlas@3/counties-10m.json"


class BoundaryService:
    """Loads state, county, and city boundaries with a disk cache."""

    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.us_cache_dir = self.cache_dir / "us"
        self.city_cache_dir = self.cache_dir / "cities"
        self.county_cache_dir = self.cache_dir / "counties"
        self.us_cache_dir.mkdir(parents=True, exist_ok=True)
        self.city_cache_dir.mkdir(parents=True, exist_ok=True)
        self.county_cache_dir.mkdir(parents=True, exist_ok=True)

    def get_us_states_topology(self) -> dict[str, Any]:
        return self._load_or_download_json(
            self.us_cache_dir / "states-10m.json",
            US_ATLAS_STATES_URL,
        )

    def get_us_nation_topology(self) -> dict[str, Any]:
        return self._load_or_download_json(
            self.us_cache_dir / "nation-10m.json",
            US_ATLAS_NATION_URL,
        )

    def get_us_counties_topology(self) -> dict[str, Any]:
        return self._load_or_download_json(
            self.us_cache_dir / "counties-10m.json",
            US_ATLAS_COUNTIES_URL,
        )

    def get_state_bounds(self, state_code: str) -> tuple[float, float, float, float] | None:
        state_feature = self.get_state_boundary(state_code)
        return self.get_feature_bounds(state_feature)

    def get_state_boundary(self, state_code: str) -> dict[str, Any] | None:
        state_fips = STATE_CODE_TO_FIPS.get(state_code)
        if not state_fips:
            return None

        topology = self.get_us_states_topology()
        states_object = topology.get("objects", {}).get("states", {})
        geometries = states_object.get("geometries", [])

        for geometry in geometries:
            geometry_fips = str(geometry.get("id", "")).zfill(2)
            if geometry_fips != state_fips:
                continue
            return {
                "type": "Feature",
                "properties": {
                    "state": state_code,
                    "source": "us-atlas-states",
                    "state_fips": state_fips,
                },
                "geometry": self._topology_geometry_to_geojson(geometry, topology),
            }
        return None

    def get_city_boundary(
        self,
        state_code: str,
        city_name: str,
        city_df: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        cache_path = self.city_cache_dir / f"{state_code}__{self._sanitize(city_name)}.geojson"
        if cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))

        candidates = [f"{city_name}, {STATE_CODE_TO_NAME.get(state_code, state_code)}, United States"]
        boundary = self._fetch_boundary_candidates(candidates)
        if boundary is None:
            boundary = self._fallback_boundary(city_df, source="fallback-city")

        cache_path.write_text(json.dumps(boundary), encoding="utf-8")
        return boundary

    def get_county_boundary(
        self,
        state_code: str,
        county_name: str,
        county_df: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        cache_path = self.county_cache_dir / f"{state_code}__{self._sanitize(county_name)}.geojson"

        atlas_boundary = self._county_boundary_from_us_atlas(state_code, county_name)
        if atlas_boundary is not None:
            cache_path.write_text(json.dumps(atlas_boundary), encoding="utf-8")
            return atlas_boundary

        if cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))

        boundary = self._fetch_boundary_candidates(
            self._county_query_candidates(state_code, county_name)
        )
        if boundary is None:
            boundary = self._fallback_boundary(county_df, source="fallback-county")

        cache_path.write_text(json.dumps(boundary), encoding="utf-8")
        return boundary

    def filter_points_to_boundary(
        self,
        df: pd.DataFrame,
        feature: dict[str, Any] | None,
    ) -> pd.DataFrame:
        """Return only points that fall inside the given GeoJSON feature."""
        if df is None or df.empty or not feature:
            return df

        valid_df = df.dropna(subset=["Start_Lng", "Start_Lat"])
        kept_index = [
            row_index
            for row_index, lng, lat in valid_df[["Start_Lng", "Start_Lat"]].itertuples()
            if self.point_in_feature(float(lng), float(lat), feature)
        ]
        return df.loc[kept_index].reset_index(drop=True)

    def get_feature_bounds(self, feature: dict[str, Any] | None) -> tuple[float, float, float, float] | None:
        """Return (min_lng, min_lat, max_lng, max_lat) for a GeoJSON feature."""
        if not feature:
            return None
        geometry = feature.get("geometry") if feature.get("type") == "Feature" else feature
        if not geometry:
            return None
        return self._geometry_bounds(geometry)

    @classmethod
    def point_in_feature(cls, lng: float, lat: float, feature: dict[str, Any]) -> bool:
        geometry = feature.get("geometry") if feature.get("type") == "Feature" else feature
        if not geometry:
            return True
        return cls.point_in_geometry(lng, lat, geometry)

    @classmethod
    def point_in_geometry(cls, lng: float, lat: float, geometry: dict[str, Any]) -> bool:
        geometry_type = geometry.get("type")
        coordinates = geometry.get("coordinates", [])

        if geometry_type == "Polygon":
            return cls._point_in_polygon_with_holes(lng, lat, coordinates)
        if geometry_type == "MultiPolygon":
            return any(
                cls._point_in_polygon_with_holes(lng, lat, polygon)
                for polygon in coordinates
            )
        return True

    @classmethod
    def _point_in_polygon_with_holes(
        cls,
        lng: float,
        lat: float,
        polygon_coordinates: list,
    ) -> bool:
        if not polygon_coordinates:
            return False
        if not cls._point_in_ring(lng, lat, polygon_coordinates[0]):
            return False
        for hole in polygon_coordinates[1:]:
            if cls._point_in_ring(lng, lat, hole):
                return False
        return True

    @staticmethod
    def _point_in_ring(lng: float, lat: float, ring: list) -> bool:
        inside = False
        if len(ring) < 3:
            return False

        prev_lng, prev_lat = ring[-1]
        for curr_lng, curr_lat in ring:
            intersects = ((curr_lat > lat) != (prev_lat > lat)) and (
                lng < ((prev_lng - curr_lng) * (lat - curr_lat) / ((prev_lat - curr_lat) or 1e-12) + curr_lng)
            )
            if intersects:
                inside = not inside
            prev_lng, prev_lat = curr_lng, curr_lat
        return inside

    def _fetch_boundary_candidates(self, candidates: list[str]) -> dict[str, Any] | None:
        for query in candidates:
            boundary = self._fetch_boundary_query(query)
            if boundary is not None:
                return boundary
        return None

    def _fetch_boundary_query(self, query: str) -> dict[str, Any] | None:
        params = urlencode(
            {
                "q": query,
                "polygon_geojson": 1,
                "format": "jsonv2",
                "limit": 5,
            }
        )
        url = f"https://nominatim.openstreetmap.org/search?{params}"
        request = Request(
            url,
            headers={"User-Agent": "AdaptiveOpenLayers/1.0 (research visualization)"},
        )
        try:
            with urlopen(request, timeout=30) as response:
                results = json.loads(response.read().decode("utf-8"))
        except URLError:
            return None

        preferred = None
        for row in results:
            geojson = row.get("geojson")
            if row.get("osm_type") == "relation" and geojson and geojson.get("type") in {"Polygon", "MultiPolygon"}:
                preferred = geojson
                break

        if preferred is None:
            for row in results:
                geojson = row.get("geojson")
                if geojson and geojson.get("type") in {"Polygon", "MultiPolygon"}:
                    preferred = geojson
                    break

        if preferred is None:
            return None

        return {
            "type": "Feature",
            "properties": {
                "query": query,
                "source": "nominatim",
            },
            "geometry": preferred,
        }

    def _load_or_download_json(self, cache_path: Path, url: str) -> dict[str, Any]:
        if cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))

        request = Request(
            url,
            headers={"User-Agent": "AdaptiveOpenLayers/1.0 (research visualization)"},
        )
        with urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))

        cache_path.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    def _county_boundary_from_us_atlas(self, state_code: str, county_name: str) -> dict[str, Any] | None:
        state_fips = STATE_CODE_TO_FIPS.get(state_code)
        if not state_fips:
            return None

        topology = self.get_us_counties_topology()
        counties_object = topology.get("objects", {}).get("counties", {})
        geometries = counties_object.get("geometries", [])
        target_name = self._normalize_county_name(county_name)

        for geometry in geometries:
            county_fips = str(geometry.get("id", "")).zfill(5)
            if not county_fips.startswith(state_fips):
                continue

            name = (geometry.get("properties") or {}).get("name", "")
            if self._normalize_county_name(name) != target_name:
                continue

            return {
                "type": "Feature",
                "properties": {
                    "name": name,
                    "source": "us-atlas-counties",
                    "county_fips": county_fips,
                },
                "geometry": self._topology_geometry_to_geojson(geometry, topology),
            }
        return None

    def _topology_geometry_to_geojson(
        self,
        geometry: dict[str, Any],
        topology: dict[str, Any],
    ) -> dict[str, Any]:
        geometry_type = geometry.get("type")
        arcs = geometry.get("arcs", [])

        if geometry_type == "Polygon":
            return {
                "type": "Polygon",
                "coordinates": [self._topology_ring_to_coordinates(ring, topology) for ring in arcs],
            }
        if geometry_type == "MultiPolygon":
            return {
                "type": "MultiPolygon",
                "coordinates": [
                    [self._topology_ring_to_coordinates(ring, topology) for ring in polygon]
                    for polygon in arcs
                ],
            }
        raise ValueError(f"Unsupported topology geometry type: {geometry_type}")

    def _topology_ring_to_coordinates(
        self,
        ring_arc_indexes: list[int],
        topology: dict[str, Any],
    ) -> list[list[float]]:
        coordinates: list[list[float]] = []
        for position, arc_index in enumerate(ring_arc_indexes):
            arc_coordinates = self._topology_arc_coordinates(arc_index, topology)
            if position > 0:
                arc_coordinates = arc_coordinates[1:]
            coordinates.extend(arc_coordinates)
        if coordinates and coordinates[0] != coordinates[-1]:
            coordinates.append(coordinates[0])
        return coordinates

    def _topology_arc_coordinates(
        self,
        arc_index: int,
        topology: dict[str, Any],
    ) -> list[list[float]]:
        arcs = topology.get("arcs", [])
        transform = topology.get("transform", {})
        scale_x, scale_y = transform.get("scale", [1.0, 1.0])
        translate_x, translate_y = transform.get("translate", [0.0, 0.0])

        reverse = arc_index < 0
        resolved_index = ~arc_index if reverse else arc_index
        raw_arc = arcs[resolved_index]

        x = 0
        y = 0
        coordinates: list[list[float]] = []
        for dx, dy in raw_arc:
            x += dx
            y += dy
            coordinates.append([
                x * scale_x + translate_x,
                y * scale_y + translate_y,
            ])

        if reverse:
            coordinates.reverse()
        return coordinates

    @classmethod
    def _geometry_bounds(cls, geometry: dict[str, Any]) -> tuple[float, float, float, float] | None:
        coordinates = geometry.get("coordinates", []) if geometry else []
        points: list[list[float]] = []
        cls._collect_coordinate_positions(coordinates, points)
        if not points:
            return None

        lngs = [float(point[0]) for point in points]
        lats = [float(point[1]) for point in points]
        return (min(lngs), min(lats), max(lngs), max(lats))

    @classmethod
    def _collect_coordinate_positions(cls, node: Any, out: list[list[float]]) -> None:
        if not isinstance(node, list) or not node:
            return

        first = node[0]
        if isinstance(first, (int, float)) and len(node) >= 2:
            out.append([float(node[0]), float(node[1])])
            return

        for child in node:
            cls._collect_coordinate_positions(child, out)

    def _county_query_candidates(self, state_code: str, county_name: str) -> list[str]:
        state_name = STATE_CODE_TO_NAME.get(state_code, state_code)
        base = county_name.strip()
        candidates = [f"{base}, {state_name}, United States"]

        lowered = base.lower()
        has_suffix = any(
            suffix in lowered
            for suffix in [" county", " parish", " borough", " census area", " municipality", " city and borough"]
        )
        if not has_suffix:
            candidates.append(f"{base} County, {state_name}, United States")
            if state_code == "LA":
                candidates.append(f"{base} Parish, {state_name}, United States")
            if state_code == "AK":
                candidates.append(f"{base} Borough, {state_name}, United States")
                candidates.append(f"{base} Census Area, {state_name}, United States")

        return candidates

    def _fallback_boundary(self, df: pd.DataFrame | None, source: str) -> dict[str, Any]:
        if df is None or df.empty:
            return self._bbox_feature(-124.0, 24.0, -66.0, 49.0, source=f"{source}-empty")

        points = df[["Start_Lng", "Start_Lat"]].dropna().to_numpy(dtype=float)
        if len(points) < 3:
            min_lng, min_lat = np.min(points, axis=0)
            max_lng, max_lat = np.max(points, axis=0)
            return self._bbox_feature(min_lng, min_lat, max_lng, max_lat, source=f"{source}-bbox")

        hull = self._convex_hull(points)
        if len(hull) < 3:
            min_lng, min_lat = np.min(points, axis=0)
            max_lng, max_lat = np.max(points, axis=0)
            return self._bbox_feature(min_lng, min_lat, max_lng, max_lat, source=f"{source}-bbox")

        padded = self._pad_ring(hull)
        coordinates = [[list(coord) for coord in padded] + [list(padded[0])]]
        return {
            "type": "Feature",
            "properties": {"source": f"{source}-convex-hull"},
            "geometry": {
                "type": "Polygon",
                "coordinates": coordinates,
            },
        }

    @staticmethod
    def _bbox_feature(min_lng: float, min_lat: float, max_lng: float, max_lat: float, source: str) -> dict[str, Any]:
        lng_pad = max((max_lng - min_lng) * 0.08, 0.01)
        lat_pad = max((max_lat - min_lat) * 0.08, 0.01)
        min_lng -= lng_pad
        max_lng += lng_pad
        min_lat -= lat_pad
        max_lat += lat_pad
        coordinates = [[
            [min_lng, min_lat],
            [max_lng, min_lat],
            [max_lng, max_lat],
            [min_lng, max_lat],
            [min_lng, min_lat],
        ]]
        return {
            "type": "Feature",
            "properties": {"source": source},
            "geometry": {"type": "Polygon", "coordinates": coordinates},
        }

    @staticmethod
    def _convex_hull(points: np.ndarray) -> list[tuple[float, float]]:
        unique_points = sorted({(float(x), float(y)) for x, y in points})
        if len(unique_points) <= 1:
            return unique_points

        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        lower: list[tuple[float, float]] = []
        for point in unique_points:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
                lower.pop()
            lower.append(point)

        upper: list[tuple[float, float]] = []
        for point in reversed(unique_points):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
                upper.pop()
            upper.append(point)

        return lower[:-1] + upper[:-1]

    @staticmethod
    def _pad_ring(ring: list[tuple[float, float]]) -> list[tuple[float, float]]:
        arr = np.asarray(ring, dtype=float)
        center = arr.mean(axis=0)
        span = np.ptp(arr, axis=0)
        scale = np.maximum(span * 0.04, np.array([0.005, 0.005]))
        adjusted = []
        for lng, lat in arr:
            direction = np.array([lng, lat]) - center
            norm = np.linalg.norm(direction)
            if norm < 1e-9:
                adjusted.append((float(lng), float(lat)))
                continue
            delta = direction / norm * scale
            adjusted.append((float(lng + delta[0]), float(lat + delta[1])))
        return adjusted

    @staticmethod
    def _normalize_county_name(value: str) -> str:
        normalized = value.strip().lower()
        for suffix in [
            " county",
            " parish",
            " borough",
            " census area",
            " municipality",
            " city and borough",
        ]:
            if normalized.endswith(suffix):
                normalized = normalized[: -len(suffix)]
                break
        return " ".join(normalized.replace(".", " ").split())

    @staticmethod
    def _sanitize(value: str) -> str:
        safe = value.replace("\\", "_").replace("/", "_").replace(":", "_")
        return safe.replace(" ", "_")



