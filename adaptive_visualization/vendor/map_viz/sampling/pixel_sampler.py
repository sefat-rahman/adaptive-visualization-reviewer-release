"""
PixelSampler - keeps map-view points without screen-space overlap.

The sampler approximates the rendered dashboard view as a fixed pixel grid:
each accepted point claims screen-space area, and later points whose projected
marker disks would intersect with an accepted marker are skipped.

Country view is modeled after the OpenLayers map used in the current
`adaptive_openlayers` app: a contiguous-US Web-Mercator-style viewport fitted
into the left map panel of the desktop layout. State view keeps the existing
local Mercator-style viewport logic.
"""

import numpy as np

from adaptive_visualization.vendor.map_viz.sampling.base import BaseSampler


class PixelSampler(BaseSampler):
    """
    Screen-space occupancy sampler for the map views.

    Points are projected into a fixed viewport and inserted one-by-one. A point
    is kept only if its marker would not intersect any previously accepted
    marker in pixel space.
    """

    # OpenLayers desktop country-view map panel approximation.
    COUNTRY_VIEWPORT_WIDTH_PX = 1248
    COUNTRY_VIEWPORT_HEIGHT_PX = 896

    # Default state-view approximation used by the legacy pipeline.
    STATE_VIEWPORT_WIDTH_PX = 780
    STATE_VIEWPORT_HEIGHT_PX = 590
    STATE_VIEWPORT_PADDING_PX = (32.0, 32.0, 32.0, 32.0)
    COUNTY_VIEWPORT_WIDTH_PX = 1248
    COUNTY_VIEWPORT_HEIGHT_PX = 896
    COUNTY_VIEWPORT_PADDING_PX = (34.0, 34.0, 34.0, 34.0)

    # Match the rendered marker footprint used in adaptive_openlayers/static/app.js.
    COUNTRY_MARKER_RADIUS_PX = 3.2
    COUNTRY_MARKER_STROKE_WIDTH_PX = 0.4
    STATE_MARKER_RADIUS_PX = 4.2
    STATE_MARKER_STROKE_WIDTH_PX = 1.2
    COUNTY_MARKER_RADIUS_PX = 4.2
    COUNTY_MARKER_STROKE_WIDTH_PX = 1.2

    COUNTRY_LAT_BOUNDS = (24.0, 49.5)
    COUNTRY_LON_BOUNDS = (-125.0, -66.0)
    STATE_PADDING_RATIO = 0.28
    STATE_MIN_SPAN_DEG = 1.5

    def __init__(self):
        self._country_min_center_distance_px = 2.0 * (
            self.COUNTRY_MARKER_RADIUS_PX + (self.COUNTRY_MARKER_STROKE_WIDTH_PX / 2.0)
        )
        self._state_min_center_distance_px = 2.0 * (
            self.STATE_MARKER_RADIUS_PX + (self.STATE_MARKER_STROKE_WIDTH_PX / 2.0)
        )
        self._county_min_center_distance_px = 2.0 * (
            self.COUNTY_MARKER_RADIUS_PX + (self.COUNTY_MARKER_STROKE_WIDTH_PX / 2.0)
        )
        self._country_mercator_y_bounds = self._mercator_latitudes(
            np.asarray(self.COUNTRY_LAT_BOUNDS, dtype=float)
        )

    def sample(self, points: np.ndarray, n_keep: int) -> np.ndarray:
        """
        Select as many non-overlapping country-view points as possible.

        Args:
            points: (N, 2) array of [lat, lng] coordinates.
            n_keep: Upper bound on how many points may be kept.

        Returns:
            Integer indices of accepted points in deterministic source order.
        """
        if len(points) == 0 or n_keep <= 0:
            return np.empty(0, dtype=int)

        pixel_xy = self._to_country_pixel_coordinates(points)
        return self._sample_pixel_positions(
            pixel_xy=pixel_xy,
            n_keep=n_keep,
            min_center_distance_px=self._country_min_center_distance_px,
        )

    def sample_country_openlayers(
        self,
        points: np.ndarray,
        viewport_width_px: float,
        viewport_height_px: float,
        padding_px: tuple[float, float, float, float],
        n_keep: int,
    ) -> np.ndarray:
        """
        Select non-overlapping country-view points using the OpenLayers fit-to-extent model.
        """
        if len(points) == 0 or n_keep <= 0:
            return np.empty(0, dtype=int)

        pixel_xy = self._to_fitted_state_pixel_coordinates(
            points=points,
            lat_range=[float(self.COUNTRY_LAT_BOUNDS[0]), float(self.COUNTRY_LAT_BOUNDS[1])],
            lon_range=[float(self.COUNTRY_LON_BOUNDS[0]), float(self.COUNTRY_LON_BOUNDS[1])],
            viewport_width_px=float(viewport_width_px),
            viewport_height_px=float(viewport_height_px),
            padding_px=padding_px,
        )
        return self._sample_pixel_positions(
            pixel_xy=pixel_xy,
            n_keep=n_keep,
            min_center_distance_px=self._country_min_center_distance_px,
        )

    def sample_state(self, points: np.ndarray, n_keep: int) -> np.ndarray:
        """
        Select as many non-overlapping state-view points as possible.

        Args:
            points: (N, 2) array of [lat, lng] coordinates.
            n_keep: Upper bound on how many points may be kept.

        Returns:
            Integer indices of accepted points in deterministic source order.
        """
        if len(points) == 0 or n_keep <= 0:
            return np.empty(0, dtype=int)

        lat_range, lon_range = self._compute_padded_ranges(
            points,
            padding_ratio=self.STATE_PADDING_RATIO,
            min_span=self.STATE_MIN_SPAN_DEG,
        )
        pixel_xy = self._to_state_pixel_coordinates(points, lat_range, lon_range)
        return self._sample_pixel_positions(
            pixel_xy=pixel_xy,
            n_keep=n_keep,
            min_center_distance_px=self._state_min_center_distance_px,
        )

    def sample_state_openlayers(
        self,
        points: np.ndarray,
        state_lat_range: list[float],
        state_lon_range: list[float],
        n_keep: int,
        viewport_width_px: float | None = None,
        viewport_height_px: float | None = None,
        padding_px: tuple[float, float, float, float] | None = None,
    ) -> np.ndarray:
        """
        Select non-overlapping state-view points using the OpenLayers fit-to-extent model.

        This matches the current adaptive_openlayers state view more closely than
        the legacy point-extent approximation by fitting the full state boundary
        extent into a padded viewport with preserved aspect ratio.
        """
        if len(points) == 0 or n_keep <= 0:
            return np.empty(0, dtype=int)

        resolved_width = float(viewport_width_px or self.STATE_VIEWPORT_WIDTH_PX)
        resolved_height = float(viewport_height_px or self.STATE_VIEWPORT_HEIGHT_PX)
        resolved_padding = padding_px or self.STATE_VIEWPORT_PADDING_PX

        pixel_xy = self._to_fitted_state_pixel_coordinates(
            points=points,
            lat_range=state_lat_range,
            lon_range=state_lon_range,
            viewport_width_px=resolved_width,
            viewport_height_px=resolved_height,
            padding_px=resolved_padding,
        )
        return self._sample_pixel_positions(
            pixel_xy=pixel_xy,
            n_keep=n_keep,
            min_center_distance_px=self._state_min_center_distance_px,
        )

    def sample_county_openlayers(
        self,
        points: np.ndarray,
        county_lat_range: list[float],
        county_lon_range: list[float],
        n_keep: int,
        viewport_width_px: float | None = None,
        viewport_height_px: float | None = None,
        padding_px: tuple[float, float, float, float] | None = None,
    ) -> np.ndarray:
        """
        Select non-overlapping county-view points using the OpenLayers fit-to-extent model.
        """
        if len(points) == 0 or n_keep <= 0:
            return np.empty(0, dtype=int)

        resolved_width = float(viewport_width_px or self.COUNTY_VIEWPORT_WIDTH_PX)
        resolved_height = float(viewport_height_px or self.COUNTY_VIEWPORT_HEIGHT_PX)
        resolved_padding = padding_px or self.COUNTY_VIEWPORT_PADDING_PX

        pixel_xy = self._to_fitted_state_pixel_coordinates(
            points=points,
            lat_range=county_lat_range,
            lon_range=county_lon_range,
            viewport_width_px=resolved_width,
            viewport_height_px=resolved_height,
            padding_px=resolved_padding,
        )
        return self._sample_pixel_positions(
            pixel_xy=pixel_xy,
            n_keep=n_keep,
            min_center_distance_px=self._county_min_center_distance_px,
        )


    def _sample_pixel_positions(
        self,
        pixel_xy: np.ndarray,
        n_keep: int,
        min_center_distance_px: float,
    ) -> np.ndarray:
        """Insert points one-by-one using exact circle-distance checks."""
        selected = []
        min_distance_sq = float(min_center_distance_px) ** 2
        cell_size = max(float(min_center_distance_px), 1.0)
        buckets: dict[tuple[int, int], list[tuple[float, float]]] = {}

        for idx, (px, py) in enumerate(pixel_xy):
            if len(selected) >= n_keep:
                break

            cell_x = int(px // cell_size)
            cell_y = int(py // cell_size)
            overlaps = False

            for neighbor_x in range(cell_x - 1, cell_x + 2):
                for neighbor_y in range(cell_y - 1, cell_y + 2):
                    for other_px, other_py in buckets.get((neighbor_x, neighbor_y), []):
                        dx = px - other_px
                        dy = py - other_py
                        if (dx * dx) + (dy * dy) < min_distance_sq:
                            overlaps = True
                            break
                    if overlaps:
                        break
                if overlaps:
                    break

            if overlaps:
                continue

            selected.append(idx)
            buckets.setdefault((cell_x, cell_y), []).append((float(px), float(py)))

        return np.asarray(selected, dtype=int)

    def _to_country_pixel_coordinates(self, points: np.ndarray) -> np.ndarray:
        """Map country-view lat/lng coordinates into the OpenLayers pixel grid."""
        lats = points[:, 0].astype(float)
        lons = points[:, 1].astype(float)

        lon_min, lon_max = self.COUNTRY_LON_BOUNDS
        lon_span = max(lon_max - lon_min, 1e-9)
        x_norm = (lons - lon_min) / lon_span

        mercator_y = self._mercator_latitudes(lats)
        y_min = float(np.min(self._country_mercator_y_bounds))
        y_max = float(np.max(self._country_mercator_y_bounds))
        y_span = max(y_max - y_min, 1e-9)
        y_norm = (mercator_y - y_min) / y_span

        x_px = np.clip(
            x_norm * (self.COUNTRY_VIEWPORT_WIDTH_PX - 1),
            0.0,
            float(self.COUNTRY_VIEWPORT_WIDTH_PX - 1),
        )
        y_px = np.clip(
            (1.0 - y_norm) * (self.COUNTRY_VIEWPORT_HEIGHT_PX - 1),
            0.0,
            float(self.COUNTRY_VIEWPORT_HEIGHT_PX - 1),
        )
        return np.column_stack([x_px, y_px])

    def _to_fitted_state_pixel_coordinates(
        self,
        points: np.ndarray,
        lat_range: list[float],
        lon_range: list[float],
        viewport_width_px: float,
        viewport_height_px: float,
        padding_px: tuple[float, float, float, float],
    ) -> np.ndarray:
        """Map state-view coordinates using an OpenLayers-style fit-to-extent transform."""
        lats = points[:, 0].astype(float)
        lons = points[:, 1].astype(float)

        x_values = self._mercator_longitudes(lons)
        y_values = self._mercator_latitudes(lats)

        x_bounds = self._mercator_longitudes(np.asarray(lon_range, dtype=float))
        y_bounds = self._mercator_latitudes(np.asarray(lat_range, dtype=float))

        x_min = float(np.min(x_bounds))
        x_max = float(np.max(x_bounds))
        y_min = float(np.min(y_bounds))
        y_max = float(np.max(y_bounds))

        extent_width = max(x_max - x_min, 1e-9)
        extent_height = max(y_max - y_min, 1e-9)

        top, right, bottom, left = [float(value) for value in padding_px]
        inner_width = max(viewport_width_px - left - right, 1.0)
        inner_height = max(viewport_height_px - top - bottom, 1.0)

        scale = min(inner_width / extent_width, inner_height / extent_height)
        scaled_width = extent_width * scale
        scaled_height = extent_height * scale

        offset_x = left + ((inner_width - scaled_width) / 2.0)
        offset_y = top + ((inner_height - scaled_height) / 2.0)

        x_px = offset_x + ((x_values - x_min) * scale)
        y_px = offset_y + ((y_max - y_values) * scale)

        return np.column_stack([x_px, y_px])

    def _to_state_pixel_coordinates(
        self,
        points: np.ndarray,
        lat_range: list[float],
        lon_range: list[float],
    ) -> np.ndarray:
        """Map latitude/longitude coordinates into the state-view pixel grid."""
        lats = points[:, 0].astype(float)
        lons = points[:, 1].astype(float)

        lon_min, lon_max = lon_range
        lon_span = max(lon_max - lon_min, 1e-9)
        x_norm = (lons - lon_min) / lon_span

        mercator_y = self._mercator_latitudes(lats)
        mercator_bounds = self._mercator_latitudes(np.asarray(lat_range, dtype=float))
        y_min = float(np.min(mercator_bounds))
        y_max = float(np.max(mercator_bounds))
        y_span = max(y_max - y_min, 1e-9)
        y_norm = (mercator_y - y_min) / y_span

        x_px = np.clip(
            x_norm * (self.STATE_VIEWPORT_WIDTH_PX - 1),
            0.0,
            float(self.STATE_VIEWPORT_WIDTH_PX - 1),
        )
        y_px = np.clip(
            (1.0 - y_norm) * (self.STATE_VIEWPORT_HEIGHT_PX - 1),
            0.0,
            float(self.STATE_VIEWPORT_HEIGHT_PX - 1),
        )
        return np.column_stack([x_px, y_px])

    @staticmethod
    def _mercator_longitudes(longitudes_deg: np.ndarray) -> np.ndarray:
        """Convert longitudes to Mercator x coordinates in radians-scale units."""
        return np.radians(longitudes_deg.astype(float))

    @staticmethod
    def _mercator_latitudes(latitudes_deg: np.ndarray) -> np.ndarray:
        """Convert latitudes to Mercator y coordinates."""
        clipped = np.clip(latitudes_deg.astype(float), -85.0, 85.0)
        radians = np.radians(clipped)
        return np.log(np.tan((np.pi / 4.0) + (radians / 2.0)))

    @staticmethod
    def _compute_padded_ranges(
        points: np.ndarray,
        padding_ratio: float,
        min_span: float,
    ) -> tuple[list[float], list[float]]:
        """Compute symmetric lat/lon padding for the state viewport."""
        lats = points[:, 0].astype(float)
        lons = points[:, 1].astype(float)

        lat_min, lat_max = float(np.min(lats)), float(np.max(lats))
        lon_min, lon_max = float(np.min(lons)), float(np.max(lons))

        lat_center = (lat_min + lat_max) / 2.0
        lon_center = (lon_min + lon_max) / 2.0

        lat_span = max(lat_max - lat_min, min_span)
        lon_span = max(lon_max - lon_min, min_span)

        lat_half = lat_span * (0.5 + padding_ratio)
        lon_half = lon_span * (0.5 + padding_ratio)

        return (
            [lat_center - lat_half, lat_center + lat_half],
            [lon_center - lon_half, lon_center + lon_half],
        )
