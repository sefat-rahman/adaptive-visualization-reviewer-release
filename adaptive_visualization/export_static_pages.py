"""Export a GitHub Pages viewer backed by precomputed API snapshots.

The Flask dashboard remains the full local app. This script creates
`adaptive_visualization/pages/`, a static viewer that intercepts the dashboard's
API calls and serves selected precomputed JSON snapshots from disk.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from adaptive_visualization.app import app, configure, _repo
from adaptive_visualization.paths import DEFAULT_DATASET_DIR, MODULE_DIR, resolve_data_dir


VIEWPORT_KEYS = {
    "viewport_width_px",
    "viewport_height_px",
    "padding_top_px",
    "padding_right_px",
    "padding_bottom_px",
    "padding_left_px",
}

RANDOM_RETAIN_PERCENTAGES = [3, 4, 26, 50]
FPS_THRESHOLDS = [0.25, 0.50, 0.75]
DEFAULT_FPS_BASELINE = "county_20pct_distance"
FPS_DISTANCE_BASELINES = [
    "country_1pct_distance",
    "county_10pct_distance",
    "county_20pct_distance",
]
FPS_BASELINES_BY_ZOOM = {
    "country": FPS_DISTANCE_BASELINES,
    "state": FPS_DISTANCE_BASELINES,
    "county": FPS_DISTANCE_BASELINES,
}
ANALYSIS_PROPERTIES = ["statistical", "density", "topological"]

STATIC_API_JS = r"""
(function () {
  'use strict';

  const originalFetch = window.fetch.bind(window);
  const manifestPromise = originalFetch('data/manifest.json').then((response) => response.json());
  const VIEWPORT_KEYS = new Set([
    'viewport_width_px',
    'viewport_height_px',
    'padding_top_px',
    'padding_right_px',
    'padding_bottom_px',
    'padding_left_px',
  ]);
  const FPS_BASELINES = new Set([
    'country_1pct_distance',
    'county_10pct_distance',
    'county_20pct_distance',
  ]);

  function jsonResponse(payload, status = 200) {
    return new Response(JSON.stringify(payload), {
      status,
      headers: { 'Content-Type': 'application/json' },
    });
  }

  function buildKey(pathname, pairs) {
    const params = new URLSearchParams();
    pairs.forEach(([key, value]) => params.set(key, value == null ? '' : String(value)));
    const query = params.toString();
    return query ? `${pathname}?${query}` : pathname;
  }

  function normalizeThreshold(value) {
    const numberValue = Number(value);
    return Number.isFinite(numberValue) ? numberValue.toFixed(2) : '0.50';
  }

  function normalizeBaseline(value) {
    return FPS_BASELINES.has(value) ? value : 'county_20pct_distance';
  }

  function normalizeApiKey(url) {
    const pathname = url.pathname.replace(/\/+$/, '') || '/';
    const params = new URLSearchParams(url.search);
    VIEWPORT_KEYS.forEach((key) => params.delete(key));

    if (pathname === '/api/meta') {
      return pathname;
    }
    if (pathname === '/api/boundaries/us/nation'
      || pathname === '/api/boundaries/us/states'
      || pathname === '/api/boundaries/us/counties') {
      return pathname;
    }
    if (pathname === '/api/labels/counties') {
      return buildKey(pathname, [['state', params.get('state') || '']]);
    }
    if (pathname === '/api/boundaries/county') {
      return buildKey(pathname, [
        ['state', params.get('state') || ''],
        ['county', params.get('county') || ''],
      ]);
    }
    if (pathname === '/api/fps-threshold/status') {
      return buildKey(pathname, [
        ['zoom', params.get('zoom') || 'state'],
        ['state', params.get('state') || ''],
        ['county', params.get('county') || ''],
      ]);
    }
    if (pathname === '/api/data') {
      const method = params.get('method') || 'pixel';
      const pairs = [
        ['zoom', params.get('zoom') || 'country'],
        ['method', method],
        ['state', params.get('state') || ''],
        ['county', params.get('county') || ''],
      ];
      if (method === 'random') {
        pairs.push(['retain_pct', params.get('retain_pct') || '50']);
      }
      if (method === 'fps_threshold') {
        pairs.push(['error_threshold', normalizeThreshold(params.get('error_threshold'))]);
        pairs.push(['topology_baseline_pct', normalizeBaseline(params.get('topology_baseline_pct'))]);
      }
      return buildKey(pathname, pairs);
    }
    if (pathname === '/api/analysis') {
      const method = params.get('method') || 'pixel';
      const pairs = [
        ['property', params.get('property') || 'topological'],
        ['zoom', params.get('zoom') || 'country'],
        ['method', method],
        ['state', params.get('state') || ''],
        ['county', params.get('county') || ''],
      ];
      if (method === 'random') {
        pairs.push(['retain_pct', params.get('retain_pct') || '50']);
      }
      if (method === 'fps_threshold') {
        pairs.push(['error_threshold', normalizeThreshold(params.get('error_threshold'))]);
        pairs.push(['topology_baseline_pct', normalizeBaseline(params.get('topology_baseline_pct'))]);
      }
      return buildKey(pathname, pairs);
    }
    return pathname;
  }

  function staticUnavailablePayload(url) {
    const params = new URLSearchParams(url.search);
    return {
      available: false,
      static_demo_unavailable: true,
      generation_supported: false,
      generation_status: 'static_demo',
      exact_count: 0,
      message: 'This GitHub Pages demo includes only selected precomputed snapshots. Clone the repository and run the local dashboard to generate additional views.',
      status: {
        zoom: params.get('zoom') || '',
        state: params.get('state') || '',
        county: params.get('county') || '',
        exact_count: 0,
        row_count: 0,
        expected_row_count: 100,
      },
    };
  }

  window.fetch = async function staticApiFetch(resource, options) {
    const requestUrl = typeof resource === 'string' ? resource : resource.url;
    const url = new URL(requestUrl, window.location.href);
    if (!url.pathname.startsWith('/api/')) {
      return originalFetch(resource, options);
    }

    if (url.pathname === '/api/fps-threshold/generate') {
      return jsonResponse(staticUnavailablePayload(url), 409);
    }

    const manifest = await manifestPromise;
    const key = normalizeApiKey(url);
    const snapshotPath = manifest.routes[key];
    if (!snapshotPath) {
      return jsonResponse(staticUnavailablePayload(url), 404);
    }

    return originalFetch(`data/${snapshotPath}`, { cache: 'force-cache' });
  };
}());
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Export static GitHub Pages snapshots.")
    parser.add_argument(
        "--data",
        default=str(DEFAULT_DATASET_DIR),
        help="Generated-data directory. Defaults to adaptive_visualization/data/generated_data/2016-12.",
    )
    parser.add_argument(
        "--output",
        default=str(MODULE_DIR / "pages"),
        help="Static output directory. Defaults to adaptive_visualization/pages.",
    )
    args = parser.parse_args()

    data_dir = resolve_data_dir(args.data)
    output_dir = Path(args.output)
    configure(data_dir)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    (output_dir / "static").mkdir(parents=True, exist_ok=True)
    (output_dir / "data" / "api").mkdir(parents=True, exist_ok=True)

    client = app.test_client()
    scopes = build_static_scopes()
    static_snapshot_coverage = build_static_snapshot_coverage(scopes)
    source_data_label = str(data_dir)
    try:
        source_data_label = data_dir.relative_to(MODULE_DIR).as_posix()
    except ValueError:
        pass
    manifest: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_data": source_data_label,
        "static_snapshot_coverage": static_snapshot_coverage,
        "routes": {},
    }

    def add_snapshot(path: str, params: dict[str, str] | None = None) -> dict[str, Any]:
        params = params or {}
        query = urlencode(params)
        url = f"{path}?{query}" if query else path
        response = client.get(url)
        if response.status_code != 200:
            raise RuntimeError(f"Snapshot failed: {url} -> HTTP {response.status_code}")
        payload = response.get_json()
        if payload is None:
            raise RuntimeError(f"Snapshot returned non-JSON payload: {url}")
        payload = sanitize_payload(payload, data_dir=data_dir)

        key = normalize_api_key(path, params)
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
        relative_path = f"api/{digest}.json"
        with (output_dir / "data" / relative_path).open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, separators=(",", ":"), allow_nan=False)
        manifest["routes"][key] = relative_path
        return payload

    meta_payload = add_snapshot("/api/meta")
    for boundary_path in [
        "/api/boundaries/us/nation",
        "/api/boundaries/us/states",
        "/api/boundaries/us/counties",
    ]:
        add_snapshot(boundary_path)

    label_states = sorted({
        scope["state"]
        for scope in scopes
        if scope["state"] and scope["zoom"] in {"state", "county"}
    })
    for state in label_states:
        add_snapshot("/api/labels/counties", {"state": state})
    for scope in scopes:
        if scope["zoom"] == "county":
            add_snapshot(
                "/api/boundaries/county",
                {"state": scope["state"], "county": scope["county"]},
            )

    for scope in scopes:
        add_scope_snapshots(add_snapshot, scope)

    for scope in scopes:
        if scope["zoom"] == "country":
            continue
        add_snapshot(
            "/api/fps-threshold/status",
            {
                "zoom": scope["zoom"],
                "state": scope["state"],
                "county": scope["county"],
            },
        )

    manifest["route_count"] = len(manifest["routes"])
    with (output_dir / "data" / "manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    shutil.copy2(MODULE_DIR / "static" / "app.js", output_dir / "static" / "app.js")
    shutil.copy2(MODULE_DIR / "static" / "style.css", output_dir / "static" / "style.css")
    (output_dir / "static" / "static-api.js").write_text(STATIC_API_JS.strip() + "\n", encoding="utf-8")
    (output_dir / ".nojekyll").write_text("", encoding="utf-8")
    write_static_index(output_dir, meta_payload, static_snapshot_coverage)
    write_static_readme(output_dir, manifest)

    print(f"Static GitHub Pages export written to {output_dir}")
    print(f"Snapshots: {manifest['route_count']}")
    print(f"Scopes: {len(scopes)}")


def build_static_scopes() -> list[dict[str, str]]:
    """Export country plus every state/county with saved FPS-threshold data."""
    coverage = _repo().get_fps_threshold_coverage()
    scopes: list[dict[str, str]] = [{"zoom": "country", "state": "", "county": ""}]
    scopes.extend(
        {"zoom": "state", "state": state, "county": ""}
        for state in coverage.get("states", [])
    )
    for state, counties in coverage.get("counties", {}).items():
        scopes.extend(
            {"zoom": "county", "state": state, "county": county}
            for county in counties
        )
    return scopes


def build_static_snapshot_coverage(scopes: list[dict[str, str]]) -> dict[str, Any]:
    """Describe which map scopes exist in the static GitHub Pages export."""
    states = sorted(
        scope["state"]
        for scope in scopes
        if scope["zoom"] == "state" and scope["state"]
    )
    county_map: dict[str, list[str]] = {}
    for scope in scopes:
        if scope["zoom"] != "county" or not scope["state"] or not scope["county"]:
            continue
        county_map.setdefault(scope["state"], []).append(scope["county"])
    county_map = {
        state: sorted(counties)
        for state, counties in sorted(county_map.items())
    }

    methods: dict[str, Any] = {}
    for method in ["all", "pixel", "fps_threshold"]:
        methods[method] = {
            "states": states,
            "counties": county_map,
        }
    methods["random"] = {
        "states": states,
        "counties": county_map,
        "retainPercentages": RANDOM_RETAIN_PERCENTAGES,
    }
    return {"methods": methods}


def add_scope_snapshots(add_snapshot, scope: dict[str, str]) -> None:
    base_scope = {
        "zoom": scope["zoom"],
        "state": scope["state"],
        "county": scope["county"],
    }
    method_queries: list[dict[str, str]] = [
        {**base_scope, "method": "all"},
        {**base_scope, "method": "pixel"},
    ]
    method_queries.extend(
        {**base_scope, "method": "random", "retain_pct": str(retain_pct)}
        for retain_pct in RANDOM_RETAIN_PERCENTAGES
    )

    baselines = FPS_BASELINES_BY_ZOOM[scope["zoom"]]
    for threshold in FPS_THRESHOLDS:
        for baseline in baselines:
            method_queries.append(
                {
                    **base_scope,
                    "method": "fps_threshold",
                    "error_threshold": f"{threshold:.2f}",
                    "topology_baseline_pct": baseline,
                }
            )

    for query in method_queries:
        add_snapshot("/api/data", query)
        for analysis_property in ANALYSIS_PROPERTIES:
            analysis_query = {
                "property": analysis_property,
                "zoom": query["zoom"],
                "method": query["method"],
                "state": query["state"],
                "county": query["county"],
            }
            if query["method"] == "random":
                analysis_query["retain_pct"] = query["retain_pct"]
            if query["method"] == "fps_threshold":
                analysis_query["error_threshold"] = query["error_threshold"]
                analysis_query["topology_baseline_pct"] = query["topology_baseline_pct"]
            add_snapshot("/api/analysis", analysis_query)


def normalize_api_key(path: str, params: dict[str, str]) -> str:
    if path in {
        "/api/meta",
        "/api/boundaries/us/nation",
        "/api/boundaries/us/states",
        "/api/boundaries/us/counties",
    }:
        return path
    if path == "/api/labels/counties":
        return with_query(path, {"state": params.get("state", "")})
    if path == "/api/boundaries/county":
        return with_query(path, {"state": params.get("state", ""), "county": params.get("county", "")})
    if path == "/api/fps-threshold/status":
        return with_query(
            path,
            {
                "zoom": params.get("zoom", "state"),
                "state": params.get("state", ""),
                "county": params.get("county", ""),
            },
        )
    if path == "/api/data":
        method = params.get("method", "pixel")
        relevant = {
            "zoom": params.get("zoom", "country"),
            "method": method,
            "state": params.get("state", ""),
            "county": params.get("county", ""),
        }
        if method == "random":
            relevant["retain_pct"] = params.get("retain_pct", "50")
        if method == "fps_threshold":
            relevant["error_threshold"] = f"{float(params.get('error_threshold', 0.5)):.2f}"
            relevant["topology_baseline_pct"] = params.get("topology_baseline_pct", DEFAULT_FPS_BASELINE)
        return with_query(path, relevant)
    if path == "/api/analysis":
        method = params.get("method", "pixel")
        relevant = {
            "property": params.get("property", "statistical"),
            "zoom": params.get("zoom", "country"),
            "method": method,
            "state": params.get("state", ""),
            "county": params.get("county", ""),
        }
        if method == "random":
            relevant["retain_pct"] = params.get("retain_pct", "50")
        if method == "fps_threshold":
            relevant["error_threshold"] = f"{float(params.get('error_threshold', 0.5)):.2f}"
            relevant["topology_baseline_pct"] = params.get("topology_baseline_pct", DEFAULT_FPS_BASELINE)
        return with_query(path, relevant)
    return path


def with_query(path: str, params: dict[str, str]) -> str:
    return f"{path}?{urlencode(params)}" if params else path


def sanitize_payload(value: Any, *, data_dir: Path) -> Any:
    """Remove local machine paths from static JSON payloads."""
    local_prefix = str(data_dir.parent.parent.parent)
    if isinstance(value, dict):
        sanitized = {}
        for key, item in value.items():
            if key in {"cache_path", "sample_abs_path"}:
                continue
            sanitized[key] = sanitize_payload(item, data_dir=data_dir)
        return sanitized
    if isinstance(value, list):
        return [sanitize_payload(item, data_dir=data_dir) for item in value]
    if isinstance(value, str) and local_prefix and local_prefix in value:
        return value.replace(local_prefix, "<adaptive_visualization>")
    return value


def write_static_index(
    output_dir: Path,
    meta_payload: dict[str, Any],
    static_snapshot_coverage: dict[str, Any],
) -> None:
    asset_version = int(
        max(
            (MODULE_DIR / "static" / "app.js").stat().st_mtime,
            (MODULE_DIR / "static" / "style.css").stat().st_mtime,
            Path(__file__).stat().st_mtime,
        )
    )
    template = (MODULE_DIR / "templates" / "index.html").read_text(encoding="utf-8")
    html = template.replace(
        "{{ url_for('static', filename='style.css') }}?v={{ style_css_version }}",
        f"static/style.css?v={asset_version}",
    )
    html = html.replace(
        "{{ url_for('static', filename='app.js') }}?v={{ app_js_version }}",
        f"static/app.js?v={asset_version}",
    )
    html = html.replace(
        "OpenLayers research viewer",
        "Precomputed GitHub Pages viewer",
    )
    app_config = {
        "stateMeta": meta_payload["state_meta"],
        "fpsThresholdCoverage": meta_payload["fps_threshold_coverage"],
        "staticDemo": True,
        "staticSnapshotCoverage": static_snapshot_coverage,
    }
    config_script = (
        "<script>\n"
        "    window.APP_CONFIG = "
        + json.dumps(app_config, separators=(",", ":"))
        + ";\n"
        "  </script>"
    )
    html = re.sub(
        r"<script>\s*window\.APP_CONFIG\s*=\s*\{.*?\};\s*</script>",
        config_script,
        html,
        flags=re.DOTALL,
    )
    html = html.replace(
        f'<script src="static/app.js?v={asset_version}"></script>',
        f'<script src="static/static-api.js?v={asset_version}"></script>\n  <script src="static/app.js?v={asset_version}"></script>',
    )
    (output_dir / "index.html").write_text(html, encoding="utf-8")


def write_static_readme(output_dir: Path, manifest: dict[str, Any]) -> None:
    random_percentages = (
        manifest.get("static_snapshot_coverage", {})
        .get("methods", {})
        .get("random", {})
        .get("retainPercentages", [])
    )
    random_percentages_label = ", ".join(f"{percentage}%" for percentage in random_percentages)
    readme = f"""# Adaptive Visualization Static Demo

This folder is a GitHub Pages export of the bundled precomputed dashboard views.
It does not run Flask or Python in the browser.

- Open `index.html` through GitHub Pages.
- Snapshot routes: `{manifest['route_count']}`
- Generated: `{manifest['generated_at']}`
- Random snapshots: `{random_percentages_label}`

If a state, county, threshold, or analysis is not bundled here, clone the
repository and run `python run_dashboard.py` from the parent
`adaptive_visualization/` folder to generate additional data.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")


if __name__ == "__main__":
    main()
