"""Flask app for the OpenLayers-based adaptive accident visualization."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from threading import Lock, Thread
import traceback

from flask import Flask, jsonify, render_template, request

from adaptive_visualization.core.boundaries import BoundaryService
from adaptive_visualization.paths import CACHE_DIR, MODULE_DIR
from adaptive_visualization.core.repository import AdaptiveDataRepository
from adaptive_visualization.core.state_metadata import build_state_payload


TEMPLATE_DIR = MODULE_DIR / "templates"
STATIC_DIR = MODULE_DIR / "static"


app = Flask(
    __name__,
    template_folder=str(TEMPLATE_DIR),
    static_folder=str(STATIC_DIR),
)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

_repository: AdaptiveDataRepository | None = None
_boundaries = BoundaryService(CACHE_DIR)
_fps_generation_jobs: dict[str, dict] = {}
_fps_generation_lock = Lock()
RANDOM_PRECOMPUTED_RETAIN_PERCENTAGES = [3, 4, 26, 50]


def configure(data_dir: str | Path) -> None:
    """Configure the server to use a generated-data directory."""
    global _repository
    _repository = AdaptiveDataRepository(data_dir)
    with _fps_generation_lock:
        _fps_generation_jobs.clear()


def _repo() -> AdaptiveDataRepository:
    if _repository is None:
        raise RuntimeError("Adaptive Visualization app is not configured with a data directory.")
    return _repository


def _filter_county_points_to_boundary(df, state: str, county: str):
    if df is None or df.empty or not state or not county:
        return df
    boundary = _boundaries.get_county_boundary(state, county, county_df=df)
    return _boundaries.filter_points_to_boundary(df, boundary)


def _parse_viewport_spec(req) -> dict | None:
    width = req.args.get('viewport_width_px', type=float)
    height = req.args.get('viewport_height_px', type=float)
    if not width or not height or width <= 0 or height <= 0:
        return None

    return {
        'width': width,
        'height': height,
        'padding': [
            req.args.get('padding_top_px', default=28.0, type=float),
            req.args.get('padding_right_px', default=28.0, type=float),
            req.args.get('padding_bottom_px', default=28.0, type=float),
            req.args.get('padding_left_px', default=28.0, type=float),
        ],
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fps_generation_key(zoom: str, state: str = "", county: str = "") -> str:
    zoom = (zoom or "").strip().lower()
    state = (state or "").strip().upper()
    county = (county or "").strip().lower()
    return f"{zoom}:{state}:{county}"


def _fps_generation_job(zoom: str, state: str = "", county: str = "") -> dict | None:
    key = _fps_generation_key(zoom, state, county)
    with _fps_generation_lock:
        job = _fps_generation_jobs.get(key)
        return dict(job) if job else None


def _fps_threshold_missing_payload(
    repo: AdaptiveDataRepository,
    *,
    zoom: str,
    method: str,
    state: str = "",
    county: str = "",
) -> dict | None:
    if method != "fps_threshold" or zoom not in {"state", "county"}:
        return None

    scope_type = "County" if zoom == "county" else "State"
    missing_message = f"Data is not precomputed for this {scope_type}."
    job = _fps_generation_job(zoom, state, county)
    if job and job.get("status") in {"queued", "running"}:
        status = repo.get_fps_threshold_cache_status(zoom=zoom, state=state, county=county)
        return {
            "available": False,
            "missing_cache": True,
            "generation_supported": True,
            "generation_status": job.get("status"),
            "message": missing_message,
            "status": status,
            "job": job,
        }

    status = repo.get_fps_threshold_cache_status(zoom=zoom, state=state, county=county)
    if status.get("available"):
        return None

    message = (
        job.get("message")
        if job and job.get("status") == "error"
        else missing_message
    )
    return {
        "available": False,
        "missing_cache": True,
        "generation_supported": True,
        "generation_status": job.get("status") if job else "missing",
        "message": message,
        "status": status,
        "job": job,
    }


def _method_precomputed_coverage(repo: AdaptiveDataRepository) -> dict:
    """Expose map scopes with saved reviewer/demo snapshots by method."""
    fps_coverage = repo.get_fps_threshold_coverage()
    states = sorted(fps_coverage.get("states", []))
    counties = {
        state: sorted(county_list or [])
        for state, county_list in sorted((fps_coverage.get("counties") or {}).items())
    }
    methods = {
        method: {
            "states": states,
            "counties": counties,
        }
        for method in ["all", "pixel", "fps_threshold"]
    }
    methods["random"] = {
        "states": states,
        "counties": counties,
        "retainPercentages": RANDOM_PRECOMPUTED_RETAIN_PERCENTAGES,
    }
    return {"methods": methods}


@app.route("/")
def index():
    repo = _repo()
    state_meta = build_state_payload(repo.get_available_states())
    method_precomputed_coverage = _method_precomputed_coverage(repo)
    return render_template(
        "index.html",
        app_js_version=int((STATIC_DIR / "app.js").stat().st_mtime),
        style_css_version=int((STATIC_DIR / "style.css").stat().st_mtime),
        state_meta=state_meta,
        fps_threshold_coverage=repo.get_fps_threshold_coverage(),
        static_snapshot_coverage=method_precomputed_coverage,
    )


@app.route("/api/meta")
def api_meta():
    repo = _repo()
    return jsonify(
        {
            "available_states": repo.get_available_states(),
            "state_meta": build_state_payload(repo.get_available_states()),
            "fps_threshold_coverage": repo.get_fps_threshold_coverage(),
            "static_snapshot_coverage": _method_precomputed_coverage(repo),
        }
    )


@app.route("/api/data")
def api_data():
    repo = _repo()
    zoom = request.args.get("zoom", "country")
    method = request.args.get("method", "pixel")
    state = request.args.get("state", "")
    county = request.args.get("county", "")
    analysis_property = request.args.get("analysis_property", "statistical")
    retain_percentage = request.args.get("retain_pct", default=50, type=int)
    error_threshold = request.args.get("error_threshold", default=0.5, type=float)
    topology_baseline_percentage = request.args.get("topology_baseline_pct", default=None)
    viewport_spec = _parse_viewport_spec(request)

    missing = _fps_threshold_missing_payload(
        repo,
        zoom=zoom,
        method=method,
        state=state,
        county=county,
    )
    if missing is not None:
        exact_count = int(missing.get("status", {}).get("exact_count", 0))
        return jsonify({
            **missing,
            "count": 0,
            "displayed_count": 0,
            "exact_count": exact_count,
            "points": [],
        })

    df = repo.get_display_df(
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
    exact_df = repo.get_reference_df(zoom=zoom, state=state, county=county)

    if zoom == "county":
        df = _filter_county_points_to_boundary(df, state, county)
        exact_df = _filter_county_points_to_boundary(exact_df, state, county)

    renamed = df.rename(columns={"Start_Lat": "lat", "Start_Lng": "lng"})
    exact_renamed = exact_df.rename(columns={"Start_Lat": "lat", "Start_Lng": "lng"})
    record_columns = [
        column
        for column in ["lat", "lng", "County", "City", "State", "ID"]
        if column in renamed.columns or column in {"lat", "lng"}
    ]
    records = json.loads(
        renamed[record_columns]
        .dropna(subset=["lat", "lng"])
        .to_json(orient="records")
    )
    exact_count = int(len(exact_renamed.dropna(subset=["lat", "lng"])))
    return jsonify({
        "count": len(records),
        "displayed_count": len(records),
        "exact_count": exact_count,
        "points": records,
    })


@app.route("/api/labels/counties")
def api_county_labels():
    repo = _repo()
    state = request.args.get("state", "")
    labels = repo.get_county_labels(state)
    return jsonify({"labels": labels, "count": len(labels)})


@app.route("/api/analysis")
def api_analysis():
    repo = _repo()
    analysis_property = request.args.get("property", "statistical")
    zoom = request.args.get("zoom", "country")
    method = request.args.get("method", "pixel")
    state = request.args.get("state", "")
    county = request.args.get("county", "")
    retain_percentage = request.args.get("retain_pct", default=50, type=int)
    error_threshold = request.args.get("error_threshold", default=0.5, type=float)
    topology_baseline_percentage = request.args.get("topology_baseline_pct", default=None)
    viewport_spec = _parse_viewport_spec(request)

    missing = _fps_threshold_missing_payload(
        repo,
        zoom=zoom,
        method=method,
        state=state,
        county=county,
    )
    if missing is not None:
        return jsonify({
            **missing,
            "displayed_count": 0,
            "exact_count": int(missing.get("status", {}).get("exact_count", 0)),
        })

    payload = repo.get_analysis(
        analysis_property=analysis_property,
        zoom=zoom,
        method=method,
        state=state,
        county=county,
        viewport_spec=viewport_spec,
        retain_percentage=retain_percentage,
        error_threshold=error_threshold,
        topology_baseline_percentage=topology_baseline_percentage,
    )
    displayed_df = repo.get_display_df(
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
    exact_df = repo.get_reference_df(zoom=zoom, state=state, county=county)
    if zoom == "county":
        displayed_df = _filter_county_points_to_boundary(displayed_df, state, county)
        exact_df = _filter_county_points_to_boundary(exact_df, state, county)
    payload["displayed_count"] = int(len(displayed_df))
    payload["exact_count"] = int(len(exact_df))
    return jsonify(payload)


@app.route("/api/fps-threshold/status")
def api_fps_threshold_status():
    repo = _repo()
    zoom = request.args.get("zoom", "state")
    state = request.args.get("state", "")
    county = request.args.get("county", "")
    status = repo.get_fps_threshold_cache_status(zoom=zoom, state=state, county=county)
    job = _fps_generation_job(zoom, state, county)
    if job:
        status["job"] = job
        status["generation_status"] = job.get("status")
    elif not status.get("available"):
        status["generation_status"] = "missing"
    else:
        status["generation_status"] = "available"
    return jsonify(status)


@app.route("/api/fps-threshold/generate", methods=["POST"])
def api_generate_fps_threshold():
    repo = _repo()
    payload = request.get_json(silent=True) or {}
    zoom = str(payload.get("zoom") or request.args.get("zoom", "state")).strip().lower()
    state = str(payload.get("state") or request.args.get("state", "")).strip().upper()
    county = str(payload.get("county") or request.args.get("county", "")).strip()

    if zoom not in {"state", "county"}:
        return jsonify({
            "available": False,
            "generation_supported": False,
            "message": "FPS-threshold generation is only supported for state and county views.",
        }), 400
    if zoom == "state" and not state:
        return jsonify({
            "available": False,
            "generation_supported": False,
            "message": "A state abbreviation is required before generating state FPS-threshold data.",
        }), 400
    if zoom == "county" and (not state or not county):
        return jsonify({
            "available": False,
            "generation_supported": False,
            "message": "Both state and county are required before generating county FPS-threshold data.",
        }), 400

    key = _fps_generation_key(zoom, state, county)
    status = repo.get_fps_threshold_cache_status(zoom=zoom, state=state, county=county)
    if status.get("available") and status.get("complete"):
        return jsonify({
            **status,
            "generation_status": "done",
            "message": "Data precomputation is complete.",
        })

    with _fps_generation_lock:
        existing = _fps_generation_jobs.get(key)
        if existing and existing.get("status") in {"queued", "running"}:
            return jsonify({
                **status,
                "generation_status": existing.get("status"),
                "message": existing.get("message", "FPS-threshold data generation is already running."),
                "job": dict(existing),
            })

        job = {
            "key": key,
            "zoom": zoom,
            "state": state,
            "county": county,
            "status": "queued",
            "message": "Data precomputation is queued.",
            "started_at": _utc_now(),
            "finished_at": None,
        }
        _fps_generation_jobs[key] = job

    def _worker() -> None:
        try:
            with _fps_generation_lock:
                _fps_generation_jobs[key]["status"] = "running"
                _fps_generation_jobs[key]["message"] = "Data precomputation is running."
            result = repo.generate_fps_threshold_cache(zoom=zoom, state=state, county=county)
            with _fps_generation_lock:
                _fps_generation_jobs[key].update({
                    "status": "done",
                    "message": "Data precomputation is complete.",
                    "finished_at": _utc_now(),
                    "result": result,
                })
        except Exception as exc:  # pragma: no cover - surfaced through the dashboard.
            with _fps_generation_lock:
                _fps_generation_jobs[key].update({
                    "status": "error",
                    "message": f"Data precomputation failed: {exc}",
                    "finished_at": _utc_now(),
                    "traceback": traceback.format_exc(),
                })

    Thread(target=_worker, name=f"fps-threshold-{key}", daemon=True).start()
    return jsonify({
        **status,
        "generation_status": "queued",
        "message": "Data precomputation is queued.",
        "job": dict(job),
    })


@app.route("/api/boundaries/us/states")
def api_us_states():
    return jsonify(_boundaries.get_us_states_topology())


@app.route("/api/boundaries/us/nation")
def api_us_nation():
    return jsonify(_boundaries.get_us_nation_topology())


@app.route("/api/boundaries/us/counties")
def api_us_counties():
    return jsonify(_boundaries.get_us_counties_topology())


@app.route("/api/boundaries/county")
def api_county_boundary():
    repo = _repo()
    state = request.args.get("state", "")
    county = request.args.get("county", "")
    county_df = repo.get_reference_df("county", state=state, county=county)
    feature = _boundaries.get_county_boundary(state, county, county_df=county_df)
    return jsonify(feature)



