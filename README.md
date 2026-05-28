# Adaptive Visualization Dashboard

This repository contains the reviewer-facing OpenLayers dashboard used for the
adaptive accident-map visualization experiments.

The runnable app is in `adaptive_visualization/`. It supports four map display
modes: all points, pixel occupancy, random reduction, and FPS error-threshold
sampling with saved topological caches.

## Run

```powershell
pip install -r adaptive_visualization/requirements.txt
python adaptive_visualization/run_dashboard.py
```

The launcher defaults to:

```text
adaptive_visualization/data/generated_data/2016-12
```

For the reviewer-facing static webpage, publish:

```text
adaptive_visualization/pages/
```

That static page shows bundled precomputed snapshots only. New state/county
computation requires cloning the repository and running the local dashboard.

Open the dashboard at:

```text
http://127.0.0.1:8060/
```

For more details about runtime files and cache expectations, see
`adaptive_visualization/README.md`.

## Repository Notes

- `map_viz/` provides shared data loading, sampling, and analysis utilities used
  by the dashboard.
- `release_artifacts/` contains paper/evaluation/support material that is not
  required to run the dashboard.
- Runtime boundary caches are written under `adaptive_visualization/data/cache/`
  and are ignored by git.
