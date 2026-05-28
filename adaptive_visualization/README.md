# Adaptive Visualization

Reviewer-facing OpenLayers dashboard for adaptive accident-map sampling. The app
compares four display modes over the same source accident dataset:

- `All Points`: the complete filtered point set for the current map scope.
- `Pixel Occupancy`: viewport-aware point selection based on visible marker
  occupancy.
- `Random`: deterministic random baseline at the selected retain percentage.
- `FPS - Error Threshold`: farthest-point-sampling prefixes selected by a
  normalized topological-error threshold and configurable FPS baselines.

## Run

Install dependencies from inside this folder:

```powershell
pip install -r requirements.txt
```

Start the dashboard from inside this folder:

```powershell
python run_dashboard.py
```

If you are running from the repository root during development:

```powershell
python adaptive_visualization/run_dashboard.py
```

By default, the launcher uses:

```text
data/generated_data/2016-12
```

You can pass an explicit generated-data directory if needed:

```powershell
python run_dashboard.py --data data/generated_data/2016-12
```

Then open:

```text
http://127.0.0.1:8060/
```

## GitHub Pages Static Demo

The `pages/` folder is a static export for reviewers. It uses precomputed JSON
snapshots and does not run Flask or Python in the browser.

To refresh the static export locally:

```powershell
python export_static_pages.py
```

Then publish the contents of:

```text
pages/
```

to GitHub Pages, for example through a `gh-pages` branch or a GitHub Pages
Action that uploads `adaptive_visualization/pages` as the artifact.

The static demo intentionally supports only bundled precomputed snapshots. To
generate additional state/county FPS-threshold data, clone the repository and
run the local dashboard with `python run_dashboard.py`.

## Runtime Layout

- `app.py`: Flask routes, API payloads, and FPS-threshold generation jobs.
- `run_dashboard.py`: dashboard launcher kept inside the release folder.
- `export_static_pages.py`: builds the static GitHub Pages snapshot viewer.
- `core/`: data loading, sampling dispatch, analysis dispatch, boundaries, and
  cache helpers.
- `vendor/`: small bundled runtime subset of the original `map_viz` utilities,
  so this folder can run without the surrounding research repository.
- `static/` and `templates/`: OpenLayers dashboard frontend.
- `pages/`: static GitHub Pages export with selected precomputed API snapshots.
- `requirements.txt`: Python dependencies for this portable folder.
- `data/generated_data/2016-12/original.csv`: source accident records.
- `data/generated_data/2016-12/*_fps_topology_cache`: saved FPS-threshold
  samples, persistence diagrams, and topology-distance curves.
- `data/generated_data/2016-12/fps_distance_baseline_cache`: shared
  pixel-distance FPS baselines.
- `data/generated_data/2016-12/pixel_openlayers_*_cache`: viewport-aware pixel
  occupancy caches.
- `data/generated_data/2016-12/runtime_sampling_cache`: random baseline cache.

Boundary caches are written under `data/cache/` at runtime and are ignored by
git.

## Data Generation In The Dashboard

Country-level FPS-threshold data is expected to be available in the included
cache. For state and county views, if an FPS-threshold topology cache has not
been generated yet, the dashboard shows a compact message, exact point count,
progress bar, and a `Generate Data` button. After generation finishes, reload
the dashboard to use the new saved cache.

Paper-writing and one-off evaluation artifacts are not required by this
portable dashboard folder.
