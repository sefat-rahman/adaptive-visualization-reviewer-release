"""Run the standalone OpenLayers adaptive visualization app.

Usage from the repository root:
    python adaptive_visualization/run_dashboard.py

Usage from inside adaptive_visualization/:
    python run_dashboard.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from adaptive_visualization.app import app, configure
from adaptive_visualization.paths import DEFAULT_DATASET_DIR, resolve_data_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the OpenLayers adaptive visualization.")
    parser.add_argument(
        "--data",
        type=str,
        default=str(DEFAULT_DATASET_DIR),
        help="Path to a generated dataset directory. Defaults to adaptive_visualization/data/generated_data/2016-12.",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8060)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    data_dir = resolve_data_dir(args.data)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    print("Launcher started.", flush=True)
    print(f"Configuring OpenLayers app with data: {data_dir}", flush=True)
    configure(data_dir)
    print(f"Starting dashboard at http://{args.host}:{args.port}/", flush=True)
    print("Press Ctrl+C to stop.", flush=True)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()

