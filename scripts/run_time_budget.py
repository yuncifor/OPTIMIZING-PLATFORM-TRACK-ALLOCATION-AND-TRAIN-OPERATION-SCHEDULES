from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from station_dispatch.experiment_runner import METHODS, run_manifest


def run_time_budget(manifest_path: Path, output_root: Path, methods: tuple[str, ...]) -> list[Path]:
    return run_manifest(manifest_path, output_root, "time_budget", methods)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the station-dispatching time-budget experiment.")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "experiments/manifest.csv")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "raw_results")
    parser.add_argument("--method", choices=(*METHODS, "all"), default="all")
    args = parser.parse_args()

    methods = METHODS if args.method == "all" else (args.method,)
    for result_path in run_time_budget(args.manifest, args.output_root, methods):
        print(result_path)


if __name__ == "__main__":
    main()
