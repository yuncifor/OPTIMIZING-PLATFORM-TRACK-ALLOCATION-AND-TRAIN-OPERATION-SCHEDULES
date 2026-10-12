from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from station_dispatch.limits import TIME_LIMIT_S

MAIN_INSTANCES = ("t030-01", "t030-02", "t030-03")
SCALABILITY_SIZES = (35, 40, 45, 50)
FIXED_QEA_LNS_SEED = 42
SEED_SENSITIVITY_SEEDS = (*range(37, 42), *range(43, 48))


def build_manifest(data_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for instance_id in MAIN_INSTANCES:
        rows.append(
            {
                "experiment_group": "main",
                "instance_id": instance_id,
                "instance_path": (data_dir / f"{instance_id}.dzn").as_posix(),
                "train_count": 30,
                "seeds": str(FIXED_QEA_LNS_SEED),
                "time_limit_s": TIME_LIMIT_S,
            }
        )
    for size in SCALABILITY_SIZES:
        for replicate in range(1, 4):
            instance_id = f"t{size:03d}-{replicate:02d}"
            rows.append(
                {
                    "experiment_group": "scalability",
                    "instance_id": instance_id,
                    "instance_path": (data_dir / f"{instance_id}.dzn").as_posix(),
                    "train_count": size,
                    "seeds": str(FIXED_QEA_LNS_SEED),
                    "time_limit_s": TIME_LIMIT_S,
                }
            )
    sensitivity_seeds = ";".join(str(seed) for seed in SEED_SENSITIVITY_SEEDS)
    for instance_id in ("t050-01",):
        rows.append(
            {
                "experiment_group": "seed_sensitivity",
                "instance_id": instance_id,
                "instance_path": (data_dir / f"{instance_id}.dzn").as_posix(),
                "train_count": 50,
                "seeds": sensitivity_seeds,
                "time_limit_s": TIME_LIMIT_S,
            }
        )
    for budget in (10, 20, 30, 40, 50, 60, 120, 300, 600):
        rows.append({
            "experiment_group": "time_budget", "instance_id": "t050-01",
            "instance_path": (data_dir / "t050-01.dzn").as_posix(), "train_count": 50,
            "seeds": str(FIXED_QEA_LNS_SEED), "time_limit_s": budget,
        })
    for row in rows:
        row["repetitions"] = 1 if row["experiment_group"] == "main" else 3
    return rows


def write_manifest(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf8") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=("experiment_group", "instance_id", "instance_path", "train_count", "seeds", "time_limit_s", "repetitions"),
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the station-dispatching experiment manifest.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/cp2025"))
    parser.add_argument("--output", type=Path, default=Path("experiments/manifest.csv"))
    args = parser.parse_args()
    write_manifest(build_manifest(args.data_dir), args.output)


if __name__ == "__main__":
    main()
