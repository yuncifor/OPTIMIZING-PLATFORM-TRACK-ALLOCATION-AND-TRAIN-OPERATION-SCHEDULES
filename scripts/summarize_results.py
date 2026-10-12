from __future__ import annotations

import argparse
import csv
import warnings
import json
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import t

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from station_dispatch.experiment_runner import METHODS

GROUPS = ("main", "scalability", "seed_sensitivity", "time_budget")
QUALITY_COLUMNS = (
    "total_station_wait", "entry_wait_total", "excess_dwell_total", "makespan", "end_sum",
    "completion_deviation_total", "route_change_count", "max_entry_wait", "entry_wait_std",
    "station_sojourn_total",
)
TIME_COLUMNS = ("first_feasible_s", "best_found_s")
RUN_COLUMNS = (
    "experiment_group", "instance_id", "train_count", "time_limit_s", "method",
    "condition_id", "source_sha256", "input_sha256", "seed", "repetition", "repetitions", "run_status",
    "best_found_s", "first_feasible_s", "iterations", "optimal", "lower_bound", "gap",
    "feasible", *QUALITY_COLUMNS,
)
STAT_FILES = {
    "main": "main_summary.csv", "scalability": "scalability_scale_stats.csv",
    "seed_sensitivity": "seed_sensitivity_seed_stats.csv", "time_budget": "time_budget_stats.csv",
}


def load_runs(raw_results: Path, manifest_path: Path = PROJECT_ROOT / "experiments/manifest.csv") -> list[dict]:
    with manifest_path.open(newline="", encoding="utf8") as stream:
        approved = {
            (row["experiment_group"], row["instance_id"], method, int(seed), int(row["time_limit_s"]), int(row["train_count"])): int(row.get("repetitions", 1))
            for row in csv.DictReader(stream)
            for method in METHODS
            for seed in row["seeds"].split(";")
        }
    rows: list[dict] = []
    seen = {}
    for group in GROUPS:
        for result_file in sorted((raw_results / group).glob("**/*.json")):
            payload = json.loads(result_file.read_text(encoding="utf8"))
            inputs = payload.get("input", {})
            key = (inputs.get("experiment_group"), inputs.get("instance_id"), payload.get("method"),
                   payload.get("seed"), inputs.get("time_limit_s"), inputs.get("train_count"))
            if key not in approved:
                warnings.warn(f"Skipping result outside current manifest: {result_file}")
                continue
            repetitions = approved[key]
            repetition = payload.get("repetition", 1)
            if ((repetitions > 1 and ("repetition" not in payload or inputs.get("repetitions", 0) < repetitions))
                    or not 1 <= repetition <= repetitions):
                warnings.warn(f"Skipping result outside current repetition design: {result_file}")
                continue
            key = (*key, repetition)
            if key in seen:
                raise ValueError(f"Duplicate experiment key in {seen[key]} and {result_file}")
            seen[key] = result_file
            run = payload.get("run", {})
            objectives = payload.get("objectives", {})
            metrics = payload.get("metrics") or {}
            validation = payload.get("validation", {})
            if ("sat" in objectives and "feasible" in validation
                    and objectives["sat"] != validation["feasible"]):
                raise ValueError(f"Conflicting feasibility flags in {result_file}")
            rows.append(
                {
                    "experiment_group": payload.get("input", {}).get("experiment_group", group),
                    "instance_id": payload.get("input", {}).get("instance_id", result_file.parent.name),
                    "train_count": payload.get("input", {}).get("train_count"),
                    "time_limit_s": payload.get("input", {}).get("time_limit_s"),
                    "method": payload.get("method"),
                    "condition_id": payload.get("environment", {}).get("condition_id", "legacy_unrecorded"),
                    "source_sha256": payload.get("environment", {}).get("source_sha256", "legacy_unrecorded"),
                    "input_sha256": inputs.get("sha256", "legacy_unrecorded"),
                    "seed": payload.get("seed"),
                    "repetition": repetition,
                    "repetitions": repetitions,
                    "run_status": run.get("status"),
                    "best_found_s": run.get("best_found_s"),
                    "first_feasible_s": run.get("first_feasible_s"),
                    "iterations": run.get("iterations"),
                    "optimal": run.get("optimal"),
                    "lower_bound": run.get("lower_bound"),
                    "gap": run.get("gap"),
                    "feasible": objectives.get("sat", validation.get("feasible", False)),
                    "total_station_wait": objectives.get("total_station_wait"),
                    "entry_wait_total": metrics.get("entry_wait_total"),
                    "excess_dwell_total": metrics.get("excess_dwell_total"),
                    "makespan": objectives.get("makespan"),
                    "end_sum": objectives.get("end_sum"),
                    "completion_deviation_total": metrics.get("completion_deviation_total"),
                    "route_change_count": metrics.get("route_change_count"),
                    "max_entry_wait": metrics.get("max_entry_wait"),
                    "entry_wait_std": metrics.get("entry_wait_std"),
                    "station_sojourn_total": metrics.get("station_sojourn_total"),
                }
            )
    return rows


def _statistics(frame: pd.DataFrame, keys: list[str], unit: str) -> pd.DataFrame:
    grouped = frame.groupby(keys, dropna=False)
    columns = (*QUALITY_COLUMNS, "wait_per_train", *TIME_COLUMNS)
    aggregations = {
        "run_count": ("feasible", "size"), "feasible_count": ("feasible", "sum"),
        "instance_count": ("instance_id", "nunique"),
        "instance_ids": ("instance_id", lambda values: ";".join(sorted(set(values)))),
        "seeds": ("seed", lambda values: ";".join(str(int(value)) for value in sorted(set(values)))),
        "repetition_ids": ("repetition", lambda values: ";".join(str(int(value)) for value in sorted(set(values)))),
        "repetitions_expected": ("repetitions", "max"),
    }
    if "input_sha256" not in keys:
        aggregations["input_sha256"] = ("input_sha256", lambda values: ";".join(sorted(set(values))))
    aggregations.update({
        f"{column}_{suffix}": (column, operation)
        for column in columns
        for suffix, operation in (("n", "count"), ("mean", "mean"), ("min", "min"),
                                  ("max", "max"), ("std", "std"), ("sem", "sem"))
    })
    stats = grouped.agg(**aggregations)
    stats["success_rate"] = stats["feasible_count"] / stats["run_count"]
    stats["replication_unit"] = unit
    stats["ci_method"] = "student_t" if unit == "repeat" else "not_estimated"
    for column in columns:
        stats[f"{column}_range"] = stats[f"{column}_max"] - stats[f"{column}_min"]
        stats[f"{column}_ci95_low"] = float("nan")
        stats[f"{column}_ci95_high"] = float("nan")
        if unit == "repeat":
            valid = stats[f"{column}_n"] >= 2
            if valid.any():
                half_width = t.ppf(0.975, stats.loc[valid, f"{column}_n"] - 1) * stats.loc[valid, f"{column}_sem"]
                stats.loc[valid, f"{column}_ci95_low"] = stats.loc[valid, f"{column}_mean"] - half_width
                stats.loc[valid, f"{column}_ci95_high"] = stats.loc[valid, f"{column}_mean"] + half_width
    return stats.reset_index()


def summarize(rows: list[dict], output_dir: Path) -> None:
    frame = pd.DataFrame(rows).reindex(columns=RUN_COLUMNS)
    for column in ("condition_id", "source_sha256", "input_sha256"):
        frame[column] = frame[column].fillna("legacy_unrecorded")
    for column in ("repetition", "repetitions"):
        frame[column] = frame[column].fillna(1).astype(int)
    duplicate_keys = ["experiment_group", "instance_id", "train_count", "method", "seed", "repetition",
                      "time_limit_s", "condition_id", "source_sha256", "input_sha256"]
    if frame.duplicated(duplicate_keys).any():
        raise ValueError("Duplicate experiment rows")
    frame["feasible"] = frame["feasible"].fillna(False).astype(bool)
    scale_mask = frame["experiment_group"].eq("scalability")
    if scale_mask.any():
        comparison = frame.loc[scale_mask]
        if (comparison["condition_id"].nunique(dropna=False) != 1
                or comparison["condition_id"].eq("legacy_unrecorded").any()
                or comparison["source_sha256"].nunique(dropna=False) != 1
                or not comparison["seed"].eq(42).all()
                or not comparison["time_limit_s"].eq(300).all()):
            raise ValueError("Scalability comparison requires one recorded condition, source version, seed=42 and 300s")
    frame.loc[~frame["feasible"], [*QUALITY_COLUMNS, *TIME_COLUMNS]] = float("nan")
    frame["wait_per_train"] = frame["total_station_wait"] / frame["train_count"]
    outputs = {"all_runs.csv": frame}
    base_keys = ["train_count", "method", "time_limit_s", "condition_id", "source_sha256"]
    for group in GROUPS:
        group_frame = frame.loc[frame["experiment_group"].eq(group)]
        if group_frame.empty:
            continue
        outputs[f"{group}_runs.csv"] = group_frame
        unit = "instance" if group == "main" else "repeat"
        keys = base_keys + (["instance_id", "input_sha256", "seed"] if group != "main" else ["seed"])
        outputs[STAT_FILES[group]] = _statistics(group_frame, keys, unit)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in outputs.items():
        table.to_csv(output_dir / name, index=False)
    for name in (*STAT_FILES.values(), *(f"{group}_runs.csv" for group in GROUPS)):
        if name not in outputs:
            (output_dir / name).unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize station-dispatching experiment results.")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "experiments/manifest.csv")
    parser.add_argument("--raw-results", type=Path, default=PROJECT_ROOT / "raw_results")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_runs(args.raw_results, args.manifest)
    summarize(rows, args.output_dir)
    print(f"summarized {len(rows)} runs into {args.output_dir}")


if __name__ == "__main__":
    main()
