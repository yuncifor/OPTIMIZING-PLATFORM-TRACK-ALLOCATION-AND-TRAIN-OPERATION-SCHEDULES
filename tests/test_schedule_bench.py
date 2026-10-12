from __future__ import annotations

import importlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


def load_runner_module():
    return importlib.import_module("station_dispatch.experiment_runner")


def load_script_module(filename: str):
    path = Path("scripts") / filename
    spec = importlib.util.spec_from_file_location(f"test_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ExperimentRunnerTest(unittest.TestCase):
    def test_runs_fcfs_on_an_approved_main_instance_and_writes_traceable_json(self) -> None:
        runner = load_runner_module()
        row = {
            "experiment_group": "main",
            "instance_id": "t030-01",
            "instance_path": "data/cp2025/t030-01.dzn",
            "train_count": "30",
            "seeds": "42",
            "time_limit_s": "300",
        }

        with tempfile.TemporaryDirectory() as temporary_directory:
            result_path = runner.run_case(row, "fcfs", 42, Path(temporary_directory))
            payload = json.loads(result_path.read_text(encoding="utf8"))

        self.assertTrue(payload["validation"]["feasible"])
        self.assertEqual(payload["input"]["experiment_group"], "main")
        self.assertEqual(payload["input"]["instance_id"], "t030-01")
        self.assertEqual(payload["input"]["time_limit_s"], 300)
        self.assertEqual(payload["method"], "fcfs")
        self.assertEqual(payload["seed"], 42)
        self.assertEqual(payload["objectives"]["end_sum"], 112317)
        self.assertEqual(payload["objectives"]["makespan"], 6117)
        self.assertEqual(
            payload["objectives"]["total_station_wait"],
            payload["metrics"]["entry_wait_total"] + payload["metrics"]["excess_dwell_total"],
        )
        self.assertEqual(len(payload["schedule"]), 30)
        details = payload["schedule_details"]
        self.assertEqual(details["time_unit"], "seconds")
        self.assertEqual(len(details["trains"]), 30)
        self.assertEqual({t["train_id"] for t in details["trains"]}, {a["train_id"] for a in payload["schedule"]})
        self.assertTrue(details["occupancies"])
        for train in details["trains"]:
            self.assertEqual(train["completion"], payload["metrics"]["by_train"][str(train["train_id"])]["actual_completion"])
        self.assertIsNotNone(payload["run"]["first_feasible_s"])
        self.assertIsNotNone(payload["run"]["best_found_s"])
        self.assertLessEqual(payload["run"]["first_feasible_s"], payload["run"]["best_found_s"])

    def test_writes_a_truthful_milp_no_solution_record(self) -> None:
        from station_dispatch.milp import MilpResult

        runner = load_runner_module()
        row = {
            "experiment_group": "main",
            "instance_id": "t030-01",
            "instance_path": "data/cp2025/t030-01.dzn",
            "train_count": "30",
            "seeds": "42",
            "time_limit_s": "300",
        }
        no_solution = MilpResult(
            schedule=None,
            status="no_solution",
            feasible=False,
            optimal=False,
            runtime_s=300.1,
            lower_bound=None,
            gap=None,
            end_sum=None,
            makespan=None,
            incumbent_source=None,
            message="SCIP found no feasible solution within the time limit",
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            with patch("station_dispatch.milp.solve_milp", return_value=no_solution) as solve_milp:
                result_path = runner.run_case(row, "milp", 42, Path(temporary_directory))
            payload = json.loads(result_path.read_text(encoding="utf8"))

        solve_milp.assert_called_once()
        self.assertEqual(solve_milp.call_args.kwargs["seed"], 42)
        self.assertFalse(payload["validation"]["feasible"])
        self.assertEqual(payload["validation"]["violations"], [no_solution.message])
        self.assertFalse(payload["objectives"]["sat"])
        self.assertIsNone(payload["objectives"]["end_sum"])
        self.assertIsNone(payload["objectives"]["makespan"])
        self.assertIsNone(payload["objectives"]["total_station_wait"])
        self.assertIsNone(payload["metrics"])
        self.assertEqual(payload["schedule"], [])
        self.assertIsNone(payload["schedule_details"])
        self.assertEqual(payload["run"]["status"], "no_solution")

    def test_milp_does_not_report_runtime_as_an_incumbent_timestamp(self) -> None:
        from station_dispatch.dzn import Block, Route, StationInstance, Train
        from station_dispatch.milp import MilpResult
        from station_dispatch.schedule import Assignment, Schedule

        runner = load_runner_module()
        row = {
            "experiment_group": "main",
            "instance_id": "synthetic",
            "instance_path": "data/synthetic.dzn",
            "train_count": "1",
            "seeds": "42",
            "time_limit_s": "300",
        }
        instance = StationInstance(
            path=Path("synthetic.dzn"),
            edge_count=1,
            edge_names=("edge",),
            edge_types=("inter",),
            platform_names=(),
            trains=(Train(1, "T1", 0, "pass", (1,)),),
            routes=(Route(1, 1, "", 0, 5, (1,)),),
            blocks=(Block(1, 1, 5, 0, False),),
        )
        schedule = Schedule((Assignment(1, 1, 0, 0),))
        result = MilpResult(
            schedule=schedule,
            status="feasible",
            feasible=True,
            optimal=False,
            runtime_s=12.3,
            lower_bound=4,
            gap=0.2,
            end_sum=5,
            makespan=5,
            incumbent_source="SCIP",
            message="feasible incumbent returned",
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            with patch.object(runner, "load_instance", return_value=instance), patch(
                "station_dispatch.milp.solve_milp", return_value=result
            ):
                result_path = runner.run_case(row, "milp", 42, Path(temporary_directory))
            payload = json.loads(result_path.read_text(encoding="utf8"))

        self.assertIsNone(payload["run"]["first_feasible_s"])
        self.assertIsNone(payload["run"]["best_found_s"])
        self.assertEqual(payload["run"]["status"], "feasible")
        self.assertFalse(payload["run"]["optimal"])
        self.assertEqual(payload["run"]["lower_bound"], 4)
        self.assertEqual(payload["run"]["gap"], 0.2)

    def test_importing_the_runner_does_not_require_the_milp_backend(self) -> None:
        command = (
            "import sys; "
            "from station_dispatch import experiment_runner; "
            "print('station_dispatch.milp' in sys.modules)"
        )
        completed = subprocess.run(
            [sys.executable, "-c", command],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.stdout.strip(), "False")

    def test_selects_the_manifest_seed_for_every_main_method(self) -> None:
        runner = load_runner_module()
        main_row = {"experiment_group": "main", "seeds": "42"}
        sensitivity_row = {
            "experiment_group": "seed_sensitivity",
            "seeds": ";".join(str(seed) for seed in (*range(37, 42), *range(43, 48))),
        }

        for method in runner.METHODS:
            self.assertEqual(runner.seeds_for_method(main_row, method), (42,))
        self.assertEqual(runner.seeds_for_method(sensitivity_row, "qea_lns"), tuple((*range(37, 42), *range(43, 48))))
        self.assertEqual(runner.seeds_for_method(sensitivity_row, "fcfs"), tuple((*range(37, 42), *range(43, 48))))
        self.assertEqual(runner.seeds_for_method(sensitivity_row, "milp"), tuple((*range(37, 42), *range(43, 48))))

    def test_main_and_seed_sensitivity_expand_to_their_approved_qea_runs(self) -> None:
        runner = load_runner_module()
        calls: list[tuple[str, str, str, int]] = []

        def record_run(command, **kwargs):
            request = json.loads(kwargs["input"])
            row, method, seed = request["row"], request["method"], request["seed"]
            self.assertEqual(Path(command[0]), runner.PROJECT_ROOT / ".venv/Scripts/python.exe")
            self.assertEqual(kwargs["env"]["PYTHONHASHSEED"], "0")
            self.assertEqual(kwargs["env"]["OMP_NUM_THREADS"], "1")
            self.assertEqual(kwargs["env"]["OPENBLAS_NUM_THREADS"], "1")
            self.assertTrue(kwargs["check"])
            calls.append((row["experiment_group"], method, row["instance_id"], seed))

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory)
            with patch.object(runner.subprocess, "run", side_effect=record_run):
                runner.run_manifest(
                    Path("experiments/manifest.csv"), output_root, "main", runner.METHODS
                )
                main_calls = list(calls)
                calls.clear()
                runner.run_manifest(
                    Path("experiments/manifest.csv"), output_root, "seed_sensitivity", runner.METHODS
                )
                sensitivity_calls = list(calls)

        self.assertEqual(
            main_calls,
            [
                ("main", method, instance_id, 42)
                for instance_id in ("t030-01", "t030-02", "t030-03")
                for method in runner.METHODS
            ],
        )
        self.assertEqual(
            sensitivity_calls,
            [("seed_sensitivity", method, "t050-01", seed)
             for method in runner.METHODS
             for seed in (*range(37, 42), *range(43, 48))
             for repetition in range(1, 4)],
        )


class ExperimentEntryPointTest(unittest.TestCase):
    def test_main_script_runs_only_the_main_group(self) -> None:
        module = load_script_module("schedule_bench.py")
        with patch.object(module, "run_manifest", return_value=[]) as run_manifest:
            module.run_main(Path("manifest.csv"), Path("raw_results"), ("qea_lns",))
        run_manifest.assert_called_once_with(
            Path("manifest.csv"), Path("raw_results"), "main", ("qea_lns",)
        )

    def test_scalability_script_runs_only_the_scalability_group(self) -> None:
        module = load_script_module("run_scalability.py")
        with patch.object(module, "run_manifest", return_value=[]) as run_manifest:
            module.run_scalability(Path("manifest.csv"), Path("raw_results"), ("milp",))
        run_manifest.assert_called_once_with(
            Path("manifest.csv"), Path("raw_results"), "scalability", ("milp",)
        )

    def test_seed_sensitivity_script_runs_only_selected_methods(self) -> None:
        module = load_script_module("run_seed_sensitivity.py")
        with patch.object(module, "run_manifest", return_value=[]) as run_manifest:
            module.run_seed_sensitivity(Path("manifest.csv"), Path("raw_results"), ("fcfs", "milp"))
        run_manifest.assert_called_once_with(
            Path("manifest.csv"), Path("raw_results"), "seed_sensitivity", ("fcfs", "milp"),
        )


if __name__ == "__main__":
    unittest.main()


class ResultSummaryTimingTest(unittest.TestCase):
    def test_summary_exposes_only_the_two_experiment_timing_measures(self) -> None:
        module = load_script_module("summarize_results.py")
        payload = {
            "input": {"experiment_group": "main", "instance_id": "t030-01", "train_count": 30, "time_limit_s": 300},
            "method": "fcfs",
            "seed": 42,
            "run": {"status": "completed", "runtime_s": 0.01, "first_feasible_s": 0.01, "best_found_s": 0.01},
            "objectives": {"sat": True, "total_station_wait": 0, "makespan": 1, "end_sum": 1},
            "metrics": {},
            "validation": {"feasible": True},
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            raw = Path(temporary_directory) / "raw_results" / "main" / "fcfs" / "t030-01"
            raw.mkdir(parents=True)
            (raw / "seed-042.json").write_text(json.dumps(payload), encoding="utf8")
            rows = module.load_runs(Path(temporary_directory) / "raw_results")
        self.assertEqual(rows[0]["first_feasible_s"], 0.01)
        self.assertEqual(rows[0]["best_found_s"], 0.01)
        self.assertNotIn("runtime_s", rows[0])


class ExperimentIsolationTest(unittest.TestCase):
    def test_expected_no_solution_is_recorded_but_programming_errors_propagate(self):
        from station_dispatch.limits import NoFeasibleSchedule
        runner = load_runner_module()
        row = runner.load_manifest(Path("experiments/manifest.csv"), "main")[0]
        with tempfile.TemporaryDirectory() as directory:
            for method in ("fcfs", "qea_lns"):
                with patch.object(runner, "solve_" + method, side_effect=NoFeasibleSchedule("budget exhausted")):
                    path = runner.run_case(row, method, 42, Path(directory))
                payload = json.loads(path.read_text(encoding="utf8"))
                self.assertEqual(payload["run"]["status"], "no_solution")
                self.assertIsNone(payload["objectives"]["total_station_wait"])
                self.assertIsNone(payload["run"]["first_feasible_s"])
                with patch.object(runner, "solve_" + method, side_effect=RuntimeError("implementation bug")):
                    with self.assertRaisesRegex(RuntimeError, "implementation bug"):
                        runner.run_case(row, method, 42, Path(directory))

    def test_timeout_without_message_is_still_a_no_solution_record(self):
        runner = load_runner_module()
        row = runner.load_manifest(Path("experiments/manifest.csv"), "main")[0]
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(runner, "solve_fcfs", side_effect=TimeoutError()):
                path = runner.run_case(row, "fcfs", 42, Path(directory))
            payload = json.loads(path.read_text(encoding="utf8"))
        self.assertEqual(payload["run"]["status"], "no_solution")
        self.assertTrue(payload["validation"]["violations"][0])

    def test_environment_rejects_version_drift(self):
        runner = load_runner_module()
        with patch.dict(runner.os.environ, runner.WORKER_ENV):
            with patch.object(runner.platform, "python_version", return_value="3.13.0"):
                with self.assertRaisesRegex(RuntimeError, "Python 3.12.14"):
                    runner.experiment_environment()
            with patch.object(runner.importlib.metadata, "version", return_value="0.0"):
                with self.assertRaisesRegex(RuntimeError, "dependency mismatch"):
                    runner.experiment_environment()

    def test_manifest_worker_records_actual_fixed_runtime(self):
        runner = load_runner_module()
        with tempfile.TemporaryDirectory() as directory:
            paths = runner.run_manifest(Path("experiments/manifest.csv"), Path(directory), "main", ("fcfs",))
            payloads = [json.loads(path.read_text(encoding="utf8")) for path in paths]
        self.assertEqual(len(payloads), 3)
        self.assertEqual(len({p["environment"]["process_id"] for p in payloads}), 3)
        self.assertEqual(len({p["environment"]["condition_id"] for p in payloads}), 1)
        for p in payloads:
            self.assertTrue(p["validation"]["feasible"])
            self.assertEqual(p["environment"]["python"], "3.12.14")
            self.assertEqual(p["environment"]["thread_environment"]["PYTHONHASHSEED"], "0")
            self.assertEqual(len(p["input"]["sha256"]), 64)

    def test_summary_ignores_off_manifest_and_counts_failed_runs(self):
        module = load_script_module("summarize_results.py")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "seed_sensitivity/milp/t050-01"
            target.mkdir(parents=True)
            for seed in (37, 38, 42):
                payload = {"input": {"experiment_group": "seed_sensitivity", "instance_id": "t050-01", "train_count": 50, "time_limit_s": 300, "repetitions": 5}, "method": "milp", "seed": seed, "repetition": 1, "objectives": {"sat": False, "total_station_wait": None}}
                (target / f"seed-{seed:03}.json").write_text(json.dumps(payload), encoding="utf8")
            with self.assertWarnsRegex(UserWarning, "outside current manifest"):
                rows = module.load_runs(root)
            self.assertEqual(len(rows), 2)
            module.summarize(rows, root)
            import pandas as pd
            stats = pd.read_csv(root / "seed_sensitivity_seed_stats.csv")
            self.assertEqual(stats["run_count"].tolist(), [1, 1])
            self.assertEqual(stats.iloc[0]["feasible_count"], 0)
            self.assertEqual(stats.iloc[0]["success_rate"], 0)
            self.assertTrue(pd.isna(stats.iloc[0]["total_station_wait_mean"]))


class ResultConditionGroupingTest(unittest.TestCase):
    def test_different_conditions_are_not_averaged_together(self):
        module = load_script_module("summarize_results.py")
        base = dict(experiment_group="seed_sensitivity", instance_id="t050-01", method="qea_lns", seed=37, feasible=True, input_sha256="same-input", entry_wait_total=0, excess_dwell_total=0, makespan=1, end_sum=1)
        rows = [dict(base, condition_id="legacy_unrecorded", total_station_wait=100), dict(base, condition_id="new-runtime", total_station_wait=20)]
        with tempfile.TemporaryDirectory() as directory:
            module.summarize(rows, Path(directory))
            import pandas as pd
            stats = pd.read_csv(Path(directory) / "seed_sensitivity_seed_stats.csv")
        self.assertEqual(len(stats), 2)
        self.assertEqual(set(stats["total_station_wait_mean"]), {20, 100})


class ScheduleDetailsTest(unittest.TestCase):
    def test_exports_actual_movements_and_origin_destination_reservations(self):
        from station_dispatch.schedule import Assignment, Schedule, build_block_occupancies, resource_occupancies, planning_horizon
        from test_schedule import small_instance
        runner = load_runner_module()
        instance = small_instance()
        schedule = Schedule((Assignment(3, 3, 10, 0), Assignment(5, 5, 20, 0)))
        details = runner.schedule_details(instance, schedule)
        self.assertEqual(details["planning_horizon"], list(planning_horizon(instance)))
        actual = build_block_occupancies(instance, schedule)
        reserved = resource_occupancies(instance, schedule)
        self.assertEqual(len(details["occupancies"]), len(actual))
        for row, movement, resource in zip(details["occupancies"], actual, reserved):
            self.assertEqual((row["operation_start"], row["operation_end"]), (movement.start, movement.end))
            self.assertEqual((row["resource_start"], row["resource_end"]), (resource.start, resource.end))
            self.assertEqual(row["edge_name"], instance.edge_names[movement.edge_id - 1])
        self.assertTrue(any(r["resource_start"] < r["operation_start"] for r in details["occupancies"]))
        self.assertTrue(any(r["resource_end"] > r["operation_end"] for r in details["occupancies"]))
        for train_id in (3, 5):
            rows = [r for r in details["occupancies"] if r["train_id"] == train_id]
            self.assertEqual([r["sequence"] for r in rows], list(range(1, len(rows) + 1)))


class ScalabilitySummaryTest(unittest.TestCase):
    def rows(self):
        return [dict(experiment_group=group, instance_id=f"t{n:03d}-01", train_count=n,
                     method="milp", condition_id="same", seed=42, time_limit_s=300,
                     feasible=ok, total_station_wait=wait, entry_wait_total=wait,
                     excess_dwell_total=0, makespan=100, end_sum=200,
                     first_feasible_s=None, best_found_s=None)
                for group,n,ok,wait in [("main",30,True,60),("scalability",35,False,0)]]

    def test_scale_summary_does_not_borrow_main_and_keeps_failed_quality_missing(self):
        import pandas as pd
        module = load_script_module("summarize_results.py")
        with tempfile.TemporaryDirectory() as tmp:
            module.summarize(self.rows(), Path(tmp))
            table = pd.read_csv(Path(tmp) / "scalability_scale_stats.csv")
        self.assertEqual(table["train_count"].tolist(), [35])
        self.assertEqual(table["feasible_count"].tolist(), [0])
        self.assertTrue(pd.isna(table.iloc[0]["total_station_wait_mean"]))

    def test_scale_summary_rejects_mixed_environment_before_writing(self):
        module = load_script_module("summarize_results.py")
        rows = self.rows()
        rows.append(dict(rows[1], instance_id="t035-02", condition_id="other"))
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "condition"):
                module.summarize(rows, Path(tmp))
            self.assertEqual(list(Path(tmp).iterdir()), [])


class TimeBudgetTest(unittest.TestCase):
    def test_budget_paths_do_not_overwrite_and_all_methods_receive_budget(self):
        from station_dispatch.limits import NoFeasibleSchedule
        runner = load_runner_module()
        with tempfile.TemporaryDirectory() as tmp:
            for method, target in [("fcfs", "station_dispatch.experiment_runner.solve_fcfs"),
                                   ("milp", "station_dispatch.milp.solve_milp"),
                                   ("qea_lns", "station_dispatch.experiment_runner.solve_qea_lns")]:
                paths = []
                for budget in (10,600):
                    row = dict(experiment_group="time_budget", instance_id="t050-01",
                               instance_path="data/cp2025/t050-01.dzn",train_count="50",
                               seeds="42",time_limit_s=str(budget))
                    with patch(target, side_effect=NoFeasibleSchedule("budget exhausted")) as solve:
                        path = runner.run_case(row,method,42,Path(tmp))
                    self.assertEqual(solve.call_args.kwargs["time_limit_s"],budget)
                    self.assertEqual(path.parent.name,f"budget-{budget:03d}")
                    paths.append(path)
                self.assertEqual(len(set(paths)),2)
                self.assertTrue(all(p.exists() for p in paths))

    def test_time_budget_entry_expands_81_distinct_paths(self):
        runner = load_runner_module()
        module = load_script_module("run_time_budget.py")
        with tempfile.TemporaryDirectory() as tmp, patch.object(runner.subprocess,"run") as process:
            paths = module.run_time_budget(Path("experiments/manifest.csv"),Path(tmp),runner.METHODS)
        self.assertEqual(len(paths),81)
        self.assertEqual(len(set(paths)),81)
        self.assertEqual(process.call_count,81)
        summary = load_script_module("summarize_results.py")
        self.assertIn("time_budget",summary.GROUPS)




class ResultStatisticsTest(unittest.TestCase):
    def rows(self, group="seed_sensitivity", values=(10, 20, 30)):
        return [dict(experiment_group=group, instance_id="t050-01", train_count=50,
                     method="qea_lns", condition_id="same", source_sha256="version-a",
                     input_sha256="input-a", seed=37, repetition=i+1, repetitions=len(values),
                     time_limit_s=300, feasible=True,
                     total_station_wait=value, entry_wait_total=value, excess_dwell_total=0,
                     makespan=100, end_sum=200, completion_deviation_total=value,
                     route_change_count=1, max_entry_wait=value, entry_wait_std=value,
                     station_sojourn_total=200, first_feasible_s=0.01, best_found_s=1.0)
                for i, value in enumerate(values)]

    def summary(self, rows, filename="seed_sensitivity_seed_stats.csv"):
        import pandas as pd
        module = load_script_module("summarize_results.py")
        with tempfile.TemporaryDirectory() as directory:
            module.summarize(rows, Path(directory))
            return pd.read_csv(Path(directory) / filename)

    def test_run_statistics_and_point_repeat_mean_t_interval(self):
        row = self.summary(self.rows()).iloc[0]
        self.assertEqual(row["replication_unit"], "repeat")
        self.assertEqual(row["total_station_wait_n"], 3)
        self.assertEqual(row["total_station_wait_mean"], 20)
        self.assertEqual(row["total_station_wait_min"], 10)
        self.assertEqual(row["total_station_wait_max"], 30)
        self.assertEqual(row["total_station_wait_range"], 20)
        self.assertEqual(row["total_station_wait_std"], 10)
        self.assertAlmostEqual(row["total_station_wait_sem"], 5.773502691896258)
        self.assertAlmostEqual(row["total_station_wait_ci95_low"], -4.841377117, places=6)
        self.assertAlmostEqual(row["total_station_wait_ci95_high"], 44.841377117, places=6)
        self.assertEqual(row["entry_wait_std_std"], 10)
        self.assertEqual(row["ci_method"], "student_t")

    def test_budget_single_run_has_no_dispersion_or_ci(self):
        import pandas as pd
        row = self.summary(self.rows("time_budget", (10,)), "time_budget_stats.csv").iloc[0]
        self.assertEqual(row["total_station_wait_n"], 1)
        for suffix in ("std", "sem", "ci95_low", "ci95_high"):
            self.assertTrue(pd.isna(row[f"total_station_wait_{suffix}"]))

    def test_main_statistics_use_instances_without_inferential_ci(self):
        import pandas as pd
        rows = self.rows("main")
        for i, row in enumerate(rows):
            row.update(instance_id=f"t030-{i+1:02}", input_sha256=f"input-{i}", train_count=30, seed=42)
        stats = self.summary(rows, "main_summary.csv").iloc[0]
        self.assertEqual(stats["replication_unit"], "instance")
        self.assertEqual(stats["instance_count"], 3)
        self.assertEqual(stats["run_count"], 3)
        self.assertEqual(stats["total_station_wait_std"], 10)
        self.assertTrue(pd.isna(stats["total_station_wait_ci95_low"]))

    def test_budgets_and_source_versions_are_separate_groups(self):
        rows = self.rows()
        rows[1]["time_limit_s"] = 60
        rows[2]["source_sha256"] = "version-b"
        table = self.summary(rows)
        self.assertEqual(len(table), 3)
        self.assertEqual(set(table["time_limit_s"]), {60, 300})
        self.assertEqual(set(table["source_sha256"]), {"version-a", "version-b"})

    def test_failed_runs_have_no_schedule_metrics_or_discovery_times(self):
        import pandas as pd
        rows = self.rows(values=(10,))
        rows[0]["feasible"] = False
        table = self.summary(rows, "all_runs.csv")
        for field in ("completion_deviation_total", "route_change_count", "max_entry_wait",
                      "entry_wait_std", "station_sojourn_total", "first_feasible_s", "best_found_s"):
            self.assertTrue(pd.isna(table.iloc[0][field]), field)
        stats = self.summary(rows).iloc[0]
        self.assertEqual(stats["run_count"], 1)
        self.assertEqual(stats["feasible_count"], 0)
        self.assertEqual(stats["total_station_wait_n"], 0)
        self.assertEqual(stats["success_rate"], 0)

    def test_empty_input_replaces_only_owned_summary_files(self):
        module = load_script_module("summarize_results.py")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            module.summarize(self.rows(), root)
            unrelated = root / "scalability_instance_features.csv"
            unrelated.write_text("keep", encoding="utf8")
            module.summarize([], root)
            self.assertFalse((root / "seed_sensitivity_seed_stats.csv").exists())
            self.assertFalse((root / "seed_sensitivity_runs.csv").exists())
            self.assertEqual(unrelated.read_text(encoding="utf8"), "keep")

    def test_duplicate_experiment_keys_are_rejected(self):
        module = load_script_module("summarize_results.py")
        source = json.dumps({"input": {"experiment_group": "main", "instance_id": "t030-01", "train_count": 30, "time_limit_s": 300}, "method": "fcfs", "seed": 42, "objectives": {"sat": True}, "validation": {"feasible": True}})
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "main/fcfs/t030-01"
            target.mkdir(parents=True)
            for name in ("seed-042.json", "duplicate.json"):
                (target / name).write_text(source, encoding="utf8")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                module.load_runs(Path(directory))

    def test_conflicting_feasibility_flags_are_rejected(self):
        module = load_script_module("summarize_results.py")
        payload = {"input": {"experiment_group": "main", "instance_id": "t030-01", "train_count": 30, "time_limit_s": 300}, "method": "fcfs", "seed": 42, "objectives": {"sat": True}, "validation": {"feasible": True}}
        payload["validation"]["feasible"] = False
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "main/fcfs/t030-01"
            target.mkdir(parents=True)
            (target / "seed-042.json").write_text(json.dumps(payload), encoding="utf8")
            with self.assertRaisesRegex(ValueError, "feasibility"):
                module.load_runs(Path(directory))


class PortableRuntimeTest(unittest.TestCase):
    def test_environment_condition_ignores_checkout_absolute_path(self):
        import shutil
        runner = load_runner_module()
        with patch.dict(runner.os.environ, runner.WORKER_ENV):
            first = runner.experiment_environment()
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                for name in (".python-version", "requirements-experiments.txt"):
                    shutil.copyfile(runner.PROJECT_ROOT / name, root / name)
                shutil.copytree(runner.PROJECT_ROOT / "station_dispatch", root / "station_dispatch",
                                ignore=shutil.ignore_patterns("__pycache__"))
                executable = root / ".venv/Scripts/python.exe"
                with patch.object(runner, "PROJECT_ROOT", root), patch.object(runner, "PROJECT_PYTHON", executable), patch.object(runner.sys, "executable", str(executable)):
                    second = runner.experiment_environment()
        self.assertNotEqual(first["executable"], second["executable"])
        self.assertEqual(first["condition_id"], second["condition_id"])

    def test_runtime_rejects_solver_backend_version_drift(self):
        from ortools.linear_solver import pywraplp
        runner = load_runner_module()
        backend = pywraplp.Solver.CreateSolver("SCIP_MIXED_INTEGER_PROGRAMMING")
        with patch.dict(runner.os.environ, runner.WORKER_ENV), patch.object(pywraplp.Solver, "CreateSolver", return_value=backend), patch.object(backend, "SolverVersion", return_value="SCIP 9.0.0"):
            with self.assertRaisesRegex(RuntimeError, "backend mismatch"):
                runner.experiment_environment()


class SummaryDuplicateTest(unittest.TestCase):
    def test_duplicate_rows_are_rejected_before_writing(self):
        module = load_script_module("summarize_results.py")
        row = ResultStatisticsTest().rows(values=(10,))[0]
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                module.summarize([row, dict(row)], Path(directory))
            self.assertEqual(list(Path(directory).iterdir()), [])


class PointRepeatExperimentTest(unittest.TestCase):
    def test_workers_keep_seed_and_budget_and_use_unique_repeat_paths(self):
        runner = load_runner_module()
        for group, count in (("main", 9), ("scalability", 108), ("seed_sensitivity", 90), ("time_budget", 81)):
            with self.subTest(group=group), tempfile.TemporaryDirectory() as tmp, patch.object(runner.subprocess, "run") as process:
                paths = runner.run_manifest(Path("experiments/manifest.csv"), Path(tmp), group, runner.METHODS)
                self.assertEqual(len(paths), count)
                self.assertEqual(len(set(paths)), count)
                for call in process.call_args_list:
                    request = json.loads(call.kwargs["input"])
                    self.assertIn(request["seed"], runner.seeds_for_method(request["row"], request["method"]))
                    self.assertIn(request["repetition"], range(1, (1 if group == "main" else 3) + 1))
                    self.assertEqual(call.kwargs["env"]["OMP_NUM_THREADS"], "1")

    def test_point_statistics_do_not_pool_seeds_or_budgets(self):
        import pandas as pd
        module = load_script_module("summarize_results.py")
        for group in ("scalability", "seed_sensitivity", "time_budget"):
            rows = []
            for point, seed, budget in ((0, 42, 300), (1, 43 if group == "seed_sensitivity" else 42, 60 if group == "time_budget" else 300)):
                for repetition, value in enumerate((10, 20, 30, 40, 50), 1):
                    rows.append(dict(experiment_group=group, instance_id=f"t050-{point+1:02d}" if group == "scalability" else "t050-01",
                                     train_count=50, method="qea_lns", seed=seed, time_limit_s=budget,
                                     repetition=repetition, repetitions=5, condition_id="fixed", source_sha256="v", input_sha256=f"i{point}" if group == "scalability" else "i",
                                     feasible=True, total_station_wait=value + 100 * point, entry_wait_total=value, excess_dwell_total=0,
                                     makespan=100, end_sum=200, first_feasible_s=.01, best_found_s=float(value)))
            with self.subTest(group=group), tempfile.TemporaryDirectory() as tmp:
                module.summarize(rows, Path(tmp))
                table = pd.read_csv(Path(tmp) / module.STAT_FILES[group])
                self.assertEqual(len(table), 2)
                self.assertEqual(set(table["replication_unit"]), {"repeat"})
                self.assertEqual(table["run_count"].tolist(), [5, 5])
                self.assertEqual(set(table["total_station_wait_mean"]), {30, 130})
                for sem in table["total_station_wait_sem"]:
                    self.assertAlmostEqual(sem, 7.0710678118654755)


class PointRepeatRecordTest(unittest.TestCase):
    def test_historical_single_record_is_not_a_repeat_and_repeat_files_load_distinctly(self):
        module = load_script_module("summarize_results.py")
        payload = {"input": {"experiment_group": "seed_sensitivity", "instance_id": "t050-01",
                             "train_count": 50, "time_limit_s": 300, "repetitions": 5},
                   "method": "fcfs", "seed": 37, "repetition": 1,
                   "objectives": {"sat": True, "total_station_wait": 100},
                   "validation": {"feasible": True}}
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "seed_sensitivity/fcfs/t050-01"
            folder.mkdir(parents=True)
            historical = dict(payload)
            historical.pop("repetition")
            (folder / "seed-037.json").write_text(json.dumps(historical), encoding="utf8")
            repeat_folder = folder / "seed-037"
            repeat_folder.mkdir()
            for repetition in range(1, 6):
                (repeat_folder / f"repeat-{repetition:02d}.json").write_text(
                    json.dumps(dict(payload, repetition=repetition)), encoding="utf8")
            with self.assertWarnsRegex(UserWarning, "repetition design"):
                rows = module.load_runs(Path(directory))
            self.assertEqual([row["repetition"] for row in rows], [1, 2, 3])
            self.assertEqual({row["seed"] for row in rows}, {37})
            self.assertEqual({row["repetitions"] for row in rows}, {3})
            (repeat_folder / "duplicate.json").write_text(json.dumps(payload), encoding="utf8")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                module.load_runs(Path(directory))

    def test_three_fresh_fcfs_workers_retain_schedules_and_point_statistics(self):
        import csv
        import pandas as pd
        runner = load_runner_module()
        summary = load_script_module("summarize_results.py")
        row = runner.load_manifest(Path("experiments/manifest.csv"), "seed_sensitivity")[0]
        row["seeds"] = "37"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.csv"
            with manifest.open("w", newline="", encoding="utf8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)
            paths = runner.run_manifest(manifest, root / "raw", "seed_sensitivity", ("fcfs",))
            payloads = [json.loads(path.read_text(encoding="utf8")) for path in paths]
            self.assertEqual(len({p["environment"]["process_id"] for p in payloads}), 3)
            self.assertEqual({p["seed"] for p in payloads}, {37})
            self.assertEqual(len({p["environment"]["condition_id"] for p in payloads}), 1)
            self.assertEqual(len({json.dumps(p["schedule"], sort_keys=True) for p in payloads}), 1)
            for p in payloads:
                self.assertTrue(p["validation"]["feasible"])
                self.assertEqual(len(p["schedule"]), 50)
                self.assertEqual(len(p["schedule_details"]["trains"]), 50)
                self.assertTrue(p["schedule_details"]["occupancies"])
            runs = summary.load_runs(root / "raw", manifest)
            summary.summarize(runs, root / "results")
            stat = pd.read_csv(root / "results/seed_sensitivity_seed_stats.csv").iloc[0]
            self.assertEqual(stat["run_count"], 3)
            self.assertEqual(stat["total_station_wait_sem"], 0)
            self.assertEqual(stat["repetition_ids"], "1;2;3")
