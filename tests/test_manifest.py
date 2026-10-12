from pathlib import Path
import importlib.util
import unittest


def load_script_module():
    path = Path("scripts/create_instance_files.py")
    spec = importlib.util.spec_from_file_location("create_instance_files", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ExperimentManifestTest(unittest.TestCase):
    def test_manifest_separates_single_seed_experiments_from_seed_sensitivity(self) -> None:
        module = load_script_module()

        rows = module.build_manifest(Path("data/cp2025"))
        main_rows = [row for row in rows if row["experiment_group"] == "main"]
        scalability_rows = [row for row in rows if row["experiment_group"] == "scalability"]
        sensitivity_rows = [row for row in rows if row["experiment_group"] == "seed_sensitivity"]

        self.assertEqual(len(rows), 25)
        self.assertEqual({row["experiment_group"] for row in rows},
                         {"main", "scalability", "seed_sensitivity", "time_budget"})
        self.assertEqual([row["instance_id"] for row in main_rows], ["t030-01", "t030-02", "t030-03"])
        self.assertEqual(len(scalability_rows), 12)
        self.assertEqual([row["instance_id"] for row in sensitivity_rows], ["t050-01"])
        self.assertEqual({row["seeds"] for row in main_rows}, {"42"})
        self.assertEqual({row["seeds"] for row in scalability_rows}, {"42"})
        self.assertEqual(
            {row["seeds"] for row in sensitivity_rows},
            {";".join(str(seed) for seed in (*range(37, 42), *range(43, 48)))},
        )
        self.assertEqual({row["train_count"] for row in sensitivity_rows}, {50})
        self.assertEqual({row["time_limit_s"] for row in rows if row["experiment_group"] != "time_budget"}, {300})
        budgets = [row for row in rows if row["experiment_group"] == "time_budget"]
        self.assertEqual([r["time_limit_s"] for r in budgets], [10,20,30,40,50,60,120,300,600])
        self.assertEqual({r["instance_id"] for r in budgets}, {"t050-01"})
        self.assertEqual({r["seeds"] for r in budgets}, {"42"})


if __name__ == "__main__":
    unittest.main()



class PortableManifestTest(unittest.TestCase):
    def test_generated_paths_are_project_relative_posix_paths(self):
        module = load_script_module()
        rows = module.build_manifest(Path("data/cp2025"))
        self.assertEqual(rows[0]["instance_path"], "data/cp2025/t030-01.dzn")
        self.assertTrue(all(not Path(row["instance_path"]).is_absolute() for row in rows))


class PointRepeatManifestTest(unittest.TestCase):
    def test_only_sensitivity_groups_have_three_repeats(self):
        rows = load_script_module().build_manifest(Path("data/cp2025"))
        for row in rows:
            self.assertEqual(row["repetitions"], 1 if row["experiment_group"] == "main" else 3)
