from pathlib import Path
import unittest


class DznInputTest(unittest.TestCase):
    def test_t030_01_keeps_original_station_and_train_data(self) -> None:
        from station_dispatch.dzn import load_instance

        instance = load_instance(Path("data/cp2025/t030-01.dzn"))

        self.assertEqual(instance.edge_count, 45)
        self.assertEqual(instance.train_count, 30)
        self.assertEqual(instance.platform_names, ("S_I", "S_II", "S_III", "S_IV", "S_V"))
        self.assertEqual(instance.trains[0].route_ids, (1, 2, 3, 4, 5))
        self.assertEqual(instance.trains[0].earliest_start, 1350)
        self.assertEqual(instance.routes[0].platform_name, "S_I")


if __name__ == "__main__":
    unittest.main()
