from pathlib import Path
import unittest

from station_dispatch.dzn import Block, Route, StationInstance, Train


class FcfsTest(unittest.TestCase):
    def test_fcfs_scans_all_allowed_routes_and_uses_the_earliest_safe_one(self) -> None:
        from station_dispatch.fcfs import solve_fcfs
        from station_dispatch.schedule import validate_schedule

        instance = StationInstance(
            path=Path("fcfs.dzn"),
            edge_count=3,
            edge_names=("entry", "alternate", "exit"),
            edge_types=("border", "inter", "border"),
            platform_names=(),
            trains=(
                Train(1, "T1", 0, "pass", (1,)),
                Train(2, "T2", 0, "pass", (2, 3)),
            ),
            routes=(
                Route(1, 1, "", 0, 5, (1,)),
                Route(2, 2, "", 0, 5, (2,)),
                Route(3, 2, "", 0, 5, (3,)),
            ),
            blocks=(
                Block(1, 1, 5, 0, False),
                Block(2, 1, 5, 0, False),
                Block(3, 2, 5, 0, False),
            ),
        )

        schedule = solve_fcfs(instance)

        self.assertEqual([(item.train_id, item.route_id, item.start) for item in schedule.assignments], [(1, 1, 0), (2, 3, 0)])
        self.assertTrue(validate_schedule(instance, schedule).feasible)

    def test_fcfs_does_not_give_a_later_origin_unconditional_throat_priority(self) -> None:
        from station_dispatch.fcfs import solve_fcfs
        from station_dispatch.schedule import validate_schedule

        instance = StationInstance(
            path=Path("origin-throat-order.dzn"),
            edge_count=3,
            edge_names=("origin_platform", "arrival_platform", "shared_throat"),
            edge_types=("platform", "platform", "inter"),
            platform_names=("S_I", "S_II"),
            trains=(
                Train(1, "arrival", 0, "pass", (1,)),
                Train(2, "origin", 10, "origin", (2,)),
            ),
            routes=(
                Route(1, 1, "S_II", 0, 20, (1, 2)),
                Route(2, 2, "S_I", 0, 5, (3, 4)),
            ),
            blocks=(
                Block(1, 2, 0, 0, True),
                Block(2, 3, 20, 0, False),
                Block(3, 1, 0, 0, True),
                Block(4, 3, 5, 0, False),
            ),
        )

        schedule = solve_fcfs(instance)
        by_train = {item.train_id: item for item in schedule.assignments}

        self.assertEqual(by_train[1].start, 0)
        self.assertEqual(by_train[2].start, 20)
        self.assertTrue(validate_schedule(instance, schedule).feasible)

    def test_fcfs_does_not_schedule_a_same_entry_successor_while_its_predecessor_is_blocked(self) -> None:
        from station_dispatch.fcfs import solve_fcfs
        from station_dispatch.schedule import validate_schedule

        instance = StationInstance(
            path=Path("fcfs-blocked-entry-fifo.dzn"),
            edge_count=4,
            edge_names=("entry", "origin_platform", "other_platform", "exit"),
            edge_types=("border", "platform", "platform", "border"),
            platform_names=("S_I", "S_II"),
            trains=(
                Train(1, "predecessor", 0, "pass", (1,)),
                Train(2, "successor", 1, "pass", (2,)),
                Train(3, "origin", 10, "origin", (3,)),
            ),
            routes=(
                Route(1, 1, "S_I", 0, 10, (1, 2)),
                Route(2, 2, "S_II", 0, 10, (3, 4)),
                Route(3, 3, "S_I", 0, 10, (5, 6)),
            ),
            blocks=(
                Block(1, 1, 5, 0, False),
                Block(2, 2, 5, 0, True),
                Block(3, 1, 5, 0, False),
                Block(4, 3, 5, 0, True),
                Block(5, 2, 5, 0, True),
                Block(6, 4, 5, 0, False),
            ),
        )

        schedule = solve_fcfs(instance)
        by_train = {item.train_id: item for item in schedule.assignments}

        self.assertLessEqual(by_train[1].start, by_train[2].start)
        self.assertTrue(validate_schedule(instance, schedule).feasible)

    def test_fcfs_reserves_an_origin_platform_before_processing_arriving_trains(self) -> None:
        from station_dispatch.fcfs import solve_fcfs
        from station_dispatch.schedule import validate_schedule

        instance = StationInstance(
            path=Path("origin-before-entry.dzn"),
            edge_count=1,
            edge_names=("platform",),
            edge_types=("platform",),
            platform_names=("S_I",),
            trains=(
                Train(1, "arrival", 0, "pass", (1,)),
                Train(2, "origin", 10, "origin", (2,)),
            ),
            routes=(
                Route(1, 1, "S_I", 0, 5, (1,)),
                Route(2, 2, "S_I", 0, 5, (2,)),
            ),
            blocks=(
                Block(1, 1, 5, 0, False),
                Block(2, 1, 5, 0, True),
            ),
        )

        schedule = solve_fcfs(instance)
        by_train = {item.train_id: item for item in schedule.assignments}

        self.assertEqual(by_train[2].start, 10)
        self.assertEqual(by_train[1].start, 15)
        self.assertTrue(validate_schedule(instance, schedule).feasible)


if __name__ == "__main__":
    unittest.main()


class FcfsTimeLimitTest(unittest.TestCase):
    def test_fcfs_uses_the_common_300_second_default(self) -> None:
        import inspect

        from station_dispatch.fcfs import solve_fcfs
        from station_dispatch.limits import TIME_LIMIT_S

        self.assertEqual(TIME_LIMIT_S, 300)
        self.assertEqual(inspect.signature(solve_fcfs).parameters["time_limit_s"].default, TIME_LIMIT_S)


class FcfsValidationCostTest(unittest.TestCase):
    def test_no_repeated_prefix_checks_but_final_schedule_is_validated(self):
        from unittest.mock import patch
        import station_dispatch.fcfs as fcfs
        from station_dispatch.dzn import load_instance
        instance = load_instance(Path("data/cp2025/t030-01.dzn"))
        with patch.object(fcfs, "earliest_feasible_assignment", wraps=fcfs.earliest_feasible_assignment) as assign, patch.object(fcfs, "validate_schedule", wraps=fcfs.validate_schedule) as validate:
            schedule = fcfs.solve_fcfs(instance)
        self.assertEqual(len(schedule.assignments), 30)
        validate.assert_called_once_with(instance, schedule)
        for call in assign.call_args_list:
            self.assertIs(call.kwargs.get("validate_existing"), False)
            self.assertIs(call.kwargs.get("validate_candidate"), False)


    def test_removing_redundant_checks_preserves_all_four_experiment_case_schedules(self):
        from unittest.mock import patch
        import station_dispatch.fcfs as fcfs
        from station_dispatch.dzn import load_instance
        original_assign = fcfs.earliest_feasible_assignment
        def with_repeated_validation(*args, **kwargs):
            kwargs.update(validate_existing=True, validate_candidate=True)
            return original_assign(*args, **kwargs)
        for case in ("t030-01", "t030-02", "t030-03", "t050-01"):
            with self.subTest(case=case):
                instance = load_instance(Path(f"data/cp2025/{case}.dzn"))
                with patch.object(fcfs, "earliest_feasible_assignment", side_effect=with_repeated_validation):
                    before = fcfs.solve_fcfs(instance)
                after = fcfs.solve_fcfs(instance)
                self.assertEqual(before, after)
