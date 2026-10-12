from pathlib import Path
import unittest

from station_dispatch.dzn import Block, Route, StationInstance, Train


def small_instance() -> StationInstance:
    return StationInstance(
        path=Path("synthetic.dzn"),
        edge_count=4,
        edge_names=("entry", "exit", "platform", "side"),
        edge_types=("border", "border", "platform", "inter"),
        platform_names=("S_I",),
        trains=(
            Train(1, "T1", 10, "pass", (1,)),
            Train(2, "T2", 20, "pass", (2,)),
            Train(3, "T3", 10, "dest", (3,)),
            Train(4, "T4", 20, "pass", (4,)),
            Train(5, "T5", 10, "origin", (5,)),
            Train(6, "T6", 20, "pass", (6,)),
        ),
        routes=(
            Route(1, 1, "S_I", 5, 10, (1, 2, 3)),
            Route(2, 2, "S_I", 0, 10, (4, 5)),
            Route(3, 3, "S_I", 0, 5, (6,)),
            Route(4, 4, "S_I", 0, 5, (7,)),
            Route(5, 5, "S_I", 0, 5, (8,)),
            Route(6, 6, "S_I", 0, 5, (9,)),
        ),
        blocks=(
            Block(1, 1, 3, 0, False),
            Block(2, 3, 4, 0, True),
            Block(3, 2, 3, 0, False),
            Block(4, 1, 5, 0, False),
            Block(5, 2, 5, 0, False),
            Block(6, 3, 5, 0, True),
            Block(7, 3, 5, 0, False),
            Block(8, 3, 5, 0, True),
            Block(9, 3, 5, 0, False),
        ),
    )


class ScheduleValidationTest(unittest.TestCase):
    def test_block_timing_includes_minimum_dwell_and_route_completion(self) -> None:
        from station_dispatch.schedule import Assignment, Schedule, build_block_occupancies

        instance = small_instance()
        schedule = Schedule((Assignment(1, 1, 10, 5),))

        occupancies = build_block_occupancies(instance, schedule)

        self.assertEqual(
            [(item.block_id, item.start, item.end) for item in occupancies],
            [(1, 10, 13), (2, 13, 22), (3, 22, 25)],
        )
        self.assertEqual(schedule.completion_times(instance), {1: 25})

    def test_validator_rejects_early_start_and_unavailable_route(self) -> None:
        from station_dispatch.schedule import Assignment, Schedule, validate_schedule

        instance = small_instance()
        schedule = Schedule((Assignment(1, 2, 9, 0),))

        result = validate_schedule(instance, schedule)

        self.assertFalse(result.feasible)
        self.assertIn("train 1 starts before its earliest time", result.violations)
        self.assertIn("train 1 uses an unavailable route", result.violations)

    def test_validator_rejects_overlapping_use_of_one_controlled_segment(self) -> None:
        from station_dispatch.schedule import Assignment, Schedule, validate_schedule

        instance = small_instance()
        schedule = Schedule((Assignment(1, 1, 10, 5), Assignment(2, 2, 12, 0)))

        result = validate_schedule(instance, schedule)

        self.assertFalse(result.feasible)
        self.assertIn("edge 1 has overlapping train occupations", result.violations)

    def test_validator_rejects_reversed_order_at_one_station_entry(self) -> None:
        from station_dispatch.schedule import Assignment, Schedule, validate_schedule

        instance = small_instance()
        schedule = Schedule((Assignment(1, 1, 30, 5), Assignment(2, 2, 20, 0)))

        result = validate_schedule(instance, schedule)

        self.assertFalse(result.feasible)
        self.assertIn("entry edge 1 reverses earliest-time order", result.violations)

    def test_validator_keeps_destination_platform_reserved_to_horizon_end(self) -> None:
        from station_dispatch.schedule import Assignment, Schedule, validate_schedule

        instance = small_instance()
        schedule = Schedule((Assignment(3, 3, 10, 0), Assignment(4, 4, 20, 0)))

        result = validate_schedule(instance, schedule)

        self.assertFalse(result.feasible)
        self.assertIn("edge 3 has overlapping train occupations", result.violations)

    def test_validator_keeps_origin_platform_reserved_from_horizon_start(self) -> None:
        from station_dispatch.schedule import Assignment, Schedule, validate_schedule

        instance = small_instance()
        schedule = Schedule((Assignment(5, 5, 20, 0), Assignment(6, 6, 10, 0)))

        result = validate_schedule(instance, schedule)

        self.assertFalse(result.feasible)
        self.assertIn("edge 3 has overlapping train occupations", result.violations)

if __name__ == "__main__":
    unittest.main()



class PartialScheduleValidationTest(unittest.TestCase):
    def test_validator_accepts_a_safe_partial_schedule_when_requested(self) -> None:
        from station_dispatch.schedule import Assignment, Schedule, validate_schedule

        result = validate_schedule(small_instance(), Schedule((Assignment(1, 1, 10, 5),)), require_complete=False)

        self.assertTrue(result.feasible)




if __name__ == "__main__":
    unittest.main()



