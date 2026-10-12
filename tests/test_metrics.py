from pathlib import Path
import unittest

from station_dispatch.dzn import Block, Route, StationInstance, Train
from station_dispatch.schedule import Assignment, Schedule


def metrics_instance() -> StationInstance:
    return StationInstance(
        path=Path("metrics.dzn"),
        edge_count=2,
        edge_names=("platform", "throat"),
        edge_types=("platform", "inter"),
        platform_names=("S_I",),
        trains=(
            Train(1, "T1", 0, "pass", (1,)),
            Train(2, "T2", 20, "pass", (2,)),
        ),
        routes=(
            Route(1, 1, "S_I", 5, 5, (1,)),
            Route(2, 2, "", 0, 10, (2,)),
        ),
        blocks=(
            Block(1, 1, 5, 0, True),
            Block(2, 2, 10, 0, False),
        ),
    )


class ScheduleMetricsTest(unittest.TestCase):
    def test_reports_plan_actual_deviation_without_changing_the_schedule(self) -> None:
        from station_dispatch.metrics import compute_metrics

        instance = metrics_instance()
        schedule = Schedule((Assignment(1, 1, 2, 7), Assignment(2, 2, 20, 0)))

        metrics = compute_metrics(instance, schedule)

        self.assertTrue(metrics.sat)
        self.assertEqual(metrics.end_sum, 44)
        self.assertEqual(metrics.makespan, 30)
        self.assertEqual(metrics.entry_wait_total, 2)
        self.assertEqual(metrics.max_entry_wait, 2)
        self.assertEqual(metrics.station_sojourn_total, 24)
        self.assertEqual(metrics.excess_dwell_total, 2)
        self.assertEqual(metrics.total_station_wait, 4)
        self.assertEqual(metrics.completion_deviation_total, 4)
        self.assertEqual(metrics.by_train[1].entry_wait, 2)
        self.assertEqual(metrics.by_train[1].station_sojourn, 14)
        self.assertEqual(metrics.by_train[1].excess_dwell, 2)
        self.assertEqual(metrics.by_train[1].completion_deviation, 4)
        self.assertEqual(metrics.platform_utilization["platform"], 0.4)
        self.assertEqual(metrics.throat_utilization["throat"], round(10 / 30, 6))


if __name__ == "__main__":
    unittest.main()


