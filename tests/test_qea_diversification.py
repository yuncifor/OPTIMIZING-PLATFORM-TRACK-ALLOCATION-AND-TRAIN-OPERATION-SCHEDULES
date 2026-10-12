from pathlib import Path
import math
import unittest

from station_dispatch.dzn import Block, Route, StationInstance, Train
from station_dispatch.schedule import Assignment


def qea_two_train_instance() -> StationInstance:
    return StationInstance(
        path=Path("qea-diversification-synthetic.dzn"),
        edge_count=2,
        edge_names=("shared_throat", "platform_route"),
        edge_types=("inter", "platform"),
        platform_names=("S_I",),
        trains=(
            Train(1, "T1", 0, "pass", (1, 2)),
            Train(2, "T2", 0, "pass", (3,)),
        ),
        routes=(
            Route(1, 1, "", 0, 10, (1,)),
            Route(2, 1, "S_I", 0, 20, (2,)),
            Route(3, 2, "", 0, 5, (3,)),
        ),
        blocks=(
            Block(1, 1, 10, 0, False),
            Block(2, 2, 20, 0, False),
            Block(3, 1, 5, 0, False),
        ),
    )


class SearchDiversificationTest(unittest.TestCase):
    def test_rotation_scale_is_adaptive_to_member_gap(self) -> None:
        import station_dispatch.qea_lns as qea_lns

        self.assertEqual(qea_lns._rotation_scale((10, 20), (10, 20)), 2.0)
        self.assertEqual(qea_lns._rotation_scale((11, 20), (10, 20)), 1.0)
        self.assertEqual(qea_lns._rotation_scale((12, 20), (10, 20)), 0.5)

    def test_fixed_entry_floors_include_every_fixed_fifo_predecessor(self) -> None:
        import station_dispatch.qea_lns as qea_lns

        instance = StationInstance(
            path=Path("fixed-entry-floors.dzn"),
            edge_count=1,
            edge_names=("entry",),
            edge_types=("inter",),
            platform_names=(),
            trains=(
                Train(1, "A", 0, "pass", (1,)),
                Train(2, "B", 5, "pass", (2,)),
                Train(3, "C", 10, "pass", (3,)),
                Train(4, "D", 15, "pass", (4,)),
            ),
            routes=(
                Route(1, 1, "", 0, 1, (1,)),
                Route(2, 2, "", 0, 1, (2,)),
                Route(3, 3, "", 0, 1, (3,)),
                Route(4, 4, "", 0, 1, (4,)),
            ),
            blocks=(
                Block(1, 1, 1, 0, False),
                Block(2, 1, 1, 0, False),
                Block(3, 1, 1, 0, False),
                Block(4, 1, 1, 0, False),
            ),
        )
        trains = {train.id: train for train in instance.trains}
        floors = qea_lns._fixed_entry_floors(
            instance,
            (Assignment(1, 1, 20, 0), Assignment(3, 3, 50, 0)),
            qea_lns.entry_edges_by_train(instance),
            trains,
        )

        self.assertEqual(floors[2], 20)
        self.assertEqual(floors[4], 50)
    def test_stagnation_reinitialization_resets_all_but_the_best_members(self) -> None:
        import station_dispatch.qea_lns as qea_lns

        instance = qea_two_train_instance()
        population = qea_lns._initial_population(instance)
        population[0].route_angles[1][0] = 1.2
        population[0].priority_angles[2][3] = 0.9
        population[1].route_angles[1][0] = 1.3
        population[1].priority_angles[2][3] = 1.0
        population[2].route_angles[1][0] = 1.4
        population[2].priority_angles[2][3] = 1.1
        population[3].route_angles[1][0] = 1.5
        population[3].priority_angles[2][3] = 1.2
        member_scores = [(0, (5, 10)), (2, (1, 10)), (3, (9, 10))]

        qea_lns._reinitialize_weak_members(instance, population, member_scores)

        self.assertEqual(population[1].route_angles[1][0], math.pi / 4)
        self.assertEqual(population[1].priority_angles[2][3], math.pi / 4)
        self.assertEqual(population[3].route_angles[1][0], math.pi / 4)
        self.assertEqual(population[3].priority_angles[2][3], math.pi / 4)
        self.assertEqual(population[0].route_angles[1][0], 1.2)
        self.assertEqual(population[0].priority_angles[2][3], 0.9)
        self.assertEqual(population[2].route_angles[1][0], 1.4)
        self.assertEqual(population[2].priority_angles[2][3], 1.1)


if __name__ == "__main__":
    unittest.main()

class QuantumRestartTriggerTest(unittest.TestCase):
    def test_keeps_learning_when_stagnant_population_still_produces_distinct_schedules(self):
        self._check_restart(collapsed=False)

    def test_restarts_when_population_repeatedly_produces_the_same_schedule(self):
        self._check_restart(collapsed=True)

    def test_different_schedules_across_generations_do_not_count_as_collapse(self):
        self._check_restart(collapsed=False, alternating_generations=True)

    def _check_restart(self, collapsed, alternating_generations=False):
        from unittest.mock import patch
        from station_dispatch.schedule import Schedule
        import station_dispatch.qea_lns as q
        instance = qea_two_train_instance()
        best = Schedule((Assignment(1, 2, 0, 0), Assignment(2, 3, 0, 0)))
        worse = Schedule((Assignment(1, 1, 0, 0), Assignment(2, 3, 10, 0)))
        candidate = q._Candidate({1: 2, 2: 3}, {1: (0,) * 16, 2: (1,) * 16}, (1, 2))
        clock = 0.0
        calls = 0
        def construct(*args, **kwargs):
            nonlocal clock, calls
            clock += 0.01
            calls += 1
            if alternating_generations:
                return best if ((calls - 1) // 2) % 2 == 0 else worse
            return best if collapsed or calls % 2 else worse
        with patch.object(q, '_initial_population', return_value=q._initial_population(instance)[:2]), patch.object(
            q, '_sample_candidate', return_value=candidate
        ), patch.object(q, '_construct_schedule', side_effect=construct), patch.object(
            q.time, 'monotonic', side_effect=lambda: clock
        ), patch.object(q, '_reinitialize_weak_members', wraps=q._reinitialize_weak_members) as restart:
            q.solve_qea_lns(instance, 42, 0.5)
        self.assertEqual(restart.called, collapsed)
