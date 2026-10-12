from pathlib import Path
import inspect
import unittest
from unittest.mock import patch

from station_dispatch.dzn import Block, Route, StationInstance, Train
from station_dispatch.schedule import validate_schedule


def milp_instance() -> StationInstance:
    return StationInstance(
        path=Path("milp-synthetic.dzn"),
        edge_count=2,
        edge_names=("shared_throat", "clear_route"),
        edge_types=("inter", "inter"),
        platform_names=(),
        trains=(
            Train(1, "T1", 0, "pass", (1, 2)),
            Train(2, "T2", 0, "pass", (3,)),
        ),
        routes=(
            Route(1, 1, "", 0, 10, (1,)),
            Route(2, 1, "", 0, 20, (2,)),
            Route(3, 2, "", 0, 5, (3,)),
        ),
        blocks=(
            Block(1, 1, 10, 0, False),
            Block(2, 2, 20, 0, False),
            Block(3, 1, 5, 0, False),
        ),
    )


def reservation_instance() -> StationInstance:
    return StationInstance(
        path=Path("milp-reservation.dzn"),
        edge_count=4,
        edge_names=("S_I", "S_II", "S_III", "exit"),
        edge_types=("platform", "platform", "platform", "border"),
        platform_names=("S_I", "S_II", "S_III"),
        trains=(
            Train(1, "Origin", 10, "origin", (1,)),
            Train(2, "Pass", 8, "pass", (2, 3)),
            Train(3, "Destination", 0, "dest", (4,)),
        ),
        routes=(
            Route(1, 1, "S_I", 0, 2, (1, 2)),
            Route(2, 2, "S_I", 0, 5, (3,)),
            Route(3, 2, "S_II", 0, 5, (4,)),
            Route(4, 3, "S_III", 0, 5, (5,)),
        ),
        blocks=(
            Block(1, 1, 1, 0, True),
            Block(2, 4, 1, 0, False),
            Block(3, 1, 5, 0, False),
            Block(4, 2, 5, 0, False),
            Block(5, 3, 5, 0, True),
        ),
    )


class MilpTest(unittest.TestCase):
    def test_minimizes_total_station_wait_before_makespan(self) -> None:
        from station_dispatch.milp import solve_milp

        result = solve_milp(milp_instance(), time_limit_s=2)

        self.assertTrue(result.feasible)
        self.assertTrue(result.optimal)
        self.assertEqual(result.status, "optimal")
        self.assertEqual(result.end_sum, 25)
        self.assertEqual(result.makespan, 20)
        self.assertEqual(result.lower_bound, 0)
        self.assertEqual(result.gap, 0.0)
        self.assertEqual(result.incumbent_source, "milp")
        self.assertIsNotNone(result.schedule)
        self.assertTrue(validate_schedule(milp_instance(), result.schedule).feasible)
        assignments = {item.train_id: item for item in result.schedule.assignments}
        self.assertEqual(assignments[1].route_id, 2)
        self.assertEqual(assignments[1].start, 0)
        self.assertEqual(assignments[2].start, 0)

    def test_alternative_routes_share_one_order_per_train_pair_and_resource_visit(self) -> None:
        import station_dispatch.milp as milp
        from station_dispatch.metrics import compute_metrics

        instance = StationInstance(
            path=Path("milp-shared-orders.dzn"),
            edge_count=5,
            edge_names=("entry_a", "entry_b", "shared", "platform_a", "platform_b"),
            edge_types=("inter",) * 5,
            platform_names=(),
            trains=(Train(1, "A", 0, "pass", (1, 2)), Train(2, "B", 0, "pass", (3, 4))),
            routes=(
                Route(1, 1, "", 0, 3, (1, 3, 4)), Route(2, 1, "", 0, 3, (1, 3, 5)),
                Route(3, 2, "", 0, 3, (2, 3, 4)), Route(4, 2, "", 0, 3, (2, 3, 5)),
            ),
            blocks=tuple(Block(edge, edge, 1, 0, False) for edge in range(1, 6)),
        )
        solver = milp.pywraplp.Solver.CreateSolver("SCIP_MIXED_INTEGER_PROGRAMMING")
        with patch.object(milp.pywraplp.Solver, "CreateSolver", return_value=solver):
            result = milp.solve_milp(instance, time_limit_s=2, seed=42)

        self.assertTrue(result.optimal)
        self.assertTrue(validate_schedule(instance, result.schedule).feasible)
        self.assertEqual(compute_metrics(instance, result.schedule).total_station_wait, 1)
        self.assertEqual(result.makespan, 4)
        orders = [variable for variable in solver.variables() if variable.name().startswith("order_")]
        self.assertEqual(len(orders), 3)

    def test_separate_visits_to_one_resource_can_have_different_train_orders(self) -> None:
        from station_dispatch.milp import solve_milp
        from station_dispatch.metrics import compute_metrics

        instance = StationInstance(
            path=Path("milp-repeated-resource.dzn"),
            edge_count=1, edge_names=("shared",), edge_types=("inter",), platform_names=(),
            trains=(Train(1, "A", 0, "pass", (1,)), Train(2, "B", 1, "pass", (2,))),
            routes=(Route(1, 1, "", 0, 4, (1, 2)), Route(2, 2, "", 0, 1, (3,))),
            blocks=(Block(1, 1, 1, 0, False), Block(2, 1, 1, 2, False), Block(3, 1, 1, 0, False)),
        )
        result = solve_milp(instance, time_limit_s=2, seed=42)

        self.assertTrue(result.optimal)
        self.assertTrue(validate_schedule(instance, result.schedule).feasible)
        self.assertEqual(compute_metrics(instance, result.schedule).total_station_wait, 0)
        self.assertEqual(result.makespan, 4)

    def test_accepts_the_formal_scip_seed(self) -> None:
        from station_dispatch.milp import solve_milp

        result = solve_milp(milp_instance(), time_limit_s=2, seed=42)

        self.assertTrue(result.feasible)
        self.assertTrue(validate_schedule(milp_instance(), result.schedule).feasible)
    def test_constructs_its_solution_without_calling_fcfs(self) -> None:
        import station_dispatch.milp as milp

        with patch.object(
            milp,
            "solve_fcfs",
            side_effect=AssertionError("MILP must not call FCFS"),
            create=True,
        ):
            result = milp.solve_milp(milp_instance(), time_limit_s=2)

        self.assertTrue(result.feasible)
        self.assertTrue(validate_schedule(milp_instance(), result.schedule).feasible)
        self.assertEqual(result.incumbent_source, "milp")

    def test_keeps_origin_and_destination_platform_reservations(self) -> None:
        from station_dispatch.milp import solve_milp

        instance = reservation_instance()
        result = solve_milp(instance, time_limit_s=2)

        self.assertTrue(result.feasible)
        self.assertTrue(validate_schedule(instance, result.schedule).feasible)
        assignments = {item.train_id: item for item in result.schedule.assignments}
        self.assertEqual(assignments[1].start, 10)
        self.assertEqual(assignments[2].route_id, 3)
        self.assertEqual(assignments[2].start, 8)

    def test_uses_the_common_300_second_default(self) -> None:
        from station_dispatch.limits import TIME_LIMIT_S
        from station_dispatch.milp import solve_milp

        self.assertEqual(inspect.signature(solve_milp).parameters["time_limit_s"].default, TIME_LIMIT_S)


if __name__ == "__main__":
    unittest.main()


class MilpThreadTest(unittest.TestCase):
    def test_solver_explicitly_requests_one_thread(self):
        import station_dispatch.milp as milp
        solver = milp.pywraplp.Solver.CreateSolver("SCIP_MIXED_INTEGER_PROGRAMMING")
        with patch.object(milp.pywraplp.Solver, "CreateSolver", return_value=solver), patch.object(solver, "SetNumThreads", wraps=solver.SetNumThreads) as threads:
            result = milp.solve_milp(milp_instance(), time_limit_s=2)
        threads.assert_called_once_with(1)
        self.assertTrue(result.feasible)


class MilpInfeasibleStatusTest(unittest.TestCase):
    def test_infeasible_model_returns_distinct_status_without_schedule(self):
        from station_dispatch.milp import solve_milp
        instance = StationInstance(
            path=Path("infeasible-status.dzn"), edge_count=1, edge_names=("S_I",),
            edge_types=("platform",), platform_names=("S_I",),
            trains=(Train(1, "A", 0, "dest", (1,)), Train(2, "B", 0, "dest", (2,))),
            routes=(Route(1, 1, "S_I", 0, 5, (1,)), Route(2, 2, "S_I", 0, 5, (2,))),
            blocks=(Block(1, 1, 5, 0, True), Block(2, 1, 5, 0, True)),
        )
        result = solve_milp(instance, time_limit_s=1, seed=42)
        self.assertEqual(result.status, "infeasible")
        self.assertFalse(result.feasible)
        self.assertIsNone(result.schedule)
        self.assertIsNone(result.end_sum)
        self.assertIsNone(result.makespan)
