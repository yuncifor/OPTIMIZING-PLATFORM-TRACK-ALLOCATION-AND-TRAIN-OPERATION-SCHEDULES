from pathlib import Path
import unittest
from unittest.mock import patch

from station_dispatch.dzn import Block, Route, StationInstance, Train
from station_dispatch.schedule import Assignment, Schedule, validate_schedule


class FcfsPureRouteRuleTest(unittest.TestCase):
    def test_fcfs_scans_alternative_routes_when_they_are_earlier_and_safe(self) -> None:
        from station_dispatch.fcfs import solve_fcfs

        instance = StationInstance(
            path=Path("fcfs-pure-route.dzn"),
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
        by_train = {item.train_id: item for item in schedule.assignments}
        self.assertEqual(by_train[2].route_id, 3)
        self.assertEqual(by_train[2].start, 0)
        self.assertTrue(validate_schedule(instance, schedule).feasible)


class QeaOriginResourceStateTest(unittest.TestCase):
    def test_priority_order_can_place_an_earlier_arrival_before_a_later_origin(self) -> None:
        import time
        import station_dispatch.qea_lns as qea_lns

        instance = StationInstance(
            path=Path("qea-origin-throat-order.dzn"),
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

        schedule = qea_lns._construct_schedule(
            instance,
            priority={1: 0, 2: 1},
            route_choices={1: 1, 2: 2},
            deadline=time.monotonic() + 1,
        )
        self.assertIsNotNone(schedule)
        by_train = {item.train_id: item for item in schedule.assignments}
        self.assertEqual(by_train[1].start, 0)
        self.assertEqual(by_train[2].start, 20)
        self.assertTrue(validate_schedule(instance, schedule).feasible)


class QeaEncodedRouteTest(unittest.TestCase):
    def test_encoded_route_is_the_route_actually_scheduled(self) -> None:
        import time
        import station_dispatch.qea_lns as qea_lns

        instance = StationInstance(
            path=Path("qea-encoded-route.dzn"),
            edge_count=4,
            edge_names=("encoded", "alternative", "blocker", "horizon"),
            edge_types=("inter", "inter", "inter", "inter"),
            platform_names=(),
            trains=(
                Train(1, "T1", 0, "pass", (1, 2)),
                Train(2, "T2", 0, "pass", (3,)),
                Train(3, "T3", 100, "pass", (4,)),
            ),
            routes=(
                Route(1, 1, "", 0, 5, (1,)),
                Route(2, 1, "", 0, 5, (2,)),
                Route(3, 2, "", 0, 5, (3,)),
                Route(4, 3, "", 0, 5, (4,)),
            ),
            blocks=(
                Block(1, 1, 5, 0, False),
                Block(2, 2, 5, 0, False),
                Block(3, 1, 5, 0, False),
                Block(4, 4, 5, 0, False),
            ),
        )

        encoded = qea_lns._assign_train(
            instance,
            [Assignment(2, 3, 0, 0)],
            instance.trains[0],
            route_id=1,
            deadline=time.monotonic() + 1,
        )
        alternative = qea_lns._assign_train(
            instance,
            [Assignment(2, 3, 0, 0)],
            instance.trains[0],
            route_id=2,
            deadline=time.monotonic() + 1,
        )

        self.assertIsNotNone(encoded)
        self.assertEqual(encoded.route_id, 1)
        self.assertGreater(encoded.start, 0)
        self.assertEqual(alternative, Assignment(1, 2, 0, 0))


class QeaLearningEncodingTest(unittest.TestCase):
    def test_solver_passes_sampled_dwell_to_schedule_constructor(self) -> None:
        import station_dispatch.qea_lns as qea_lns

        instance = StationInstance(
            path=Path("qea-solver-dwell.dzn"),
            edge_count=2,
            edge_names=("platform_1", "platform_2"),
            edge_types=("platform", "platform"),
            platform_names=("P1", "P2"),
            trains=(Train(1, "T1", 1, "pass", (1,)), Train(2, "T2", 0, "pass", (2,))),
            routes=(
                Route(1, 1, "P1", 2, 3, (1,)),
                Route(2, 2, "P2", 2, 3, (2,)),
            ),
            blocks=(
                Block(1, 1, 2, 0, True),
                Block(2, 2, 2, 0, True),
            ),
        )
        candidate = qea_lns._Candidate(
            {1: 1, 2: 2},
            {1: (0,) * qea_lns._PRIORITY_BITS, 2: (0,) * qea_lns._PRIORITY_BITS},
            (2, 1),
            {1: 0, 2: 2},
            {1: (0,) * qea_lns._DWELL_BITS, 2: qea_lns._int_to_bits(2, qea_lns._DWELL_BITS)},
        )
        schedule = Schedule((Assignment(2, 2, 0, 4), Assignment(1, 1, 1, 2)))
        observed_dwell = []

        def construct(*args, **kwargs):
            observed_dwell.append(kwargs.get("extra_dwell_by_train"))
            return schedule

        with patch.object(qea_lns, "_sample_candidate", return_value=candidate), patch.object(
            qea_lns, "_construct_schedule", side_effect=construct
        ), patch.object(qea_lns, "_reschedule_wait_chain", return_value=None):
            result = qea_lns.solve_qea_lns(instance, seed=42, time_limit_s=0.01)

        self.assertTrue(observed_dwell)
        self.assertEqual(observed_dwell[0], {1: 0, 2: 2})
        self.assertEqual(result.schedule, schedule)

    def test_schedule_learning_candidate_uses_constructor_order_for_origins(self) -> None:
        import station_dispatch.qea_lns as qea_lns

        instance = StationInstance(
            path=Path("qea-learning-origin-order.dzn"),
            edge_count=2,
            edge_names=("origin_entry", "arrival_entry"),
            edge_types=("inter", "inter"),
            platform_names=(),
            trains=(
                Train(1, "origin", 0, "origin", (1,)),
                Train(2, "arrival", 0, "pass", (2,)),
            ),
            routes=(
                Route(1, 1, "", 0, 5, (1,)),
                Route(2, 2, "", 0, 5, (2,)),
            ),
            blocks=(
                Block(1, 1, 5, 0, False),
                Block(2, 2, 5, 0, False),
            ),
        )
        schedule = Schedule((Assignment(2, 2, 0, 0), Assignment(1, 1, 0, 0)))
        candidate = qea_lns._candidate_from_schedule(instance, schedule)

        self.assertEqual(candidate.priority_order, (2, 1))

    def test_component_priority_orders_interleave_origins_with_fifo_groups(self) -> None:
        import station_dispatch.qea_lns as qea_lns

        instance = StationInstance(
            path=Path("qea-origin-interleaving.dzn"),
            edge_count=2,
            edge_names=("origin_entry", "arrival_entry"),
            edge_types=("inter", "inter"),
            platform_names=(),
            trains=(
                Train(1, "origin", 0, "origin", (1,)),
                Train(2, "arrival", 0, "pass", (2,)),
            ),
            routes=(
                Route(1, 1, "", 0, 5, (1,)),
                Route(2, 2, "", 0, 5, (2,)),
            ),
            blocks=(
                Block(1, 1, 5, 0, False),
                Block(2, 2, 5, 0, False),
            ),
        )

        orders = set(qea_lns._component_priority_orders(instance, {1, 2}))

        self.assertEqual(orders, {(1, 2), (2, 1)})

    def test_schedule_learning_candidate_preserves_route_order_and_extra_dwell(self) -> None:
        import station_dispatch.qea_lns as qea_lns

        instance = StationInstance(
            path=Path("qea-learning-encoding.dzn"),
            edge_count=2,
            edge_names=("platform_1", "platform_2"),
            edge_types=("platform", "platform"),
            platform_names=("P1", "P2"),
            trains=(Train(1, "T1", 1, "pass", (1,)), Train(2, "T2", 0, "pass", (2,))),
            routes=(
                Route(1, 1, "P1", 2, 3, (1,)),
                Route(2, 2, "P2", 2, 3, (2,)),
            ),
            blocks=(
                Block(1, 1, 2, 0, True),
                Block(2, 2, 2, 0, True),
            ),
        )
        schedule = Schedule((
            Assignment(2, 2, 0, 4),
            Assignment(1, 1, 1, 2),
        ))
        candidate = qea_lns._candidate_from_schedule(instance, schedule)
        self.assertEqual(candidate.priority_order, (2, 1))
        self.assertEqual(candidate.route_choices, {2: 2, 1: 1})
        self.assertEqual(candidate.extra_dwell_by_train, {2: 2, 1: 0})
        self.assertEqual(
            sum(bit << position for position, bit in enumerate(candidate.dwell_bits[2])),
            2,
        )
        state = qea_lns._QuantumState(
            route_angles={1: [qea_lns._MIN_ANGLE], 2: [qea_lns._MIN_ANGLE]},
            priority_angles={
                1: [qea_lns._MIN_ANGLE] * qea_lns._PRIORITY_BITS,
                2: [qea_lns._MIN_ANGLE] * qea_lns._PRIORITY_BITS,
            },
            dwell_angles={
                1: [qea_lns._MIN_ANGLE] * qea_lns._DWELL_BITS,
                2: [qea_lns._MIN_ANGLE] * qea_lns._DWELL_BITS,
            },
        )
        observed = qea_lns._Candidate(
            candidate.route_choices,
            candidate.priority_bits,
            candidate.priority_order,
            {1: 0, 2: 0},
            {1: (0,) * qea_lns._DWELL_BITS, 2: (0,) * qea_lns._DWELL_BITS},
        )
        qea_lns._update_member_toward(instance, state, observed, candidate)
        self.assertGreater(state.dwell_angles[2][1], qea_lns._MIN_ANGLE)
        replay = qea_lns._construct_schedule(
            instance,
            candidate.priority(),
            candidate.route_choices,
            __import__("time").monotonic() + 100,
            extra_dwell_by_train=candidate.extra_dwell_by_train,
        )
        self.assertEqual(replay, schedule)


if __name__ == "__main__":
    unittest.main()


