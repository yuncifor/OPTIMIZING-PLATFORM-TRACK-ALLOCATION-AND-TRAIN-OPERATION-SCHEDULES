from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import math
import random
import time

from station_dispatch.dzn import Block, Route, StationInstance, Train
from station_dispatch.schedule import Assignment, Schedule, validate_schedule


def qea_instance() -> StationInstance:
    return StationInstance(
        path=Path("qea-synthetic.dzn"),
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


def qea_forced_route_instance() -> StationInstance:
    return StationInstance(
        path=Path("qea-forced-route-synthetic.dzn"),
        edge_count=2,
        edge_names=("free_route", "blocked_route"),
        edge_types=("inter", "inter"),
        platform_names=(),
        trains=(
            Train(1, "T1", 0, "pass", (1, 2)),
            Train(2, "T2", 0, "pass", (3,)),
        ),
        routes=(
            Route(1, 1, "", 0, 1, (1,)),
            Route(2, 1, "", 0, 2, (2,)),
            Route(3, 2, "", 0, 5, (3,)),
        ),
        blocks=(
            Block(1, 1, 1, 0, False),
            Block(2, 2, 2, 0, False),
            Block(3, 2, 5, 0, False),
        ),
    )


def qea_route_choice_instance() -> StationInstance:
    return StationInstance(
        path=Path("qea-route-choice-synthetic.dzn"),
        edge_count=4,
        edge_names=("entry_a", "shared_section", "alternative_section", "entry_b"),
        edge_types=("inter", "inter", "inter", "inter"),
        platform_names=(),
        trains=(
            Train(1, "T1", 0, "pass", (1, 2)),
            Train(2, "T2", 0, "pass", (3,)),
        ),
        routes=(
            Route(1, 1, "", 0, 11, (1, 2)),
            Route(2, 1, "", 0, 11, (1, 3)),
            Route(3, 2, "", 0, 11, (4, 5)),
        ),
        blocks=(
            Block(1, 1, 1, 0, False),
            Block(2, 2, 10, 0, False),
            Block(3, 3, 10, 0, False),
            Block(4, 4, 1, 0, False),
            Block(5, 2, 10, 0, False),
        ),
    )


def qea_dwell_reordering_instance() -> StationInstance:
    return StationInstance(
        path=Path("qea-dwell-reordering-synthetic.dzn"),
        edge_count=5,
        edge_names=("entry_a", "platform", "shared_exit", "entry_b", "exit_c"),
        edge_types=("inter", "platform", "inter", "inter", "inter"),
        platform_names=("S_I",),
        trains=(
            Train(1, "A", 0, "pass", (1,)),
            Train(2, "B", 0, "pass", (2,)),
            Train(3, "C", 0, "pass", (3,)),
        ),
        routes=(
            Route(1, 1, "S_I", 0, 6, (1, 2, 3)),
            Route(2, 2, "", 0, 7, (4, 5)),
            Route(3, 3, "", 0, 2, (6, 7)),
        ),
        blocks=(
            Block(1, 1, 4, 0, False),
            Block(2, 2, 1, 0, True),
            Block(3, 3, 1, 0, False),
            Block(4, 4, 3, 0, False),
            Block(5, 3, 4, 0, False),
            Block(6, 1, 1, 0, False),
            Block(7, 5, 1, 0, False),
        ),
    )


class QeaLnsTest(unittest.TestCase):
    def test_normal_construction_uses_free_route_outside_busy_encoding(self) -> None:
        import station_dispatch.qea_lns as qea
        instance = qea_forced_route_instance()
        schedule = qea._construct_schedule(
            instance, {1: 1, 2: 0}, {1: 2, 2: 3}, time.monotonic() + 2,
            fixed_assignments=(Assignment(2, 3, 0, 0),),
        )
        self.assertIsNotNone(schedule)
        self.assertIn(Assignment(1, 1, 0, 0), schedule.assignments)
        self.assertTrue(validate_schedule(instance, schedule).feasible)

    def test_repaired_route_feedback_rotates_away_from_the_busy_sample(self) -> None:
        import station_dispatch.qea_lns as qea
        instance = qea_route_choice_instance()
        state = qea._initial_population(instance)[0]
        candidate = qea._Candidate({1: 1, 2: 3}, {1: (1,) * 16, 2: (0,) * 16}, (2, 1))
        with patch.object(qea, '_initial_population', return_value=[state]), patch.object(
            qea, '_sample_candidate', return_value=candidate
        ):
            qea.solve_qea_lns(instance, 42, 0.03)
        self.assertLess(state.route_angles[1][0], math.pi / 4)
        self.assertGreater(state.route_angles[1][1], math.pi / 4)

    def test_normal_construction_keeps_encoded_route_when_equally_early(self) -> None:
        import station_dispatch.qea_lns as qea
        instance = qea_route_choice_instance()
        schedules = [qea._construct_schedule(
            instance, {1: 0, 2: 1}, {1: route_id, 2: 3}, time.monotonic() + 2,
        ) for route_id in (1, 2)]
        self.assertEqual([s.assignments[0].route_id for s in schedules], [1, 2])
        self.assertLess(qea._score(instance, schedules[1]), qea._score(instance, schedules[0]))

    def test_learning_observes_successful_dispatch_order_not_sampled_order(self) -> None:
        import station_dispatch.qea_lns as qea
        instance = qea_instance()
        candidate = qea._Candidate(
            {1: 2, 2: 3}, {1: (0,) * qea._PRIORITY_BITS, 2: (1,) * qea._PRIORITY_BITS}, (1, 2),
        )
        schedule = Schedule((Assignment(2, 3, 0, 0), Assignment(1, 2, 0, 0)))
        with patch.object(qea, '_sample_candidate', return_value=candidate), patch.object(
            qea, '_construct_schedule', return_value=schedule
        ), patch.object(qea, '_update_member_toward', wraps=qea._update_member_toward) as learn:
            qea.solve_qea_lns(instance, 42, 0.03)
        self.assertTrue(learn.called)
        for call in learn.call_args_list:
            self.assertEqual(call.args[2].priority_order, (1, 2))
            self.assertEqual(call.args[3].priority_order, (2, 1))

    def test_learning_recovers_resource_order_from_a_reordered_assignment_container(self) -> None:
        import station_dispatch.qea_lns as qea
        instance = qea_dwell_reordering_instance()
        schedule = Schedule((Assignment(1, 1, 2, 0), Assignment(2, 2, 0, 0), Assignment(3, 3, 6, 0)))
        self.assertTrue(validate_schedule(instance, schedule).feasible)
        candidate = qea._candidate_from_schedule(instance, schedule)
        self.assertIsNotNone(candidate)
        replay = qea._construct_schedule(instance, candidate.priority(), candidate.route_choices, time.monotonic() + 2, extra_dwell_by_train=candidate.extra_dwell_by_train)
        self.assertEqual(set(replay.assignments), set(schedule.assignments))

    def test_learning_does_not_claim_unrepresentable_idle_time_is_encoded(self) -> None:
        import station_dispatch.qea_lns as qea
        instance = qea_instance()
        schedule = Schedule((Assignment(1, 2, 1, 0), Assignment(2, 3, 1, 0)))
        self.assertTrue(validate_schedule(instance, schedule).feasible)
        self.assertIsNone(qea._candidate_from_schedule(instance, schedule))

    def test_search_timing_reports_actual_first_solution_and_strict_improvements(self) -> None:
        import station_dispatch.qea_lns as qea
        result = qea.solve_qea_lns(qea_instance(), seed=42, time_limit_s=0.08)
        self.assertLessEqual(0, result.first_feasible_s)
        self.assertLessEqual(result.first_feasible_s, result.best_found_s)
        self.assertLessEqual(result.best_found_s, result.runtime_s)
        self.assertEqual(result.improvement_history[0][0], result.first_feasible_s)
        self.assertEqual(result.improvement_history[-1][0], result.best_found_s)
        self.assertEqual(result.improvement_history[-1][1:], qea._score(qea_instance(), result.schedule))
        for previous, current in zip(result.improvement_history, result.improvement_history[1:]):
            self.assertLess(current[1:], previous[1:])
            self.assertLessEqual(previous[0], current[0])

    def test_search_timing_uses_high_resolution_reporting_clock(self) -> None:
        import station_dispatch.qea_lns as qea

        instance = qea_instance()
        state = qea._initial_population(instance)[0]
        candidate = qea._Candidate(
            route_choices={1: 2, 2: 3},
            priority_bits={1: (0,) * qea._PRIORITY_BITS, 2: (1,) * qea._PRIORITY_BITS},
            priority_order=(1, 2),
        )
        schedule = Schedule((
            Assignment(train_id=1, route_id=2, start=0, dwell=0),
            Assignment(train_id=2, route_id=3, start=0, dwell=0),
        ))
        coarse_calls = 0
        precise_times = iter((100.0, 100.002, 100.005))

        def monotonic() -> float:
            nonlocal coarse_calls
            coarse_calls += 1
            return 0.0 if coarse_calls <= 7 else 11.0

        clock = SimpleNamespace(monotonic=monotonic)
        with patch.object(qea, '_REPORTING_CLOCK', side_effect=precise_times), patch.object(
            qea, '_initial_population', return_value=[state],
        ), patch.object(
            qea, '_sample_candidate', return_value=candidate,
        ), patch.object(
            qea, '_construct_schedule', return_value=schedule,
        ), patch.object(
            qea, '_reschedule_wait_chain', return_value=None,
        ), patch.object(qea, 'time', clock):
            result = qea.solve_qea_lns(instance, seed=42, time_limit_s=10)

        self.assertAlmostEqual(result.first_feasible_s, 0.002)
        self.assertAlmostEqual(result.best_found_s, 0.002)
        self.assertAlmostEqual(result.runtime_s, 0.005)

    def test_necessary_dwell_checks_later_pairs_and_resumes_without_repeating(self) -> None:
        import station_dispatch.qea_lns as qea
        instance = StationInstance(
            path=Path("three-platform-exit-pairs.dzn"), edge_count=4,
            edge_names=("P1", "P2", "P3", "exit"),
            edge_types=("platform", "platform", "platform", "inter"),
            platform_names=("P1", "P2", "P3"),
            trains=tuple(Train(i, str(i), 0, "pass", (i,)) for i in (1, 2, 3)),
            routes=tuple(Route(i, i, f"P{i}", 0, 2, (2*i-1, 2*i)) for i in (1, 2, 3)),
            blocks=tuple(block for i in (1, 2, 3) for block in (Block(2*i-1, i, 1, 0, True), Block(2*i, 4, 1, 0, False))),
        )
        schedule = Schedule(tuple(Assignment(i, i, 2*(i-1), 0) for i in (1, 2, 3)))
        self.assertTrue(validate_schedule(instance, schedule).feasible)
        original = qea._construct_schedule
        clock = time.monotonic()
        calls = []
        def construct(*args, **kwargs):
            nonlocal clock
            calls.append(kwargs["extra_dwell_by_train"])
            result = original(*args, **kwargs)
            clock += 0.1
            return result
        progress = {}
        with patch.object(qea, "time", SimpleNamespace(monotonic=lambda: clock)), patch.object(qea, "_construct_schedule", side_effect=construct):
            qea._reschedule_with_necessary_dwell(instance, schedule, {1, 2, 3}, (), {1: 1, 2: 2, 3: 3}, 4, clock + 0.05, progress=progress)
            self.assertFalse(progress["complete"])
            repair = qea._reschedule_with_necessary_dwell(instance, schedule, {1, 2, 3}, (), {1: 1, 2: 2, 3: 3}, 4, clock + 1, progress=progress)
        self.assertTrue(progress["complete"])
        self.assertEqual(len(calls), 2)
        self.assertEqual(len({tuple(item.items()) for item in calls}), 2)
        self.assertIsNotNone(repair)
        self.assertTrue(validate_schedule(instance, repair.schedule).feasible)
        self.assertLess(qea._score(instance, repair.schedule), qea._score(instance, schedule))

    def test_score_matches_reporting_without_building_the_full_report(self) -> None:
        import station_dispatch.qea_lns as qea
        from station_dispatch.metrics import compute_metrics

        instance = qea_dwell_reordering_instance()
        schedule = qea._construct_schedule(instance, {1: 1, 2: 0, 3: 2}, {1: 1, 2: 2, 3: 3}, time.monotonic() + 2, extra_dwell_by_train={1: 2})
        metrics = compute_metrics(instance, schedule)
        with patch.object(qea, "compute_metrics", side_effect=AssertionError("full reporting in search")):
            self.assertEqual(qea._score(instance, schedule), (metrics.total_station_wait, metrics.makespan))

    def test_zero_extra_dwell_checks_only_the_encoded_route(self) -> None:
        import station_dispatch.qea_lns as qea
        instance = qea_forced_route_instance()
        with patch.object(qea, "earliest_feasible_assignment", wraps=qea.earliest_feasible_assignment) as assign:
            actual = qea._assign_train(instance, [Assignment(2, 3, 0, 0)], instance.trains[0], 2, time.monotonic() + 2)
        self.assertEqual(actual, Assignment(1, 2, 5, 0))
        self.assertEqual(assign.call_count, 1)
        self.assertEqual(assign.call_args.kwargs["route_ids"], (2,))

    def test_initial_dwell_exploration_does_not_invent_unlearned_high_bits(self) -> None:
        import station_dispatch.qea_lns as qea
        instance = qea_dwell_reordering_instance()
        state = qea._initial_population(instance)[0]
        rng = random.Random(42)
        for _ in range(250):
            candidate = qea._sample_candidate(instance, state, rng)
            self.assertEqual(set(candidate.extra_dwell_by_train.values()), {0})

    def test_feedback_records_executed_dwell_even_when_route_did_not_change(self) -> None:
        import station_dispatch.qea_lns as qea
        instance = qea_dwell_reordering_instance()
        candidate = qea._Candidate({1: 1, 2: 2, 3: 3}, {i: (0,) * 16 for i in (1, 2, 3)}, (2, 1, 3), {1: 65535, 2: 0, 3: 0}, {i: qea._int_to_bits(65535 if i == 1 else 0, 16) for i in (1, 2, 3)})
        schedule = qea._construct_schedule(instance, {1: 1, 2: 0, 3: 2}, candidate.route_choices, time.monotonic() + 2)
        actual = qea._schedule_extra_dwell(instance, schedule, {1, 2, 3})
        with patch.object(qea, "_sample_candidate", return_value=candidate), patch.object(qea, "_construct_schedule", return_value=schedule), patch.object(qea, "_reschedule_wait_chain", return_value=None), patch.object(qea, "_update_member_toward") as update:
            qea.solve_qea_lns(instance, seed=42, time_limit_s=0.03)
        self.assertTrue(update.called)
        for call in update.call_args_list:
            observed = call.args[2]
            target = call.args[3]
            self.assertEqual(observed.extra_dwell_by_train,
                             {t: candidate.extra_dwell_by_train[t] for t in observed.route_choices})
            self.assertEqual(target.extra_dwell_by_train, actual)
            self.assertEqual(qea._bits_to_int(target.dwell_bits[1]), actual[1])

    def test_wait_targets_include_extra_dwell_and_origin_wait(self) -> None:
        import station_dispatch.qea_lns as qea
        from dataclasses import replace
        instance = qea_dwell_reordering_instance()
        instance = replace(instance, trains=(instance.trains[0], replace(instance.trains[1], train_type="origin"), instance.trains[2]))
        schedule = Schedule((Assignment(1, 1, 0, 2), Assignment(2, 2, 10, 0), Assignment(3, 3, 0, 0)))
        self.assertEqual([t.id for t in qea._waiting_trains(instance, schedule)], [2, 1])

    def test_dwell_only_wait_chain_detects_the_blocked_exit(self) -> None:
        import station_dispatch.qea_lns as qea
        instance = qea_dwell_reordering_instance()
        schedule = Schedule((Assignment(2, 2, 0, 0), Assignment(1, 1, 0, 2), Assignment(3, 3, 4, 0)))
        self.assertTrue(validate_schedule(instance, schedule).feasible)
        waiting = qea._waiting_trains(instance, schedule)
        self.assertIn(1, [t.id for t in waiting])
        component, edge = qea._wait_chain(instance, schedule, [t.id for t in waiting].index(1))
        self.assertEqual(component, {1, 2})
        self.assertEqual(edge, 3)

    def test_short_time_slices_resume_joint_search_until_a_real_improvement(self) -> None:
        import station_dispatch.qea_lns as qea_lns

        instance = qea_instance()
        initial = Schedule((Assignment(1, 1, 0, 0), Assignment(2, 3, 10, 0)))
        candidate = qea_lns._Candidate({1: 1, 2: 3}, {1: (0,) * 16, 2: (0,) * 16}, (1, 2))
        clock = time.monotonic()
        original_assign = qea_lns._assign_train
        calls = []
        original_encode = qea_lns._candidate_from_schedule

        def encode(*args):
                                                                                
            with patch.object(qea_lns, "_assign_train", original_assign):
                return original_encode(*args)

        def construct(*args, **kwargs):
            nonlocal clock
            clock += 0.01
            return initial

        def assign(_instance, partial, train, preference, deadline, **kwargs):
            nonlocal clock
            calls.append((tuple(partial), train.id, preference))
            clock += 0.05
            return original_assign(_instance, partial, train, preference, deadline)

        with patch.object(qea_lns, "time", SimpleNamespace(monotonic=lambda: clock)), patch.object(
            qea_lns, "_sample_candidate", return_value=candidate,
        ), patch.object(qea_lns, "_construct_schedule", side_effect=construct), patch.object(
            qea_lns, "_assign_train", side_effect=assign,
        ), patch.object(qea_lns, "_candidate_from_schedule", side_effect=encode):
            result = qea_lns.solve_qea_lns(instance, seed=42, time_limit_s=2)

        self.assertTrue(validate_schedule(instance, result.schedule).feasible)
        self.assertEqual(qea_lns._score(instance, result.schedule)[0], 0)
        self.assertEqual(len(calls), len(set(calls)))
        self.assertGreater(result.iterations, qea_lns._POPULATION)

    def test_resumed_joint_search_matches_uninterrupted_search_without_repeating_work(self) -> None:
        import station_dispatch.qea_lns as qea_lns

        instance = qea_alternative_route_component_instance()
        current = Schedule((Assignment(1, 1, 10, 0), Assignment(2, 3, 0, 0), Assignment(3, 4, 0, 0)))
        component, decisions = qea_lns._alternative_route_component(instance, current, 0)
        choices = {a.train_id: a.route_id for a in current.assignments if a.train_id in component}
        fixed = tuple(a for a in current.assignments if a.train_id not in component)
        arguments = (instance, component, decisions, choices, fixed, qea_lns._score(instance, current))
        original_assign = qea_lns._assign_train
        calls = []
        clock = time.monotonic()

        def assign(_instance, partial, train, preference, deadline, **kwargs):
            nonlocal clock
            calls.append((tuple(partial), train.id, preference))
            clock += 0.02
            return original_assign(_instance, partial, train, preference, deadline)

        with patch.object(qea_lns, "time", SimpleNamespace(monotonic=lambda: clock)), patch.object(
            qea_lns, "_assign_train", side_effect=assign,
        ):
            uninterrupted = qea_lns._search_joint_component(*arguments, clock + 10)
            expected_calls = list(calls)
            calls.clear()
            progress = {}
            for _ in range(100):
                resumed = qea_lns._search_joint_component(*arguments, clock + 0.03, progress=progress)
                if progress["complete"]:
                    break

        self.assertTrue(progress["complete"])
        self.assertIsNotNone(uninterrupted)
        self.assertEqual(resumed, uninterrupted)
        self.assertEqual(calls, expected_calls)
        self.assertTrue(validate_schedule(instance, resumed.schedule).feasible)

    def test_local_search_yields_budget_for_quantum_learning_and_next_generation(self) -> None:
        import station_dispatch.qea_lns as qea_lns

        instance = qea_instance()
        schedule = Schedule((Assignment(1, 1, 0, 0), Assignment(2, 3, 10, 0)))
        candidate = qea_lns._Candidate(
            {1: 1, 2: 3}, {1: (0,) * 16, 2: (0,) * 16}, (1, 2),
        )
        clock = 0.0
        feedback_times = []

        def construct(*args, **kwargs):
            nonlocal clock
            clock += 0.01
            return schedule

        def repair(_instance, _schedule, _rng, deadline, **kwargs):
            nonlocal clock
            self.assertEqual(kwargs["max_steps"], qea_lns._LNS_STEPS)
            clock += 0.1
            return None

        def update(*args):
            feedback_times.append(clock)

        with patch.object(qea_lns.time, "monotonic", side_effect=lambda: clock), patch.object(
            qea_lns, "_sample_candidate", return_value=candidate,
        ), patch.object(qea_lns, "_construct_schedule", side_effect=construct), patch.object(
            qea_lns, "_reschedule_wait_chain", side_effect=repair,
        ), patch.object(qea_lns, "_update_member_toward", side_effect=update):
            result = qea_lns.solve_qea_lns(instance, seed=42, time_limit_s=1)

        self.assertLess(feedback_times[0], 1)
        self.assertGreater(result.iterations, qea_lns._POPULATION)
        self.assertTrue(validate_schedule(instance, result.schedule).feasible)

    def test_joint_search_visits_lower_wait_branch_before_input_route_order(self) -> None:
        import station_dispatch.qea_lns as qea_lns

        instance = qea_forced_route_instance()
        visited = []
        validate = qea_lns.validate_schedule

        def record(_instance, schedule):
            visited.append(next(a.route_id for a in schedule.assignments if a.train_id == 1))
            return validate(_instance, schedule)

        with patch.object(qea_lns, "validate_schedule", side_effect=record):
            repair = qea_lns._search_joint_component(
                instance, {1}, {1}, {1: 2}, (Assignment(2, 3, 0, 0),),
                (6, 8), time.monotonic() + 2,
            )

        self.assertEqual(visited[0], 1)
        self.assertIsNotNone(repair)
        self.assertEqual(qea_lns._score(instance, repair.schedule), (0, 5))
        self.assertTrue(validate_schedule(instance, repair.schedule).feasible)

    def test_joint_search_does_not_repeat_identical_executed_route_branches(self) -> None:
        import station_dispatch.qea_lns as qea_lns

        instance = qea_route_choice_instance()
        calls = []

        def assign(_instance, partial, train, preference, deadline, **kwargs):
            calls.append((tuple(partial), train.id, preference))
            return Assignment(train.id, train.route_ids[0], 0, 0)

        with patch.object(qea_lns, "_assign_train", side_effect=assign), patch.object(
            qea_lns, "validate_schedule", return_value=SimpleNamespace(feasible=True),
        ), patch.object(qea_lns, "_score", return_value=(100, 100)):
            qea_lns._search_joint_component(
                instance, {1, 2}, {1}, {1: 1, 2: 3}, (), (50, 50), time.monotonic() + 1,
            )

        self.assertTrue(calls)
        self.assertEqual(len(calls), len(set(calls)))

    def test_keeps_a_safe_best_schedule_under_the_common_rules(self) -> None:
        from station_dispatch.qea_lns import solve_qea_lns

        instance = qea_instance()
        result = solve_qea_lns(instance, seed=7, time_limit_s=1)

        self.assertTrue(result.feasible)
        self.assertTrue(validate_schedule(instance, result.schedule).feasible)
        self.assertEqual(result.end_sum, 25)
        self.assertEqual(result.makespan, 20)
        self.assertGreaterEqual(result.iterations, 1)

    def test_busy_encoded_route_waits_on_that_route_until_it_is_safe(self) -> None:
        from station_dispatch.qea_lns import _assign_train

        instance = qea_forced_route_instance()
        train = instance.trains[0]
        existing = [Assignment(2, 3, 0, 0)]

        assignment = _assign_train(
            instance,
            existing,
            train,
            route_id=2,
            deadline=time.monotonic() + 1,
        )

        self.assertIsNotNone(assignment)
        self.assertEqual(assignment.route_id, 2)
        self.assertEqual(assignment.start, 5)

    def test_waits_for_the_encoded_route_release(self) -> None:
        from dataclasses import replace
        from station_dispatch.qea_lns import _assign_train

        instance = qea_route_choice_instance()
        instance = replace(
            instance,
            trains=instance.trains + (Train(3, "T3", 0, "pass", (4,)),),
            routes=instance.routes + (Route(4, 3, "", 0, 4, (6,)),),
            blocks=instance.blocks + (Block(6, 3, 4, 0, False),),
        )
        fixed = [Assignment(2, 3, 0, 0), Assignment(3, 4, 0, 0)]
        assignment = _assign_train(instance, fixed, instance.trains[0], 1, time.monotonic() + 1)

        self.assertEqual(assignment, Assignment(1, 1, 10, 0))
        self.assertTrue(validate_schedule(instance, Schedule(tuple(fixed + [assignment]))).feasible)

    def test_equal_entry_route_preferences_change_downstream_resource_wait(self) -> None:
        from station_dispatch.qea_lns import _construct_schedule, _score

        instance = qea_route_choice_instance()
        first = _construct_schedule(instance, {1: 0, 2: 1}, {1: 1, 2: 3}, time.monotonic() + 1)
        second = _construct_schedule(instance, {1: 0, 2: 1}, {1: 2, 2: 3}, time.monotonic() + 1)

        self.assertTrue(validate_schedule(instance, first).feasible)
        self.assertTrue(validate_schedule(instance, second).feasible)
        self.assertEqual(first.assignments[0], Assignment(1, 1, 0, 0))
        self.assertEqual(second.assignments[0], Assignment(1, 2, 0, 0))
        self.assertEqual(_score(instance, first), (10, 21))
        self.assertEqual(_score(instance, second), (0, 11))

    def test_order_search_uses_earlier_alternative_under_changed_resource_conditions(self) -> None:
        from station_dispatch.qea_lns import _construct_schedule, _score

        instance = qea_route_choice_instance()
        first = _construct_schedule(instance, {1: 0, 2: 1}, {1: 1, 2: 3}, time.monotonic() + 1)
        second = _construct_schedule(instance, {1: 1, 2: 0}, {1: 1, 2: 3}, time.monotonic() + 1)

        self.assertTrue(validate_schedule(instance, second).feasible)
        self.assertEqual(second.assignments, (Assignment(2, 3, 0, 0), Assignment(1, 2, 0, 0)))
        self.assertLess(_score(instance, second), _score(instance, first))

    def test_quantum_feedback_learns_an_improved_route_without_displacing_the_sample(self) -> None:
        import station_dispatch.qea_lns as qea_lns

        instance = qea_route_choice_instance()
        candidate = qea_lns._Candidate({1: 1, 2: 3}, {1: (1,) * 16, 2: (0,) * 16}, (2, 1))
        updates = []
        with patch.object(qea_lns, "_sample_candidate", return_value=candidate), patch.object(
            qea_lns, "_update_member_toward", side_effect=lambda *args: updates.append(args),
        ):
            result = qea_lns.solve_qea_lns(instance, seed=42, time_limit_s=0.05)

        self.assertTrue(validate_schedule(instance, result.schedule).feasible)
        self.assertTrue(updates)
        for _, _, observed, target, _rotation_scale in updates:
            self.assertEqual(observed.route_choices, {1: 1, 2: 3})
            self.assertEqual(target.route_choices, {1: 2, 2: 3})
            self.assertEqual(observed.priority_bits, candidate.priority_bits)

    def test_scores_lower_wait_before_a_lower_makespan(self) -> None:
        import station_dispatch.qea_lns as qea_lns

        instance = qea_instance()
        lower_makespan = Schedule((
            Assignment(1, 1, 0, 0),
            Assignment(2, 3, 10, 0),
        ))
        lower_wait = Schedule((
            Assignment(1, 2, 0, 0),
            Assignment(2, 3, 0, 0),
        ))

        self.assertTrue(validate_schedule(instance, lower_makespan).feasible)
        self.assertTrue(validate_schedule(instance, lower_wait).feasible)
        self.assertLess(qea_lns._score(instance, lower_wait), qea_lns._score(instance, lower_makespan))

    def test_initial_population_gives_each_qea_member_independent_quantum_state(self) -> None:
        from station_dispatch.qea_lns import _POPULATION, _initial_population

        population = _initial_population(qea_instance())

        self.assertEqual(len(population), _POPULATION)
        self.assertTrue(all(member is not population[0] for member in population[1:]))
        self.assertTrue(all(member.route_angles is not population[0].route_angles for member in population[1:]))
        self.assertTrue(all(member.priority_angles is not population[0].priority_angles for member in population[1:]))
        self.assertTrue(all(member.route_angles[1] is not population[0].route_angles[1] for member in population[1:]))
        self.assertTrue(all(member.priority_angles[1] is not population[0].priority_angles[1] for member in population[1:]))

        population[0].route_angles[1][0] = 0.1
        population[0].priority_angles[1][0] = 0.1

        for member in population[1:]:
            self.assertEqual(member.route_angles[1][0], math.pi / 4)
            self.assertEqual(member.priority_angles[1][0], math.pi / 4)

    def test_qea_member_update_changes_only_its_own_quantum_state(self) -> None:
        from station_dispatch.qea_lns import _Candidate, _PRIORITY_BITS, _initial_population, _update_member_toward

        instance = qea_instance()
        population = _initial_population(instance)
        observed_candidate = _Candidate(
            route_choices={1: 1, 2: 3},
            priority_bits={1: (0,) * _PRIORITY_BITS, 2: (0,) * _PRIORITY_BITS},
            priority_order=(1, 2),
        )
        incumbent_candidate = _Candidate(
            route_choices={1: 2, 2: 3},
            priority_bits={1: (1,) * _PRIORITY_BITS, 2: (0,) * _PRIORITY_BITS},
            priority_order=(2, 1),
        )
        observed_schedule = Schedule((Assignment(1, 1, 0, 0), Assignment(2, 3, 0, 0)))
        incumbent_schedule = Schedule((Assignment(1, 2, 0, 0), Assignment(2, 3, 0, 0)))
        other_route_angles = {train_id: list(angles) for train_id, angles in population[1].route_angles.items()}
        other_priority_angles = {train_id: list(angles) for train_id, angles in population[1].priority_angles.items()}

        _update_member_toward(
            instance,
            population[0],
            observed_candidate,
            incumbent_candidate,
        )

        self.assertGreater(population[0].route_angles[1][1], math.pi / 4)
        self.assertLess(population[0].route_angles[1][0], math.pi / 4)
        self.assertGreater(population[0].priority_angles[1][0], math.pi / 4)
        self.assertEqual(population[1].route_angles, other_route_angles)
        self.assertEqual(population[1].priority_angles, other_priority_angles)

    def test_route_only_lns_feedback_preserves_priority_encoding(self) -> None:
        from station_dispatch.qea_lns import _Candidate, _LnsRepair, _merge_lns_repair_into_candidate

        keys = {1: 7000, 2: 29000, 3: 63000}
        original = _Candidate(
            {1: 1, 2: 2, 3: 3},
            {train_id: tuple((key >> bit) & 1 for bit in range(16)) for train_id, key in keys.items()},
            (1, 2, 3),
        )
        feedback = _merge_lns_repair_into_candidate(
            original, _LnsRepair(Schedule(()), {2: 4}, (2,)),
        )

        self.assertEqual(feedback.route_choices, {1: 1, 2: 4, 3: 3})
        self.assertEqual(feedback.priority_order, original.priority_order)
        self.assertEqual(feedback.priority_bits, original.priority_bits)

    def test_order_repair_reassigns_existing_keys_without_reencoding_other_trains(self) -> None:
        from station_dispatch.qea_lns import _Candidate, _LnsRepair, _merge_lns_repair_into_candidate

        keys = {1: 7000, 2: 29000, 3: 63000}
        original = _Candidate(
            {1: 1, 2: 2, 3: 3},
            {train_id: tuple((key >> bit) & 1 for bit in range(16)) for train_id, key in keys.items()},
            (1, 2, 3),
        )
        feedback = _merge_lns_repair_into_candidate(
            original, _LnsRepair(Schedule(()), {1: 1, 3: 3}, (3, 1)),
        )

        self.assertEqual(feedback.priority_order, (3, 2, 1))
        self.assertEqual(feedback.priority_bits[2], original.priority_bits[2])
        self.assertEqual(feedback.priority_bits[3], original.priority_bits[1])
        self.assertEqual(feedback.priority_bits[1], original.priority_bits[3])

    def test_accumulates_lns_repairs_into_one_complete_learning_candidate(self) -> None:
        from station_dispatch.qea_lns import _Candidate, _LnsRepair, _PRIORITY_BITS, _merge_lns_repair_into_candidate

        original = _Candidate(
            route_choices={1: 1, 2: 2, 3: 3},
            priority_bits={train_id: (0,) * _PRIORITY_BITS for train_id in (1, 2, 3)},
            priority_order=(1, 2, 3),
        )
        first_repair = _LnsRepair(
            schedule=Schedule(()),
            route_choices={1: 4, 2: 2},
            priority_order=(2, 1),
        )
        second_repair = _LnsRepair(
            schedule=Schedule(()),
            route_choices={2: 5, 3: 6},
            priority_order=(3, 2),
        )

        accumulated = _merge_lns_repair_into_candidate(original, first_repair)
        accumulated = _merge_lns_repair_into_candidate(accumulated, second_repair)

        self.assertEqual(accumulated.route_choices, {1: 4, 2: 5, 3: 6})
        self.assertEqual(accumulated.priority_order, (3, 1, 2))
        self.assertEqual(
            {
                train_id: sum(bit << position for position, bit in enumerate(bits))
                for train_id, bits in accumulated.priority_bits.items()
            },
            {3: 0, 1: 1, 2: 2},
        )

    def test_strict_lns_feedback_updates_all_observed_members_toward_the_complete_global_target(self) -> None:
        import station_dispatch.qea_lns as qea_lns

        instance = qea_instance()
        source_state, other_state = qea_lns._initial_population(instance)[:2]
        source_candidate = qea_lns._Candidate(
            route_choices={1: 1, 2: 3},
            priority_bits={1: (0,) * qea_lns._PRIORITY_BITS, 2: (0,) * qea_lns._PRIORITY_BITS},
            priority_order=(1, 2),
        )
        other_candidate = qea_lns._Candidate(
            route_choices={1: 1, 2: 3},
            priority_bits={1: (1,) * qea_lns._PRIORITY_BITS, 2: (1,) * qea_lns._PRIORITY_BITS},
            priority_order=(2, 1),
        )
        initial = Schedule((
            Assignment(train_id=1, route_id=1, start=0, dwell=0),
            Assignment(train_id=2, route_id=3, start=10, dwell=0),
        ))
        repair = qea_lns._LnsRepair(
            schedule=Schedule((
                Assignment(train_id=1, route_id=2, start=0, dwell=0),
                Assignment(train_id=2, route_id=3, start=0, dwell=0),
            )),
            route_choices={1: 2, 2: 3},
            priority_order=(2, 1),
        )
        learning_candidate = qea_lns._candidate_from_schedule(instance, repair.schedule)
        updates = []
        def monotonic() -> float:
            return 11.0 if len(updates) >= 2 else 0.0

        def record_update(*args) -> None:
            updates.append(args)

        def score(_, schedule):
            return (0, 20) if schedule == repair.schedule else (10, 20)

        with patch.object(qea_lns, '_initial_population', return_value=[source_state, other_state]), patch.object(
            qea_lns,
            '_sample_candidate',
            side_effect=[source_candidate, other_candidate],
        ), patch.object(
            qea_lns,
            '_construct_schedule',
            return_value=initial,
        ), patch.object(
            qea_lns,
            '_reschedule_wait_chain',
            side_effect=[repair, None, None],
        ), patch.object(
            qea_lns,
            '_update_member_toward',
            side_effect=record_update,
        ), patch.object(
            qea_lns,
            '_score',
            side_effect=score,
        ), patch.object(qea_lns, '_candidate_from_schedule', return_value=learning_candidate), patch.object(qea_lns.time, 'monotonic', side_effect=monotonic):
            qea_lns.solve_qea_lns(instance, seed=42, time_limit_s=10)

        lns_updates = [
            update
            for update in updates
            if update[3].route_choices == {1: 2, 2: 3}
        ]
        self.assertEqual(len(lns_updates), 2)
        self.assertCountEqual([update[1] for update in lns_updates], [source_state, other_state])
        self.assertCountEqual([update[2] for update in lns_updates], [source_candidate, other_candidate])

    def test_unreplayable_lns_repair_updates_target_with_its_retained_context(self) -> None:
        import station_dispatch.qea_lns as qea_lns

        instance = qea_instance()
        states = qea_lns._initial_population(instance)[:2]
        candidates = [
            qea_lns._Candidate(
                route_choices={1: 1, 2: 3},
                priority_bits={1: (bit,) * qea_lns._PRIORITY_BITS, 2: (bit,) * qea_lns._PRIORITY_BITS},
                priority_order=(1, 2) if bit == 0 else (2, 1),
            )
            for bit in (0, 1)
        ]
        initial = Schedule((
            Assignment(train_id=1, route_id=1, start=0, dwell=0),
            Assignment(train_id=2, route_id=3, start=10, dwell=0),
        ))
        repair = qea_lns._LnsRepair(
            schedule=Schedule((
                Assignment(train_id=1, route_id=2, start=0, dwell=0),
                Assignment(train_id=2, route_id=3, start=0, dwell=0),
            )),
            route_choices={1: 2, 2: 3},
            priority_order=(2, 1),
        )
        updates = []
        def monotonic() -> float:
            return 11.0 if len(updates) >= 2 else 0.0

        def score(_, schedule):
            return (0, 20) if schedule == repair.schedule else (10, 20)

        with patch.object(qea_lns, '_initial_population', return_value=states), patch.object(
            qea_lns, '_sample_candidate', side_effect=candidates,
        ), patch.object(
            qea_lns, '_construct_schedule', return_value=initial,
        ), patch.object(
            qea_lns, '_reschedule_wait_chain', side_effect=[repair, None, None],
        ), patch.object(
            qea_lns, '_candidate_from_schedule', return_value=None,
        ), patch.object(
            qea_lns, '_update_member_toward', side_effect=lambda *args: updates.append(args),
        ), patch.object(
            qea_lns, '_score', side_effect=score,
        ), patch.object(qea_lns.time, 'monotonic', side_effect=monotonic):
            qea_lns.solve_qea_lns(instance, seed=42, time_limit_s=10)

        self.assertTrue(updates)
        self.assertTrue(all(update[3].route_choices == {1: 2, 2: 3} for update in updates))

    def test_retains_the_global_lns_target_after_a_later_worse_local_result(self) -> None:
        import station_dispatch.qea_lns as qea_lns

        instance = qea_instance()
        state = qea_lns._initial_population(instance)[0]
        first_candidate = qea_lns._Candidate(
            route_choices={1: 1, 2: 3},
            priority_bits={1: (0,) * qea_lns._PRIORITY_BITS, 2: (0,) * qea_lns._PRIORITY_BITS},
            priority_order=(1, 2),
        )
        later_candidate = qea_lns._Candidate(
            route_choices={1: 1, 2: 3},
            priority_bits={1: (1,) * qea_lns._PRIORITY_BITS, 2: (1,) * qea_lns._PRIORITY_BITS},
            priority_order=(2, 1),
        )
        first_raw = Schedule((
            Assignment(train_id=1, route_id=1, start=0, dwell=0),
            Assignment(train_id=2, route_id=3, start=10, dwell=0),
        ))
        first_repaired = Schedule((
            Assignment(train_id=1, route_id=2, start=0, dwell=0),
            Assignment(train_id=2, route_id=3, start=0, dwell=0),
        ))
        later_raw = Schedule((
            Assignment(train_id=1, route_id=1, start=0, dwell=0),
            Assignment(train_id=2, route_id=3, start=9, dwell=0),
        ))
        later_repaired = Schedule((
            Assignment(train_id=1, route_id=1, start=0, dwell=0),
            Assignment(train_id=2, route_id=3, start=1, dwell=0),
        ))
        first_repair = qea_lns._LnsRepair(
            schedule=first_repaired,
            route_choices={1: 2, 2: 3},
            priority_order=(2, 1),
        )
        later_repair = qea_lns._LnsRepair(
            schedule=later_repaired,
            route_choices={1: 1, 2: 3},
            priority_order=(1, 2),
        )
        updates = []

        def monotonic() -> float:
            return 11.0 if len(updates) >= 2 else 0.0

        def score(_, schedule):
            if schedule is first_repaired:
                return (0, 20)
            if schedule is later_repaired:
                return (1, 20)
            if schedule is later_raw:
                return (9, 20)
            return (10, 20)

        with patch.object(qea_lns, '_initial_population', return_value=[state]), patch.object(
            qea_lns,
            '_sample_candidate',
            side_effect=[first_candidate, later_candidate],
        ), patch.object(
            qea_lns,
            '_construct_schedule',
            side_effect=[first_raw, later_raw],
        ), patch.object(
            qea_lns,
            '_reschedule_wait_chain',
            side_effect=[first_repair, None, None, later_repair, None, None],
        ), patch.object(
            qea_lns,
            '_update_member_toward',
            side_effect=lambda *args: updates.append(args),
        ), patch.object(
            qea_lns,
            '_score',
            side_effect=score,
        ), patch.object(qea_lns.time, 'monotonic', side_effect=monotonic):
            qea_lns.solve_qea_lns(instance, seed=42, time_limit_s=10)

        global_target_updates = [
            update
            for update in updates
            if update[3].route_choices == {1: 2, 2: 3}
        ]
        self.assertEqual(len(global_target_updates), 2)

    def test_equal_score_lns_repair_does_not_add_quantum_feedback(self) -> None:
        import station_dispatch.qea_lns as qea_lns

        instance = qea_instance()
        state = qea_lns._initial_population(instance)[0]
        candidate = qea_lns._Candidate(
            route_choices={1: 1, 2: 3},
            priority_bits={1: (0,) * qea_lns._PRIORITY_BITS, 2: (0,) * qea_lns._PRIORITY_BITS},
            priority_order=(1, 2),
        )
        initial = Schedule((
            Assignment(train_id=1, route_id=1, start=0, dwell=0),
            Assignment(train_id=2, route_id=3, start=10, dwell=0),
        ))
        repair = qea_lns._LnsRepair(
            schedule=Schedule((
                Assignment(train_id=1, route_id=2, start=0, dwell=0),
                Assignment(train_id=2, route_id=3, start=0, dwell=0),
            )),
            route_choices={1: 2, 2: 3},
            priority_order=(2, 1),
        )
        updates = []
        calls = 0

        def monotonic() -> float:
            nonlocal calls
            calls += 1
            return 0.0 if calls <= 5 else 11.0

        with patch.object(qea_lns, '_initial_population', return_value=[state]), patch.object(
            qea_lns,
            '_sample_candidate',
            return_value=candidate,
        ), patch.object(
            qea_lns,
            '_construct_schedule',
            return_value=initial,
        ), patch.object(
            qea_lns,
            '_reschedule_wait_chain',
            return_value=repair,
        ), patch.object(
            qea_lns,
            '_update_member_toward',
            side_effect=lambda *args: updates.append(args),
        ), patch.object(
            qea_lns,
            '_score',
            return_value=(10, 20),
        ), patch.object(qea_lns.time, 'monotonic', side_effect=monotonic):
            qea_lns.solve_qea_lns(instance, seed=42, time_limit_s=10)

        self.assertEqual(len(updates), 1)
        self.assertIs(updates[0][1], state)
        self.assertIs(updates[0][2], candidate)
        self.assertIs(updates[0][3], candidate)

    def test_uses_sixteen_bits_for_the_qea_priority_encoding(self) -> None:
        from station_dispatch.qea_lns import _PRIORITY_BITS, _initial_population, _sample_candidate

        instance = qea_instance()
        candidate = _sample_candidate(instance, _initial_population(instance)[0], random.Random(42))

        self.assertEqual(_PRIORITY_BITS, 16)
        for train in instance.trains:
            self.assertEqual(len(candidate.priority_bits[train.id]), 16)
    def test_samples_one_actual_allowed_route_for_every_train(self) -> None:
        from station_dispatch.qea_lns import _initial_population, _sample_candidate

        instance = qea_instance()
        candidate = _sample_candidate(
            instance,
            _initial_population(instance)[0],
            random.Random(42),
        )

        for train in instance.trains:
            self.assertIn(candidate.route_choices[train.id], train.route_ids)

    def test_samples_a_unique_complete_train_priority_order(self) -> None:
        from station_dispatch.qea_lns import _initial_population, _sample_candidate

        instance = qea_instance()
        candidate = _sample_candidate(
            instance,
            _initial_population(instance)[0],
            random.Random(42),
        )

        priority = candidate.priority()
        self.assertEqual(set(priority), {train.id for train in instance.trains})
        self.assertEqual(set(priority.values()), set(range(len(instance.trains))))

    def test_constructs_its_initial_schedule_without_calling_fcfs(self) -> None:
        import station_dispatch.qea_lns as qea_lns

        with patch.object(
            qea_lns,
            "solve_fcfs",
            side_effect=AssertionError("QEA-LNS must not call FCFS"),
            create=True,
        ):
            result = qea_lns.solve_qea_lns(qea_instance(), seed=7, time_limit_s=1)

        self.assertTrue(result.feasible)
        self.assertTrue(validate_schedule(qea_instance(), result.schedule).feasible)

    def test_checks_the_encoded_route_under_the_common_entry_and_safety_rules(self) -> None:
        from station_dispatch.qea_lns import _assign_train

        instance = qea_route_choice_instance()
        assignment = _assign_train(
            instance,
            [Assignment(train_id=2, route_id=3, start=0, dwell=0)],
            instance.trains[0],
            route_id=1,
            deadline=time.monotonic() + 1,
        )

        self.assertIsNotNone(assignment)
        self.assertEqual(assignment.route_id, 1)
        self.assertEqual(assignment.start, 10)

    def test_reorders_a_platform_departure_with_only_the_needed_extra_dwell(self) -> None:
        from station_dispatch.qea_lns import solve_qea_lns

        instance = qea_dwell_reordering_instance()
        result = solve_qea_lns(instance, seed=42, time_limit_s=1)
        assignments = {assignment.train_id: assignment for assignment in result.schedule.assignments}

        self.assertTrue(validate_schedule(instance, result.schedule).feasible)
        self.assertEqual(result.end_sum, 21)
        self.assertEqual(result.makespan, 8)
        self.assertEqual(assignments[2].start, 0)
        self.assertEqual(assignments[1].start, 0)
        self.assertEqual(assignments[1].dwell, 2)
        self.assertEqual(assignments[3].start, 4)

    def test_continues_later_wait_targets_after_work_quota_without_new_raw_improvement(self) -> None:
        import station_dispatch.qea_lns as qea_lns

        instance = qea_wait_chain_instance()
        initial = Schedule((Assignment(1, 1, 12, 0), Assignment(2, 2, 1, 0), Assignment(3, 3, 1, 0)))
        candidate = qea_lns._Candidate({1: 1, 2: 2, 3: 3}, {t: (0,) * 16 for t in (1, 2, 3)}, (1, 2, 3))
        clock = 0.0
        targets = []

        def construct(*args, **kwargs):
            nonlocal clock
            clock += 0.1
            return initial

        def repair(_instance, schedule, _rng, deadline, *, target_index, progress=None, max_steps=None):
            nonlocal clock
            targets.append(target_index)
            self.assertEqual(max_steps, qea_lns._LNS_STEPS)
            clock += 0.1
            return None

        with patch.object(qea_lns.time, "monotonic", side_effect=lambda: clock), patch.object(
            qea_lns, "_sample_candidate", return_value=candidate,
        ), patch.object(qea_lns, "_construct_schedule", side_effect=construct), patch.object(
            qea_lns, "_reschedule_wait_chain", side_effect=repair,
        ):
            result = qea_lns.solve_qea_lns(instance, seed=42, time_limit_s=7)

        self.assertTrue(validate_schedule(instance, result.schedule).feasible)
        self.assertEqual(targets[:4], [0, 1, 2, 0])
        self.assertGreater(result.iterations, qea_lns._POPULATION)

    def test_continues_lns_from_its_improvement_across_generations(self) -> None:
        import station_dispatch.qea_lns as qea_lns

        instance = qea_wait_chain_instance()
        initial = Schedule((Assignment(1, 1, 12, 0), Assignment(2, 2, 1, 0), Assignment(3, 3, 1, 0)))
        improved = Schedule((Assignment(1, 1, 11, 0), Assignment(2, 2, 1, 0), Assignment(3, 3, 1, 0)))
        candidate = qea_lns._Candidate({1: 1, 2: 2, 3: 3}, {t: (0,) * 16 for t in (1, 2, 3)}, (1, 2, 3))
        clock = 0.0
        inputs = []
        states = []

        def construct(*args, **kwargs):
            nonlocal clock
            clock += 0.1
            return initial

        def repair(_instance, schedule, _rng, deadline, *, target_index, progress=None, max_steps=None):
            nonlocal clock
            inputs.append(schedule)
            states.append(progress)
            self.assertEqual(max_steps, qea_lns._LNS_STEPS)
            clock += 0.1
            if len(inputs) == 1:
                return qea_lns._LnsRepair(improved, {1: 1}, (1,))
            return None

        with patch.object(qea_lns.time, "monotonic", side_effect=lambda: clock), patch.object(
            qea_lns, "_sample_candidate", return_value=candidate,
        ), patch.object(qea_lns, "_construct_schedule", side_effect=construct), patch.object(
            qea_lns, "_reschedule_wait_chain", side_effect=repair,
        ):
            result = qea_lns.solve_qea_lns(instance, seed=42, time_limit_s=5)

        self.assertTrue(validate_schedule(instance, result.schedule).feasible)
        self.assertEqual(result.schedule, qea_lns._advance_schedule(instance, improved, math.inf))
        self.assertGreater(len(inputs), 1)
        self.assertIsNot(states[0], states[1])
        self.assertTrue(all(schedule is result.schedule for schedule in inputs[1:]))

    def test_scans_all_wait_chains_once_then_stops_without_a_new_qea_candidate(self) -> None:
        import station_dispatch.qea_lns as qea_lns

        instance = qea_wait_chain_instance()
        initial = Schedule((
            Assignment(train_id=1, route_id=1, start=12, dwell=0),
            Assignment(train_id=2, route_id=2, start=1, dwell=0),
            Assignment(train_id=3, route_id=3, start=1, dwell=0),
        ))
        candidate = qea_lns._Candidate(
            route_choices={1: 1, 2: 2, 3: 3},
            priority_bits={
                1: (0,) * qea_lns._PRIORITY_BITS,
                2: (0,) * qea_lns._PRIORITY_BITS,
                3: (0,) * qea_lns._PRIORITY_BITS,
            },
            priority_order=(1, 2, 3),
        )
        lns_inputs: list[Schedule] = []
        target_indices: list[int] = []
        timer_calls = 0

        self.assertTrue(validate_schedule(instance, initial).feasible)

        def monotonic() -> float:
            nonlocal timer_calls
            timer_calls += 1
            return 0.0 if timer_calls <= 40 else 11.0

        def reschedule(
            _instance: StationInstance,
            current: Schedule,
            _rng: random.Random,
            _deadline: float,
            *,
            target_index: int,
            progress=None,
            max_steps=None,
        ) -> qea_lns._LnsRepair:
            progress["complete"] = True
            lns_inputs.append(current)
            target_indices.append(target_index)
            return qea_lns._LnsRepair(
                schedule=initial,
                route_choices={1: 1, 2: 2, 3: 3},
                priority_order=(1, 2, 3),
            )

        with patch.object(qea_lns, "_initial_population", return_value=[qea_lns._QuantumState({}, {})]), patch.object(
            qea_lns,
            "_sample_candidate",
            return_value=candidate,
        ), patch.object(
            qea_lns,
            "_construct_schedule",
            return_value=initial,
        ), patch.object(
            qea_lns,
            "_reschedule_wait_chain",
            side_effect=reschedule,
        ), patch.object(
            qea_lns,
            "_update_member_toward",
        ), patch.object(
            qea_lns,
            "_score",
            return_value=(10, 20),
        ), patch.object(qea_lns.time, "monotonic", side_effect=monotonic):
            result = qea_lns.solve_qea_lns(instance, seed=42, time_limit_s=10)

        self.assertEqual(lns_inputs, [initial, initial, initial])
        self.assertEqual(target_indices, [0, 1, 2])
        self.assertIs(result.schedule, initial)

    def test_restarts_at_the_largest_wait_chain_after_each_strict_lns_improvement(self) -> None:
        import station_dispatch.qea_lns as qea_lns

        instance = qea_instance()
        initial = Schedule((
            Assignment(train_id=1, route_id=1, start=0, dwell=0),
            Assignment(train_id=2, route_id=3, start=10, dwell=0),
        ))
        strict_one = Schedule((
            Assignment(train_id=1, route_id=1, start=0, dwell=0),
            Assignment(train_id=2, route_id=3, start=11, dwell=0),
        ))
        strict_two = Schedule((
            Assignment(train_id=1, route_id=1, start=0, dwell=0),
            Assignment(train_id=2, route_id=3, start=12, dwell=0),
        ))
        candidate = qea_lns._Candidate(
            route_choices={1: 1, 2: 3},
            priority_bits={1: (0,) * qea_lns._PRIORITY_BITS, 2: (0,) * qea_lns._PRIORITY_BITS},
            priority_order=(1, 2),
        )
        lns_inputs: list[Schedule] = []
        target_indices: list[int] = []
        timer_calls = 0

        def monotonic() -> float:
            nonlocal timer_calls
            timer_calls += 1
            return 0.0 if timer_calls <= 40 else 11.0

        def reschedule(
            _instance: StationInstance,
            current: Schedule,
            _rng: random.Random,
            _deadline: float,
            *,
            target_index: int,
            progress=None,
            max_steps=None,
        ) -> qea_lns._LnsRepair:
            progress["complete"] = True
            lns_inputs.append(current)
            target_indices.append(target_index)
            if current is initial:
                schedule = strict_one
            elif current is strict_one:
                schedule = strict_two
            else:
                schedule = strict_two
            return qea_lns._LnsRepair(
                schedule=schedule,
                route_choices={1: 1, 2: 3},
                priority_order=(2, 1),
            )

        def score(_instance: StationInstance, schedule: Schedule) -> tuple[int, int]:
            if schedule is strict_two:
                return (8, 20)
            if schedule is strict_one:
                return (9, 20)
            return (10, 20)

        with patch.object(qea_lns, "_initial_population", return_value=[qea_lns._QuantumState({}, {})]), patch.object(
            qea_lns,
            "_sample_candidate",
            return_value=candidate,
        ), patch.object(
            qea_lns,
            "_construct_schedule",
            return_value=initial,
        ), patch.object(
            qea_lns,
            "_reschedule_wait_chain",
            side_effect=reschedule,
        ), patch.object(
            qea_lns,
            "_update_member_toward",
        ) as update_member, patch.object(
            qea_lns,
            "_score",
            side_effect=score,
        ), patch.object(qea_lns.time, "monotonic", side_effect=monotonic):
            result = qea_lns.solve_qea_lns(instance, seed=42, time_limit_s=10)

        self.assertEqual(lns_inputs, [initial, strict_one, strict_two])
        self.assertEqual(target_indices, [0, 0, 0])
        self.assertIs(result.schedule, strict_two)

    def test_joint_route_domain_keeps_current_and_all_allowed_target_routes(self) -> None:
        from station_dispatch.qea_lns import _component_route_choices

        instance = qea_route_choice_instance()
        route_choices = list(
            _component_route_choices(
                instance,
                {1: 1, 2: 3},
                {1},
                time.monotonic() + 1,
            )
        )

        self.assertEqual(route_choices, [{1: 1, 2: 3}, {1: 2, 2: 3}])


    def test_keeps_the_better_of_a_joint_and_necessary_dwell_repair(self) -> None:
        import station_dispatch.qea_lns as qea_lns

        instance = qea_dwell_reordering_instance()
        current = Schedule((
            Assignment(train_id=1, route_id=1, start=0, dwell=0),
            Assignment(train_id=2, route_id=2, start=3, dwell=0),
            Assignment(train_id=3, route_id=3, start=4, dwell=0),
        ))
        dwell_repair = Schedule((
            Assignment(train_id=1, route_id=1, start=0, dwell=2),
            Assignment(train_id=2, route_id=2, start=0, dwell=0),
            Assignment(train_id=3, route_id=3, start=4, dwell=0),
        ))
        joint_repair = qea_lns._LnsRepair(current, {2: 2, 3: 3}, (2, 3))

        self.assertTrue(validate_schedule(instance, current).feasible)
        self.assertTrue(validate_schedule(instance, dwell_repair).feasible)
        with patch.object(
            qea_lns,
            "_alternative_route_component",
            return_value=({2, 3}, {3}),
        ), patch.object(
            qea_lns,
            "_wait_chain",
            return_value=({2, 3}, 3),
        ), patch.object(
            qea_lns,
            "_search_joint_component",
            return_value=joint_repair,
        ), patch.object(
            qea_lns,
            "_reschedule_with_necessary_dwell",
            return_value=qea_lns._LnsRepair(dwell_repair, {2: 2, 3: 3}, (2, 3)),
        ):
            result = qea_lns._reschedule_wait_chain(
                instance,
                current,
                random.Random(42),
                time.monotonic() + 1,
                target_index=0,
            )

        self.assertIsNotNone(result)
        self.assertEqual(qea_lns._score(instance, result.schedule), (6, 8))


    def test_keeps_joint_repair_when_following_dwell_search_hits_deadline(self) -> None:
        import station_dispatch.qea_lns as qea_lns

        instance = qea_alternative_route_component_instance()
        current = Schedule((
            Assignment(train_id=1, route_id=1, start=10, dwell=0),
            Assignment(train_id=2, route_id=3, start=0, dwell=0),
            Assignment(train_id=3, route_id=4, start=0, dwell=0),
        ))
        joint_schedule = Schedule((
            Assignment(train_id=1, route_id=2, start=0, dwell=0),
            Assignment(train_id=2, route_id=3, start=0, dwell=0),
            Assignment(train_id=3, route_id=5, start=0, dwell=0),
        ))
        joint_repair = qea_lns._LnsRepair(joint_schedule, {1: 2, 2: 3, 3: 5}, (1, 2, 3))

        with patch.object(
            qea_lns,
            "_search_joint_component",
            return_value=joint_repair,
        ), patch.object(
            qea_lns,
            "_reschedule_with_necessary_dwell",
            side_effect=TimeoutError("station-dispatching method exceeded its single-instance time limit"),
        ):
            result = qea_lns._reschedule_wait_chain(
                instance,
                current,
                random.Random(42),
                time.monotonic() + 1,
                target_index=0,
            )

        self.assertIsNotNone(result)
        self.assertIs(result.schedule, joint_schedule)


    def test_returns_a_strictly_better_joint_component_repair(self) -> None:
        import station_dispatch.qea_lns as qea_lns

        instance = qea_alternative_route_component_instance()
        current = Schedule((
            Assignment(train_id=1, route_id=1, start=10, dwell=0),
            Assignment(train_id=2, route_id=3, start=0, dwell=0),
            Assignment(train_id=3, route_id=4, start=0, dwell=0),
        ))
        joint_schedule = Schedule((
            Assignment(train_id=1, route_id=2, start=0, dwell=0),
            Assignment(train_id=2, route_id=3, start=0, dwell=0),
            Assignment(train_id=3, route_id=5, start=0, dwell=0),
        ))
        joint_repair = qea_lns._LnsRepair(joint_schedule, {1: 2, 2: 3, 3: 5}, (1, 2, 3))

        self.assertTrue(validate_schedule(instance, current).feasible)
        self.assertTrue(validate_schedule(instance, joint_schedule).feasible)
        with patch.object(
            qea_lns,
            "_search_joint_component",
            return_value=joint_repair,
        ), patch.object(
            qea_lns,
            "_reschedule_with_necessary_dwell",
            return_value=None,
        ):
            result = qea_lns._reschedule_wait_chain(
                instance,
                current,
                random.Random(42),
                time.monotonic() + 1,
                target_index=0,
            )

        self.assertIsNotNone(result)
        self.assertIs(result.schedule, joint_schedule)





    def test_resumed_wait_chain_reuses_its_unchanged_conflict_context(self) -> None:
        import station_dispatch.qea_lns as qea_lns

        instance = qea_instance()
        schedule = Schedule((
            Assignment(train_id=1, route_id=1, start=0, dwell=0),
            Assignment(train_id=2, route_id=3, start=10, dwell=0),
        ))
        progress = {}
        component_result = qea_lns._alternative_route_component(instance, schedule, 0)
        with patch.object(
            qea_lns, '_alternative_route_component', return_value=component_result,
        ) as component, patch.object(
            qea_lns, '_wait_chain', wraps=qea_lns._wait_chain,
        ) as wait_chain, patch.object(
            qea_lns, '_reschedule_with_necessary_dwell', return_value=None,
        ), patch.object(qea_lns, '_search_joint_component', return_value=None):
            for _ in range(2):
                qea_lns._reschedule_wait_chain(
                    instance, schedule, random.Random(42), time.monotonic() + 1, 0, progress,
                )

        self.assertEqual(component.call_count, 1)
        self.assertEqual(wait_chain.call_count, 1)


def qea_six_train_order_instance() -> StationInstance:
    return StationInstance(
        path=Path("qea-six-train-order-synthetic.dzn"),
        edge_count=10,
        edge_names=tuple(f"edge_{index}" for index in range(1, 11)),
        edge_types=("inter",) * 10,
        platform_names=(),
        trains=(
            Train(1, "target", 0, "pass", (1, 2, 3, 4, 5)),
            Train(2, "T2", 0, "pass", (6,)),
            Train(3, "T3", 0, "pass", (7,)),
            Train(4, "T4", 0, "pass", (8,)),
            Train(5, "T5", 0, "pass", (9,)),
            Train(6, "T6", 10, "pass", (10,)),
        ),
        routes=tuple(Route(route_id, 1 if route_id <= 5 else route_id - 4, "", 0, 1, (route_id,)) for route_id in range(1, 11)),
        blocks=tuple(Block(block_id, block_id, 1, 0, False) for block_id in range(1, 11)),
    )


class ConflictPriorityNeighborhoodTest(unittest.TestCase):
    def test_enumerates_all_fifo_legal_orders_in_the_component(self) -> None:
        from station_dispatch.qea_lns import _component_priority_orders

        instance = qea_six_train_order_instance()
        orders = list(_component_priority_orders(instance, {1, 2, 3, 4, 5, 6}))

        self.assertEqual(len(orders), math.factorial(6))
        self.assertEqual(len(set(orders)), math.factorial(6))



def qea_wait_chain_instance() -> StationInstance:
    return StationInstance(
        path=Path("qea-wait-chain-synthetic.dzn"),
        edge_count=4,
        edge_names=("entry_a", "shared_throat", "entry_b", "isolated"),
        edge_types=("inter", "inter", "inter", "inter"),
        platform_names=(),
        trains=(
            Train(1, "waiting", 0, "pass", (1,)),
            Train(2, "blocker", 0, "pass", (2,)),
            Train(3, "unrelated", 0, "pass", (3,)),
        ),
        routes=(
            Route(1, 1, "", 0, 11, (1, 2)),
            Route(2, 2, "", 0, 11, (3, 2)),
            Route(3, 3, "", 0, 10, (4,)),
        ),
        blocks=(
            Block(1, 1, 1, 0, False),
            Block(2, 2, 10, 0, False),
            Block(3, 3, 1, 0, False),
            Block(4, 4, 10, 0, False),
        ),
    )



def qea_minimal_wait_chain_instance() -> StationInstance:
    return StationInstance(
        path=Path("qea-minimal-wait-chain-synthetic.dzn"),
        edge_count=3,
        edge_names=("entry_a", "shared_throat", "entry_b"),
        edge_types=("inter", "inter", "inter"),
        platform_names=(),
        trains=(
            Train(1, "A1", 0, "pass", (1,)),
            Train(2, "A2", 1, "pass", (2,)),
            Train(3, "A3_waiting", 2, "pass", (3,)),
            Train(4, "A4_successor", 9, "pass", (4,)),
            Train(5, "B_blocker", 0, "pass", (5,)),
        ),
        routes=(
            Route(1, 1, "", 0, 1, (1,)),
            Route(2, 2, "", 0, 1, (2,)),
            Route(3, 3, "", 0, 11, (3, 4)),
            Route(4, 4, "", 0, 1, (5,)),
            Route(5, 5, "", 0, 11, (6, 7)),
        ),
        blocks=(
            Block(1, 1, 1, 0, False),
            Block(2, 1, 1, 0, False),
            Block(3, 1, 1, 0, False),
            Block(4, 2, 10, 0, False),
            Block(5, 1, 1, 0, False),
            Block(6, 3, 1, 0, False),
            Block(7, 2, 10, 0, False),
        ),
    )


def qea_alternative_route_component_instance() -> StationInstance:
    return StationInstance(
        path=Path("qea-alternative-route-component-synthetic.dzn"),
        edge_count=5,
        edge_names=("entry_target", "current_shared", "entry_blocker", "alternative_shared", "alternative_free"),
        edge_types=("inter", "inter", "inter", "inter", "inter"),
        platform_names=(),
        trains=(
            Train(1, "target", 0, "pass", (1, 2)),
            Train(2, "current_route_blocker", 0, "pass", (3,)),
            Train(3, "alternative_route_blocker", 0, "pass", (4, 5)),
        ),
        routes=(
            Route(1, 1, "", 0, 11, (1, 2)),
            Route(2, 1, "", 0, 11, (1, 3)),
            Route(3, 2, "", 0, 11, (4,)),
            Route(4, 3, "", 0, 11, (5, 6)),
            Route(5, 3, "", 0, 11, (5, 7)),
        ),
        blocks=(
            Block(1, 1, 1, 0, False),
            Block(2, 2, 10, 0, False),
            Block(3, 4, 10, 0, False),
            Block(4, 2, 11, 0, False),
            Block(5, 3, 1, 0, False),
            Block(6, 4, 10, 0, False),
            Block(7, 5, 10, 0, False),
        ),
    )


class AlternativeRouteComponentTest(unittest.TestCase):
    def test_repairs_wait_by_jointly_changing_target_and_alternative_blocker(self) -> None:
        import station_dispatch.qea_lns as qea_lns

        instance = qea_alternative_route_component_instance()
        current = Schedule((
            Assignment(train_id=1, route_id=1, start=10, dwell=0),
            Assignment(train_id=2, route_id=3, start=0, dwell=0),
            Assignment(train_id=3, route_id=4, start=0, dwell=0),
        ))

        self.assertTrue(validate_schedule(instance, current).feasible)
        component, decision_ids = qea_lns._alternative_route_component(instance, current, target_index=0)
        self.assertEqual(component, {1, 2, 3})
        self.assertEqual(decision_ids, {1, 3})

        repair = qea_lns._reschedule_wait_chain(
            instance,
            current,
            random.Random(42),
            time.monotonic() + 5,
            target_index=0,
        )

        self.assertIsNotNone(repair)
        self.assertTrue(validate_schedule(instance, repair.schedule).feasible)
        self.assertLess(qea_lns._score(instance, repair.schedule), qea_lns._score(instance, current))
        assignments = {assignment.train_id: assignment for assignment in repair.schedule.assignments}
        self.assertEqual(assignments[1].route_id, 2)
        self.assertEqual(assignments[1].start, 0)
        self.assertEqual(assignments[3].route_id, 5)


    def test_returns_found_joint_repair_when_deadline_arrives_after_improvement(self) -> None:
        import station_dispatch.qea_lns as qea_lns

        instance = qea_alternative_route_component_instance()
        current = Schedule((
            Assignment(train_id=1, route_id=1, start=10, dwell=0),
            Assignment(train_id=2, route_id=3, start=0, dwell=0),
            Assignment(train_id=3, route_id=4, start=0, dwell=0),
        ))
        component, decision_ids = qea_lns._alternative_route_component(instance, current, target_index=0)
        assignments = {assignment.train_id: assignment for assignment in current.assignments}
        fixed_assignments = tuple(
            assignment for assignment in current.assignments if assignment.train_id not in component
        )
        base_route_choices = {train_id: assignments[train_id].route_id for train_id in component}
        started_at = time.monotonic()
        deadline = started_at + 1
        clock = started_at
        original_score = qea_lns._score
        current_score = original_score(instance, current)

        def score(_instance, schedule):
            nonlocal clock
            result = original_score(_instance, schedule)
            if result < current_score:
                clock = deadline
            return result

        with patch.object(qea_lns, "time", SimpleNamespace(monotonic=lambda: clock)), patch.object(
            qea_lns, "_score", side_effect=score,
        ):
            repair = qea_lns._search_joint_component(
                instance,
                component,
                decision_ids,
                base_route_choices,
                fixed_assignments,
                qea_lns._score(instance, current),
                deadline,
            )

        self.assertIsNotNone(repair)
        self.assertTrue(validate_schedule(instance, repair.schedule).feasible)
        self.assertLess(qea_lns._score(instance, repair.schedule), qea_lns._score(instance, current))


    def test_keeps_found_joint_repair_when_assignment_hits_deadline(self) -> None:
        import station_dispatch.qea_lns as qea_lns

        instance = qea_alternative_route_component_instance()
        current = Schedule((
            Assignment(train_id=1, route_id=1, start=10, dwell=0),
            Assignment(train_id=2, route_id=3, start=0, dwell=0),
            Assignment(train_id=3, route_id=4, start=0, dwell=0),
        ))
        component, decision_ids = qea_lns._alternative_route_component(instance, current, target_index=0)
        assignments = {assignment.train_id: assignment for assignment in current.assignments}
        fixed_assignments = tuple(
            assignment for assignment in current.assignments if assignment.train_id not in component
        )
        base_route_choices = {train_id: assignments[train_id].route_id for train_id in component}
        assign_calls = 0
        original_assign = qea_lns._assign_train

        def assign_until_deadline(*args, **kwargs):
            nonlocal assign_calls
            assign_calls += 1
            if assign_calls == 10:
                raise TimeoutError("station-dispatching method exceeded its single-instance time limit")
            return original_assign(*args, **kwargs)

        with patch.object(qea_lns, "_assign_train", side_effect=assign_until_deadline):
            repair = qea_lns._search_joint_component(
                instance,
                component,
                decision_ids,
                base_route_choices,
                fixed_assignments,
                qea_lns._score(instance, current),
                time.monotonic() + 1,
            )

        self.assertIsNotNone(repair)
        self.assertTrue(validate_schedule(instance, repair.schedule).feasible)
        self.assertLess(qea_lns._score(instance, repair.schedule), qea_lns._score(instance, current))


    def test_searches_a_joint_component_without_rebuilding_each_full_candidate(self) -> None:
        import station_dispatch.qea_lns as qea_lns

        instance = qea_alternative_route_component_instance()
        current = Schedule((
            Assignment(train_id=1, route_id=1, start=10, dwell=0),
            Assignment(train_id=2, route_id=3, start=0, dwell=0),
            Assignment(train_id=3, route_id=4, start=0, dwell=0),
        ))

        with patch.object(qea_lns, "_construct_schedule", side_effect=AssertionError("full reconstruction is forbidden")):
            repair = qea_lns._reschedule_wait_chain(
                instance,
                current,
                random.Random(42),
                time.monotonic() + 5,
                target_index=0,
            )

        self.assertIsNotNone(repair)
        self.assertTrue(validate_schedule(instance, repair.schedule).feasible)
        self.assertLess(qea_lns._score(instance, repair.schedule), qea_lns._score(instance, current))


    def test_includes_every_multi_route_train_in_the_current_conflict_component(self) -> None:
        import station_dispatch.qea_lns as qea_lns

        instance = StationInstance(
            path=Path("qea-complete-component-synthetic.dzn"),
            edge_count=5,
            edge_names=("entry_target", "current_shared", "entry_blocker", "alternative_shared", "alternative_free"),
            edge_types=("inter", "inter", "inter", "inter", "inter"),
            platform_names=(),
            trains=(
                Train(1, "target", 0, "pass", (1, 2)),
                Train(2, "current_route_blocker", 0, "pass", (3, 6)),
                Train(3, "alternative_route_blocker", 0, "pass", (4, 5)),
            ),
            routes=(
                Route(1, 1, "", 0, 11, (1, 2)),
                Route(2, 1, "", 0, 11, (1, 3)),
                Route(3, 2, "", 0, 11, (4,)),
                Route(4, 3, "", 0, 11, (5, 6)),
                Route(5, 3, "", 0, 11, (5, 7)),
                Route(6, 2, "", 0, 11, (8,)),
            ),
            blocks=(
                Block(1, 1, 1, 0, False),
                Block(2, 2, 10, 0, False),
                Block(3, 4, 10, 0, False),
                Block(4, 2, 11, 0, False),
                Block(5, 3, 1, 0, False),
                Block(6, 4, 10, 0, False),
                Block(7, 5, 10, 0, False),
                Block(8, 5, 10, 0, False),
            ),
        )
        current = Schedule((
            Assignment(train_id=1, route_id=1, start=10, dwell=0),
            Assignment(train_id=2, route_id=3, start=0, dwell=0),
            Assignment(train_id=3, route_id=4, start=0, dwell=0),
        ))

        self.assertTrue(validate_schedule(instance, current).feasible)
        component, decision_ids = qea_lns._alternative_route_component(instance, current, target_index=0)

        self.assertEqual(component, {1, 2, 3})
        self.assertEqual(decision_ids, {1, 2, 3})

class WaitNeighborhoodTest(unittest.TestCase):
    def test_selects_the_waiting_train_and_its_shared_resource_blocker(self) -> None:
        import station_dispatch.qea_lns as qea_lns

        instance = qea_wait_chain_instance()
        schedule = Schedule((
            Assignment(train_id=2, route_id=2, start=0, dwell=0),
            Assignment(train_id=1, route_id=1, start=10, dwell=0),
            Assignment(train_id=3, route_id=3, start=0, dwell=0),
        ))

        self.assertTrue(validate_schedule(instance, schedule).feasible)
        neighborhood = getattr(qea_lns, "_wait_neighborhood", None)
        self.assertIsNotNone(neighborhood, "QEA-LNS needs a wait-chain neighborhood selector")
        self.assertEqual(neighborhood(instance, schedule), {1, 2})

    def test_excludes_unrelated_fifo_successor_but_keeps_direct_predecessor_and_blocker(self) -> None:
        from station_dispatch.qea_lns import _wait_chain

        instance = qea_minimal_wait_chain_instance()
        schedule = Schedule((
            Assignment(train_id=1, route_id=1, start=0, dwell=0),
            Assignment(train_id=2, route_id=2, start=1, dwell=0),
            Assignment(train_id=3, route_id=3, start=10, dwell=0),
            Assignment(train_id=4, route_id=4, start=11, dwell=0),
            Assignment(train_id=5, route_id=5, start=0, dwell=0),
        ))

        self.assertTrue(validate_schedule(instance, schedule).feasible)
        neighborhood, conflict_edge = _wait_chain(instance, schedule)

        self.assertEqual(neighborhood, {2, 3, 5})
        self.assertEqual(conflict_edge, 2)

    def test_cycles_to_the_next_waiting_conflict_chain(self) -> None:
        from station_dispatch.qea_lns import _wait_chain

        instance = qea_dwell_reordering_instance()
        schedule = Schedule((
            Assignment(train_id=1, route_id=1, start=0, dwell=0),
            Assignment(train_id=2, route_id=2, start=3, dwell=0),
            Assignment(train_id=3, route_id=3, start=4, dwell=0),
        ))

        self.assertTrue(validate_schedule(instance, schedule).feasible)
        neighborhood, conflict_edge = _wait_chain(instance, schedule, target_index=1)
        self.assertTrue({1, 2}.issubset(neighborhood))
        self.assertEqual(conflict_edge, 3)


if __name__ == "__main__":
    unittest.main()


def qea_joint_component_instance() -> StationInstance:
    return StationInstance(
        path=Path("qea-joint-component-synthetic.dzn"),
        edge_count=6,
        edge_names=tuple(f"edge_{edge_id}" for edge_id in range(1, 7)),
        edge_types=("inter",) * 6,
        platform_names=(),
        trains=(
            Train(1, "A1", 0, "pass", (1, 2)),
            Train(2, "A2", 1, "pass", (3, 4)),
            Train(3, "B1", 0, "pass", (5, 6)),
        ),
        routes=(
            Route(1, 1, "", 0, 1, (1,)),
            Route(2, 1, "", 0, 1, (2,)),
            Route(3, 2, "", 0, 1, (3,)),
            Route(4, 2, "", 0, 1, (4,)),
            Route(5, 3, "", 0, 1, (5,)),
            Route(6, 3, "", 0, 1, (6,)),
        ),
        blocks=tuple(
            Block(block_id, 1 if block_id <= 4 else 5, 1, 0, False)
            for block_id in range(1, 7)
        ),
    )


class JointComponentNeighborhoodTest(unittest.TestCase):
    def test_enumerates_all_joint_route_combinations_for_three_multi_route_trains(self) -> None:
        from station_dispatch.qea_lns import _component_route_choices

        instance = qea_joint_component_instance()
        route_choices = list(
            _component_route_choices(
                instance,
                {1: 1, 2: 3, 3: 5},
                {1, 2, 3},
                time.monotonic() + 1,
            )
        )

        self.assertEqual(len(route_choices), 8)
        self.assertIn({1: 2, 2: 4, 3: 6}, route_choices)
        self.assertIn({1: 1, 2: 3, 3: 5}, route_choices)

    def test_enumerates_cross_entry_interleavings_without_reversing_fifo(self) -> None:
        from station_dispatch.qea_lns import _component_priority_orders

        instance = qea_joint_component_instance()
        orders = list(_component_priority_orders(instance, {1, 2, 3}))

        self.assertEqual(set(orders), {(1, 2, 3), (1, 3, 2), (3, 1, 2)})
        for order in orders:
            self.assertLess(order.index(1), order.index(2))



def qea_delayed_alternative_blocker_instance() -> StationInstance:
    return StationInstance(
        path=Path("qea-delayed-alternative-blocker-synthetic.dzn"),
        edge_count=4,
        edge_names=("shared_entry", "current_section", "alternative_section", "horizon"),
        edge_types=("inter", "inter", "inter", "inter"),
        platform_names=(),
        trains=(
            Train(1, "fifo_predecessor", 0, "pass", (1,)),
            Train(2, "target", 0, "pass", (2, 3)),
            Train(3, "later_alternative_blocker", 0, "origin", (4,)),
            Train(4, "horizon", 40, "pass", (5,)),
        ),
        routes=(
            Route(1, 1, "", 0, 1, (1,)),
            Route(2, 2, "", 0, 11, (2, 3)),
            Route(3, 2, "", 0, 11, (4, 5)),
            Route(4, 3, "", 0, 10, (6,)),
            Route(5, 4, "", 0, 1, (7,)),
        ),
        blocks=(
            Block(1, 1, 1, 0, False),
            Block(2, 1, 1, 0, False),
            Block(3, 2, 10, 0, False),
            Block(4, 1, 1, 0, False),
            Block(5, 3, 10, 0, False),
            Block(6, 3, 10, 0, False),
            Block(7, 4, 1, 0, False),
        ),
    )


class DelayedAlternativeRouteBlockerTest(unittest.TestCase):
    def test_includes_blocker_encountered_after_fifo_delays_an_alternative_route(self) -> None:
        import station_dispatch.qea_lns as qea_lns

        instance = qea_delayed_alternative_blocker_instance()
        current = Schedule((
            Assignment(train_id=1, route_id=1, start=10, dwell=0),
            Assignment(train_id=2, route_id=2, start=12, dwell=0),
            Assignment(train_id=3, route_id=4, start=13, dwell=0),
            Assignment(train_id=4, route_id=5, start=40, dwell=0),
        ))

        self.assertTrue(validate_schedule(instance, current).feasible)
        target_index = [train.id for train in qea_lns._waiting_trains(instance, current)].index(2)
        component, decision_ids = qea_lns._alternative_route_component(instance, current, target_index=target_index)

        self.assertEqual(component, {1, 2, 3})
        self.assertEqual(decision_ids, {2})


def qea_origin_arrival_interleaving_instance() -> StationInstance:
    return StationInstance(
        path=Path("qea-origin-arrival-interleaving-synthetic.dzn"),
        edge_count=1,
        edge_names=("shared_throat",),
        edge_types=("inter",),
        platform_names=(),
        trains=(
            Train(1, "short_arrival", 0, "pass", (1,)),
            Train(2, "long_origin", 0, "origin", (2,)),
        ),
        routes=(
            Route(1, 1, "", 0, 1, (1,)),
            Route(2, 2, "", 0, 10, (2,)),
        ),
        blocks=(
            Block(1, 1, 1, 0, False),
            Block(2, 1, 10, 0, False),
        ),
    )


class JointComponentOriginArrivalInterleavingTest(unittest.TestCase):
    def test_search_can_schedule_an_arrival_before_an_origin_when_wait_is_lower(self) -> None:
        import station_dispatch.qea_lns as qea_lns

        instance = qea_origin_arrival_interleaving_instance()
        current = Schedule((
            Assignment(train_id=2, route_id=2, start=0, dwell=0),
            Assignment(train_id=1, route_id=1, start=10, dwell=0),
        ))

        self.assertTrue(validate_schedule(instance, current).feasible)
        repair = qea_lns._search_joint_component(
            instance,
            neighborhood={1, 2},
            decision_ids=set(),
            base_route_choices={1: 1, 2: 2},
            fixed_assignments=(),
            current_score=qea_lns._score(instance, current),
            deadline=time.monotonic() + 1,
        )

        self.assertIsNotNone(repair)
        assignments = {assignment.train_id: assignment for assignment in repair.schedule.assignments}
        self.assertEqual(assignments[1].start, 0)
        self.assertEqual(assignments[2].start, 1)
        self.assertEqual(qea_lns._score(instance, repair.schedule), (1, 11))


class LnsAdvanceBeforeLearningTest(unittest.TestCase):
    def test_adopts_safe_advance_before_encoding_without_an_extra_empty_replay(self) -> None:
        import station_dispatch.qea_lns as qea_lns

        instance = qea_instance()
        states = qea_lns._initial_population(instance)[:2]
        candidates = [
            qea_lns._Candidate(
                route_choices={1: 1, 2: 3},
                priority_bits={1: (bit,) * qea_lns._PRIORITY_BITS, 2: (bit,) * qea_lns._PRIORITY_BITS},
                priority_order=(1, 2) if bit == 0 else (2, 1),
            )
            for bit in (0, 1)
        ]
        initial = Schedule((
            Assignment(train_id=1, route_id=1, start=0, dwell=0),
            Assignment(train_id=2, route_id=3, start=10, dwell=0),
        ))
        repaired = Schedule((
            Assignment(train_id=1, route_id=2, start=0, dwell=0),
            Assignment(train_id=2, route_id=3, start=5, dwell=0),
        ))
        replayed = Schedule((
            Assignment(train_id=1, route_id=2, start=0, dwell=0),
            Assignment(train_id=2, route_id=3, start=0, dwell=0),
        ))
        repair = qea_lns._LnsRepair(
            schedule=repaired,
            route_choices={1: 2, 2: 3},
            priority_order=(2, 1),
        )
        clock = 0.0
        update = qea_lns._update_member_toward
        def finish_generation(*args):
            nonlocal clock
            update(*args)
            clock = 11.0

        with patch.object(qea_lns, '_initial_population', return_value=states), patch.object(
            qea_lns, '_sample_candidate', side_effect=candidates,
        ), patch.object(
            qea_lns, '_construct_schedule', side_effect=[initial, initial],
        ) as construct, patch.object(
            qea_lns, '_reschedule_wait_chain', side_effect=[repair, None, None],
        ), patch.object(
            qea_lns, '_candidate_from_schedule', return_value=None,
        ), patch.object(qea_lns, '_update_member_toward', side_effect=finish_generation), patch.object(qea_lns.time, 'monotonic', side_effect=lambda: clock):
            result = qea_lns.solve_qea_lns(instance, seed=42, time_limit_s=10)

        self.assertEqual(result.schedule, replayed)
        self.assertEqual(construct.call_count, 2)
        self.assertEqual(result.improvement_history[-1][1:], (0, 20))


class LearningScoreConsistencyTest(unittest.TestCase):
    def test_sample_worse_than_contextual_lns_does_not_replace_learning_target(self) -> None:
        import station_dispatch.qea_lns as q

        instance = qea_instance()
        initial = Schedule((Assignment(1, 1, 0, 0), Assignment(2, 3, 10, 0)))
        repaired = Schedule((Assignment(1, 2, 0, 0), Assignment(2, 3, 5, 0)))
        sampled = Schedule((Assignment(1, 2, 3, 0), Assignment(2, 3, 4, 0)))
        for schedule in (initial, repaired, sampled):
            self.assertTrue(validate_schedule(instance, schedule).feasible)
        self.assertLess(q._score(instance, repaired), q._score(instance, sampled))
        self.assertLess(q._score(instance, sampled), q._score(instance, initial))
        candidate = q._Candidate({1: 1, 2: 3}, {1: (0,) * 16, 2: (1,) * 16}, (1, 2))
        repair = q._LnsRepair(repaired, {1: 2, 2: 3}, (1, 2))
        clock = 0.0
        targets = []

        def update(instance, state, observed, target, rotation_scale):
            nonlocal clock
            targets.append(target)
            if len(targets) == 2:
                clock = 11.0

        with patch.object(q, '_initial_population', return_value=q._initial_population(instance)[:1]), patch.object(
            q, '_sample_candidate', return_value=candidate,
        ), patch.object(
            q, '_construct_schedule', side_effect=[initial, sampled],
        ), patch.object(
            q, '_reschedule_wait_chain', side_effect=[repair, None],
        ), patch.object(
            q, '_advance_schedule', side_effect=lambda instance, schedule, deadline: schedule,
        ), patch.object(
            q, '_candidate_from_schedule', return_value=None,
        ), patch.object(
            q, '_update_member_toward', side_effect=update,
        ), patch.object(
            q, '_rotation_scale', wraps=q._rotation_scale,
        ) as rotation, patch.object(q.time, 'monotonic', side_effect=lambda: clock):
            result = q.solve_qea_lns(instance, seed=42, time_limit_s=10)

        self.assertEqual(result.schedule, repaired)
        self.assertEqual(targets[-1].route_choices, {1: 2, 2: 3})
        self.assertEqual(rotation.call_args.args[1], q._score(instance, repaired))

    def test_contextual_lns_target_keeps_the_score_of_its_actual_schedule(self) -> None:
        import station_dispatch.qea_lns as qea_lns

        instance = qea_instance()
        candidate = qea_lns._Candidate(
            route_choices={1: 1, 2: 3},
            priority_bits={1: (0,) * qea_lns._PRIORITY_BITS, 2: (1,) * qea_lns._PRIORITY_BITS},
            priority_order=(1, 2),
        )
        initial = Schedule((Assignment(1, 1, 0, 0), Assignment(2, 3, 10, 0)))
        repaired = Schedule((Assignment(1, 2, 0, 0), Assignment(2, 3, 5, 0)))
        repair = qea_lns._LnsRepair(repaired, {1: 2, 2: 3}, (1, 2))
        def monotonic() -> float:
            return 11.0 if rotation.call_count >= 2 else 0.0

        with patch.object(qea_lns, '_advance_schedule', side_effect=lambda instance, schedule, deadline: schedule), patch.object(qea_lns, '_initial_population', return_value=qea_lns._initial_population(instance)[:2]), patch.object(
            qea_lns, '_sample_candidate', return_value=candidate,
        ), patch.object(
            qea_lns, '_construct_schedule', side_effect=[initial, initial, None],
        ), patch.object(
            qea_lns, '_reschedule_wait_chain', side_effect=[repair, None, None],
        ), patch.object(
            qea_lns, '_candidate_from_schedule', return_value=None,
        ), patch.object(
            qea_lns, '_rotation_scale', wraps=qea_lns._rotation_scale,
        ) as rotation, patch.object(qea_lns.time, 'monotonic', side_effect=monotonic):
            result = qea_lns.solve_qea_lns(instance, seed=42, time_limit_s=10)

        self.assertEqual(result.schedule, repaired)
        self.assertTrue(rotation.called)
        for call in rotation.call_args_list:
            self.assertEqual(call.args[1], qea_lns._score(instance, repaired))


class LnsResidualWaitTest(unittest.TestCase):
    def test_keeps_necessary_platform_dwell_while_advancing_another_train(self) -> None:
        import station_dispatch.qea_lns as q

        instance = qea_dwell_reordering_instance()
        current = Schedule((Assignment(1, 1, 0, 2), Assignment(2, 2, 0, 0), Assignment(3, 3, 8, 0)))
        self.assertTrue(validate_schedule(instance, current).feasible)
        advanced = q._advance_schedule(instance, current, math.inf)
        actual = {assignment.train_id: assignment for assignment in advanced.assignments}
        self.assertEqual(actual[1], current.assignments[0])
        self.assertEqual(actual[3].start, 4)
        self.assertTrue(validate_schedule(instance, advanced).feasible)

    def test_does_not_advance_past_same_entry_predecessor(self) -> None:
        import station_dispatch.qea_lns as q

        instance = qea_dwell_reordering_instance()
        current = Schedule((Assignment(1, 1, 2, 0), Assignment(2, 2, 0, 0), Assignment(3, 3, 8, 0)))
        self.assertTrue(validate_schedule(instance, current).feasible)
        advanced = q._advance_schedule(instance, current, math.inf)
        actual = {assignment.train_id: assignment for assignment in advanced.assignments}
                                                                                      
        self.assertEqual(actual[1].start, 2)
        self.assertEqual(actual[3].start, 6)
        self.assertTrue(validate_schedule(instance, advanced).feasible)

    def test_advances_safe_wait_without_changing_routes_or_dwell(self) -> None:
        import station_dispatch.qea_lns as q

        instance = qea_instance()
        current = Schedule((Assignment(1, 2, 0, 0), Assignment(2, 3, 10, 0)))
        self.assertTrue(validate_schedule(instance, current).feasible)
        advanced = q._advance_schedule(instance, current, math.inf)
        self.assertEqual(advanced, Schedule((Assignment(1, 2, 0, 0), Assignment(2, 3, 0, 0))))
        self.assertTrue(validate_schedule(instance, advanced).feasible)
        self.assertLess(q._score(instance, advanced), q._score(instance, current))
        candidate = q._candidate_from_schedule(instance, advanced)
        self.assertIsNotNone(candidate)
        replay = q._construct_schedule(instance, candidate.priority(), candidate.route_choices, math.inf,
                                       extra_dwell_by_train=candidate.extra_dwell_by_train)
        self.assertEqual(set(replay.assignments), set(advanced.assignments))

    def test_does_not_remove_wait_required_by_another_train(self) -> None:
        import station_dispatch.qea_lns as q

        instance = qea_instance()
        current = Schedule((Assignment(1, 1, 0, 0), Assignment(2, 3, 10, 0)))
        self.assertEqual(q._advance_schedule(instance, current, math.inf), current)

    def test_expired_budget_keeps_the_valid_schedule(self) -> None:
        import station_dispatch.qea_lns as q

        instance = qea_instance()
        current = Schedule((Assignment(1, 2, 0, 0), Assignment(2, 3, 10, 0)))
        self.assertEqual(q._advance_schedule(instance, current, -math.inf), current)


class JointEquivalentStateTest(unittest.TestCase):
    def test_equivalent_construction_orders_do_not_expand_the_same_state_twice(self) -> None:
        import station_dispatch.qea_lns as q
        from collections import Counter

        instance = StationInstance(
            path=Path("independent-origin-order.dzn"), edge_count=3,
            edge_names=("exit_a", "exit_b", "exit_c"), edge_types=("inter",) * 3,
            platform_names=(),
            trains=tuple(Train(i, str(i), 0, "origin", (i,)) for i in (1, 2, 3)),
            routes=tuple(Route(i, i, "", 0, 1, (i,)) for i in (1, 2, 3)),
            blocks=tuple(Block(i, i, 1, 0, False) for i in (1, 2, 3)),
        )
        calls = Counter()
        original = q._assign_train

        def assign(*args, **kwargs):
            calls[(frozenset(args[1]), args[2].id, args[3], kwargs.get("earliest_start_floor"))] += 1
            return original(*args, **kwargs)

        progress = {}
                                                                                   
        with patch.object(q, "_assign_train", side_effect=assign), patch.object(q, "_score", return_value=(100, 100)):
            q._search_joint_component(instance, {1, 2, 3}, set(), {1: 1, 2: 2, 3: 3}, (),
                                      (100, 100), math.inf, progress=progress)
        self.assertTrue(progress["complete"])
        self.assertTrue(calls)
        self.assertEqual(max(calls.values()), 1)


class AlternativeBlockerOccupancyReuseTest(unittest.TestCase):
    def test_reuses_unchanged_occupations_and_refreshes_changed_assignments(self) -> None:
        import station_dispatch.qea_lns as q
        import station_dispatch.dispatch as dispatch

        instance = qea_alternative_route_component_instance()
        current = Schedule((Assignment(1, 1, 10, 0), Assignment(2, 3, 0, 0), Assignment(3, 4, 0, 0)))
        target = instance.trains[0]
        self.assertEqual(q._alternative_route_blockers(instance, current, target, 2), {3})
        with patch.object(q, 'resource_occupancies', wraps=q.resource_occupancies) as direct, patch.object(
            dispatch, 'resource_occupancies', wraps=dispatch.resource_occupancies,
        ) as cached:
            self.assertEqual(q._alternative_route_blockers(instance, current, target, 2), {3})
        self.assertEqual(direct.call_count + cached.call_count, 0)
                                                                                       
        changed = Schedule((Assignment(1, 1, 10, 0), Assignment(2, 3, 0, 0), Assignment(3, 5, 0, 0)))
        self.assertTrue(validate_schedule(instance, changed).feasible)
        self.assertEqual(q._alternative_route_blockers(instance, changed, target, 2), set())


class AbandonedJointSearchTest(unittest.TestCase):
    def test_closes_joint_search_when_replaced_or_solver_finishes(self) -> None:
        import station_dispatch.qea_lns as q

        for improves in (False, True):
            with self.subTest(improves=improves):
                instance = qea_instance()
                initial = Schedule((Assignment(1, 1, 0, 0), Assignment(2, 3, 10, 0)))
                improved = Schedule((Assignment(1, 2, 0, 0), Assignment(2, 3, 5, 0)))
                candidate = q._Candidate({1: 1, 2: 3}, {1: (0,) * 16, 2: (1,) * 16}, (1, 2))
                clock = 0.0
                iterators = []

                def repair(instance, schedule, rng, deadline, target_index, progress, max_steps=None):
                    nonlocal clock
                    joint = progress.setdefault('joint', {})
                    q._search_joint_component(instance, {1, 2}, {1}, {1: 1, 2: 3}, (),
                                              (100, 100), -math.inf, progress=joint)
                    iterators.append(joint['iterator'])
                    clock = 11.0
                    return q._LnsRepair(improved, {1: 2, 2: 3}, (1, 2)) if improves else None

                with patch.object(q, '_initial_population', return_value=q._initial_population(instance)[:1]), patch.object(
                    q, '_sample_candidate', return_value=candidate,
                ), patch.object(q, '_construct_schedule', side_effect=[initial, None]), patch.object(
                    q, '_reschedule_wait_chain', side_effect=repair,
                ), patch.object(q.time, 'monotonic', side_effect=lambda: clock):
                    result = q.solve_qea_lns(instance, seed=42, time_limit_s=10)

                self.assertEqual(result.schedule, improved if improves else initial)
                self.assertEqual(len(iterators), 1)
                self.assertIsNone(iterators[0].gi_frame)


    def test_releases_iterator_and_recursive_search_without_waiting_for_gc(self) -> None:
        import gc
        import weakref
        import station_dispatch.qea_lns as q

        for started in (False, True):
            with self.subTest(started=started):
                searches = {0: {'joint': {}}}
                joint = searches[0]['joint']
                q._search_joint_component(qea_instance(), {1, 2}, {1}, {1: 1, 2: 3}, (),
                                          (100, 100), -math.inf, progress=joint)
                if started:
                    joint['deadline'] = math.inf
                    next(joint['iterator'])
                iterator_ref = weakref.ref(joint['iterator'])
                search_ref = weakref.ref(joint['iterator'].gi_frame.f_locals['search']) if started else None
                gc_was_enabled = gc.isenabled()
                gc.disable()
                try:
                    q._clear_target_searches(searches)
                    self.assertIsNone(iterator_ref())
                    if search_ref is not None:
                        self.assertIsNone(search_ref())
                finally:
                    if gc_was_enabled:
                        gc.enable()


class JointRouteDwellTest(unittest.TestCase):
    def test_joint_search_changes_route_and_holds_at_platform_in_one_branch(self):
        from dataclasses import replace
        import station_dispatch.qea_lns as qea
        instance = qea_dwell_reordering_instance()
                                                                            
                                                                           
                                                                             
        instance = replace(
            instance,
            trains=(replace(instance.trains[0], route_ids=(1, 4)), *instance.trains[1:]),
            routes=(replace(instance.routes[0], duration_min=9, block_ids=(8, 2, 3)),
                    *instance.routes[1:], Route(4, 1, "S_I", 0, 6, (1, 2, 3))),
            blocks=(*instance.blocks, Block(8, 1, 7, 0, False)),
        )
        current = Schedule((Assignment(2, 2, 0, 0), Assignment(1, 1, 0, 0), Assignment(3, 3, 7, 0)))
        self.assertTrue(validate_schedule(instance, current).feasible)
        self.assertEqual(qea._score(instance, current)[0], 7)
        repair = qea._search_joint_component(instance, {1, 2, 3}, {1}, {1: 1, 2: 2, 3: 3}, (), qea._score(instance, current), time.monotonic() + 2)
        self.assertIsNotNone(repair)
        self.assertTrue(validate_schedule(instance, repair.schedule).feasible)
        self.assertEqual(qea._score(instance, repair.schedule), (6, 8))
        assignments = {a.train_id: a for a in repair.schedule.assignments}
        self.assertEqual(assignments[1], Assignment(1, 4, 0, 2))
        self.assertEqual(assignments[3].start, 4)
        learned = qea._candidate_from_schedule(instance, repair.schedule)
        self.assertIsNotNone(learned)
        replay = qea._construct_schedule(instance, learned.priority(), learned.route_choices, time.monotonic() + 2, extra_dwell_by_train=learned.extra_dwell_by_train)
        self.assertEqual(set(replay.assignments), set(repair.schedule.assignments))

class PriorityLearningRepresentationTest(unittest.TestCase):
    def test_same_order_with_different_numeric_keys_does_not_rotate(self):
        import station_dispatch.qea_lns as q
        instance = qea_instance()
        state = q._initial_population(instance)[0]
        observed = q._Candidate({1: 1, 2: 3}, {1: q._int_to_bits(12000, 16), 2: q._int_to_bits(50000, 16)}, (1, 2))
        recovered = q._Candidate(dict(observed.route_choices), {1: q._int_to_bits(0, 16), 2: q._int_to_bits(1, 16)}, (1, 2))
        before = {k: list(v) for k, v in state.priority_angles.items()}
        q._update_member_toward(instance, state, observed, recovered)
        self.assertEqual(state.priority_angles, before)

    def test_changed_order_learns_swapped_keys_not_tiny_rank_values(self):
        import station_dispatch.qea_lns as q
        instance = qea_instance()
        state = q._initial_population(instance)[0]
        observed = q._Candidate({1: 1, 2: 3}, {1: q._int_to_bits(12000, 16), 2: q._int_to_bits(50000, 16)}, (1, 2))
        recovered = q._Candidate(dict(observed.route_choices), {2: q._int_to_bits(0, 16), 1: q._int_to_bits(1, 16)}, (2, 1))
        before = {k: list(v) for k, v in state.priority_angles.items()}
        q._update_member_toward(instance, state, observed, recovered)
        for train_id, other_id in ((1, 2), (2, 1)):
            for bit, (old, target) in enumerate(zip(observed.priority_bits[train_id], observed.priority_bits[other_id])):
                expected = q._rotate(before[train_id][bit], bool(target)) if old != target else before[train_id][bit]
                self.assertEqual(state.priority_angles[train_id][bit], expected)


class JointFifoLowerBoundTest(unittest.TestCase):
    def test_prunes_unavoidable_remaining_entry_wait_before_assignment(self):
        import station_dispatch.qea_lns as q
        instance = qea_instance()
                                                                           
                                                                            
        with patch.object(q, "_assign_train", side_effect=AssertionError("dominated branch expanded")):
            progress = {}
            repair = q._search_joint_component(instance, {2}, set(), {2: 3},
                                               (Assignment(1, 1, 20, 0),),
                                               (39, 100), math.inf, progress=progress)
        self.assertIsNone(repair)
        self.assertTrue(progress["complete"])

    def test_equal_wait_bound_still_allows_makespan_improvement_search(self):
        import station_dispatch.qea_lns as q
        instance = qea_instance()
        with patch.object(q, "_assign_train", wraps=q._assign_train) as assign:
            q._search_joint_component(instance, {2}, set(), {2: 3},
                                      (Assignment(1, 1, 20, 0),),
                                      (40, 100), math.inf)
        self.assertGreater(assign.call_count, 0)


class ContextualRouteRepairTest(unittest.TestCase):
    def test_advance_switches_from_busy_encoded_route_to_safe_alternative(self):
        import station_dispatch.qea_lns as q
        instance = qea_forced_route_instance()
        current = Schedule((Assignment(2, 3, 0, 0), Assignment(1, 2, 5, 0)))
        self.assertTrue(validate_schedule(instance, current).feasible)
        advanced = q._advance_schedule(instance, current, math.inf)
        self.assertEqual({a.train_id: a for a in advanced.assignments}[1], Assignment(1, 1, 0, 0))
        self.assertTrue(validate_schedule(instance, advanced).feasible)
        self.assertLess(q._score(instance, advanced), q._score(instance, current))

    def test_context_keeps_other_trains_occupations_when_rebuilding(self):
        import station_dispatch.qea_lns as q
        instance = qea_forced_route_instance()
        fixed = (Assignment(2, 3, 0, 0),)
        result = q._construct_schedule(instance, {1: 0, 2: 1}, {1: 2, 2: 3}, math.inf,
                                       fixed_assignments=fixed, train_ids={1})
        self.assertEqual({a.train_id: a for a in result.assignments}[1], Assignment(1, 1, 0, 0))
        self.assertIn(fixed[0], result.assignments)
        self.assertTrue(validate_schedule(instance, result).feasible)

    def test_solver_retains_lns_dwell_and_occupations_in_next_generation(self):
        import station_dispatch.qea_lns as q
        instance = qea_dwell_reordering_instance()
        improved = Schedule((Assignment(1, 1, 0, 2), Assignment(2, 2, 0, 0), Assignment(3, 3, 4, 0)))
        self.assertTrue(validate_schedule(instance, improved).feasible)
        repair = q._LnsRepair(improved, {1: 1, 2: 2, 3: 3}, (2, 1, 3), {1: 2, 2: 0, 3: 0})
        original = q._construct_schedule
        contexts = []
        observed_fixed_dwell = []
        def construct(*args, **kwargs):
            if kwargs.get('fixed_assignments'):
                contexts.append(kwargs['fixed_assignments'])
                observed_fixed_dwell.append(kwargs['extra_dwell_by_train'][1])
            return original(*args, **kwargs)
        with patch.object(q, '_construct_schedule', side_effect=construct), patch.object(
            q, '_alternative_route_component', return_value=({3}, set())
        ) as component, patch.object(q, '_reschedule_wait_chain', return_value=repair), patch.object(
            q, '_candidate_from_schedule', return_value=None
        ):
            result = q.solve_qea_lns(instance, 42, 0.1)
        self.assertEqual(set(result.schedule.assignments), set(improved.assignments))
        self.assertTrue(contexts, 'later QEA generations discarded LNS occupations')
        self.assertTrue(all(set(fixed) == set(improved.assignments[:2]) for fixed in contexts))
        self.assertTrue(all(dwell == 2 for dwell in observed_fixed_dwell))
        self.assertEqual(component.call_count, len(q._waiting_trains(instance, improved)))

    def test_advance_does_not_treat_another_trains_reserved_route_as_free(self):
        from dataclasses import replace
        import station_dispatch.qea_lns as q
        instance = qea_forced_route_instance()
        instance = replace(instance,
            trains=(replace(instance.trains[0], earliest_start=1), instance.trains[1], Train(3, 'T3', 0, 'pass', (4,))),
            routes=(*instance.routes, Route(4, 3, '', 0, 10, (4,))),
            blocks=(*instance.blocks, Block(4, 1, 10, 0, False)))
        current = Schedule((Assignment(2, 3, 0, 0), Assignment(3, 4, 0, 0), Assignment(1, 2, 5, 0)))
        self.assertTrue(validate_schedule(instance, current).feasible)
        advanced = q._advance_schedule(instance, current, math.inf)
        self.assertEqual({a.train_id: a for a in advanced.assignments}[1], Assignment(1, 2, 5, 0))
        self.assertTrue(validate_schedule(instance, advanced).feasible)


class GlobalExplorationAndSuccessorBoundTest(unittest.TestCase):
    def test_later_generations_keep_global_and_contextual_constructions(self):
        import station_dispatch.qea_lns as q
        instance = qea_dwell_reordering_instance()
        schedule = Schedule((Assignment(1, 1, 0, 2), Assignment(2, 2, 0, 0), Assignment(3, 3, 4, 0)))
        clock = [0.0]
        constructions = []
        def construct(*args, **kwargs):
            constructions.append(bool(kwargs.get("fixed_assignments")))
            if len(constructions) == 24:
                clock[0] = 2.0
            return schedule
        with patch.object(q.time, "monotonic", side_effect=lambda: clock[0]), patch.object(
            q, "_construct_schedule", side_effect=construct
        ), patch.object(q, "_alternative_route_component", return_value=({3}, set())), patch.object(
            q, "_reschedule_wait_chain", return_value=None
        ), patch.object(q, "_advance_schedule", return_value=schedule):
            result = q.solve_qea_lns(instance, 42, 1.0)
        self.assertEqual(constructions[:8], [False] * 8)
        for generation in (constructions[8:16], constructions[16:24]):
            self.assertEqual(generation.count(False), 1, "complete exploration stopped after first generation")
            self.assertEqual(generation.count(True), 7)
        self.assertNotEqual(constructions[8:16].index(False), constructions[16:24].index(False))
        self.assertTrue(validate_schedule(instance, result.schedule).feasible)

    def test_fixed_successor_conflict_stops_before_remaining_trains_and_validation(self):
        import station_dispatch.qea_lns as q
        instance = qea_dwell_reordering_instance()
        with patch.object(q, "_assign_train", wraps=q._assign_train) as assign, patch.object(
            q, "validate_schedule", wraps=q.validate_schedule
        ) as validate:
            result = q._construct_schedule(instance, {1: 0, 2: 1, 3: 2}, {1: 1, 2: 2, 3: 3}, math.inf,
                                           fixed_assignments=(Assignment(3, 3, 0, 0),), train_ids={1, 2})
        self.assertIsNone(result)
        self.assertEqual([call.args[2].id for call in assign.call_args_list], [1])
        validate.assert_not_called()

    def test_all_routes_are_checked_before_rejecting_successor_bound_and_equal_is_allowed(self):
        import station_dispatch.qea_lns as q
        instance = qea_instance()
        result = q._construct_schedule(instance, {1: 0, 2: 1}, {1: 1, 2: 3}, math.inf,
                                       fixed_assignments=(Assignment(2, 3, 0, 0),), train_ids={1})
        self.assertIsNotNone(result)
        self.assertIn(Assignment(1, 2, 0, 0), result.assignments)
        self.assertTrue(validate_schedule(instance, result).feasible)


    def test_joint_search_prunes_fixed_successor_violation_before_validation(self):
        import station_dispatch.qea_lns as q
        instance = qea_dwell_reordering_instance()
        progress = {}
        with patch.object(q, "validate_schedule", wraps=q.validate_schedule) as validate:
            repair = q._search_joint_component(
                instance, {1, 2}, set(), {1: 1, 2: 2},
                (Assignment(3, 3, 0, 0),), (1000, 1000), math.inf, progress)
        self.assertIsNone(repair)
        self.assertTrue(progress["complete"])
        validate.assert_not_called()

    def test_joint_successor_pruning_keeps_other_route_and_equal_start(self):
        import station_dispatch.qea_lns as q
        instance = qea_instance()
        repair = q._search_joint_component(
            instance, {1}, {1}, {1: 1}, (Assignment(2, 3, 0, 0),),
            (1000, 1000), math.inf)
        self.assertIsNotNone(repair)
        self.assertIn(Assignment(1, 2, 0, 0), repair.schedule.assignments)
        self.assertTrue(validate_schedule(instance, repair.schedule).feasible)

    def test_successor_bound_retries_after_pending_origin_releases_platform(self):
        import station_dispatch.qea_lns as q
        instance = StationInstance(
            path=Path("successor-origin-release.dzn"), edge_count=4,
            edge_names=("entry", "platform", "exit", "bypass"),
            edge_types=("inter", "platform", "inter", "inter"), platform_names=("S_I",),
            trains=(Train(1, "arrival", 0, "pass", (1, 2)), Train(2, "fixed_successor", 0, "pass", (3,)),
                    Train(3, "pending_origin", 0, "origin", (4,)), Train(4, "bypass_blocker", 0, "pass", (5,))),
            routes=(Route(1, 1, "S_I", 0, 3, (1, 2, 3)), Route(2, 1, "", 0, 2, (1, 4)),
                    Route(3, 2, "", 0, 1, (5,)), Route(4, 3, "S_I", 0, 2, (6,)),
                    Route(5, 4, "", 0, 10, (7,))),
            blocks=(Block(1, 1, 1, 0, False), Block(2, 2, 1, 0, True), Block(3, 3, 1, 0, False),
                    Block(4, 4, 1, 0, False), Block(5, 1, 1, 0, False), Block(6, 2, 2, 0, True),
                    Block(7, 4, 10, 0, False)))
        with patch.object(q, "_assign_train", wraps=q._assign_train) as assign:
            result = q._construct_schedule(instance, {1: 0, 2: 2, 3: 1, 4: 3}, {1: 1, 2: 3, 3: 4, 4: 5}, math.inf,
                fixed_assignments=(Assignment(2, 3, 5, 0), Assignment(4, 5, 0, 0)), train_ids={1, 3})
        self.assertIsNotNone(result)
        self.assertEqual([call.args[2].id for call in assign.call_args_list], [1, 3, 1])
        self.assertIn(Assignment(1, 1, 1, 0), result.assignments)
        self.assertTrue(validate_schedule(instance, result).feasible)


class LocalMeasuredObservationTest(unittest.TestCase):
    def test_solver_observes_raw_active_bits_not_context_remapped_bits(self):
        import station_dispatch.qea_lns as q
        instance = qea_dwell_reordering_instance()
        schedule = Schedule((Assignment(1, 1, 0, 2), Assignment(2, 2, 0, 0), Assignment(3, 3, 4, 0)))
        clock = [0.0]
        current = [None]
        measured, active = {}, {}
        observed_pairs = []
        calls = [0]
        sample, update = q._sample_candidate, q._update_member_toward
        def capture_sample(inst, state, rng):
            raw = sample(inst, state, rng)
            current[0] = id(state)
            measured[id(state)] = raw
            return raw
        def construct(*args, **kwargs):
            active[current[0]] = kwargs.get("train_ids")
            calls[0] += 1
            return schedule
        def capture_update(inst, state, observation, target, *args):
            if active[id(state)]:
                observed_pairs.append((measured[id(state)], observation, active[id(state)]))
            if len(observed_pairs) == 7:
                clock[0] = 2.0
            return update(inst, state, observation, target, *args)
        with patch.object(q.time, "monotonic", side_effect=lambda: clock[0]), patch.object(
            q, "_sample_candidate", side_effect=capture_sample
        ), patch.object(q, "_construct_schedule", side_effect=construct), patch.object(
            q, "_alternative_route_component", return_value=({1, 3}, set())
        ), patch.object(q, "_reschedule_wait_chain", return_value=None), patch.object(
            q, "_update_member_toward", side_effect=capture_update
        ):
            q.solve_qea_lns(instance, 42, 1.0)
        self.assertEqual(len(observed_pairs), 7)
        for raw, observation, neighborhood in observed_pairs:
            self.assertEqual(set(observation.route_choices), neighborhood)
            self.assertEqual(observation.priority_order, tuple(t for t in raw.priority_order if t in neighborhood))
            for train_id in neighborhood:
                self.assertEqual(observation.priority_bits[train_id], raw.priority_bits[train_id])
                self.assertEqual(observation.route_choices[train_id], raw.route_choices[train_id])
                self.assertEqual(q._candidate_dwell_bits(observation, train_id), q._candidate_dwell_bits(raw, train_id))

    def test_local_order_learning_uses_raw_scale_and_leaves_fixed_train_unchanged(self):
        import copy
        import station_dispatch.qea_lns as q
        instance = qea_dwell_reordering_instance()
        state = q._initial_population(instance)[0]
        before = copy.deepcopy(state)
        observation = q._Candidate({1: 1, 3: 3}, {1: q._int_to_bits(900, 16), 3: q._int_to_bits(100, 16)}, (3, 1))
        target = q._Candidate({1: 1, 2: 2, 3: 3}, {t: q._int_to_bits(t, 16) for t in (1, 2, 3)}, (1, 2, 3), {2: 8})
        q._update_member_toward(instance, state, observation, target)
        for train_id, desired in ((1, 100), (3, 900)):
            for index, bit in enumerate(q._int_to_bits(desired, 16)):
                old = before.priority_angles[train_id][index]
                expected = old if observation.priority_bits[train_id][index] == bit else q._rotate(old, bool(bit))
                self.assertEqual(state.priority_angles[train_id][index], expected)
        self.assertEqual(state.route_angles[2], before.route_angles[2])
        self.assertEqual(state.priority_angles[2], before.priority_angles[2])
        self.assertEqual(state.dwell_angles[2], before.dwell_angles[2])

    def test_single_active_train_learns_route_and_dwell_but_not_order(self):
        import copy
        import station_dispatch.qea_lns as q
        instance = qea_instance()
        state = q._initial_population(instance)[0]
        before = copy.deepcopy(state)
        observation = q._Candidate({1: 1}, {1: q._int_to_bits(900, 16)}, (1,), {1: 0})
        target = q._Candidate({1: 2, 2: 3}, {1: q._int_to_bits(1, 16), 2: q._int_to_bits(0, 16)}, (2, 1), {1: 2, 2: 8})
        q._update_member_toward(instance, state, observation, target)
        self.assertLess(state.route_angles[1][0], before.route_angles[1][0])
        self.assertGreater(state.route_angles[1][1], before.route_angles[1][1])
        self.assertGreater(state.dwell_angles[1][1], before.dwell_angles[1][1])
        self.assertEqual(state.priority_angles, before.priority_angles)
        self.assertEqual(state.route_angles[2], before.route_angles[2])
        self.assertEqual(state.dwell_angles[2], before.dwell_angles[2])






class FixedWorkContinuationTest(unittest.TestCase):
    def test_real_joint_search_quota_one_matches_uninterrupted_search(self):
        import station_dispatch.qea_lns as q
        instance = qea_instance()
        args = (instance, {1, 2}, {1, 2}, {1: 1, 2: 3}, (), (1000, 1000), math.inf)
        expected = q._search_joint_component(*args)
        progress = {}
        for _ in range(1000):
            actual = q._search_joint_component(*args, progress=progress, max_steps=1)
            if progress["complete"]:
                break
        self.assertTrue(progress["complete"])
        self.assertIsNotNone(expected)
        self.assertEqual(actual, expected)
        self.assertTrue(validate_schedule(instance, actual.schedule).feasible)


class DeadlineSemanticsTest(unittest.TestCase):
    def test_encoding_timeout_is_not_a_failed_replay(self):
        from station_dispatch import qea_lns as q
        schedule = Schedule((Assignment(1, 1, 0, 0), Assignment(2, 3, 10, 0)))
        with patch.object(q, "_assign_train", side_effect=TimeoutError):
            with self.assertRaises(TimeoutError):
                q._candidate_from_schedule(qea_instance(), schedule, 1)

    def test_same_logical_work_has_same_search_under_different_clock_rates(self):
        from station_dispatch import qea_lns as q
        instance = qea_instance()
        traces = []
        for elapsed in (.01, .8):
            calls = []
            clock = [0.0]
            construct = q._construct_schedule
            repair = q._reschedule_wait_chain
            def record_construct(*args, **kwargs):
                result = construct(*args, **kwargs)
                calls.append(("construct", args[1], result))
                clock[0] = 2.0 if len(calls) >= 24 else elapsed
                return result
            def record_repair(*args, **kwargs):
                result = repair(*args, **kwargs)
                calls.append(("repair", kwargs["target_index"], result))
                clock[0] = 2.0 if len(calls) >= 24 else elapsed
                return result
            with patch.object(q.time, "monotonic", side_effect=lambda: clock[0]), patch.object(
                q, "_construct_schedule", side_effect=record_construct
            ), patch.object(q, "_reschedule_wait_chain", side_effect=record_repair):
                result = q.solve_qea_lns(instance, 42, 1)
            traces.append((calls, result.schedule))
        self.assertEqual(traces[0], traces[1])


    def test_interrupted_encoding_keeps_repair_without_fallback_or_feedback(self):
        from station_dispatch import qea_lns as q
        instance = qea_instance()
        initial = Schedule((Assignment(1, 1, 0, 0), Assignment(2, 3, 10, 0)))
        improved = Schedule((Assignment(1, 2, 0, 0), Assignment(2, 3, 0, 0)))
        repair = q._LnsRepair(improved, {1: 2, 2: 3}, (1, 2))
        clock = [0.0]
        def expire(*args):
            clock[0] = 2.0
            raise TimeoutError
        with patch.object(q.time, "monotonic", side_effect=lambda: clock[0]), patch.object(
            q, "_construct_schedule", return_value=initial
        ), patch.object(q, "_reschedule_wait_chain", return_value=repair), patch.object(
            q, "_candidate_from_schedule", side_effect=expire
        ), patch.object(q, "_merge_lns_repair_into_candidate", side_effect=AssertionError("timeout fallback")), patch.object(
            q, "_update_member_toward", side_effect=AssertionError("late feedback")
        ):
            result = q.solve_qea_lns(instance, 42, 1)
        self.assertEqual(result.schedule, improved)
        self.assertTrue(result.feasible)
        self.assertEqual(result.status, "time_limit")
