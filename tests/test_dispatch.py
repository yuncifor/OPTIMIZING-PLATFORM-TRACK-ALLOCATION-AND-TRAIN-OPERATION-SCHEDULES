from pathlib import Path
import unittest
from unittest.mock import patch

from station_dispatch.dzn import Block, Route, StationInstance, Train
from station_dispatch.schedule import Assignment


def dispatch_instance() -> StationInstance:
    return StationInstance(
        path=Path("dispatch.dzn"),
        edge_count=2,
        edge_names=("shared_throat", "clear_throat"),
        edge_types=("inter", "inter"),
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


class DynamicDispatchTest(unittest.TestCase):
    def test_selects_an_immediately_available_route_from_all_allowed_routes(self) -> None:
        from station_dispatch.dispatch import earliest_feasible_assignment

        instance = dispatch_instance()
        assignment = earliest_feasible_assignment(
            instance,
            (Assignment(1, 1, 0, 0),),
            instance.trains[1],
        )

        self.assertEqual(assignment, Assignment(2, 3, 0, 0))

    def test_waits_until_shared_resource_is_released_when_no_route_is_immediately_available(self) -> None:
        from station_dispatch.dispatch import earliest_feasible_assignment

        instance = dispatch_instance()
        assignment = earliest_feasible_assignment(
            instance,
            (Assignment(1, 1, 0, 0),),
            instance.trains[1],
            route_ids=(2,),
        )

        self.assertEqual(assignment, Assignment(2, 2, 5, 0))

    def test_uses_cached_route_and_block_indexes_after_template_initialization(self) -> None:
        import station_dispatch.dispatch as dispatch
        from station_dispatch.schedule import planning_horizon

        instance = dispatch_instance()
        planning_horizon(instance)
        dispatch._assignment_resource_occupancies(instance, Assignment(1, 1, 0, 0))
        with patch.object(dispatch, "_routes_by_id", wraps=dispatch._routes_by_id) as routes_by_id, patch.object(
            dispatch, "_blocks_by_id", wraps=dispatch._blocks_by_id,
        ) as blocks_by_id:
            dispatch.earliest_feasible_assignment(instance, (), instance.trains[0])

        routes_by_id.assert_called_once_with(instance)
        blocks_by_id.assert_called_once_with(instance)

    def test_precomputed_resource_index_preserves_the_default_assignment(self) -> None:
        from station_dispatch.dispatch import _indexed_resource_occupancies, earliest_feasible_assignment

        instance = dispatch_instance()
        assignments = (Assignment(1, 1, 0, 0),)
        default = earliest_feasible_assignment(instance, assignments, instance.trains[1])
        indexed = earliest_feasible_assignment(
            instance,
            assignments,
            instance.trains[1],
            validate_existing=False,
            validate_candidate=False,
            resource_occupations=_indexed_resource_occupancies(instance, assignments),
        )

        self.assertEqual(indexed, default)


if __name__ == "__main__":
    unittest.main()

class FcfsRouteChoiceTest(unittest.TestCase):
    def test_uses_earliest_start_instead_of_earliest_completion(self) -> None:
        from station_dispatch.dispatch import earliest_feasible_assignment

        instance = StationInstance(
            path=Path("fcfs-route-choice.dzn"),
            edge_count=2,
            edge_names=("shared_throat", "clear_throat"),
            edge_types=("inter", "inter"),
            platform_names=(),
            trains=(
                Train(1, "T1", 0, "pass", (1,)),
                Train(2, "T2", 0, "pass", (2, 3)),
            ),
            routes=(
                Route(1, 1, "", 0, 5, (1,)),
                Route(2, 2, "", 0, 1, (2,)),
                Route(3, 2, "", 0, 10, (3,)),
            ),
            blocks=(
                Block(1, 1, 5, 0, False),
                Block(2, 1, 1, 0, False),
                Block(3, 2, 10, 0, False),
            ),
        )

        assignment = earliest_feasible_assignment(
            instance,
            (Assignment(1, 1, 0, 0),),
            instance.trains[1],
        )

        self.assertEqual(assignment, Assignment(2, 3, 0, 0))


class PreferredRouteTieBreakTest(unittest.TestCase):
    def test_prefers_the_encoded_route_only_when_all_routes_share_the_earliest_safe_start(self) -> None:
        from station_dispatch.dispatch import earliest_feasible_assignment

        instance = dispatch_instance()
        assignment = earliest_feasible_assignment(
            instance,
            (),
            instance.trains[1],
            preferred_route_id=3,
        )

        self.assertEqual(assignment, Assignment(2, 3, 0, 0))



class OccupancyTemplateTest(unittest.TestCase):
    def test_identical_assignment_reuses_immutable_occupancies(self):
        import station_dispatch.dispatch as dispatch
        from test_schedule import small_instance
        instance = small_instance()
        assignment = Assignment(1, 1, 10, 5)
        first = dispatch._assignment_resource_occupancies(instance, assignment)
        with patch.object(dispatch, "BlockOccupancy", side_effect=AssertionError("duplicate occupancy construction")):
            self.assertIs(dispatch._assignment_resource_occupancies(instance, assignment), first)

    def test_assignment_memo_is_bounded_and_evicted_values_recompute_exactly(self):
        import station_dispatch.dispatch as dispatch
        from station_dispatch.schedule import Schedule, resource_occupancies
        from test_schedule import small_instance
        instance = small_instance()
        for start in range(8193):
            dispatch._assignment_resource_occupancies(instance, Assignment(1, 1, start, 5))
        memo = dispatch._ASSIGNMENT_OCCUPANCY_CACHE[id(instance)][2]
        self.assertEqual(len(memo), 8192)
        first = Assignment(1, 1, 0, 5)
        self.assertNotIn(first, memo)
        self.assertEqual(dispatch._assignment_resource_occupancies(instance, first),
                         resource_occupancies(instance, Schedule((first,))))
        self.assertEqual(len(memo), 8192)

    def test_many_start_and_dwell_values_share_one_route_template(self):
        import station_dispatch.dispatch as dispatch
        from station_dispatch.schedule import Schedule, resource_occupancies
        from test_schedule import small_instance
        instance = small_instance()
        dispatch._ASSIGNMENT_OCCUPANCY_CACHE.pop(id(instance), None)
        for start in range(10, 30):
            for dwell in (0, 5, 17):
                assignment = Assignment(1, 1, start, dwell)
                self.assertEqual(dispatch._assignment_resource_occupancies(instance, assignment),
                                 resource_occupancies(instance, Schedule((assignment,))))
        self.assertEqual(len(dispatch._ASSIGNMENT_OCCUPANCY_CACHE[id(instance)][1]), 1)

    def test_template_preserves_origin_destination_and_multi_stop_intervals(self):
        from dataclasses import replace
        import station_dispatch.dispatch as dispatch
        from station_dispatch.schedule import Schedule, resource_occupancies
        from test_schedule import small_instance
        instance = small_instance()
                                                                           
                                                           
        instance = replace(instance, blocks=tuple(
            replace(b, is_stop=True, start_offset=3) if b.id == 3 else b
            for b in instance.blocks))
        for train_id, route_id in ((1, 1), (3, 3), (5, 5)):
            for start in (10, 20, 100):
                for dwell in (0, 5, 17):
                    assignment = Assignment(train_id, route_id, start, dwell)
                    self.assertEqual(dispatch._assignment_resource_occupancies(instance, assignment),
                                     resource_occupancies(instance, Schedule((assignment,))))
