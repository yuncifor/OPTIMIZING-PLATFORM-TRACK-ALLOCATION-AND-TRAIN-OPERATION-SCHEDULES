import importlib.util
import unittest
import math
import statistics
import tempfile
from pathlib import Path


def module():
    spec = importlib.util.spec_from_file_location('plot_experiments', Path(__file__).resolve().parents[1] / 'scripts/plot_experiments.py')
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def repeated_rows(group):
    if group == 'scalability':
        points = [(f't{size:03d}-{i:02d}', size, 42, 300)
                  for size in (35, 40, 45, 50) for i in (1, 2, 3)]
    elif group == 'seed_sensitivity':
        points = [('t050-01', 50, seed, 300) for seed in (*range(37, 42), *range(43, 48))]
    else:
        points = [('t050-01', 50, 42, budget) for budget in (10, 20, 30, 40, 50, 60, 120, 300, 600)]
    rows = []
    for point, (instance, size, seed, budget) in enumerate(points):
        for method in ('fcfs', 'qea_lns', 'milp'):
            for repetition in range(1, 6):
                change = (repetition - 3) * (point + 1)
                wait = 1000 + point * 100 if method == 'fcfs' else 800 + point * 80 + change * 10
                rows.append(dict(experiment_group=group, instance_id=instance, train_count=size,
                                 seed=seed, time_limit_s=budget, method=method, repetition=repetition,
                                 repetitions=5, condition_id='fixed', source_sha256='version',
                                 input_sha256=instance, feasible=method != 'milp',
                                 total_station_wait=wait if method != 'milp' else None,
                                 makespan=2000 + point * 30 + change,
                                 best_found_s=(.005 + repetition * .0001 if method == 'fcfs' else 1 + point + change * .1)))
    return rows


def point_stats(rows):
    groups = {}
    keys = ('method', 'train_count', 'time_limit_s', 'condition_id', 'source_sha256',
            'instance_id', 'input_sha256', 'seed')
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)
    result = []
    for key, group in groups.items():
        stat = dict(zip(keys, key))
        stat.update(run_count=len(group), feasible_count=sum(r['feasible'] for r in group),
                    instance_count=1, instance_ids=group[0]['instance_id'], seeds=str(int(group[0]['seed'])),
                    repetition_ids=';'.join(str(int(i)) for i in sorted({r.get('repetition', 1) for r in group})),
                    repetitions_expected=group[0].get('repetitions', 1), replication_unit='repeat', ci_method='student_t')
        for metric in ('total_station_wait', 'makespan', 'best_found_s'):
            values = [r[metric] for r in group if r['feasible'] and r.get(metric) is not None]
            stat[metric + '_n'] = len(values)
            stat[metric + '_mean'] = statistics.mean(values) if values else ''
            stat[metric + '_sem'] = statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else ''
        result.append(stat)
    return result


class FigureDataTest(unittest.TestCase):
    def test_pairing_preserves_ties_and_rejects_mixed_conditions(self):
        p = module()
        rows = [dict(instance_id='a', seed=42, time_limit_s=300, input_sha256='i',
                     condition_id='v', method=m, feasible=True, total_station_wait=10.)
                for m in ('fcfs', 'qea_lns')]
        pairs = p.paired(rows)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0][0]['total_station_wait'], pairs[0][1]['total_station_wait'])
        rows[1]['condition_id'] = 'other'
        with self.assertRaises(ValueError):
            p.paired(rows)

    def test_platform_occupancy_uses_reservation_not_movement(self):
        p = module()
        row = dict(edge_type='platform', resource_start=1, resource_end=9,
                   operation_start=3, operation_end=4, train_id=1)
        result = p.platform_rows({'schedule_details': {'occupancies': [row, dict(row, edge_type='switch')]}})
        self.assertEqual(result, [row])
        self.assertEqual((result[0]['resource_start'], result[0]['resource_end']), (1, 9))

    def test_milp_status_is_compact_but_explicit(self):
        p = module()
        self.assertEqual(p.status_note([{'method': 'milp', 'feasible': False}]),
                         'MILP: 0/1 feasible')

    def test_long_caption_is_retained_outside_figure(self):
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            self.skipTest('Rendering checks use the separate plotting environment.')
        p = module()
        p.plt = plt
        fig, ax = plt.subplots(figsize=(4, 2))
        note = 'Detailed interpretation belongs in the companion caption.'
        qa = {}
        with tempfile.TemporaryDirectory() as folder:
            p.export(fig, 'sample', Path(folder), note, qa)
        self.assertNotIn(note, [t.get_text() for t in fig.texts])
        self.assertEqual(qa['sample']['note'], note)

    def test_main_uses_zero_based_grouped_bars_and_lower_titles(self):
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            self.skipTest('Rendering checks use the separate plotting environment.')
        p = module()
        p.plt = plt
        captured = []
        p.export = lambda fig, *args, **kwargs: captured.append(fig)
        rows = p.read_runs(p.ROOT / 'results/main_runs.csv')
        p.main_quality(rows, Path('.'), {})
        fig = captured[0]
        self.addCleanup(plt.close, fig)
        self.assertFalse(fig.texts)
        for ax, metric in zip(fig.axes, ('total_station_wait', 'makespan')):
            self.assertEqual(ax.get_xlim()[0], 0)
            self.assertEqual(len(ax.patches), 6)
            self.assertEqual(ax.get_title(loc='left'), '')
            self.assertEqual(len(ax.texts), 1)
            self.assertLess(ax.texts[0].get_position()[1], 0)
            expected = sorted(row[metric] / 60 for row in rows if row['method'] in ('fcfs', 'qea_lns'))
            self.assertEqual(sorted(bar.get_width() for bar in ax.patches), expected)

    def test_track_usage_covers_all_trains_and_matches_source(self):
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            self.skipTest('Rendering checks use the separate plotting environment.')
        import csv
        p = module()
        p.plt = plt
        captured = []
        p.export = lambda fig, *args: captured.append(fig)
        rows = p.read_runs(p.ROOT / 'results/main_runs.csv')
        with tempfile.TemporaryDirectory() as folder:
            p.schedule_plot(rows, p.ROOT / 'raw_results', Path(folder), {})
            with (Path(folder) / 'fig02_occupancy_source.csv').open() as stream:
                exported = list(csv.DictReader(stream))
        fig = captured[0]
        self.addCleanup(plt.close, fig)
        for ax, method in zip(fig.axes[:2], ('fcfs', 'qea_lns')):
            self.assertEqual([t.get_text() for t in ax.get_yticklabels()], ['I', 'II', 'III', 'IV', 'V'])
            row = next(r for r in rows if r['method'] == method and r['instance_id'] == 't030-03')
            record = p.read_schedule(p.ROOT / 'raw_results', row)
            source = p.platform_rows(record)
            actual = [o for o in exported if o['method'] == method]
            self.assertEqual(len(actual), 30)
            self.assertEqual({int(o['train_id']) for o in actual},
                             {t['train_id'] for t in record['schedule_details']['trains']})
            self.assertEqual(len(ax.containers), 30)
            for bar, usage in zip([c.patches[0] for c in ax.containers], actual):
                parts = sorted([o for o in source if o['train_id'] == int(usage['train_id'])],
                               key=lambda o: o['resource_start'])
                start = parts[0]['resource_start']
                end = parts[0]['resource_end']
                for part in parts[1:]:
                    self.assertLessEqual(part['resource_start'], end)
                    end = max(end, part['resource_end'])
                self.assertEqual(int(usage['usage_start']), start)
                self.assertEqual(int(usage['usage_end']), end)
                self.assertEqual(usage['platform_name'], parts[0]['platform_name'])
                self.assertAlmostEqual(bar.get_x(), start / 60)
                self.assertAlmostEqual(bar.get_width(), (end - start) / 60)
                track_index = ['S_I', 'S_II', 'S_III', 'S_IV', 'S_V'].index(usage['platform_name'])
                self.assertAlmostEqual(bar.get_y() + bar.get_height() / 2, track_index)
                from matplotlib.colors import to_rgba
                colors = {'pass': '#496D87', 'origin': '#E1E4E6', 'vanish': '#B58A73'}
                self.assertEqual(bar.get_facecolor(), to_rgba(colors[usage['train_type']]))
                self.assertGreater(bar.get_linewidth(), 0)
                self.assertEqual(bar.get_edgecolor()[3], 1)
            for track in {o['platform_name'] for o in actual}:
                group = sorted([o for o in actual if o['platform_name'] == track], key=lambda o: int(o['usage_start']))
                for previous, current in zip(group, group[1:]):
                    self.assertLessEqual(int(previous['usage_end']), int(current['usage_start']))
            self.assertEqual(len(ax.texts), 1)
        self.assertEqual(len(fig.axes), 2)
        self.assertEqual(fig.axes[0].get_xlim(), fig.axes[1].get_xlim())
        for ax, method in zip(fig.axes, ('FCFS', 'QEA-LNS')):
            self.assertEqual(ax.texts[0].get_text(), f'({"ab"[fig.axes.index(ax)]}) {method}')
            self.assertLess(ax.texts[0].get_position()[1], 0)
            self.assertGreater(ax.get_position().width, .85)
            self.assertFalse(any(line.get_visible() for line in ax.get_xgridlines() + ax.get_ygridlines()))
            self.assertEqual(ax.get_xlim(), (0., 100.))
            self.assertEqual(ax.get_xlabel(), 'Time (min)')
            self.assertEqual(ax.get_ylabel(), 'Track')
            self.assertEqual(ax.xaxis.label.get_fontsize(), 8)
            self.assertEqual(ax.yaxis.label.get_fontsize(), 8)
            self.assertFalse(ax.spines['top'].get_visible())
            self.assertFalse(ax.spines['right'].get_visible())
            ticks = ax.get_xticks()
            self.assertTrue(all(b-a == 10 for a, b in zip(ticks, ticks[1:])))
            self.assertTrue(all(t % 10 == 5 for t in ax.xaxis.get_minorticklocs()))
        self.assertEqual(len(fig.legends), 1)
        self.assertEqual([t.get_text() for t in fig.legends[0].get_texts()],
                         ['Arrival–departure', 'Originating', 'Terminating'])
        self.assertFalse(fig.legends[0].get_frame_on())
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        legend_box = fig.legends[0].get_window_extent(renderer)
        self.assertGreater(legend_box.y0, fig.axes[0].get_window_extent(renderer).y1)
        for ax in fig.axes:
            labels = [t for t in ax.get_xticklabels() if t.get_visible()
                      and ax.get_xlim()[0] <= t.get_position()[0] <= ax.get_xlim()[1]]
            boxes = [t.get_window_extent(renderer) for t in labels]
            self.assertTrue(all(a.x1 < b.x0 for a, b in zip(boxes, boxes[1:])))
            xlabel = ax.xaxis.label.get_window_extent(renderer)
            self.assertLess(xlabel.y1, min(b.y0 for b in boxes))
            title = ax.texts[0].get_window_extent(renderer)
            self.assertLess(title.y1, xlabel.y0)
            self.assertGreaterEqual(title.y0, 0)
        self.assertGreater(fig.axes[0].texts[0].get_window_extent(renderer).y0,
                           fig.axes[1].get_window_extent(renderer).y1)

    def test_summary_mismatch_is_rejected(self):
        p = module()
        rows = [r for r in repeated_rows('seed_sensitivity') if r['method'] == 'qea_lns' and r['seed'] == 37]
        stats = point_stats(rows)
        for field, replacement in (('source_sha256', 'other'), ('seeds', '42'),
                                   ('run_count', '9'), ('input_sha256', 'other'),
                                   ('total_station_wait_mean', '0'), ('total_station_wait_sem', '0')):
            with self.subTest(field=field):
                altered = [dict(r, **{field: replacement}) if r['method'] == 'qea_lns' else r for r in stats]
                with self.assertRaises(ValueError):
                    p.statistic_row(altered, rows, 'total_station_wait')

    def test_missing_milp_quality_is_not_zero(self):
        p = module()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'runs.csv'
            path.write_text('method,feasible,total_station_wait\nmilp,false,\n', encoding='utf8')
            rows = p.read_runs(path)
        milp = [r for r in rows if r['method'] == 'milp']
        self.assertEqual(len(milp), 1)
        self.assertTrue(all(not r['feasible'] and r['total_station_wait'] is None for r in milp))


class PointRepeatPlotValidationTest(unittest.TestCase):
    def test_comparisons_reject_mismatched_inputs_between_methods(self):
        p = module()
        rows = repeated_rows('seed_sensitivity')
        for row in rows:
            if row['method'] == 'qea_lns' and row['seed'] == 37:
                row['input_sha256'] = 'other-input'
        with self.assertRaisesRegex(ValueError, 'input'):
            p.repeat_points(rows, point_stats(rows), ('total_station_wait',))

    def test_scale_plot_rejects_pooling_seeds_or_main_results(self):
        p = module()
        for field, value in (('seed', 43), ('experiment_group', 'main')):
            rows = repeated_rows('scalability')
            for row in rows:
                if row['instance_id'] == 't035-01':
                    row[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                p.scale_plot(rows, Path('.'), {}, point_stats(rows))

def synthetic_schedule(row):
    count = int(row['train_count'])
    tracks = ('S_I', 'S_II', 'S_III', 'S_IV', 'S_V')
    trains, occupancies, by_train = [], [], {}
    for i in range(count):
        name = tracks[(i + int(row['repetition']) - 1) % 5]
        trains.append(dict(train_id=i, platform_name=name))
        by_train[str(i)] = dict(entry_wait=row['total_station_wait'] / count, excess_dwell=0)
        occupancies.append(dict(train_id=i, edge_type='platform', platform_name=name,
                                resource_start=i * 2, resource_end=i * 2 + 10))
    return dict(schedule_details=dict(trains=trains, occupancies=occupancies, planning_horizon=[0, 1000]),
                metrics=dict(by_train=by_train))


class ThreeDimensionalSensitivityTest(unittest.TestCase):
    def setUp(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        self.p = module()
        self.p.plt = plt
        self.p.read_schedule = lambda raw, row: synthetic_schedule(row)
        self.figures = []
        self.p.export = lambda fig, name, out, note, qa, **kwargs: self.figures.append((name, fig))
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.out = Path(self.folder.name)
        self.addCleanup(lambda: [plt.close(fig) for _, fig in self.figures])

    def nodes(self, name):
        import csv
        with (self.out / (name + '_source.csv')).open(encoding='utf8', newline='') as stream:
            return list(csv.DictReader(stream))

    def test_track_occupancy_unions_subresources_and_keeps_origin_reservation(self):
        p = self.p
        record = dict(schedule_details=dict(planning_horizon=[0, 100],
                      trains=[dict(train_id=1, platform_name='S_I'), dict(train_id=2, platform_name='S_I')],
                      occupancies=[dict(edge_type='platform', platform_name='S_I', resource_start=a, resource_end=b)
                                   for a, b in ((0, 20), (0, 20), (10, 30), (40, 50))]),
                      metrics=dict(by_train={'1': dict(entry_wait=60, excess_dwell=30),
                                             '2': dict(entry_wait=0, excess_dwell=90)}))
        result = p.track_metrics(record, dict(train_count=2, total_station_wait=180))
        self.assertEqual([r['track'] for r in result], ['I', 'II', 'III', 'IV', 'V'])
        self.assertEqual(result[0]['assigned_trains'], 2)
        self.assertAlmostEqual(result[0]['occupancy_pct'], 40)
        self.assertAlmostEqual(result[0]['mean_wait_min'], 1.5)
        self.assertEqual(sum(r['assigned_trains'] for r in result), 2)
        self.assertTrue(math.isnan(result[1]['mean_wait_min']))
        self.assertEqual(result[1]['occupancy_pct'], 0)

    def test_track_metrics_rejects_incomplete_trains_and_waits(self):
        row = repeated_rows('seed_sensitivity')[0]
        record = synthetic_schedule(row)
        record['schedule_details']['trains'].pop()
        with self.assertRaises(ValueError):
            self.p.track_metrics(record, row)
        record = synthetic_schedule(row)
        record['metrics']['by_train']['0']['entry_wait'] += 1
        with self.assertRaises(ValueError):
            self.p.track_metrics(record, row)

    def test_track_points_use_same_point_repetitions_and_common_horizon(self):
        rows = repeated_rows('seed_sensitivity')
        points = self.p.track_points(rows, self.out)
        group = [r for r in rows if r['method'] == 'qea_lns' and r['seed'] == 37]
        selected = next(r for r in points if r['method'] == 'qea_lns' and r['seed'] == 37 and r['track'] == 'I')
        values = [self.p.track_metrics(synthetic_schedule(r), r)[0]['mean_wait_min'] for r in group]
        self.assertAlmostEqual(selected['mean_wait_min'], statistics.mean(values))
        self.assertAlmostEqual(selected['mean_wait_min_sem'], statistics.stdev(values) / math.sqrt(5))
        self.assertEqual(selected['mean_wait_min_n'], 5)
        self.assertFalse(any(r['method'] == 'milp' for r in points))
        def changed_horizon(raw, row):
            record = synthetic_schedule(row)
            if row['method'] == 'qea_lns':
                record['schedule_details']['planning_horizon'][1] += 1
            return record
        self.p.read_schedule = changed_horizon
        with self.assertRaisesRegex(ValueError, 'horizon'):
            self.p.track_points(rows, self.out)

    def test_scale_mesh_covers_all_instances_and_tracks(self):
        rows = repeated_rows('scalability')
        self.p.scale_plot_3d(rows, self.out, {}, point_stats(rows), self.out)
        self.assertEqual([name for name, _ in self.figures], ['fig03_scalability_3d', 'fig03_scalability_tracks_3d'])
        mean_nodes = self.nodes('fig03_scalability_3d')
        self.assertEqual(len(mean_nodes), 24)
        self.assertEqual({r['instance_id'] for r in mean_nodes}, {r['instance_id'] for r in rows})
        stat = next(r for r in point_stats(rows) if r['method'] == 'qea_lns' and r['instance_id'] == 't035-01')
        node = next(r for r in mean_nodes if r['method'] == 'qea_lns' and r['instance_id'] == 't035-01')
        self.assertAlmostEqual(float(node['value']), stat['total_station_wait_mean'] / 35 / 60)
        self.assertAlmostEqual(float(node['sem']), stat['total_station_wait_sem'] / 35 / 60)
        tracks = self.nodes('fig03_scalability_tracks_3d')
        self.assertEqual(len(tracks), 120)
        self.assertEqual({r['track'] for r in tracks}, {'I', 'II', 'III', 'IV', 'V'})
        self.assertEqual(len(self.figures[1][1].axes), 3)
        self.assertEqual([t.get_text() for t in self.figures[0][1].axes[0].get_yticklabels()], ['01', '02', '03'])

    def test_seed_mesh_retains_all_ten_seeds_and_five_tracks(self):
        rows = repeated_rows('seed_sensitivity')
        self.p.seed_plot_3d(rows, self.out, {}, point_stats(rows), self.out)
        self.assertEqual([name for name, _ in self.figures], ['fig04_seed_tracks_3d'])
        nodes = self.nodes('fig04_seed_tracks_3d')
        self.assertEqual(len(nodes), 200)
        fig = self.figures[0][1]
        self.assertEqual(len(fig.axes), 2)
        for ax in fig.axes:
            self.assertEqual(ax.name, '3d')
            self.assertEqual([t.get_text() for t in ax.get_xticklabels()], ['I', 'II', 'III', 'IV', 'V'])
            self.assertEqual([t.get_text() for t in ax.get_yticklabels()],
                             [str(s) for s in (*range(37, 42), *range(43, 48))])
        self.assertEqual({r['method'] for r in nodes}, {'fcfs', 'qea_lns'})
        self.assertEqual({r['metric'] for r in nodes}, {'assigned_trains', 'mean_wait_min'})
        allocations = [r for r in nodes if r['metric'] == 'assigned_trains']
        for seed in (*range(37, 42), *range(43, 48)):
            for method in ('fcfs', 'qea_lns'):
                self.assertAlmostEqual(sum(float(r['value']) for r in allocations if int(float(r['seed'])) == seed and r['method'] == method), 50)

    def test_budget_mesh_uses_actual_seconds_and_repeat_values(self):
        rows = repeated_rows('time_budget')
        self.p.budget_plot_3d(rows, self.out, {}, point_stats(rows), self.out)
        self.assertEqual([name for name, _ in self.figures], ['fig05_time_budget_tracks_3d', 'fig05_time_budget_efficiency_3d'])
        waiting = self.nodes('fig05_time_budget_tracks_3d')
        efficiency = self.nodes('fig05_time_budget_efficiency_3d')
        self.assertEqual(len(waiting), 90)
        self.assertEqual(len(efficiency), 90)
        budgets = [10, 20, 30, 40, 50, 60, 120, 300, 600]
        for name, fig in self.figures:
            self.assertEqual(len(fig.axes), 2)
            self.assertEqual(list(fig.axes[0].get_xticks()), budgets[:6])
            self.assertEqual(list(fig.axes[1].get_xticks()), budgets[5:])
            self.assertEqual(fig.axes[0].get_zlim(), fig.axes[1].get_zlim())
            surfaces = [c for ax in fig.axes for c in ax.collections if c.get_label() == '_QEA-LNS surface']
            self.assertEqual(len(surfaces), 2)
            self.assertTrue(all(c.get_clim() == fig.axes[0].get_zlim() for c in surfaces))
            self.assertTrue(all(ax.get_xscale() == 'linear' for ax in fig.axes))
            self.assertEqual({float(r['x']) for r in self.nodes(name)}, set(budgets))
        for node in efficiency:
            row = next(r for r in rows if r['method'] == node['method'] and r['time_limit_s'] == float(node['time_limit_s']) and r['repetition'] == int(node['repetition']))
            self.assertEqual(float(node['value']), row['best_found_s'])
            self.assertEqual(node['sem'], '')
            self.assertEqual(int(node['n']), 1)
        ax = self.figures[1][1].axes[0]
        self.assertEqual([t.get_text() for t in ax.get_yticklabels()], ['1', '2', '3', '4', '5'])

    def test_mesh_z_ticks_stay_within_displayed_bounds(self):
        rows = repeated_rows('scalability')
        self.p.scale_plot_3d(rows, self.out, {}, point_stats(rows), self.out)
        for _, fig in self.figures:
            for ax in fig.axes:
                lower, upper = ax.get_zlim()
                self.assertTrue(all(lower <= tick <= upper for tick in ax.get_zticks()))

    def test_mesh_export_includes_three_dimensional_axis_labels(self):
        extras = []
        def capture(fig, name, out, note, qa, **kwargs):
            self.figures.append((name, fig))
            extras.append((kwargs.get('extra_artists', ()), kwargs.get('bbox_inches')))
        self.p.export = capture
        rows = repeated_rows('scalability')
        self.p.scale_plot_3d(rows, self.out, {}, point_stats(rows), self.out)
        for (_, fig), (artists, bounds) in zip(self.figures, extras):
            self.assertTrue(all(ax.zaxis.label in artists for ax in fig.axes))
            self.assertEqual(len(fig.legends), 0)
            fig.canvas.draw()
            self.assertIsNotNone(bounds)
            pixels = bounds.transformed(fig.dpi_scale_trans)
            for ax in fig.axes:
                self.assertLessEqual(ax.get_window_extent(fig.canvas.get_renderer()).y1, pixels.y1)
                for line in ax.lines:
                    self.assertLessEqual(line.get_window_extent(fig.canvas.get_renderer()).y1, pixels.y1)
                label = ax.zaxis.label.get_window_extent(fig.canvas.get_renderer())
                self.assertTrue(pixels.contains(label.x0, label.y0))
                self.assertTrue(pixels.contains(label.x1, label.y1))


    def test_export_checks_visible_zlabels_without_rejecting_out_of_range_ticks(self):
        from matplotlib.ticker import MaxNLocator
        from unittest.mock import patch
        p = module()
        p.plt = self.p.plt
        for misplaced in (False, True):
            fig = p.plt.figure(figsize=(8.2, 4.8))
            self.addCleanup(p.plt.close, fig)
            ax = fig.add_subplot(projection='3d')
            ax.set_zlim(0, 360)
            ax.zaxis.set_major_locator(MaxNLocator(nbins=5))
            self.assertGreater(max(ax.get_zticks()), ax.get_zlim()[1])
            if misplaced:
                ax.text2D(2, .5, 'Outside label', transform=ax.transAxes)
            with self.subTest(misplaced=misplaced), patch.object(fig, 'savefig') as save:
                if misplaced:
                    with self.assertRaisesRegex(ValueError, 'Text outside canvas: Outside label'):
                        p.export(fig, 'zlabels', self.out, '', {})
                    self.assertFalse(save.called)
                else:
                    p.export(fig, 'zlabels', self.out, '', {})
                    self.assertEqual(save.call_count, 3)

    def test_two_panel_meshes_are_horizontal_equal_size_and_wide(self):
        for group, function in (('seed_sensitivity', self.p.seed_plot_3d),
                                ('time_budget', self.p.budget_plot_3d)):
            rows = repeated_rows(group)
            function(rows, self.out, {}, point_stats(rows), self.out)
        for name, fig in self.figures:
            with self.subTest(figure=name):
                self.assertEqual(len(fig.axes), 2)
                self.assertGreater(fig.get_figwidth() / fig.get_figheight(), 1.5)
                first, second = [ax.get_position() for ax in fig.axes]
                self.assertAlmostEqual(first.y0, second.y0)
                self.assertAlmostEqual(first.width, second.width)
                self.assertAlmostEqual(first.height, second.height)
                self.assertLess(first.x1, second.x0)

    def test_budget_tick_labels_do_not_overlap(self):
        rows = repeated_rows('time_budget')
        self.p.budget_plot_3d(rows, self.out, {}, point_stats(rows), self.out)
        for _, fig in self.figures:
            ticks = fig.axes[0].xaxis.get_major_ticks()
            fig.canvas.draw()
            boxes = [tick.label1.get_window_extent(fig.canvas.get_renderer()) for tick in ticks]
            for i, first in enumerate(boxes):
                self.assertFalse(any(first.overlaps(second) for second in boxes[i+1:]))

    def test_shape_preserving_curves_keep_nodes_without_overshoot(self):
        import numpy as np
        self.assertTrue(callable(getattr(self.p, 'mesh_curve', None)))
        xs = np.array([10., 20., 30., 60., 120., 300., 600.])
        values = np.array([7., 7., 1., 9., 2., 4., 3.])
        x, z = self.p.mesh_curve(xs, values)
        self.assertGreaterEqual(len(x), 32*(len(xs)-1)+1)
        for knot, value in zip(xs, values):
            self.assertEqual(list(z[x == knot]), [value])
        for i in range(len(xs)-1):
            segment = z[(x >= xs[i]) & (x <= xs[i+1])]
            self.assertTrue(np.all(segment >= min(values[i:i+2]) - 1e-12))
            self.assertTrue(np.all(segment <= max(values[i:i+2]) + 1e-12))
        self.assertTrue(np.all(z[(x >= 10) & (x <= 20)] == 7))

    def test_shape_preserving_curves_do_not_fill_missing_intervals(self):
        import numpy as np
        self.assertTrue(callable(getattr(self.p, 'mesh_curve', None)))
        x, z = self.p.mesh_curve([0, 1, 2, 3, 4], [1, 2, math.nan, 3, 4])
        self.assertTrue(np.all(np.isnan(z[(x > 1) & (x < 3)])))
        self.assertTrue(np.all(np.isfinite(z[(x >= 0) & (x <= 1)])))
        self.assertTrue(np.all(np.isfinite(z[(x >= 3) & (x <= 4)])))
        x, z = self.p.mesh_curve([0, 1], [3, 5])
        self.assertTrue(np.allclose(z, 3 + 2*x))

    def test_mesh_surface_preserves_samples_without_points_or_connecting_lines(self):
        import numpy as np
        from unittest.mock import patch
        fig = self.p.plt.figure()
        self.addCleanup(self.p.plt.close, fig)
        ax = fig.add_subplot(projection='3d')
        nodes = [dict(method=method, x=x, y=y, value=(x+1)*(y+1), sem=0)
                 for method in ('fcfs', 'qea_lns') for y in (0, 1) for x in (0, 1, 2)]
        with patch.object(ax, 'plot_surface', wraps=ax.plot_surface) as draw:
            self.p.mesh_surface(ax, nodes, [0, 1, 2], [0, 1])
        surfaces = [c for c in ax.collections if c.get_label() == '_QEA-LNS surface']
        self.assertEqual(len(surfaces), 1)
        self.assertEqual(len(ax.collections), 1)
        self.assertEqual(len(ax.lines), 0)
        self.assertEqual(surfaces[0].get_cmap().name, 'slate_blue')
        self.assertGreaterEqual(surfaces[0].get_alpha(), .9)
        self.assertFalse(any(surfaces[0].get_antialiased()))
        self.assertTrue(surfaces[0].get_rasterized())
        self.assertEqual(list(surfaces[0].get_linewidths()), [0])
        self.assertEqual(surfaces[0].get_edgecolors().size, 0)
        self.assertGreater(len(np.unique(surfaces[0].get_array())), 1)
        grid_x, grid_y, values = draw.call_args.args[:3]
        for point in nodes:
            if point['method'] == 'qea_lns':
                at_sample = (grid_x == point['x']) & (grid_y == point['y'])
                self.assertTrue(np.any(at_sample))
                np.testing.assert_allclose(values[at_sample], point['value'])

    def test_mesh_palettes_have_restrained_saturation_and_ordered_luminance(self):
        import colorsys
        import numpy as np
        from matplotlib.colors import to_rgb
        palettes = getattr(self.p, 'MESH_COLORS', {})
        self.assertEqual(set(palettes), {'slate_blue', 'sage_green'})
        previous = {
            'slate_blue': ('#DFEAF1', '#A8C4D6', '#749AB5', '#4F7D9B', '#325F80', '#234861'),
            'sage_green': ('#E7E6C4', '#C8D8AF', '#A5C18F', '#7FA977', '#548C67', '#32614B'),
        }
        for name, colors in palettes.items():
            rgb = np.array([to_rgb(color) for color in colors])
            self.assertLessEqual(max(colorsys.rgb_to_hsv(*color)[1] for color in rgb), .65)
            linear = np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055)**2.4)
            luminance = linear @ np.array([.2126, .7152, .0722])
            self.assertTrue(np.all(np.diff(luminance) < 0))
            self.assertGreater(luminance[0] - luminance[-1], .5)
            self.assertEqual((colors[0], colors[-1]), (previous[name][0], previous[name][-1]))
            before = np.array([to_rgb(color) for color in previous[name]])
            before = np.where(before <= .04045, before / 12.92, ((before + .055) / 1.055)**2.4)
            ratio = luminance[1:-1] / (before @ np.array([.2126, .7152, .0722]))[1:-1]
            self.assertTrue(np.all((ratio >= .70) & (ratio <= .93)))

    def test_mesh_text_is_readable_at_180_mm_print_width(self):
        from matplotlib.text import Text
        bounds = []
        def capture(fig, name, out, note, qa, **kwargs):
            self.figures.append((name, fig))
            bounds.append(kwargs['bbox_inches'])
        self.p.export = capture
        for group, function in (('scalability', self.p.scale_plot_3d),
                                ('seed_sensitivity', self.p.seed_plot_3d),
                                ('time_budget', self.p.budget_plot_3d)):
            rows = repeated_rows(group)
            function(rows, self.out, {}, point_stats(rows), self.out)
        for (name, fig), crop in zip(self.figures, bounds):
            sizes = [text.get_fontsize() for text in fig.findobj(Text)
                     if text.get_visible() and text.get_text()]
            effective = min(sizes) * min(1, 180 / 25.4 / crop.width)
            with self.subTest(figure=name):
                self.assertGreaterEqual(effective, 7)
                self.assertLessEqual(crop.height * min(1, 180 / 25.4 / crop.width) * 25.4, 220)

    def test_seed_surfaces_use_green_palette_without_point_or_line_overlays(self):
        rows = repeated_rows('seed_sensitivity')
        self.p.seed_plot_3d(rows, self.out, {}, point_stats(rows), self.out)
        for ax in self.figures[0][1].axes:
            surface = next(c for c in ax.collections if c.get_label() == '_QEA-LNS surface')
            self.assertEqual(surface.get_cmap().name, 'sage_green')
            self.assertEqual(len(ax.collections), 1)
            self.assertFalse(any(line.get_label() in ('_QEA-LNS curve', '_QEA-LNS connector')
                                 for line in ax.lines))

    def test_mesh_surface_omits_error_bars_without_changing_source_nodes(self):
        p = self.p
        fig = p.plt.figure()
        self.addCleanup(p.plt.close, fig)
        ax = fig.add_subplot(projection='3d')
        nodes = [dict(method='qea_lns', x=x, y=y, value=3, sem=sem)
                 for x, y, sem in ((0, 0, 0), (1, 0, math.nan), (0, 1, .2), (1, 1, 0))]
        p.mesh_surface(ax, nodes, [0, 1], [0, 1])
        self.assertEqual(len(ax.lines), 0)
        self.assertEqual(len(ax.collections), 1)
        self.assertEqual(nodes[2]['sem'], .2)
        self.assertTrue(math.isnan(nodes[1]['sem']))

    def test_point_stats_still_reject_cross_seed_uncertainty(self):
        rows = repeated_rows('seed_sensitivity')
        stats = [dict(r, replication_unit='seed') for r in point_stats(rows)]
        with self.assertRaises(ValueError):
            self.p.seed_plot_3d(rows, self.out, {}, stats, self.out)

    def test_script_entry_invokes_argument_parser(self):
        import runpy
        import sys
        from unittest.mock import patch
        with patch.object(sys, 'argv', ['plot_experiments.py', '--help']):
            with self.assertRaises(SystemExit) as result:
                runpy.run_path(str(self.p.ROOT / 'scripts/plot_experiments.py'), run_name='__main__')
        self.assertEqual(result.exception.code, 0)

    def test_sensitivity_only_preserves_non_target_report(self):
        import json
        import sys
        from unittest.mock import patch
        p = self.p
        untouched = {'size_inches': [7.2, 3.1], 'note': 'Existing main figure.'}
        report = dict(figures={'fig01_main_quality': untouched}, sources_sha256={'main-input': 'original'},
                      conditions_by_group={'main': ['main-condition']}, records_by_group={'main': 9})
        (self.out / 'figure_report.json').write_text(json.dumps(report), encoding='utf8')
        def capture(rows, out, qa, stats, raw):
            qa[rows[0]['experiment_group']] = dict(note='Updated sensitivity.', size_inches=[8, 6])
        with patch.object(sys, 'argv', ['plot_experiments.py', '--sensitivity-only', '--output-dir', str(self.out)]), \
                patch.object(p, 'scale_plot', capture), patch.object(p, 'seed_plot', capture), \
                patch.object(p, 'budget_plot', capture), patch.object(p, 'main_quality') as main_plot, \
                patch.object(p, 'schedule_plot') as schedule_plot:
            p.main()
        saved = json.loads((self.out / 'figure_report.json').read_text(encoding='utf8'))
        self.assertEqual(saved['figures']['fig01_main_quality'], untouched)
        self.assertEqual(saved['records_by_group']['main'], 9)
        self.assertEqual(saved['sources_sha256']['main-input'], 'original')
        self.assertFalse(main_plot.called)
        self.assertFalse(schedule_plot.called)

    def test_search_history_keeps_saved_steps_without_uncertainty(self):
        self.p.read_schedule = module().read_schedule
        rows = self.p.read_runs(self.p.ROOT / 'results/time_budget_runs.csv')
        self.p.search_plot(rows, self.p.ROOT / 'raw_results', self.out, {})
        ax = self.figures[0][1].axes[0]
        self.assertEqual(ax.get_xscale(), 'log')
        self.assertFalse(ax.collections)
        self.assertGreater(len(ax.xaxis.get_minorticklocs()), 0)
        self.assertGreater(len(ax.yaxis.get_minorticklocs()), 0)


class TwoDimensionalSensitivityTest(unittest.TestCase):
    def setUp(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        self.addCleanup(plt.rcParams.update, plt.rcParams.copy())
        plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
                             'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8})
        self.p = module()
        self.p.plt = plt
        self.p.read_schedule = lambda raw, row: synthetic_schedule(row)
        self.figures = []
        self.p.export = lambda fig, name, out, note, qa, **kwargs: self.figures.append((name, fig))
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.out = Path(self.folder.name)
        self.addCleanup(lambda: [plt.close(fig) for _, fig in self.figures])


    def test_three_panel_lines_share_a_wide_aligned_layout(self):
        for group, function in (('scalability', self.p.scale_plot),
                                ('seed_sensitivity', self.p.seed_plot),
                                ('time_budget', self.p.budget_plot)):
            rows = repeated_rows(group)
            function(rows, self.out, {}, point_stats(rows), self.out)
        for name, fig in self.figures:
            with self.subTest(figure=name):
                self.assertEqual(len(fig.axes), 3)
                self.assertGreaterEqual(fig.get_figwidth(), 7.8)
                self.assertGreater(fig.get_figwidth() / fig.get_figheight(), 2)
                positions = [ax.get_position() for ax in fig.axes]
                for first, second in zip(positions, positions[1:]):
                    self.assertAlmostEqual(first.y0, second.y0)
                    self.assertAlmostEqual(first.width, second.width)
                    self.assertAlmostEqual(first.height, second.height)
                    self.assertLess(first.x1, second.x0)

    def test_scale_entry_restores_original_2d_three_panels(self):
        rows = repeated_rows('scalability')
        self.p.scale_plot(rows, self.out, {}, point_stats(rows), self.out)
        self.assertEqual(len(self.figures), 1)
        name, fig = self.figures[0]
        self.assertEqual(name, 'fig03_scalability')
        self.assertEqual(len(fig.axes), 3)
        self.assertTrue(all(ax.name == 'rectilinear' for ax in fig.axes))
        curves = [l for l in fig.axes[0].lines if not l.get_label().startswith('_')]
        self.assertEqual(len(curves), 3)
        self.assertTrue(all(list(l.get_xdata()) == [35, 40, 45, 50] for l in curves))
        for scene, line in enumerate(curves, 1):
            for size, value in zip(line.get_xdata(), line.get_ydata()):
                pairs = self.p.paired([r for r in rows if r['instance_id'] == f't{size:03d}-{scene:02d}'])
                expected = statistics.mean(100 * (a['total_station_wait'] - b['total_station_wait']) / a['total_station_wait'] for a, b in pairs)
                self.assertAlmostEqual(value, expected)

    def test_seed_entry_retains_ten_points_and_point_sem_in_2d(self):
        rows = repeated_rows('seed_sensitivity')
        self.p.seed_plot(rows, self.out, {}, point_stats(rows), self.out)
        self.assertEqual(len(self.figures), 1)
        name, fig = self.figures[0]
        self.assertEqual(name, 'fig04_seed_sensitivity')
        self.assertEqual(len(fig.axes), 3)
        self.assertTrue(all(ax.name == 'rectilinear' for ax in fig.axes))
        for ax, metric, divisor in zip(fig.axes, ('total_station_wait', 'makespan', 'best_found_s'), (60, 60, 1)):
            curves = [l for l in ax.lines if not l.get_label().startswith('_')]
            self.assertEqual(len(curves), 2)
            for method, line in zip(('fcfs', 'qea_lns'), curves):
                self.assertEqual(list(line.get_xdata()), [*range(37, 42), *range(43, 48)])
                for seed, value in zip(line.get_xdata(), line.get_ydata()):
                    group = [r for r in rows if r['method'] == method and r['seed'] == seed]
                    self.assertAlmostEqual(value, statistics.mean(r[metric] for r in group) / divisor)
            self.assertTrue(ax.collections)

    def test_budget_entry_retains_nine_points_and_original_quality_limits(self):
        rows = repeated_rows('time_budget')
        self.p.budget_plot(rows, self.out, {}, point_stats(rows), self.out)
        self.assertEqual(len(self.figures), 1)
        name, fig = self.figures[0]
        self.assertEqual(name, 'fig05_time_budget')
        self.assertEqual(len(fig.axes), 3)
        self.assertTrue(all(ax.name == 'rectilinear' for ax in fig.axes))
        for ax in fig.axes:
            curves = [l for l in ax.lines if not l.get_label().startswith('_')]
            self.assertEqual(len(curves), 2)
            self.assertTrue(all(list(l.get_xdata()) == [10, 20, 30, 40, 50, 60, 120, 300, 600] for l in curves))
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for ax in fig.axes:
            boxes = [label.get_window_extent(renderer) for label in ax.get_xticklabels() if label.get_text()]
            self.assertTrue(all(a.x1 + 1 <= b.x0 for a, b in zip(boxes, boxes[1:])))
        for ax, bounds in zip(fig.axes, ((48796, 51094), (10209, 10315))):
            margin = (bounds[1] - bounds[0]) * .05
            self.assertAlmostEqual(ax.get_ylim()[0], (bounds[0] - margin) / 60)
            self.assertAlmostEqual(ax.get_ylim()[1], (bounds[1] + margin) / 60)


    def test_3d_only_does_not_touch_original_figures_or_report(self):
        import json
        import sys
        from unittest.mock import patch
        report = dict(figures={'fig03_scalability': {'note': 'Original 2D'}})
        (self.out / 'figure_report.json').write_text(json.dumps(report), encoding='utf8')
        (self.out / 'fig03_scalability.png').write_bytes(b'original-2d')
        destinations = []
        def capture(rows, out, qa, stats, raw):
            destinations.append(out)
            qa[rows[0]['experiment_group'] + '_3d'] = dict(note='Supplementary 3D')
        with patch.object(sys, 'argv', ['plot_experiments.py', '--sensitivity-3d-only', '--output-dir', str(self.out)]), \
                patch.object(self.p, 'scale_plot_3d', capture), patch.object(self.p, 'seed_plot_3d', capture), \
                patch.object(self.p, 'budget_plot_3d', capture), patch.object(self.p, 'scale_plot') as original:
            self.p.main()
        self.assertEqual(destinations, [self.out / '3d'] * 3)
        self.assertFalse(original.called)
        self.assertEqual(json.loads((self.out / 'figure_report.json').read_text(encoding='utf8')), report)
        self.assertEqual((self.out / 'fig03_scalability.png').read_bytes(), b'original-2d')
        self.assertTrue((self.out / '3d/figure_report.json').exists())

    def test_2d_only_does_not_touch_3d_outputs(self):
        import sys
        from unittest.mock import patch
        (self.out / '3d').mkdir()
        target = self.out / '3d/figure_report.json'
        target.write_bytes(b'3d-report')
        def capture(rows, out, qa, stats, raw):
            self.assertEqual(out, self.out)
            qa[rows[0]['experiment_group']] = dict(note='Original 2D')
        with patch.object(sys, 'argv', ['plot_experiments.py', '--sensitivity-only', '--output-dir', str(self.out)]), \
                patch.object(self.p, 'scale_plot', capture), patch.object(self.p, 'seed_plot', capture), \
                patch.object(self.p, 'budget_plot', capture), patch.object(self.p, 'scale_plot_3d') as supplement:
            self.p.main()
        self.assertFalse(supplement.called)
        self.assertEqual(target.read_bytes(), b'3d-report')
        import json
        report = json.loads((self.out / 'figure_report.json').read_text(encoding='utf8'))
        self.assertFalse(any('Hermite' in note or 'Z limits' in note for note in report['limitations']))


if __name__ == '__main__':
    unittest.main()
