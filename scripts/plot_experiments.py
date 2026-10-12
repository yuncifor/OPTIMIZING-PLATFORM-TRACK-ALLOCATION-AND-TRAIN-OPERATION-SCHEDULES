from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BLUE = '#3B6F8F'
GREY = '#9EA4A8'
ORANGE = '#A86C59'
INK = '#333333'
PALE_GREY = '#BFC3C5'
MESH_COLORS = {
    'slate_blue': ('#DFEAF1', '#9AB7CA', '#648BA5', '#446F8A', '#2D5875', '#234861'),
    'sage_green': ('#E7E6C4', '#BCCDA2', '#95B280', '#6C9968', '#477E5B', '#32614B'),
}
TRAIN_TYPE_COLORS = {'pass': '#496D87', 'origin': '#E1E4E6', 'vanish': '#B58A73'}
NUMERIC = {'repetition', 'repetitions', 'seed', 'train_count', 'time_limit_s', 'total_station_wait', 'entry_wait_total',
           'excess_dwell_total', 'makespan', 'end_sum', 'first_feasible_s', 'best_found_s'}
SOURCES = {}


def track(path):
    SOURCES[str(path.resolve().relative_to(ROOT)) if path.resolve().is_relative_to(ROOT) else str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()


def read_runs(path):
    track(path)
    with path.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row['feasible'] = row['feasible'].lower() == 'true'
        row.setdefault('repetition', '1')
        row.setdefault('repetitions', '1')
        for key in NUMERIC:
            row[key] = float(row[key]) if row.get(key) not in ('', None) else None
        if not row['feasible']:
            for key in ('total_station_wait', 'entry_wait_total', 'excess_dwell_total', 'makespan', 'end_sum'):
                row[key] = None
    return rows


def read_stats(path):
    track(path)
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def statistic_row(stats, rows, metric):
    keys = ('method', 'train_count', 'time_limit_s', 'condition_id', 'source_sha256',
            'instance_id', 'input_sha256', 'seed')
    reference = rows[0]
    if any(len({row[key] for row in rows}) != 1 for key in keys):
        raise ValueError('Statistics require one fixed method, instance, seed, budget and version.')
    matches = [row for row in stats if all(
        float(row[key]) == reference[key] if key in ('train_count', 'time_limit_s', 'seed')
        else row[key] == reference[key] for key in keys)]
    if len(matches) != 1:
        raise ValueError('Statistics do not match the selected point.')
    result = matches[0]
    if result['replication_unit'] != 'repeat':
        raise ValueError('Point SEM requires repetitions of the same fixed point.')
    ids = [int(row.get('repetition', 1)) for row in rows]
    counts = {int(row.get('repetitions', 1)) for row in rows}
    if len(set(ids)) != len(ids) or len(counts) != 1 or any(i < 1 or i > max(counts) for i in ids):
        raise ValueError('Invalid point repetition identifiers.')
    expected = {
        'run_count': len(rows), 'feasible_count': sum(row['feasible'] for row in rows),
        'instance_count': 1, 'repetitions_expected': max(counts),
    }
    if any(int(result[key]) != value for key, value in expected.items()):
        raise ValueError('Statistics and runs have different sample counts.')
    for key, values in (
        ('instance_ids', {reference['instance_id']}),
        ('seeds', {str(int(reference['seed']))}),
        ('repetition_ids', {str(i) for i in ids}),
    ):
        if set(result[key].split(';')) != values:
            raise ValueError(f'Statistics and runs have different {key}.')
    values = [row[metric] for row in rows if row['feasible'] and row.get(metric) is not None]
    if int(result[metric + '_n']) != len(values):
        raise ValueError(f'Statistics and runs have different valid counts: {metric}.')
    mean = statistics.mean(values) if values else None
    sem = statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else None
    for suffix, expected_value in (('_mean', mean), ('_sem', sem)):
        saved = result[metric + suffix]
        if expected_value is None:
            if saved not in ('', None) and not math.isnan(float(saved)):
                raise ValueError(f'Undefined statistic must remain empty: {metric + suffix}.')
        elif not math.isclose(float(saved), expected_value, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(f'Statistics do not reproduce the run values: {metric + suffix}.')
    return result


def repeat_points(rows, stats, metrics):
    check_condition(rows)
    if len({row['source_sha256'] for row in rows}) != 1:
        raise ValueError('A comparison requires one source version.')
    groups, inputs = {}, {}
    for row in rows:
        point = row['instance_id'], row['seed'], row['time_limit_s']
        identity = row['train_count'], row['input_sha256']
        if inputs.setdefault(point, identity) != identity:
            raise ValueError('Methods at one point must use the same input.')
        key = (*point, row['method'])
        groups.setdefault(key, []).append(row)
    points = []
    for key, group in sorted(groups.items()):
        point = dict(group[0])
        for metric in metrics:
            stat = statistic_row(stats, group, metric)
            for suffix in ('mean', 'sem'):
                value = stat[metric + '_' + suffix]
                point[metric if suffix == 'mean' else metric + '_sem'] = float(value) if value not in ('', None) else math.nan
        points.append(point)
    return points


def track_metrics(record, row):
    details = record['schedule_details']
    trains = details['trains']
    names = ('S_I', 'S_II', 'S_III', 'S_IV', 'S_V')
    ids = [t['train_id'] for t in trains]
    waits = record['metrics']['by_train']
    if (len(trains) != int(row['train_count']) or len(set(ids)) != len(ids)
            or set(waits) != {str(i) for i in ids}
            or any(t['platform_name'] not in names for t in trains)):
        raise ValueError('Track metrics require all trains and their platform assignments.')
    total = sum(waits[str(i)]['entry_wait'] + waits[str(i)]['excess_dwell'] for i in ids)
    if not math.isclose(total, row['total_station_wait'], rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError('Train waits do not reproduce total station waiting.')
    lower, upper = details['planning_horizon']
    if upper <= lower:
        raise ValueError('Track occupancy requires a positive planning horizon.')
    result = []
    for name in names:
        selected = [t for t in trains if t['platform_name'] == name]
        values = [waits[str(t['train_id'])]['entry_wait'] + waits[str(t['train_id'])]['excess_dwell']
                  for t in selected]
        intervals = sorted((o['resource_start'], o['resource_end']) for o in platform_rows(record)
                           if o['platform_name'] == name)
        union = []
        for start, end in intervals:
            if not lower <= start <= end <= upper:
                raise ValueError('Platform reservation lies outside the planning horizon.')
            if union and start <= union[-1][1]:
                union[-1][1] = max(end, union[-1][1])
            else:
                union.append([start, end])
        result.append(dict(track=name[2:], assigned_trains=len(selected),
                           mean_wait_min=statistics.mean(values) / 60 if values else math.nan,
                           occupancy_pct=sum(end-start for start, end in union) / (upper-lower) * 100))
    return result


def track_points(rows, raw):
    groups, horizons = {}, {}
    for row in rows:
        if not row['feasible']:
            continue
        record = read_schedule(raw, row)
        horizon = tuple(record['schedule_details']['planning_horizon'])
        if horizons.setdefault(row['input_sha256'], horizon) != horizon:
            raise ValueError('Methods and repetitions require the same planning horizon.')
        for item in track_metrics(record, row):
            key = row['method'], row['instance_id'], row['seed'], row['time_limit_s'], item['track']
            groups.setdefault(key, []).append((row, item))
    result = []
    for key, group in sorted(groups.items()):
        point = dict(group[0][0], track=key[-1])
        for metric in ('assigned_trains', 'mean_wait_min', 'occupancy_pct'):
            values = [item[metric] for _, item in group if math.isfinite(item[metric])]
            point[metric] = statistics.mean(values) if values else math.nan
            point[metric + '_sem'] = statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else math.nan
            point[metric + '_n'] = len(values)
        result.append(point)
    return result


def mesh_curve(xs, values):
    import numpy as np
    xs, values = np.asarray(xs, dtype=float), np.asarray(values, dtype=float)
    if len(xs) < 2:
        return xs.copy(), values.copy()
    widths = np.diff(xs)
    secants = np.diff(values) / widths
    slopes = np.zeros_like(values)
    slopes[0], slopes[-1] = secants[0], secants[-1]
    left, right = secants[:-1], secants[1:]
    same = (left * right) > 0
    w1, w2 = 2*widths[1:] + widths[:-1], widths[1:] + 2*widths[:-1]
    slopes[1:-1][same] = (w1[same]+w2[same]) / (w1[same]/left[same]+w2[same]/right[same])
    slopes[1:-1][~np.isfinite(left)] = right[~np.isfinite(left)]
    slopes[1:-1][~np.isfinite(right)] = left[~np.isfinite(right)]
    u = np.linspace(0, 1, 33)[:-1]
    z = ((2*u**3-3*u**2+1)*values[:-1, None]
         + (u**3-2*u**2+u)*widths[:, None]*slopes[:-1, None]
         + (-2*u**3+3*u**2)*values[1:, None]
         + (u**3-u**2)*widths[:, None]*slopes[1:, None])
    z[:, 0] = values[:-1]
    x = xs[:-1, None] + widths[:, None]*u
    return np.r_[x.ravel(), xs[-1]], np.r_[z.ravel(), values[-1]]


def mesh_surface(ax, nodes, xs, ys, *, cmap='slate_blue'):
    import numpy as np
    xs, ys = list(xs), list(ys)
    from matplotlib.colors import LinearSegmentedColormap
    palette = LinearSegmentedColormap.from_list(cmap, MESH_COLORS[cmap])
    selected = [p for p in nodes if p['method'] == 'qea_lns' and math.isfinite(p['value'])]
    if not selected:
        return
    lookup = {(p['x'], p['y']): p['value'] for p in selected}
    values = np.array([[lookup.get((x, y), math.nan) for x in xs] for y in ys])
    fine_x = mesh_curve(xs, values[0])[0]
    rows = np.array([mesh_curve(xs, row)[1] for row in values])
    fine_y = mesh_curve(ys, rows[:, 0])[0]
    smooth = np.array([mesh_curve(ys, column)[1] for column in rows.T]).T
    grid_x, grid_y = np.meshgrid(fine_x, fine_y)
    ax.plot_surface(grid_x, grid_y, smooth, rstride=1, cstride=1, cmap=palette,
                    alpha=1, edgecolor='none', linewidth=0, shade=False, antialiased=False,
                    rasterized=True, label='_QEA-LNS surface', zorder=1)


def mesh_axes(ax, xs, ys, xlabels, ylabels, xlabel, ylabel, zlabel, *, budget=False):
    from matplotlib.ticker import MaxNLocator
    ax.set_xticks(xs, xlabels, fontsize=8)
    ax.set_yticks(ys, ylabels, fontsize=8)
    ax.set_xlabel(xlabel, fontsize=8, labelpad=9 if budget else 7)
    ax.set_ylabel(ylabel, fontsize=8, labelpad=7)
    ax.set_zlabel(zlabel, fontsize=8, labelpad=8)
    ax.tick_params(labelsize=8, pad=2, length=2.5, width=.55)
    locator = MaxNLocator(nbins=5, integer=zlabel == 'Assigned trains')
    ax.zaxis.set_major_locator(locator)
    ax.set_zlim(0, locator.tick_values(0, ax.get_zlim()[1])[-1])
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor('white')
        axis.line.set_color(INK)
        axis.line.set_linewidth(.55)
    ax.view_init(elev=27, azim=-58)
    ax.set_proj_type('ortho')
    ax.set_box_aspect((1.3, 1.7, 1.05) if ylabel == 'Random seed' else (1.65, 1.25, 1.05), zoom=1.06)


def mesh_export(fig, name, out, nodes, note, qa):
    write_plot_source(out, name, nodes)
    for ax in fig.axes:
        for collection in ax.collections:
            if collection.get_label() == '_QEA-LNS surface':
                collection.set_clim(*ax.get_zlim())
    colormaps = sorted({c.get_cmap().name for ax in fig.axes for c in ax.collections
                        if c.get_label() == '_QEA-LNS surface'})
    note += (' Only QEA-LNS is displayed; FCFS nodes remain in the source CSV. Surface colours map Z values through '
             + '/'.join(colormaps) + ' over the displayed Z-axis range. The rasterized gradient surface is a visual guide, '
             'not an uncertainty band. Sample points, connecting lines and SEM error bars are not displayed; '
             'node values and SEM remain in the source CSV. Labels remain vector elements.')
    extras = fig.get_default_bbox_extra_artists() + [ax.zaxis.label for ax in fig.axes]
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = fig.get_tightbbox(renderer, bbox_extra_artists=extras).frozen()
    visible = [*fig.legends, *fig.texts]
    for ax in fig.axes:
        visible.extend([*ax.lines, *ax.texts, ax.xaxis.label, ax.yaxis.label, ax.zaxis.label,
                        *ax.get_xticklabels(), *ax.get_yticklabels(), *ax.get_zticklabels()])
    bounds.y1 = max(bounds.y1, max(a.get_window_extent(renderer).y1 for a in visible if a.get_visible()) / fig.dpi)
    bounds = bounds.padded(.03)
    export(fig, name, out, note, qa, extra_artists=extras, bbox_inches=bounds)
    if name in qa:
        qa[name].update(node_count=len(nodes), sem_nonzero=sum(math.isfinite(p['sem']) and p['sem'] > 0 for p in nodes),
                        source_csv=f'{name}_source.csv', colormaps=colormaps, palette_stops={c: MESH_COLORS[c] for c in colormaps},
                        export_size_inches=[bounds.width, bounds.height], print_width_mm=180,
                        min_font_at_print_width_pt=8 * min(1, 180 / 25.4 / bounds.width),
                        displayed_methods=['qea_lns'],
                        displayed_node_count=sum(p['method'] == 'qea_lns' and math.isfinite(p['value']) for p in nodes))


def check_condition(rows):
    values = {r['condition_id'] for r in rows}
    if len(values) != 1 or values & {'', 'legacy_unrecorded', None}:
        raise ValueError('A comparison requires one recorded experimental condition.')


def paired(rows, *, feasible_only=False):
    check_condition(rows)
    groups = {}
    for row in rows:
        if row['method'] not in ('fcfs', 'qea_lns'):
            continue
        key = row['instance_id'], row['seed'], row['time_limit_s'], row.get('repetition', 1)
        group = groups.setdefault(key, {})
        if row['method'] in group:
            raise ValueError(f'Duplicate result: {key}, {row["method"]}')
        group[row['method']] = row
    result = []
    for key, group in sorted(groups.items()):
        if set(group) != {'fcfs', 'qea_lns'}:
            raise ValueError(f'Unpaired result: {key}')
        a, b = group['fcfs'], group['qea_lns']
        if (a['input_sha256'] != b['input_sha256']
                or a.get('source_sha256') != b.get('source_sha256')):
            raise ValueError(f'Paired provenance mismatch: {key}')
        if not a['feasible'] or not b['feasible']:
            if feasible_only:
                continue
            raise ValueError(f'Cannot compute paired quality: {key}')
        result.append((a, b))
    return result


def read_schedule(raw, row):
    folder = raw / row['experiment_group'] / row['method'] / row['instance_id']
    if row['experiment_group'] == 'time_budget':
        folder /= f'budget-{int(row["time_limit_s"]):03d}'
    if int(row.get('repetitions', 1)) > 1:
        path = folder / f'seed-{int(row["seed"]):03d}' / f'repeat-{int(row["repetition"]):02d}.json'
    else:
        path = folder / f'seed-{int(row["seed"]):03d}.json'
    track(path)
    record = json.loads(path.read_text(encoding='utf8'))
    if record.get('repetition', 1) != int(row.get('repetition', 1)):
        raise ValueError(f'CSV/JSON repetition mismatch: {path}')
    if record['environment']['condition_id'] != row['condition_id'] or record['input']['sha256'] != row['input_sha256']:
        raise ValueError(f'CSV/JSON provenance mismatch: {path}')
    for key in ('total_station_wait', 'makespan', 'end_sum'):
        if record['metrics'][key] != row[key]:
            raise ValueError(f'CSV/JSON metric mismatch: {path}, {key}')
    return record


def platform_rows(record):
    return [o for o in record['schedule_details']['occupancies'] if o['edge_type'] == 'platform']


def status_note(rows):
    milp = [r for r in rows if r['method'] == 'milp']
    if not milp:
        return 'MILP records not provided.'
    found = sum(r['feasible'] for r in milp)
    return f'MILP: {found}/{len(milp)} feasible'


def panel(ax, label, title, offset=38, grid_axis='y'):
    text = f'({label}) {title}' if label else title
    ax.annotate(text, (.5, 0), xycoords='axes fraction', xytext=(0, -offset),
                textcoords='offset points', ha='center', va='top',
                fontsize=8, fontweight='normal', color=INK)
    ax.grid(False, axis='both', which='both')
    ax.set_axisbelow(True)
    ax.tick_params(axis='both', which='major', direction='out', length=3, width=.55,
                   color=INK, labelcolor=INK)
    ax.tick_params(axis='both', which='minor', direction='out', length=2, width=.4,
                   color=INK, labelcolor=INK)
    for spine in ax.spines.values():
        spine.set_color(INK)
        spine.set_linewidth(.55)


def export(fig, name, out, note, qa, *, extra_artists=None, bbox_inches='tight'):
    from matplotlib.text import Text
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    omitted_zlabels = [tick.label1 for ax in fig.axes if hasattr(ax, 'zaxis')
                       for tick in ax.zaxis.get_major_ticks()
                       if not ax.get_zlim()[0] <= tick.get_loc() <= ax.get_zlim()[1]]
    issues = []
    for text in fig.findobj(Text):
        if not text.get_visible() or not text.get_text():
            continue
        if text.get_fontsize() < 5:
            issues.append(f'Font below 5 pt: {text.get_text()}')
        box = text.get_window_extent(renderer)
                                                                        
        if text in omitted_zlabels or text in [t for ax in fig.axes for t in ax.get_xticklabels() + ax.get_yticklabels()]:
            continue
        if box.x0 < -1 or box.y0 < -1 or box.x1 > fig.bbox.width + 1 or box.y1 > fig.bbox.height + 1:
            issues.append(f'Text outside canvas: {text.get_text()}')
    if issues:
        raise ValueError('\n'.join(issues))
    fig.savefig(out / f'{name}.png', dpi=300, bbox_extra_artists=extra_artists, bbox_inches=bbox_inches)
    fig.savefig(out / f'{name}.pdf', bbox_extra_artists=extra_artists, bbox_inches=bbox_inches)
    fig.savefig(out / f'{name}.svg', bbox_extra_artists=extra_artists, bbox_inches=bbox_inches)
    qa[name] = {'size_inches': list(fig.get_size_inches()), 'automated_text_check': 'passed', 'note': note}
    plt.close(fig)


def main_quality(rows, out, qa):
    pairs = paired(rows)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
    fig.subplots_adjust(left=.12, right=.98, bottom=.25, top=.86, wspace=.43)
    for ax, metric, label, title, xlabel in zip(axes, ('total_station_wait', 'makespan'), 'ab',
            ('Station waiting', 'Makespan'), ('Total station waiting (min)', 'Makespan (min)')):
        for index, method, color, offset in ((0, 'FCFS', GREY, -.15), (1, 'QEA-LNS', BLUE, .15)):
            ax.barh([y+offset for y in range(len(pairs))],
                    [pair[index][metric]/60 for pair in pairs], height=.22,
                    color=color, edgecolor='none', linewidth=0, label=method, zorder=3)
        ax.set_yticks(range(len(pairs)), [a['instance_id'] for a, _ in pairs])
        ax.set_ylim(len(pairs)-.5, -.5)
        ax.set_xlim(0, max(pair[index][metric]/60 for pair in pairs for index in (0, 1))*1.06)
        ax.set_xlabel(xlabel)
        ax.tick_params(axis='y', length=0)
        panel(ax, label, title, grid_axis='x')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.55, .985),
               ncol=2, frameon=False, handlelength=1.25, handleheight=.7,
               columnspacing=1.2, handletextpad=.5, fontsize=8)
    export(fig, 'fig01_main_quality', out,
           'Three main instances; seed 42; 300 s budget. Paired horizontal bars show FCFS and QEA-LNS '
           'with zero-based axes; values are expressed in minutes. Small or zero differences are not enlarged. '
           +status_note(rows), qa)


def schedule_plot(rows, raw, out, qa):
    a, b = next((a, b) for a, b in paired(rows) if a['instance_id'] == 't030-03')
    records = [read_schedule(raw, row) for row in (a, b)]
    usages = []
    for record in records:
        source = platform_rows(record)
        usage = []
        for train in record['schedule_details']['trains']:
            parts = [o for o in source if o['train_id'] == train['train_id']]
            usage.append({'train_id': train['train_id'], 'train_type': train['train_type'],
                          'route_id': train['route_id'], 'platform_name': train['platform_name'],
                          'usage_start': min(o['resource_start'] for o in parts),
                          'usage_end': max(o['resource_end'] for o in parts),
                          'source_block_ids': ';'.join(str(o['block_id']) for o in parts)})
        usages.append(sorted(usage, key=lambda o: (o['platform_name'], o['usage_start'], o['train_id'])))
    tracks = sorted({o['platform_name'] for usage in usages for o in usage})
    slots = {name: i for i, name in enumerate(tracks)}
    from matplotlib.patches import Patch
    from matplotlib.ticker import MultipleLocator
    fig = plt.figure(figsize=(7.2, 4.7))
    grid = fig.add_gridspec(2, 1, left=.085, right=.99, bottom=.14, top=.915, hspace=.58)
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[1, 0])]
    bounds = [v / 60 for usage in usages for o in usage for v in (o['usage_start'], o['usage_end'])]
    limits = math.floor(min(bounds)/10)*10, math.ceil(max(bounds)/10)*10
    for ax, usage, method, letter in zip(axes, usages, ('FCFS', 'QEA-LNS'), 'ab'):
        for o in usage:
            y = slots[o['platform_name']]
            start, end = o['usage_start']/60, o['usage_end']/60
            ax.barh(y, end-start, left=start, height=.46,
                    color=TRAIN_TYPE_COLORS[o['train_type']], edgecolor='#50575D', linewidth=.35)
        ax.set_yticks(range(len(tracks)), [name[2:] for name in tracks], fontsize=8)
        ax.set_ylabel('Track', fontsize=8, labelpad=7)
        ax.set_ylim(len(tracks)-.45, -.55)
        ax.set_xlim(limits)
        ax.xaxis.set_major_locator(MultipleLocator(10))
        ax.xaxis.set_minor_locator(MultipleLocator(5))
        ax.set_xlabel('Time (min)', fontsize=8)
        panel(ax, letter, method, offset=34, grid_axis='x')
        ax.tick_params(axis='both', labelsize=8)
        ax.tick_params(axis='y', length=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    handles = [Patch(facecolor=TRAIN_TYPE_COLORS[kind], edgecolor='#50575D', linewidth=.35, label=label)
               for kind, label in (('pass', 'Arrival–departure'), ('origin', 'Originating'),
                                   ('vanish', 'Terminating'))]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.54, .985), ncol=3,
               frameon=False, fontsize=8, handlelength=1.5, handleheight=.75,
               handletextpad=.5, columnspacing=1.8)
    export(fig, 'fig02_schedule', out,
           't030-03; seed 42; 300 s budget. Arrival/departure track utilization from the saved dispatch schedules. '
           'Two full-width panels show FCFS and QEA-LNS on tracks I–V, with all 30 trains represented once per method. '
           'Each interval spans the earliest reservation to the latest release among that train’s platform resources; '
           'the source intervals are continuous and the aggregated intervals do not overlap between trains on the same track in this case. '
           'These are model-based track-use intervals, not measured platform arrival/departure or head/tail passage times. '
           'Origin reservations retain the planning-window start. Aggregation is for display only and adds no solver constraint. '
           'Colours encode arrival–departure, originating and terminating operations consistently across both panels. '
           'Common time limits are rounded outward to 10-minute boundaries; major/minor ticks are 10/5 minutes. '
           'No train-number annotations, grids or decorative text.', qa)
    with (out / 'fig02_occupancy_source.csv').open('w', newline='', encoding='utf8') as stream:
        fields = ['method', *usages[0][0]]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for method, usage in zip(('fcfs', 'qea_lns'), usages):
            writer.writerows({'method': method, **o} for o in usage)



def write_plot_source(out, name, nodes):
    fields = ('experiment_group', 'method', 'instance_id', 'train_count', 'seed', 'time_limit_s',
              'repetition', 'track', 'scene', 'condition_id', 'source_sha256', 'input_sha256',
              'metric', 'unit', 'x', 'y', 'value', 'sem', 'n')
    with (out / f'{name}_source.csv').open('w', encoding='utf8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for node in nodes:
            writer.writerow({key: (value if not isinstance(value, float) or math.isfinite(value) else '')
                             for key, value in node.items() if key in fields})


def line_series(ax, nodes, label, color, *, marker='o', linestyle='-'):
    import numpy as np
    x = np.array([p['x'] for p in nodes])
    mean = np.array([p['value'] for p in nodes], dtype=float)
    sem = np.array([p['sem'] for p in nodes], dtype=float)
    ax.plot(x, mean, color=color, marker=marker, markersize=3.2, linewidth=.85,
            markeredgewidth=.55, markerfacecolor='white', linestyle=linestyle, label=label)
    valid = np.isfinite(mean) & np.isfinite(sem)
    ax.fill_between(x, mean-sem, mean+sem, where=valid, color=color, alpha=.20, linewidth=0)
    nonzero = valid & (sem > 0)
    if nonzero.any():
        ax.errorbar(x[nonzero], mean[nonzero], yerr=sem[nonzero], fmt='none',
                    ecolor=color, elinewidth=.65, capsize=2, capthick=.65)


def line_export(fig, name, out, nodes, note, qa):
    write_plot_source(out, name, nodes)
    export(fig, name, out, note, qa)
    if name in qa:
        qa[name].update(source_csv=f'{name}_source.csv', node_count=len(nodes),
                        sem_nonzero=sum(math.isfinite(p['sem']) and p['sem'] > 0 for p in nodes),
                        displayed_methods=sorted({p['method'] for p in nodes if math.isfinite(p['value'])}))


def scale_plot(rows, out, qa, stats, raw=ROOT/'raw_results'):
    from matplotlib.ticker import AutoMinorLocator, MaxNLocator
    if ({r['experiment_group'] for r in rows} != {'scalability'}
            or len({(r['seed'], r['time_limit_s']) for r in rows}) != 1):
        raise ValueError('Scale points require scale records with one seed and budget.')
    points = repeat_points(rows, stats, ('total_station_wait', 'makespan', 'best_found_s'))
    sizes = sorted({int(p['train_count']) for p in points})
    scenes = sorted({int(p['instance_id'].split('-')[-1]) for p in points})
    fig, axes = plt.subplots(1, 3, figsize=(7.8, 3.4))
    fig.subplots_adjust(left=.08, right=.985, bottom=.31, top=.85, wspace=.52)
    nodes = []
    for scene, color in zip(scenes, (BLUE, '#688B77', ORANGE)):
        improvements = []
        for size in sizes:
            group = [r for r in rows if r['instance_id'] == f't{size:03d}-{scene:02d}']
            pairs = paired(group, feasible_only=True)
            values = [100*(a['total_station_wait']-b['total_station_wait'])/a['total_station_wait'] for a, b in pairs]
            point = next(p for p in points if p['method'] == 'qea_lns' and p['instance_id'] == group[0]['instance_id'])
            improvements.append(dict(point, scene=scene, repetition='', x=size,
                                     metric='waiting_improvement', unit='%', n=len(values),
                                     value=statistics.mean(values) if values else math.nan,
                                     sem=statistics.stdev(values)/math.sqrt(len(values)) if len(values) > 1 else math.nan))
        line_series(axes[0], improvements, f'Scenario {scene:02d}', color)
        nodes.extend(improvements)
        for index, metric, divisor, unit in ((1, 'makespan', 60, 'min'), (2, 'best_found_s', 1, 's')):
            for method in (('qea_lns',) if index == 1 else ('fcfs', 'qea_lns')):
                selected = sorted((p for p in points if p['method'] == method and int(p['instance_id'].split('-')[-1]) == scene),
                                  key=lambda p: p['train_count'])
                series = [dict(p, scene=scene, repetition='', x=p['train_count'], metric=metric, unit=unit,
                               value=p[metric]/divisor, sem=p[metric+'_sem']/divisor,
                               n=sum(r['feasible'] for r in rows if (r['method'], r['instance_id']) == (p['method'], p['instance_id'])))
                          for p in selected]
                line_series(axes[index], series, '_'+method, color,
                            marker='s' if method == 'fcfs' else 'o', linestyle='--' if method == 'fcfs' else '-')
                nodes.extend(series)
    from matplotlib.lines import Line2D
    handles, labels = axes[0].get_legend_handles_labels()
    handles += [Line2D([], [], color=INK, marker='s', markersize=3.2, linestyle='--', linewidth=.85),
                Line2D([], [], color=INK, marker='o', markersize=3.2, markerfacecolor='white', linewidth=.85)]
    fig.legend(handles, labels+['FCFS', 'QEA-LNS'], loc='upper center', ncol=5, frameon=False)
    for ax, ylabel, title, letter in zip(axes, ('Waiting improvement (%)', 'Makespan (min)', 'Time to best solution (s)'),
                                        ('Waiting improvement', 'QEA-LNS makespan', 'Solution efficiency'), 'abc'):
        ax.set_xticks(sizes)
        ax.set_xlabel('Number of trains')
        ax.set_ylabel(ylabel)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        panel(ax, letter, title, offset=34)
    times = [p for p in nodes if p['metric'] == 'best_found_s' and math.isfinite(p['value'])]
    if times and all(p['value']-(p['sem'] if math.isfinite(p['sem']) else 0) > 0 for p in times):
        axes[2].set_yscale('log')
    line_export(fig, 'fig03_scalability', out, nodes,
                'Four scales and all three scenarios; seed 42; 300 s. Waiting improvement is computed within paired repetitions relative to FCFS. '
                'Each point uses its own repetition mean ± SEM; makespan is QEA-LNS and efficiency includes FCFS and QEA-LNS. '+status_note(rows), qa)


def sensitivity_lines(rows, out, qa, stats, *, group, coordinate, name):
    from matplotlib.ticker import AutoMinorLocator, MaxNLocator, MultipleLocator
    points = repeat_points(rows, stats, ('total_station_wait', 'makespan', 'best_found_s'))
    fixed = ('instance_id', 'time_limit_s') if group == 'seed_sensitivity' else ('instance_id', 'seed')
    if ({r['experiment_group'] for r in rows} != {group}
            or len({tuple(p[k] for k in fixed) for p in points}) != 1):
        raise ValueError('Sensitivity points require one instance and fixed non-varied conditions.')
    xs = sorted({int(p[coordinate]) for p in points})
    fig, axes = plt.subplots(1, 3, figsize=(7.8, 3.4))
    fig.subplots_adjust(left=.08, right=.985, bottom=.31, top=.85, wspace=.52)
    nodes = []
    for index, (metric, divisor, ylabel, title) in enumerate((
            ('total_station_wait', 60, 'Total station waiting (min)', 'Station waiting'),
            ('makespan', 60, 'Makespan (min)', 'Completion time'),
            ('best_found_s', 1, 'Time to best solution (s)', 'Solution efficiency'))):
        ax = axes[index]
        for method, label, color, marker, linestyle in (
                ('fcfs', 'FCFS', GREY, 's', '--'), ('qea_lns', 'QEA-LNS', BLUE, 'o', '-')):
            selected = sorted((p for p in points if p['method'] == method), key=lambda p: p[coordinate])
            series = [dict(p, repetition='', x=p[coordinate], metric=metric, unit='min' if divisor == 60 else 's',
                           value=p[metric]/divisor, sem=p[metric+'_sem']/divisor,
                           n=sum(r['feasible'] for r in rows if (r['method'], r['instance_id'], r['seed'], r['time_limit_s']) ==
                                 (p['method'], p['instance_id'], p['seed'], p['time_limit_s']))) for p in selected]
            if any(math.isfinite(p['value']) for p in series):
                line_series(ax, series, label, color, marker=marker, linestyle=linestyle)
            nodes.extend(series)
        if coordinate == 'time_limit_s':
            ax.set_xscale('log')
            ax.minorticks_off()
            ax.set_xticks(xs, [str(x) if x not in (20, 40, 50) else '' for x in xs],
                          rotation=45, ha='right', rotation_mode='anchor')
            ax.set_xlabel('Time budget (s)')
        else:
            ax.set_xticks(xs)
            ax.set_xlabel('Random seed')
        ax.set_ylabel(ylabel)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        panel(ax, 'abc'[index], title, offset=44 if coordinate == 'time_limit_s' else 34)
    times = [p for p in nodes if p['metric'] == 'best_found_s' and math.isfinite(p['value'])]
    if times and all(p['value']-(p['sem'] if math.isfinite(p['sem']) else 0) > 0 for p in times):
        axes[2].set_yscale('log')
    if coordinate == 'time_limit_s':
        for ax, bounds, major, minor in zip(axes, ((48796, 51094), (10209, 10315)), (10, .5), (5, .1)):
            margin = (bounds[1]-bounds[0])*.05
            ax.set_ylim((bounds[0]-margin)/60, (bounds[1]+margin)/60)
            ax.yaxis.set_major_locator(MultipleLocator(major))
            ax.yaxis.set_minor_locator(MultipleLocator(minor))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(labels), frameon=False)
    line_export(fig, name, out, nodes,
                'Each point is the mean ± SEM of repetitions at its own fixed instance, seed and budget. '
                'All measured points are retained; zero SEM is not widened. No across-seed or across-budget averaging. '+status_note(rows), qa)


def seed_plot(rows, out, qa, stats, raw=ROOT/'raw_results'):
    sensitivity_lines(rows, out, qa, stats, group='seed_sensitivity', coordinate='seed', name='fig04_seed_sensitivity')


def budget_plot(rows, out, qa, stats, raw=ROOT/'raw_results'):
    sensitivity_lines(rows, out, qa, stats, group='time_budget', coordinate='time_limit_s', name='fig05_time_budget')


def scale_plot_3d(rows, out, qa, stats, raw=ROOT/'raw_results'):
    if ({r['experiment_group'] for r in rows} != {'scalability'}
            or len({(r['seed'], r['time_limit_s']) for r in rows}) != 1):
        raise ValueError('Scale points require scale records with one seed and budget.')
    points = repeat_points(rows, stats, ('total_station_wait',))
    sizes = sorted({int(p['train_count']) for p in points})
    scenes = sorted({int(p['instance_id'].split('-')[-1]) for p in points})
    nodes = [dict(p, scene=int(p['instance_id'].split('-')[-1]), x=p['train_count'],
                  y=int(p['instance_id'].split('-')[-1]), value=p['total_station_wait'] / p['train_count'] / 60,
                  sem=p['total_station_wait_sem'] / p['train_count'] / 60, n=sum(r['feasible'] for r in rows
                  if (r['method'], r['instance_id']) == (p['method'], p['instance_id'])),
                  repetition='', metric='mean_station_wait', unit='min/train')
             for p in points if math.isfinite(p['total_station_wait'])]
    fig = plt.figure(figsize=(8.6, 5.6))
    ax = fig.add_subplot(projection='3d', computed_zorder=False)
    fig.subplots_adjust(left=.04, right=.86, bottom=.07, top=.98)
    mesh_surface(ax, nodes, sizes, scenes)
    mesh_axes(ax, sizes, scenes, [str(s) for s in sizes], [f'{s:02d}' for s in scenes],
              'Number of trains', 'Scenario', 'Mean station waiting (min/train)')
    note = ('Scale experiment; all four sizes and three scenarios; seed 42; 300 s budget. '
            'Every node is the mean ± SEM of repetitions at that fixed point. '
            'Shape-preserving curves pass through measured nodes; scenario numbers are discrete labels. ' + status_note(rows))
    mesh_export(fig, 'fig03_scalability_3d', out, nodes, note, qa)
    tracks = ('I', 'II', 'III', 'IV', 'V')
    nodes = [dict(p, scene=int(p['instance_id'].split('-')[-1]), x=p['train_count'], y=tracks.index(p['track']),
                  value=p['occupancy_pct'], sem=p['occupancy_pct_sem'], n=p['occupancy_pct_n'],
                  repetition='', metric='track_reservation_occupancy', unit='%') for p in track_points(rows, raw)]
    fig = plt.figure(figsize=(7.8, 7.8))
    positions = ((.035, .54, .37, .37), (.505, .54, .37, .37), (.27, .165, .37, .37))
    for i, scene in enumerate(scenes):
        ax = fig.add_axes(positions[i], projection='3d', computed_zorder=False)
        mesh_surface(ax, [p for p in nodes if p['scene'] == scene], sizes, range(5))
        mesh_axes(ax, sizes, range(5), [str(s) for s in sizes], tracks,
                  'Number of trains', 'Track', 'Reservation occupancy (%)')
        ax.set_zlim(0, 100)
        ax.text2D(.5, -.05, f'({"abc"[i]}) Scenario {scene:02d}', transform=ax.transAxes,
                  ha='center', va='top', fontsize=8)
    mesh_export(fig, 'fig03_scalability_tracks_3d', out, nodes,
                'All five tracks for each scale scenario. Reservation occupancy is the union of platform-resource '
                'reservation intervals on a track divided by the fixed planning-horizon duration; origin reservations are included. '
                'Multiple subresources are not added together. Nodes show within-point mean ± SEM. ' + status_note(rows), qa)


def seed_plot_3d(rows, out, qa, stats, raw=ROOT/'raw_results'):
    points = repeat_points(rows, stats, ('total_station_wait',))
    if ({r['experiment_group'] for r in rows} != {'seed_sensitivity'}
            or len({(p['instance_id'], p['time_limit_s']) for p in points}) != 1):
        raise ValueError('Seed comparison requires seed records with one instance and budget.')
    seeds = sorted({int(p['seed']) for p in points})
    tracks = ('I', 'II', 'III', 'IV', 'V')
    usage = track_points(rows, raw)
    fig = plt.figure(figsize=(8.2, 4.8))
    fig.subplots_adjust(left=.02, right=.88, bottom=.18, top=.90, wspace=.35)
    nodes = []
    for index, (metric, zlabel, unit, title) in enumerate((
            ('assigned_trains', 'Assigned trains', 'trains', 'Track allocation'),
            ('mean_wait_min', 'Mean station waiting (min/train)', 'min/train', 'Track waiting'))):
        selected = [dict(p, x=tracks.index(p['track']), y=seeds.index(int(p['seed'])),
                         value=p[metric], sem=p[metric+'_sem'], n=p[metric+'_n'],
                         repetition='', metric=metric, unit=unit) for p in usage]
        nodes.extend(selected)
        ax = fig.add_subplot(1, 2, index+1, projection='3d', computed_zorder=False)
        mesh_surface(ax, selected, range(5), range(len(seeds)), cmap='sage_green')
        mesh_axes(ax, range(5), range(len(seeds)), tracks, [str(s) for s in seeds], 'Track', 'Random seed', zlabel)
        ax.text2D(.5, -.07, f'({"ab"[index]}) {title}', transform=ax.transAxes,
                  ha='center', va='top', fontsize=8)
    mesh_export(fig, 'fig04_seed_tracks_3d', out, nodes,
                't050-01; all five tracks and ten seeds; 300 s. The two panels use distinct Z metrics and units. '
                'Each node is the mean ± SEM of repetitions at that same seed. Counts include every train once. '
                'Waiting is averaged over trains allocated to each track in each schedule; empty-track waiting is missing. '
                'No across-seed uncertainty. '+status_note(rows), qa)


def budget_plot_3d(rows, out, qa, stats, raw=ROOT/'raw_results'):
    points = repeat_points(rows, stats, ('total_station_wait', 'best_found_s'))
    if ({r['experiment_group'] for r in rows} != {'time_budget'}
            or len({(p['instance_id'], p['seed']) for p in points}) != 1):
        raise ValueError('Budget comparison requires budget records with one instance and seed.')
    budgets = sorted({int(p['time_limit_s']) for p in points})
    tracks = ('I', 'II', 'III', 'IV', 'V')
    waiting = [dict(p, x=p['time_limit_s'], y=tracks.index(p['track']),
                    value=p['mean_wait_min'], sem=p['mean_wait_min_sem'], n=p['mean_wait_min_n'],
                    repetition='', metric='mean_wait_min', unit='min/train') for p in track_points(rows, raw)]
    repeats = sorted({int(r['repetition']) for r in rows})
    efficiency = [dict(r, x=r['time_limit_s'], y=int(r['repetition']), value=r['best_found_s'],
                       sem=math.nan, n=1, metric='best_found_s', unit='s')
                  for r in rows if r['feasible'] and r['best_found_s'] is not None]
    for name, nodes, ys, ylabels, ylabel, zlabel, note in (
            ('fig05_time_budget_tracks_3d', waiting, range(5), tracks, 'Track', 'Mean station waiting (min/train)',
             'Track waiting is averaged within each repetition then summarized as the point mean ± SEM. No pooling across budgets.'),
            ('fig05_time_budget_efficiency_3d', efficiency, repeats, [str(i) for i in repeats], 'Repetition',
             'Time to best solution (s)', 'Each node is the recorded elapsed time when the final best-known solution of that run was first found. Individual repetitions have no across-repeat SEM.')):
        fig = plt.figure(figsize=(8.2, 4.8))
        fig.subplots_adjust(left=.02, right=.88, bottom=.18, top=.90, wspace=.35)
        for i, (lower, upper) in enumerate(((10, 60), (60, 600))):
            axis_budgets = [b for b in budgets if lower <= b <= upper]
            ax = fig.add_subplot(1, 2, i+1, projection='3d', computed_zorder=False)
            mesh_surface(ax, [p for p in nodes if lower <= p['x'] <= upper], axis_budgets, ys)
            mesh_axes(ax, axis_budgets, ys, [str(b) for b in axis_budgets], ylabels,
                      'Time budget (s)', ylabel, zlabel, budget=True)
            ax.text2D(.5, -.06, f'({"ab"[i]}) {lower}–{upper} s', transform=ax.transAxes,
                      ha='center', va='top', fontsize=8)
        maximum = max(ax.get_zlim()[1] for ax in fig.axes)
        for ax in fig.axes:
            ax.set_zlim(0, maximum)
        mesh_export(fig, name, out, nodes,
                    't050-01; seed 42; all nine budgets. The 10–60 s and 60–600 s panels share the 60 s boundary '
                    'and Z limits; budget coordinates remain actual seconds on a linear axis. '
                    'Shape-preserving curves are visual interpolation guides, not extra experimental results. '
                    + note + ' ' + status_note(rows), qa)


def search_plot(rows, raw, out, qa):
    a, b = next((a, b) for a, b in paired(rows) if a['time_limit_s'] == 600 and a.get('repetition', 1) == 1)
    record = read_schedule(raw, b)
    history = record['run']['improvement_history']
    scores = [(h[1], h[2]) for h in history]
    if not history or any(y > x for x, y in zip(scores, scores[1:])) or scores[-1] != (b['total_station_wait'], b['makespan']):
        raise ValueError('Invalid improvement history.')
    times = [h[0] for h in history]
    if any(y < x for x, y in zip(times, times[1:])):
        raise ValueError('Improvement timestamps are not ordered.')
    stop = max(600, times[-1])
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    fig.subplots_adjust(left=.10, right=.97, bottom=.24, top=.95)
    ax.step(times+[stop], [h[1]/60 for h in history]+[history[-1][1]/60],
            where='post', color=BLUE, lw=.9, label='QEA-LNS')
    ax.axhline(a['total_station_wait']/60, color=GREY, linestyle='--', lw=.6, label='FCFS')
    ax.set_xlabel('Elapsed solver time (s, log scale)')
    ax.set_ylabel('Total station waiting (min)')
    ax.set_xscale('log')
    ax.set_xticks([.1, 1, 10, 100, 600])
    ax.xaxis.set_major_formatter(lambda value, _: f'{value:g}')
    ax.minorticks_off()
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc='lower left', fontsize=8,
              frameon=False, handlelength=1.25, handletextpad=.45,
              borderaxespad=.5)
    from matplotlib.ticker import MultipleLocator, LogLocator, NullFormatter
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.yaxis.set_minor_locator(MultipleLocator(2))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=(2, 5)))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(which='minor', length=2, width=.4)
    export(fig, 'fig06_search_history', out, 't050-01; seed 42; repetition 1; one 600 s run. Logarithmic time axis. The curve shows the best total station waiting found so far in one saved run. No values before the first feasible solution; no interpolation.\nA plateau means no recorded improvement, not proof of optimality.', qa)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=ROOT/'results')
    parser.add_argument('--raw-results', type=Path, default=ROOT/'raw_results')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'figures')
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--include-search', action='store_true')
    selection.add_argument('--sensitivity-only', action='store_true')
    selection.add_argument('--sensitivity-3d-only', action='store_true')
    args = parser.parse_args()
    global plt
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({
        'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
        'legend.fontsize': 8, 'legend.frameon': False,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .55,
        'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK,
        'pdf.fonttype': 42, 'svg.fonttype': 'none', 'savefig.facecolor': 'white',
        'figure.facecolor': 'white', 'savefig.bbox': 'tight', 'savefig.pad_inches': .03,
    })
    sensitivity_only = args.sensitivity_only or args.sensitivity_3d_only
    SOURCES.clear()
    selected_groups = ('scalability', 'seed_sensitivity', 'time_budget') if sensitivity_only else ('main', 'scalability', 'seed_sensitivity', 'time_budget')
    groups = {g: read_runs(args.results/f'{g}_runs.csv') for g in selected_groups}
    for rows in groups.values():
        check_condition(rows)
    out = args.output_dir/'3d' if args.sensitivity_3d_only else args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    qa = {}
    if not sensitivity_only:
        main_quality(groups['main'], args.output_dir, qa)
        schedule_plot(groups['main'], args.raw_results, args.output_dir, qa)
    scale_function, seed_function, budget_function = (scale_plot_3d, seed_plot_3d, budget_plot_3d) if args.sensitivity_3d_only else (scale_plot, seed_plot, budget_plot)
    scale_function(groups['scalability'], out, qa,
               read_stats(args.results/'scalability_scale_stats.csv'), args.raw_results)
    seed_function(groups['seed_sensitivity'], out, qa,
              read_stats(args.results/'seed_sensitivity_seed_stats.csv'), args.raw_results)
    budget_function(groups['time_budget'], out, qa,
                read_stats(args.results/'time_budget_stats.csv'), args.raw_results)
    if args.include_search:
        search_plot(groups['time_budget'], args.raw_results, args.output_dir, qa)
    exported_count = len(qa)
    report_path = out/'figure_report.json'
    previous = json.loads(report_path.read_text(encoding='utf8')) if args.sensitivity_only and report_path.exists() else {}
    figures = dict(previous.get('figures', {}), **qa)
    if not args.sensitivity_3d_only:
        for name in ('fig03_scalability_tracks', 'fig04_seed_waiting', 'fig05_time_budget_efficiency'):
            figures.pop(name, None)
    sources = dict(previous.get('sources_sha256', {}), **SOURCES)
    captions = '\n\n'.join(f'## {name}\n\n{item["note"]}' for name, item in figures.items())
    (out/'figure_captions.md').write_text(captions+'\n', encoding='utf8')
    report = {'figures': figures, 'sources_sha256': sources,
        'conditions_by_group': dict(previous.get('conditions_by_group', {}),
                                    **{g: sorted({r['condition_id'] for r in rows}) for g, rows in groups.items()}),
        'records_by_group': dict(previous.get('records_by_group', {}), **{g: len(rows) for g, rows in groups.items()}),
        'limitations': ['No cross-group pooling; scale statistics use scale records only.',
            'MILP missing objectives are not zero. No global optimality claim.',
            'FCFS recorded zero discovery times do not establish zero cost; no precise speedup ratios are computed.',
            'Sensitivity nodes use point-specific mean ± SEM of process repetitions, not across-seed or across-instance uncertainty. Zero SEM is not widened. Single observations have no SEM.']
            + (['Budget-efficiency nodes are individual repetition values without cross-repeat error bars.',
                'Track reservation occupancy unions platform-resource intervals within a fixed planning horizon. Track waiting means follow the actual train allocations of each schedule.',
                'Shape-preserving Hermite interpolation is a visual guide through measured nodes, not new observations. Tracks, seeds and scenarios are discrete categories. Budget panels use actual seconds on a linear axis with a shared 60 s boundary and common Z limits.'] if args.sensitivity_3d_only else ['Scale waiting improvement is paired within each repetition before computing mean and SEM. Budget quality axes retain the fixed minute ranges; budget X axes are logarithmic.',
                'Schedule figure is illustrative t030-03, not an additional independent experiment.'])
            + ['Layout checks do not replace visual inspection.']}
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
    print(f'Exported {exported_count} figures (PNG/PDF/SVG) to {out}')


if __name__ == '__main__':
    main()
