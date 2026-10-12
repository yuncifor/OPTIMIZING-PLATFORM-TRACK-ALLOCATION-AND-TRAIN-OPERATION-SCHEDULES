# Train platforming and dispatching with QEA-LNS

Python implementations of FCFS, MILP and QEA-LNS for a five-track station with 45 controlled track sections. All solver source files are included. Inputs specify earliest operation start times, allowed routes and minimum running and dwell requirements.

The primary objective is total admission waiting plus excess dwell time. Makespan breaks ties. Common validation checks modeled resource exclusivity and operation requirements.

## Data source and license

The 15 scenario inputs are unchanged files from the CP 2025 benchmark by Andreas Schutt, Matteo Cardellini, Jip J. Dekker, Daniel Harabor, Marco Maratea and Mauro Vallati. The source paper is Constraint-Based In-Station Train Dispatching, LIPIcs 340, pages 33:1-33:24, DOI 10.4230/LIPIcs.CP.2025.33.

The benchmark authors are credited for the scenarios and original station model. Their copyright and MIT permission notice is retained in LICENSE. The source distribution uses the same MIT license.

## Usage and citation

When copying, modifying or redistributing the software or substantial portions of it, retain the applicable copyright notices and the MIT permission notice as required by the license.

If you use this project's method implementations, experimental workflow or results in a paper, research report or other public work, please cite this repository and any associated publication used, and identify the version or commit. When using the original station model or scenario inputs, also cite the benchmark paper identified above.

The relevant rights holders reserve the right to seek remedies under applicable law for license violations or infringement. The scholarly citation guidance is a request for research attribution and does not impose additional conditions on the MIT license.

## Directory contents

- station_dispatch/: solvers, scheduling, validation and metrics.
- scripts/: manifest generation, experiment entry points, aggregation and plotting.
- tests/: solver, execution, statistics and plotting checks.
- data/cp2025/: the 15 instances used by the experiment manifest.
- experiments/manifest.csv: instances, seeds, budgets and repetitions.
- raw_results/: saved reference runs and complete schedules when feasible.
- results/: run tables and statistical summaries.

## Reference results

The reference set retains 288 runs: 9 main, 108 train-count, 90 seed and 81 time-budget runs. Each method has 96 records. Successful and unsuccessful runs are included. Input hashes, software revisions and environment identifiers remain unchanged.

Saved records belong to their recorded revisions. New runs use their own source and environment identifiers. Historical executable paths describe the original environment and are not experiment inputs. Keep fresh outputs in a separate directory rather than overwriting reference files.

## Environment

The checked platform is Windows AMD64 with CPython 3.12.14. Runtime dependencies are pinned in uv.lock and requirements-experiments.txt. The experiment runner verifies SCIP 10.0.0 / SoPlex 8.0.0 and uses fresh, single-thread workers. Plotting uses a separate environment.

Run from the repository root in PowerShell:

    uv sync --locked --python 3.12.14
    uv venv .venv-figures --python 3.12.14
    uv pip install --python .venv-figures/Scripts/python.exe -r requirements-figures.txt

## Experiment design

| Experiment | Instances | Seeds | Budget (s) | Repetitions |
|---|---|---|---|---|
| Main | t030-01 to t030-03 | 42 | 300 | 1 |
| Train count | 35, 40, 45 and 50 trains; instances 01-03 | 42 | 300 | 3 |
| Random seed | t050-01 | 37-41 and 43-47 | 300 | 3 |
| Time budget | t050-01 | 42 | 10, 20, 30, 40, 50, 60, 120, 300, 600 | 3 |

Run the three methods independently and write new records to runs/:

    ./.venv/Scripts/python.exe -B scripts/schedule_bench.py --manifest experiments/manifest.csv --output-root runs --method all
    ./.venv/Scripts/python.exe -B scripts/run_scalability.py --manifest experiments/manifest.csv --output-root runs --method all
    ./.venv/Scripts/python.exe -B scripts/run_seed_sensitivity.py --manifest experiments/manifest.csv --output-root runs --method all
    ./.venv/Scripts/python.exe -B scripts/run_time_budget.py --manifest experiments/manifest.csv --output-root runs --method all
    ./.venv/Scripts/python.exe -B scripts/summarize_results.py --manifest experiments/manifest.csv --raw-results runs --output-dir analysis

The entry points also accept --method fcfs, --method milp or --method qea_lns. FCFS returns after construction. Search methods use their configured budgets. Runs without a feasible schedule retain their status and do not supply quality values.

## Aggregation and plotting

Recompute the reference tables and figures without solving:

    ./.venv/Scripts/python.exe -B scripts/summarize_results.py --manifest experiments/manifest.csv --raw-results raw_results --output-dir analysis/reference
    ./.venv-figures/Scripts/python.exe -B scripts/plot_experiments.py --results analysis/reference --raw-results raw_results --output-dir figures/reference --include-search
    ./.venv-figures/Scripts/python.exe -B scripts/plot_experiments.py --results analysis/reference --raw-results raw_results --output-dir figures/reference --sensitivity-3d-only

For fresh records, use --results analysis --raw-results runs in the plotting commands. Statistical tables use the repetitions of each fixed point. Undefined dispersion remains empty; measured zero SEM remains zero. Saved train assignments support platform occupation plots.

## Checks

The runtime pattern includes all current runtime test files and excludes the separate plotting suite:

    ./.venv/Scripts/python.exe -B -m unittest discover -s tests -p "test_[dfmqs]*.py" -v
    ./.venv-figures/Scripts/python.exe -B -m unittest discover -s tests -p test_plot_experiments.py -v

The common validate_schedule and compute_metrics functions check complete modeled feasibility and recalculate schedule metrics for every method.
