"""Figures for saved Kangaroo runs, into a new results/figures/<name>_<time>/ (D40; spec
docs/kangaroo/design/specs/2026-10-14-kangaroo-plots-design.md). No robot needed.

    python3 src/kangaroo_ring_controller/scripts/plot_results.py [--trials GLOB ...] [--calibrations GLOB ...]
        [--refit FILE ...] [--group NAME=GLOB ...] [--name figures] [--root DIR]
A GLOB may name run directories or the JSON files in them. Groups label reaches for the summary figures.
"""
import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the kangaroo_ring_controller package
from kangaroo_ring_controller import RESULTS, evaluation, new_run_dir, plots  # noqa: E402

REPO = Path(__file__).resolve().parents[3]  # relative refit_of paths (D39 refits) are relative to it


def files(pattern, filename):
    """The `filename` JSON files a glob names (directories are looked into); exits if it names none."""
    paths = [p / filename if p.is_dir() else p for p in map(Path, sorted(glob.glob(str(pattern))))]
    if not paths:
        sys.exit(f'{pattern}: matches no {filename}')
    return paths


def read(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError) as e:
        sys.exit(f'{path}: {e}')


def main():
    parser = argparse.ArgumentParser(description='Figures for saved Kangaroo runs.')
    parser.add_argument('--trials', nargs='*', default=[], help='ring_reach / ring_gui runs (trial.json)')
    parser.add_argument('--calibrations', nargs='*', default=[], help='calibration runs (calibration.json)')
    parser.add_argument('--refit', nargs='*', default=[], help='batch refits (D39), drawn over their source')
    parser.add_argument('--group', action='append', default=[], metavar='NAME=GLOB', help='reaches for the summaries')
    parser.add_argument('--evaluation', nargs='*', default=[], help='evaluate.py run dirs (D42)')
    parser.add_argument('--name', default='figures')
    parser.add_argument('--root', default=str(RESULTS), help="default: the package's results/")
    args = parser.parse_args()
    trials = [p for g in args.trials for p in files(g, 'trial.json')]
    cals = [p for g in args.calibrations for p in files(g, 'calibration.json')]
    groups = {name: files(pattern, 'trial.json') for name, pattern in (g.split('=', 1) for g in args.group)}
    records = {path.resolve(): read(path) for path in cals}
    for path, cal in records.items():
        if not cal.get('trials'):
            sys.exit(f'{path}: calibration has no trials (a refit? pass it with --refit)')
    refits = {}
    for path in map(Path, args.refit):
        record = read(path)
        if 'refit_of' not in record:
            sys.exit(f'{path}: not a refit (no refit_of)')
        source = (REPO / record['refit_of']).resolve()  # an absolute refit_of stays as it is
        if source not in records:
            sys.exit(f'{path}: its source {record["refit_of"]} is not among --calibrations')
        refits[source] = record

    out = new_run_dir('figures', args.name, args.root)
    for path in trials:
        plots.trial_figure(read(path), path.parent.name).savefig(out / f'trial_{path.parent.name}.png', dpi=120)
    for path in cals:
        cal, name = records[path.resolve()], path.parent.name
        plots.calibration_progress_figure(cal, name).savefig(out / f'calibration_progress_{name}.png', dpi=120)
        plots.decoder_fit_figure(cal, name, refits.get(path.resolve())).savefig(out / f'decoder_fit_{name}.png',
                                                                                dpi=120)
    if groups:
        data = {name: [read(p) for p in paths] for name, paths in groups.items()}
        for name, figure in (('evaluation_error', plots.evaluation_error_figure(data)),
                             ('plant_vs_decoder', plots.plant_vs_decoder_figure(data))):
            for ext in ('png', 'pdf'):
                figure.savefig(out / f'{name}.{ext}', dpi=150)
    if args.evaluation:
        evaluated = [read(p) for d in args.evaluation for p in files(Path(d) / 'trials' / '*', 'trial.json')]
        for name, figure in (('fixed_relative_error', plots.fixed_boxplot_figure(evaluated)),
                             ('sampled_absolute_error', plots.sampled_error_figure(evaluated))):
            for ext in ('png', 'pdf'):
                figure.savefig(out / f'{name}.{ext}', dpi=150)
        (out / 'summary.json').write_text(json.dumps(dict(summary=evaluation.summary(evaluated),
                                                          verdict=evaluation.verdict(evaluated)), indent=1))
    (out / 'manifest.json').write_text(json.dumps(dict(
        command=sys.argv, evaluation=args.evaluation, trials=list(map(str, trials)), calibrations=list(map(str, cals)), refits=args.refit,
        groups={name: list(map(str, paths)) for name, paths in groups.items()}), indent=1))
    print(out)


if __name__ == '__main__':
    main()
