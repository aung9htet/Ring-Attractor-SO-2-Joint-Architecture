"""Figures for Kangaroo runs from saved trial.json / calibration.json (DECISIONS.md D40; spec
docs/kangaroo/design/specs/2026-10-14-kangaroo-plots-design.md). Pure: parsed JSON in, matplotlib Figure out; the
caller saves (scripts/plot_results.py).
"""
import math
import statistics

from matplotlib.figure import Figure

from . import ring_core  # noqa: F401  (TIAGo package on sys.path)
from tiago_ring_controller.math.control import decoder_features


def _ratio(a, b):
    return a / b if b else math.nan


def relative_error(trial):
    """|q_final − q_goal| / |q_goal − q_start| (D28's measure, dimensionless); NaN if goal = start."""
    return _ratio(abs(trial['q_final'] - trial['q_goal']), abs(trial['q_goal'] - trial['q_start']))


def commanded(trial):
    """Σ decoded_velocity·dt [rad]: the distance the decoder commanded (the command state advances by v·dt a tick)."""
    return sum(trial['decoded_velocity']) * trial['dt']


def ratios(trial):
    """(moved / commanded, commanded / wanted), the plant's and the decoder's share of a reach's error (F15):
    moved = q_final − q_start, wanted = q_goal − q_start [rad]; NaN on a zero."""
    c = commanded(trial)
    return _ratio(trial['q_final'] - trial['q_start'], c), _ratio(c, trial['q_goal'] - trial['q_start'])


def predicted_dq(spikes, dt, gain_positive, gain_negative, tau, delay_steps):
    """Δq [rad] a decoder gives a drive: k_pos·S_pos + k_neg·S_neg, the model TIAGo's fit uses
    (training/calibration.py:97-113; S from math/control.py decoder_features)."""
    f = decoder_features(spikes, dt, tau, delay_steps)
    return gain_positive * f['S_pos'] + gain_negative * f['S_neg']


def trial_figure(trial, title):
    """One reach over ticks (t = tick·dt [s]): position with the goal, signed drive, decoded velocity, error."""
    t = [i * trial['dt'] for i in range(len(trial['joint_position']))]
    fig = Figure(figsize=(8, 9))
    pos, drive, vel, err = fig.subplots(4, 1, sharex=True)
    pos.plot(t, trial['joint_position'], color='C3')
    pos.axhline(trial['q_goal'], color='C2')
    pos.set_title(f'{title}: joint position [rad] (green: goal)')
    drive.plot(t, trial['signed_drive'])
    drive.set_title('signed drive')
    vel.plot(t, trial['decoded_velocity'])
    vel.set_title('decoded velocity [rad/s]')
    err.plot(t, [q - trial['q_goal'] for q in trial['joint_position']])
    err.axhline(0, color='0.6')
    err.set_title('position error q - goal [rad]')
    err.set_xlabel('t [s]')
    fig.tight_layout()
    return fig


def calibration_progress_figure(cal, title):
    """Per calibration trial: |dq_actual − dq_desired| [rad], and the gains and tau the trial ran with."""
    trials, n = cal['trials'], range(1, len(cal['trials']) + 1)
    fig = Figure(figsize=(8, 8))
    err, gains, tau = fig.subplots(3, 1, sharex=True)
    err.plot(n, [abs(t['dq_actual'] - t['dq_desired']) for t in trials], '.')
    err.set_title(f'{title}: |dq_actual - dq_desired| per trial [rad]')
    gains.plot(n, [t['decoder_used']['gain_positive'] for t in trials], label='k_pos')
    gains.plot(n, [t['decoder_used']['gain_negative'] for t in trials], label='k_neg')
    gains.legend()
    gains.set_title('decoder gains used [rad/s per drive]')
    tau.plot(n, [t['decoder_used']['tau'] for t in trials])
    tau.set_title('tau used [s]')
    tau.set_xlabel('trial')
    fig.tight_layout()
    return fig


def decoder_fit_figure(cal, title, refit=None):
    """Predicted Δq of the final decoder (and of `refit`, the same trials refitted, D39) against the desired and the
    actual Δq of every calibration trial; y = x is a perfect decoder."""
    if not cal.get('trials'):
        raise ValueError(f'{title}: calibration has no trials (a refit file? pass its source)')
    fig = Figure(figsize=(10, 4.5))
    want, got = fig.subplots(1, 2)
    for record, label in ((cal, 'final gains'), (refit, 'batch refit')):
        if record is None:
            continue
        gains = (record['decoder_gain_positive'], record['decoder_gain_negative'], record['decoder_tau'],
                 record['decoder_delay_steps'])
        pred = [predicted_dq(t['spike_history'], t['dt'], *gains) for t in cal['trials']]
        want.scatter([t['dq_desired'] for t in cal['trials']], pred, s=10, label=label)
        got.scatter([t['dq_actual'] for t in cal['trials']], pred, s=10, label=label)
    for ax, name in ((want, 'desired'), (got, 'actual')):
        lo, hi = ax.get_xlim()
        ax.plot([lo, hi], [lo, hi], color='0.6')
        ax.set_xlabel(f'{name} dq [rad]')
        ax.legend()
    want.set_ylabel('predicted dq [rad]')
    want.set_title(f'{title}: predicted vs desired dq [rad]')
    got.set_title('predicted vs actual dq [rad]')
    fig.tight_layout()
    return fig


def _column(trial):
    return f"{trial['side']} j{trial['joint']} {'up' if trial['q_goal'] > trial['q_start'] else 'down'}"


def evaluation_error_figure(groups):
    """Relative error of each reach, one column per joint and direction, one marker per group; each group's median
    as a line labelled with the group (reaches with goal = start have no error and are left out); the 0.5 pass bar
    (D28) dashed."""
    columns = sorted({_column(t) for trials in groups.values() for t in trials})
    fig = Figure(figsize=(max(6, len(columns)), 4.5))
    ax = fig.subplots()
    for i, (name, trials) in enumerate(groups.items()):
        errors = [relative_error(t) for t in trials]
        ax.scatter([columns.index(_column(t)) + 0.15 * (i - (len(groups) - 1) / 2) for t in trials], errors,
                   color=f'C{i}', label=f'{name} reaches')
        ax.axhline(statistics.median([e for e in errors if not math.isnan(e)]), color=f'C{i}', label=name)
    ax.axhline(0.5, color='0.4', linestyle='--', label='bar 0.5')
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels(columns, rotation=30)
    ax.set_ylabel('|q_final - goal| / |goal - start|')
    ax.set_title('relative error per reach (line: group median)')
    ax.legend(fontsize='small')
    fig.tight_layout()
    return fig


def plant_vs_decoder_figure(groups):
    """F15's split: moved vs commanded (the plant; on y = x if it follows the command) and commanded vs wanted (the
    decoder), Δq in rad, one colour per group."""
    fig = Figure(figsize=(10, 4.5))
    plant, decoder = fig.subplots(1, 2)
    for i, (name, trials) in enumerate(groups.items()):
        cmd = [commanded(t) for t in trials]
        plant.scatter(cmd, [t['q_final'] - t['q_start'] for t in trials], color=f'C{i}', label=name)
        decoder.scatter([t['q_goal'] - t['q_start'] for t in trials], cmd, color=f'C{i}', label=name)
    for ax, (x, y) in ((plant, ('commanded', 'moved')), (decoder, ('wanted', 'commanded'))):
        lo, hi = ax.get_xlim()
        ax.plot([lo, hi], [lo, hi], color='0.6')
        ax.set_xlabel(f'{x} [rad]')
        ax.set_ylabel(f'{y} [rad]')
        ax.legend()
    plant.set_title('plant: moved vs commanded [rad]')
    decoder.set_title('decoder: commanded vs wanted [rad]')
    fig.tight_layout()
    return fig


def _boxes(ax, groups, title, ylabel):
    names = sorted(groups)
    ax.boxplot([[x for x in groups[n] if not math.isnan(x)] for n in names], labels=names)
    ax.tick_params(axis='x', rotation=30)
    ax.set_title(title)
    ax.set_ylabel(ylabel)


def fixed_boxplot_figure(trials):
    """Fixed goals (D42): relative error of each joint and direction over its seeds; the 0.5 bar (D28) dashed."""
    groups = {}
    for t in trials:
        if t['set'] == 'fixed':
            groups.setdefault(_column(t), []).append(relative_error(t))
    fig = Figure(figsize=(max(6, len(groups)), 4.5))
    ax = fig.subplots()
    _boxes(ax, groups, 'fixed goals: relative error over seeds', '|q_final - goal| / |goal - start|')
    ax.axhline(0.5, color='0.4', linestyle='--')
    fig.tight_layout()
    return fig


def sampled_error_figure(trials):
    """Sampled goals (D42): absolute final error |q_final - goal| [rad] per joint (the paper's Fig. 5 measure)."""
    groups = {}
    for t in trials:
        if t['set'] == 'sampled':
            groups.setdefault(f"{t['side']} j{t['joint']}", []).append(abs(t['q_final'] - t['q_goal']))
    fig = Figure(figsize=(max(6, len(groups)), 4.5))
    ax = fig.subplots()
    _boxes(ax, groups, 'sampled goals: absolute final error [rad]', '|q_final - goal| [rad]')
    fig.tight_layout()
    return fig
