"""Multi-seed evaluation of Kangaroo reaches: protocol, resume, run loop, statistics and verdict (DECISIONS.md D42;
spec docs/kangaroo/design/specs/2026-10-14-kangaroo-multi-seed-evaluation-design.md). Pure: no ROS, NEST or sim.
"""
import math
import statistics

import numpy as np

from .plots import ratios, relative_error

FIXED_SEEDS = range(100, 110)
SAMPLED_SEEDS = range(300, 320)
MIN_DISTANCE = 0.1  # rad: a sampled goal at least this far from the start
JUDGED = {('left', 1), ('left', 2), ('left', 3), ('right', 0), ('right', 2), ('right', 3)}  # the D39 joints
KEY = ('set', 'side', 'joint', 'goal_index', 'seed')


def protocol(setups):
    """The reaches for each JointSetup: fixed goals start + 0.5·(max − start) (goal_index 0) and
    start + 0.5·(min − start) (1), each with FIXED_SEEDS; then len(SAMPLED_SEEDS) goals uniform over the ring range
    (rng seeded 2027 + 10·arm + joint, arm 0 left / 1 right), redrawn while closer than MIN_DISTANCE to the start."""
    reaches = []
    for s in setups:
        start, (lo, hi) = s.start[s.joint], s.limits
        if max(hi - start, start - lo) < MIN_DISTANCE:
            raise ValueError(f'{s.side} j{s.joint}: ring range {s.limits} too narrow to leave the start {start}')
        for index, goal in enumerate((start + 0.5 * (hi - start), start + 0.5 * (lo - start))):
            reaches += [dict(set='fixed', side=s.side, joint=s.joint, goal_index=index, goal=goal, seed=seed)
                        for seed in FIXED_SEEDS]
        rng = np.random.default_rng(2027 + 10 * ('left', 'right').index(s.side) + s.joint)
        for index, seed in enumerate(SAMPLED_SEEDS):
            goal = start
            while abs(goal - start) < MIN_DISTANCE:
                goal = float(rng.uniform(lo, hi))
            reaches.append(dict(set='sampled', side=s.side, joint=s.joint, goal_index=index, goal=goal, seed=seed))
    return reaches


def remaining(reaches, done):
    """The reaches without a saved trial in `done` (matched on set, side, joint, goal_index and seed)."""
    have = {tuple(t[k] for k in KEY) for t in done}
    return [r for r in reaches if tuple(r[k] for k in KEY) not in have]


def _quartiles(xs):
    """[q1, median, q3], inclusive method, NaN left out; one value gives it three times; none gives NaNs."""
    xs = [x for x in xs if not math.isnan(x)]
    if len(xs) < 2:
        return [xs[0]] * 3 if xs else [math.nan] * 3
    return list(statistics.quantiles(xs, n=4, method='inclusive'))


def _right_way(t):
    return (t['q_final'] - t['q_start']) * (t['q_goal'] - t['q_start']) > 0


def _direction(t):
    return f"{t['side']} j{t['joint']} {'up' if t['q_goal'] > t['q_start'] else 'down'}"


def summary(trials):
    """Statistics of saved reaches (spec 'Summaries'): fixed set per joint and direction, sampled set per joint."""
    fixed, sampled = {}, {}
    for t in trials:
        if t['set'] == 'fixed':
            fixed.setdefault(_direction(t), []).append(t)
        else:
            sampled.setdefault(f"{t['side']} j{t['joint']}", []).append(t)
    out = dict(fixed={}, sampled={})
    for name, ts in fixed.items():
        q1, median, q3 = _quartiles([relative_error(t) for t in ts])
        out['fixed'][name] = dict(n=len(ts), median=median, iqr=q3 - q1, right_way=sum(map(_right_way, ts)) / len(ts),
                                  commanded_over_wanted=_quartiles([ratios(t)[1] for t in ts]),
                                  moved_over_commanded=_quartiles([ratios(t)[0] for t in ts])[1])
    for name, ts in sampled.items():
        out['sampled'][name] = dict(n=len(ts), abs_error=_quartiles([abs(t['q_final'] - t['q_goal']) for t in ts]),
                                    median_relative=_quartiles([relative_error(t) for t in ts])[1],
                                    right_way=sum(map(_right_way, ts)) / len(ts))
    return out


def verdict(trials, judged=JUDGED):
    """The spec's pass bars on the `judged` joints: (1) >= 10 of their 12 fixed joint-direction medians < 0.5;
    (2) sampled overall median relative error < 0.35 and right-way fraction >= 0.9."""
    mine = [t for t in trials if (t['side'], t['joint']) in judged]
    groups = summary(mine)['fixed']
    below = sum(g['median'] < 0.5 for g in groups.values())
    complete = len(groups) == 2 * len(judged)
    sampled = [t for t in mine if t['set'] == 'sampled']
    median = _quartiles([relative_error(t) for t in sampled])[1]
    right_way = sum(map(_right_way, sampled)) / len(sampled) if sampled else math.nan
    fixed = dict(below=below, groups=len(groups), bar='>= 10 of 12 medians < 0.5', passed=complete and below >= 10,
                 note='' if complete else f'incomplete: {len(groups)} of {2 * len(judged)} groups')
    samp = dict(median_relative=median, right_way=right_way, bar='median < 0.35 and right way >= 0.9',
                passed=median < 0.35 and right_way >= 0.9)
    return dict(fixed=fixed, sampled=samp, passed=fixed['passed'] and samp['passed'])


def run(reaches, do):
    """Call do(reach) for each reach; a reach that raises is returned with its error and the others still run."""
    failed = []
    for r in reaches:
        try:
            do(r)
        except Exception as e:  # keep evaluating; the caller records it in the manifest
            failed.append(dict(r, error=repr(e)))
    return failed
