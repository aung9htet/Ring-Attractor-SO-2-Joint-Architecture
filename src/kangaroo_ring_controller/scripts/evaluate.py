"""Multi-seed evaluation of one Kangaroo arm (D42; spec docs/kangaroo/design/specs/2026-10-14-kangaroo-multi-seed-
evaluation-design.md). Each reach is saved as soon as it ends, so a crash loses at most one; --resume continues.

    python3 src/kangaroo_ring_controller/scripts/evaluate.py <left|right> [--name eval] [--resume DIR] [--dry-run]
Run one process per arm in parallel. --dry-run prints the protocol (one JSON reach per line) without ROS or NEST.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the kangaroo_ring_controller package
from kangaroo_ring_controller import new_run_dir, save_trial  # noqa: E402
from kangaroo_ring_controller.evaluation import protocol, remaining, run  # noqa: E402
from kangaroo_ring_controller.joint_setup import LIMITS, newest_setup  # noqa: E402


def git_head():
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True,
                              cwd=Path(__file__).parent).stdout.strip() or 'unknown'
    except OSError:
        return 'unknown'


def main():
    parser = argparse.ArgumentParser(description='Multi-seed evaluation of one Kangaroo arm.')
    parser.add_argument('side', choices=('left', 'right'))
    parser.add_argument('--name', default='eval')
    parser.add_argument('--resume', help='an evaluation directory to continue')
    parser.add_argument('--dry-run', action='store_true', help='print the protocol and exit (no ROS, no NEST)')
    args = parser.parse_args()
    setups = [newest_setup(args.side, joint) for joint in range(len(LIMITS))]
    reaches = protocol(setups)
    if args.resume:
        out = Path(args.resume)
        manifest = json.loads((out / 'manifest.json').read_text())
        for i, (old, new) in enumerate(zip(manifest['protocol'], reaches)):
            if old != new:
                sys.exit(f'{out}: reach {i} differs: saved {old}, now {new}')
        if len(manifest['protocol']) != len(reaches):
            sys.exit(f"{out}: {len(manifest['protocol'])} reaches saved, {len(reaches)} now")
    if args.dry_run:
        for r in reaches:
            print(json.dumps(r))
        print(f'{len(reaches)} reaches')
        return

    import rclpy
    from kangaroo_ring_controller.arm import connect
    from kangaroo_ring_controller.ring import run_trial
    if not args.resume:
        out = new_run_dir('evaluation', f'{args.name}_{args.side}')
        manifest = dict(protocol=reaches, calibrations={s.joint: s.calibration for s in setups}, git_head=git_head(),
                        seeds=dict(fixed='100-109', sampled='300-319', goals='default_rng(2027 + 10*arm + joint)'),
                        failed=[])
        (out / 'manifest.json').write_text(json.dumps(manifest, indent=1))  # before the first reach: resumable
    done = [json.loads(p.read_text()) for p in (out / 'trials').glob('*/trial.json')]
    todo = remaining(reaches, done)
    node, arms = connect(f'evaluate_{args.side}', [args.side])
    arm = arms[args.side]

    def do(r):
        setup = setups[r['joint']]
        arm.reset(setup.start)
        path, trial = save_trial('trials', setup, run_trial(arm, setup, r['goal'], r['seed']), out,
                                 set=r['set'], goal_index=r['goal_index'])
        print(f"{r['set']} j{r['joint']} goal {r['goal']:+.3f} seed {r['seed']}: q_final {trial['q_final']:+.3f} "
              f"({trial['stop_reason']}) -> {path.parent.name}", flush=True)

    manifest['failed'] = run(todo, do)  # this session's failures; --resume retries them
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=1))
    rclpy.shutdown()
    print(f"{len(todo) - len(manifest['failed'])} of {len(todo)} reaches saved -> {out}")


if __name__ == '__main__':
    main()
