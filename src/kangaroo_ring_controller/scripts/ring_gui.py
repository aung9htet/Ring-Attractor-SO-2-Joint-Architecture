"""Browser GUI: drive one Kangaroo arm joint with the ring attractor (DECISIONS.md D27).

    python3 src/kangaroo_ring_controller/scripts/ring_gui.py [--port 8765]   # then open http://localhost:8765/
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the kangaroo_ring_controller package
import rclpy  # noqa: E402
from kangaroo_ring_controller.arm import connect  # noqa: E402
from kangaroo_ring_controller.gui import Gui, make_server, process  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description='Browser GUI: ring-attractor reaches on Kangaroo, one joint at a time.')
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    node, arms = connect('ring_gui')
    gui = Gui()
    make_server(gui, args.port)
    print(f'open http://localhost:{args.port}/  (Ctrl-C quits)', flush=True)
    try:
        while True:
            process(arms, gui)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
