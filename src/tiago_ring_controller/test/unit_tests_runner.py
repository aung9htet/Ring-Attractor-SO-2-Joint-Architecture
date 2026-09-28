#!/usr/bin/env python3
"""Catkin/rosunit bridge for the standard-library unittest suite."""

import sys
import unittest
from pathlib import Path

import rosunit


TEST_DIR = Path(__file__).resolve().parent


def main():
    result_file = None
    for argument in sys.argv[1:]:
        if argument.startswith(rosunit.XML_OUTPUT_FLAG):
            result_file = argument[len(rosunit.XML_OUTPUT_FLAG):]
            break

    suite = unittest.defaultTestLoader.discover(
        str(TEST_DIR), pattern="test_*.py"
    )
    runner = rosunit.create_xml_runner(
        "tiago_ring_controller",
        "unittest-python",
        result_file,
    )
    result = runner.run(suite)
    rosunit.print_unittest_summary(result)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
