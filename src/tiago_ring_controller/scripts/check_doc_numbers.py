#!/usr/bin/env python3
"""Check that the numbers quoted in the summaries are the numbers in the generated tables.

    python3 -B scripts/check_doc_numbers.py [--repo PATH]

The summary sections of ``README.md`` (section 1), ``docs/results.md`` and
``docs/stress-test.md`` quote results.  Every decimal number and every
percentage in them must occur in one of the generated files
(``docs/results/tables.md``, ``docs/results/paper/numbers.md``,
``docs/results/stress/tables.md``), which ``scripts/results_report.py`` and
``scripts/stress_test.py report`` write from the data.  A number that does not
is stale or was typed by hand: the script lists it and exits with 1.
``test/test_doc_numbers.py`` runs this, so a stale summary fails the suite.
"""

import argparse
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
GENERATED = ("docs/results/tables.md", "docs/results/paper/numbers.md", "docs/results/stress/tables.md")
#: document -> (heading that starts the checked part, heading prefix that ends it)
SUMMARIES = {"README.md": ("## 1. The versions", "## 2."), "docs/results.md": ("## Summary", "## 1."),
             "docs/stress-test.md": ("## Summary", "## What a trial is")}
#: numbers that are definitions, not results (tolerances, dates, versions)
ALLOWED = {"0.05"}
NUMBER = re.compile(r"(?<![\w.])(\d+\.\d+|\d+%)(?![\w.]*\d)")


def find_repo(start=HERE):
    """The repository root: the folder holding ``docs/results``; ``/model_docs`` in the batch container."""

    path = start
    for _ in range(6):
        if os.path.isdir(os.path.join(path, "docs", "results")):
            return path
        path = os.path.dirname(path)
    return "/model_repo_docs" if os.path.isdir("/model_repo_docs/docs/results") else None


def section(text, start, end):
    if start not in text:
        return None
    body = text[text.index(start) + len(start):]
    for line_start in ("\n" + end,):
        if line_start in body:
            body = body[:body.index(line_start)]
    return body


def numbers_in(text):
    return [m.group(1) for m in NUMBER.finditer(text)]


def check(repo):
    generated = ""
    for relative in GENERATED:
        path = os.path.join(repo, relative)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as stream:
                generated += stream.read() + "\n"
    known = set(numbers_in(generated)) | ALLOWED
    problems, checked = [], 0
    for relative, (start, end) in SUMMARIES.items():
        path = os.path.join(repo, relative)
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as stream:
            body = section(stream.read(), start, end)
        if body is None:
            problems.append("%s: section %r not found" % (relative, start))
            continue
        for number in numbers_in(body):
            checked += 1
            if number not in known:
                problems.append("%s: %s is not in the generated tables" % (relative, number))
    return problems, checked


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default=None, help="repository root (default: found from this file)")
    args = parser.parse_args(argv)
    repo = args.repo or find_repo()
    if repo is None:
        print("no docs/results found: nothing to check")
        return 0
    problems, checked = check(repo)
    for problem in problems:
        print(problem)
    print("%d numbers checked in %s, %d not found" % (checked, repo, len(problems)))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
