"""The numbers in the summaries of README.md, docs/results.md and docs/stress-test.md are the generated ones."""

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import check_doc_numbers  # noqa: E402


class DocNumberTests(unittest.TestCase):
    def test_extraction_and_detection(self):
        self.assertEqual(check_doc_numbers.numbers_in("error 0.008 / 0.022 rad, 99% settled, 56 of 56, v1.2.3, 2026-10-04, section 6.1 no"),
                         ["0.008", "0.022", "99%", "6.1"])
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            (repo / "docs" / "results").mkdir(parents=True)
            (repo / "docs" / "results" / "tables.md").write_text("| v4 | 0.008 / 0.022 / 0.045 | 100% |\n")
            (repo / "docs" / "results.md").write_text("# R\n\n## Summary\n\nv4 ends 0.008 rad off (p90 0.022), 100% settled, PID 0.019.\n\n## 1. What\n\n0.777\n")
            problems, checked = check_doc_numbers.check(str(repo))
        self.assertEqual(checked, 4)
        self.assertEqual(problems, ["docs/results.md: 0.019 is not in the generated tables"])

    def test_the_repository_summaries_quote_generated_numbers(self):
        repo = check_doc_numbers.find_repo()
        if repo is None:
            self.skipTest("the repository's docs are not next to the package")
        problems, checked = check_doc_numbers.check(repo)
        self.assertGreater(checked, 10)
        self.assertEqual(problems, [])


if __name__ == "__main__":
    unittest.main()
