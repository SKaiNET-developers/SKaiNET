"""Regression check for #1418: the operator docs generator must emit Markdown.

Static check on the Gradle task source (build-logic has no Kotlin test source set and
Gradle cannot resolve its plugins offline here). It fails while `generateMarkdown` still
throws NotImplementedError.

Usage: python scripts/test_docs_markdown_stub.py [path/to/GenerateDocumentationTask.kt]
"""
import sys
import unittest
from pathlib import Path

DEFAULT = Path(__file__).resolve().parent.parent / "build-logic/convention/src/main/kotlin/GenerateDocumentationTask.kt"


def generate_markdown_body(source: str) -> str:
    start = source.index("private fun generateMarkdown(")
    end = source.index("\n    private fun ", start + 1)
    return source[start:end]


SOURCE = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT


class MarkdownGeneratorTest(unittest.TestCase):
    def test_generate_markdown_is_implemented(self):
        source = SOURCE.read_text(encoding="utf-8")
        body = generate_markdown_body(source)
        stubbed = [line.strip() for line in body.splitlines() if "NotImplementedError" in line]
        self.assertFalse(stubbed, "generateMarkdown still stubbed: " + " | ".join(stubbed))
        self.assertIn("index.md", body)
        self.assertIn('"ops-status-matrix.md"', source)


if __name__ == "__main__":
    # Drop the optional path argument so unittest does not treat it as a test name.
    sys.argv = sys.argv[:1]
    unittest.main(verbosity=2)
