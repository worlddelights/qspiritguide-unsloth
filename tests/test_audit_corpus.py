import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from audit_corpus import (
    Article,
    analyze_article,
    analyze_pair,
    article_pairs,
    exact_duplicates,
    load_articles,
    near_duplicates,
    normalize_text,
    parse_json_object,
    scan_json_escape_hazards,
    tokenize,
    vectorize,
)


def article(name: str, text: str) -> Article:
    from audit_corpus import sha256_text

    return Article(name, text, sha256_text(text))


class AuditCorpusTests(unittest.TestCase):
    def test_parser_repairs_invalid_json_backslash_escapes(self):
        result = parse_json_object(r'{"issue":"bad \q path and \u12xz"}')
        self.assertEqual(result["issue"], r"bad \q path and \u12xz")

    def test_preflight_finds_suspicious_source_backslashes(self):
        source = article("math.md", r"Use \(x + y\), but retain \\ as a literal slash.")

        findings = scan_json_escape_hazards([source])

        self.assertEqual([finding["sequence"] for finding in findings], [r"\(", r"\)"])

    def test_semantic_analysis_uses_model_flags_and_evidence(self):
        responses = [
            '{"concepts":[{"name":"active listening","defined_here":true,'
            '"importance":3,"evidence":"Listening with attention."}],'
            '"topics":["communication"]}',
            '{"conflicts":[{"issue":"Different definitions",'
            '"left_evidence":"Definition A","right_evidence":"Definition B",'
            '"confidence":"medium"}]}',
        ]

        class FakeClient:
            def __init__(self):
                self.chat = SimpleNamespace(completions=self)
                self.response_formats = []

            def create(self, **kwargs):
                self.response_formats.append(kwargs["response_format"])
                content = responses.pop(0)
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
                )

        client = FakeClient()
        left = article("left.md", "Listening with attention.")
        right = article("right.md", "Listening without attention.")

        article_result = analyze_article(client, "test-model", left)
        pair_result = analyze_pair(client, "test-model", left, right, 0.3)

        self.assertTrue(article_result["concepts"][0]["defined_here"])
        self.assertEqual(article_result["topics"], ["communication"])
        self.assertEqual(pair_result["conflicts"][0]["confidence"], "medium")
        self.assertEqual(client.response_formats, [{"type": "text"}, {"type": "text"}])

    def test_semantic_analysis_retries_malformed_json_once(self):
        responses = [
            '{"concepts":[] "topics":["history"]}',
            '{"concepts":[],"topics":["history"]}',
        ]

        class FakeClient:
            def __init__(self):
                self.chat = SimpleNamespace(completions=self)
                self.response_formats = []

            def create(self, **kwargs):
                self.response_formats.append(kwargs["response_format"])
                content = responses.pop(0)
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
                )

        client = FakeClient()
        result = analyze_article(
            client, "test-model", article("history.md", "An article about history.")
        )

        self.assertEqual(result["topics"], ["history"])
        self.assertEqual(client.response_formats, [{"type": "text"}, {"type": "text"}])

    def test_normalize_text_ignores_heading_and_whitespace(self):
        self.assertEqual(
            normalize_text("# Shared idea\n\nTwo   useful terms."),
            normalize_text("Shared idea Two useful terms."),
        )

    def test_exact_duplicates_and_near_duplicates_are_separate(self):
        articles = [
            article("first.md", "# Topic\n\nA careful definition of active listening."),
            article("copy.md", "Topic A careful definition of active listening."),
            article(
                "related.md",
                "Active listening is a useful communication practice for careful discussion.",
            ),
        ]
        exact = exact_duplicates(articles)
        pairs = article_pairs(articles, vectorize(articles), 0.0)
        near = near_duplicates(pairs, exact, 0.15)

        self.assertEqual(exact, [["copy.md", "first.md"]])
        self.assertTrue(any(pair["articles"] == ["first.md", "related.md"] for pair in near))
        self.assertFalse(any(pair["articles"] == ["copy.md", "first.md"] for pair in near))

    def test_tokenize_omits_common_words_and_markdown_code(self):
        tokens = tokenize("The practice is useful. ```ignore_this```")
        self.assertIn("practice", tokens)
        self.assertNotIn("the", tokens)
        self.assertNotIn("ignore_this", tokens)

    def test_load_articles_is_sorted_and_reads_only_markdown(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            (directory / "z.md").write_text("Last article", encoding="utf-8")
            (directory / "a.md").write_text("First article", encoding="utf-8")
            (directory / "ignored.txt").write_text("Not an article", encoding="utf-8")

            loaded = load_articles(directory)

        self.assertEqual([item.name for item in loaded], ["a.md", "z.md"])
        self.assertEqual(loaded[0].text, "First article")


if __name__ == "__main__":
    unittest.main()
