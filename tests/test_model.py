from subprocess import run
import unittest

from naive_bayes.model import NaiveBayesClassifier


class NaiveBayesTests(unittest.TestCase):
    def test_prediction(self) -> None:
        model = NaiveBayesClassifier()
        texts = ["goal match team", "python release package"]
        labels = ["sports", "tech"]
        model.fit(texts, labels)
        self.assertEqual(model.predict("team won the match"), "sports")

    def test_cli(self) -> None:
        result = run(
            [
                "python3",
                "cli.py",
                "--data",
                "data/sample_text.csv",
                "--text",
                "python deploy update",
            ],
            check=True,
            text=True,
            capture_output=True,
        )
        self.assertIn(result.stdout.strip(), {"sports", "tech"})


if __name__ == "__main__":
    unittest.main()
