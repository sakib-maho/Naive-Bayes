"""Simple multinomial Naive Bayes text classifier."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9']+", text.lower())


class NaiveBayesClassifier:
    def __init__(self) -> None:
        self.class_counts: Counter[str] = Counter()
        self.word_counts: dict[str, Counter[str]] = defaultdict(Counter)
        self.total_words: Counter[str] = Counter()
        self.vocabulary: set[str] = set()

    def fit(self, texts: list[str], labels: list[str]) -> None:
        for text, label in zip(texts, labels):
            self.class_counts[label] += 1
            tokens = tokenize(text)
            self.word_counts[label].update(tokens)
            self.total_words[label] += len(tokens)
            self.vocabulary.update(tokens)

    def predict(self, text: str) -> str:
        tokens = tokenize(text)
        total_docs = sum(self.class_counts.values())
        best_label = ""
        best_score = float("-inf")
        vocab_size = max(1, len(self.vocabulary))

        for label, label_count in self.class_counts.items():
            log_prob = math.log(label_count / total_docs)
            for token in tokens:
                token_count = self.word_counts[label][token]
                prob = (token_count + 1) / (self.total_words[label] + vocab_size)
                log_prob += math.log(prob)
            if log_prob > best_score:
                best_score = log_prob
                best_label = label
        return best_label
