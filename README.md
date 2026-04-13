# Naive Bayes Showcase

<!-- BrandCloud:readme-standard -->
[![Maintained](https://img.shields.io/badge/Maintained-yes-brightgreen.svg)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Showcase](https://img.shields.io/badge/Portfolio-Showcase-blue.svg)](#)

_Part of the `sakib-maho` project showcase series with consistent documentation and quality standards._

This repository has been upgraded into a reproducible Naive Bayes mini-project.
The original notebook remains available, and the repo now includes scriptable training + prediction via CLI.

## Features

- Lightweight multinomial Naive Bayes classifier
- CSV training data loader (`text`, `label`)
- CLI for classifying new text input
- Unit tests for model behavior and CLI workflow

## Quick Start

```bash
python3 cli.py --data data/sample_text.csv --text "team won the final match"
```

## Run Tests

```bash
python3 -m unittest discover -s tests -p "test_*.py"
```

## License

MIT License. See `LICENSE`.
