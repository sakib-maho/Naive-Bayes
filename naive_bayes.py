"""Backward-compatible entrypoint for Naive Bayes CLI."""

from cli import main


if __name__ == "__main__":
    raise SystemExit(main())
