"""Fetch the AI-GA dataset (Anagnostou et al.) -- 28,662 real, human-labeled
title/abstract pairs, half original and half AI-generated -- for training a
genuine AI-text classifier instead of the small hand-written demo CSV.

Source: https://github.com/panagiotisanagnostou/AI-GA
Paper: Anagnostou et al., "Bioinformatics and biomedical informatics with
       ChatGPT: Year one review" (dataset released alongside related work)
"""
from __future__ import annotations
import argparse
import urllib.request
from pathlib import Path

DATASET_URL = "https://raw.githubusercontent.com/panagiotisanagnostou/AI-GA/main/ai-ga-dataset.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/ai_ga_dataset.csv"))
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {DATASET_URL} ...")
    urllib.request.urlretrieve(DATASET_URL, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
