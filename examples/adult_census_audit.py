"""End-to-end EquiML audit on the UCI Adult dataset.

Run:
    python examples/adult_census_audit.py

Downloads ~32k rows from the UCI ML repository, runs `equiml audit` with
`sex` and `race` as sensitive features, and writes the artefacts:

    examples/adult_audit.json   metrics + _meta block
    examples/adult_audit.html   side-by-side baseline-vs-fair report

Tries OpenML via scikit-learn first (no extra dep) and falls back to the
UCI CSV mirror over plain HTTPS if OpenML is unreachable. Re-runs reuse
the cached CSV at examples/adult.csv.
"""

import subprocess
import sys
from pathlib import Path

import pandas as pd


ADULT_COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country",
    "income",
]
ADULT_KEEP = [
    "age", "workclass", "education-num", "marital-status", "occupation",
    "race", "sex", "hours-per-week", "income",
]
UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "adult/adult.data"
)


def _fetch_via_openml() -> pd.DataFrame:
    from sklearn.datasets import fetch_openml
    bunch = fetch_openml("adult", version=2, as_frame=True)
    df = bunch.frame.rename(columns={"class": "income"})
    return df[ADULT_KEEP].dropna()


def _fetch_via_uci() -> pd.DataFrame:
    df = pd.read_csv(
        UCI_URL,
        header=None,
        names=ADULT_COLUMNS,
        skipinitialspace=True,
        na_values="?",
    )
    return df[ADULT_KEEP].dropna()


def fetch_adult() -> pd.DataFrame:
    try:
        return _fetch_via_openml()
    except Exception as openml_err:
        print(f"  OpenML fetch failed ({openml_err.__class__.__name__}); "
              "falling back to UCI mirror.")
        return _fetch_via_uci()


def main() -> None:
    out_dir = Path(__file__).parent
    csv_path = out_dir / "adult.csv"
    json_path = out_dir / "adult_audit.json"
    html_path = out_dir / "adult_audit.html"

    if not csv_path.exists():
        print("Fetching UCI Adult dataset...")
        df = fetch_adult()
        df.to_csv(csv_path, index=False)
        print(f"  wrote {csv_path} ({len(df)} rows)")

    print("Running equiml audit...")
    result = subprocess.run(
        [sys.executable, "-m", "equiml.cli", "audit", str(csv_path),
         "--target", "income",
         "--sensitive", "sex", "race",
         "--output", str(json_path),
         "--report", str(html_path)],
        check=False,
    )
    if result.returncode != 0:
        print("equiml audit failed", file=sys.stderr)
        sys.exit(result.returncode)

    print(f"\nDone. JSON: {json_path}, HTML: {html_path}")


if __name__ == "__main__":
    main()
