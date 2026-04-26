"""End-to-end EquiML audit on the UCI Adult dataset.

Run:
    python examples/adult_census_audit.py

This downloads ~32k rows from OpenML, runs `equiml audit` programmatically
with `sex` as the sensitive feature, and prints metrics. Output:
- examples/adult_audit.json (metrics)
- examples/adult_audit.html (report)
"""

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml


def main():
    out_dir = Path(__file__).parent
    csv_path = out_dir / "adult.csv"
    json_path = out_dir / "adult_audit.json"
    html_path = out_dir / "adult_audit.html"

    if not csv_path.exists():
        print("Fetching UCI Adult dataset from OpenML...")
        bunch = fetch_openml("adult", version=2, as_frame=True)
        df = bunch.frame
        # Keep a manageable subset of columns and the target.
        df = df[["age", "workclass", "education-num", "marital-status",
                 "occupation", "race", "sex", "hours-per-week", "class"]]
        df = df.rename(columns={"class": "income"})
        df = df.dropna()
        df.to_csv(csv_path, index=False)
        print(f"  wrote {csv_path} ({len(df)} rows)")

    print("Running equiml audit...")
    result = subprocess.run(
        ["equiml", "audit", str(csv_path),
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
