import argparse
from pathlib import Path
from typing import List

import pandas as pd

DEFAULT_INPUT = Path("data/csv-data/ai_keywords_ixbrl.csv")
DEFAULT_OUTPUT = Path("data/csv-data/ai_keywords_ixbrl_cleaned.csv")


def load_ai_keywords(input_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    return df


def clean_ai_keywords(df: pd.DataFrame, required_years: List[str]) -> pd.DataFrame:
    base_columns = ["entity", "company_name"]
    
    year_columns = [year for year in required_years if year in df.columns]
    if not year_columns:
        raise ValueError(
            f"None of the required years ({', '.join(required_years)}) found in the input file. "
            f"Available year columns: {', '.join([col for col in df.columns if col.isdigit()])}"
        )
    
    missing_years = [year for year in required_years if year not in df.columns]
    if missing_years:
        print(f"[warn] Missing year columns: {', '.join(missing_years)}")
    
    top_keywords_columns = [f"top_keywords_{year}" for year in year_columns if f"top_keywords_{year}" in df.columns]
    
    required_columns = base_columns + year_columns + top_keywords_columns
    
    df_cleaned = df[required_columns].copy()
    
    df_cleaned = df_cleaned.replace({"": pd.NA, " ": pd.NA})
    
    for year in year_columns:
        df_cleaned[year] = pd.to_numeric(df_cleaned[year], errors="coerce")
    
    df_cleaned = df_cleaned.dropna(subset=year_columns)
    
    return df_cleaned


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Clean ai_keywords_ixbrl.csv by filtering to specific years "
            "and removing entities with missing data for any year."
        )
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help=f"Path to the input AI keywords CSV file (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help=f"Path to write the cleaned CSV file (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--years",
        nargs="+",
        required=True,
        help="Years to include. Example: --years 2023 2024 2025",
    )

    args = parser.parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"Loading AI keywords from {input_path}...")
    df = load_ai_keywords(input_path)
    print(f"Input rows: {len(df)}")

    print(f"Cleaning data for years: {', '.join(args.years)}...")
    df_cleaned = clean_ai_keywords(df, args.years)

    if df_cleaned.empty:
        print("No data remaining after cleaning.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_cleaned.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"\nCleaned AI keywords saved to: {output_path}")
    print(f"Years included: {', '.join(args.years)}")
    print(f"Entities retained: {df_cleaned['entity'].nunique()}")
    print(f"Rows written: {len(df_cleaned)}")
    print(f"Columns: {len(df_cleaned.columns)}")


if __name__ == "__main__":
    main()

