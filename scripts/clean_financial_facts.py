import argparse
from pathlib import Path

import pandas as pd

DEFAULT_INPUT = Path("data/csv-data/financial_facts.csv")
DEFAULT_OUTPUT = Path("data/csv-data/financial_facts_cleaned.csv")
REQUIRED_COLUMNS = ["entity", "company_name", "concept", "2023", "2024"]


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    present_cols = [col for col in REQUIRED_COLUMNS if col in df.columns]
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]

    if missing_cols:
        raise ValueError(
            f"Input file is missing required columns: {', '.join(missing_cols)}"
        )

    df = df[present_cols].copy()
    df = df.replace({"": pd.NA, " ": pd.NA})

    for col in ["2023", "2024"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def filter_complete_entities(df: pd.DataFrame) -> pd.DataFrame:
    def entity_is_complete(group: pd.DataFrame) -> bool:
        if group.isna().any().any():
            return False
        return True

    return df.groupby("entity", group_keys=False).filter(entity_is_complete)


def clean_financial_facts(input_path: Path, output_path: Path) -> None:
    df = pd.read_csv(input_path)
    df = normalize_dataframe(df)
    df = filter_complete_entities(df)

    if df.empty:
        print("No entities with complete 2023 and 2024 data were found.")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Cleaned financial facts saved to: {output_path}")
    print(f"Entities retained: {df['entity'].nunique()}")
    print(f"Rows written: {len(df)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter financial_facts.csv to keep only 2023 and 2024 data "
            "for entities with complete values."
        )
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help=f"Path to the input CSV file (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help=f"Path to write the cleaned CSV (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    clean_financial_facts(input_path, output_path)


if __name__ == "__main__":
    main()

