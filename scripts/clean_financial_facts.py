import argparse
from pathlib import Path
from typing import List, Optional

import pandas as pd

DEFAULT_INPUT = Path("data/csv-data/financial_facts.csv")
DEFAULT_OUTPUT = Path("data/csv-data/financial_facts_cleaned.csv")
DEFAULT_YEARS = ["2023", "2024"]


def normalize_dataframe(
    df: pd.DataFrame, 
    required_years: List[str],
    concepts: Optional[List[str]] = None
) -> pd.DataFrame:
    base_columns = ["entity", "company_name", "concept"]
    required_columns = base_columns + required_years
    
    missing_base = [col for col in base_columns if col not in df.columns]
    if missing_base:
        raise ValueError(
            f"Input file is missing required base columns: {', '.join(missing_base)}"
        )
    
    missing_years = [year for year in required_years if year not in df.columns]
    if missing_years:
        raise ValueError(
            f"Input file is missing required year columns: {', '.join(missing_years)}. "
            f"Available columns: {', '.join(df.columns.tolist())}"
        )

    df = df[required_columns].copy()
    df = df.replace({"": pd.NA, " ": pd.NA})
    
    # Filter by concepts if specified
    if concepts:
        df = df[df["concept"].isin(concepts)]

    for year in required_years:
        df[year] = pd.to_numeric(df[year], errors="coerce")

    return df


def filter_complete_entities(
    df: pd.DataFrame, 
    required_years: List[str],
    required_concepts: Optional[List[str]] = None
) -> pd.DataFrame:
    def entity_is_complete(group: pd.DataFrame) -> bool:
        # Check if entity has all required concepts
        if required_concepts:
            entity_concepts = set(group["concept"].unique())
            required_concepts_set = set(required_concepts)
            if not required_concepts_set.issubset(entity_concepts):
                # Entity is missing at least one required concept
                return False
        
        # Check if all required year columns have non-null values for all concepts
        year_cols = [col for col in required_years if col in group.columns]
        if not year_cols:
            return False
        if group[year_cols].isna().any().any():
            # At least one year value is missing for at least one concept
            return False
        return True

    return df.groupby("entity", group_keys=False).filter(entity_is_complete)


def clean_financial_facts(
    input_path: Path, 
    output_path: Path, 
    years: List[str],
    concepts: Optional[List[str]] = None
) -> None:
    df = pd.read_csv(input_path)
    df = normalize_dataframe(df, required_years=years, concepts=concepts)
    df = filter_complete_entities(df, required_years=years, required_concepts=concepts)

    year_str = ", ".join(years)
    concept_str = f" for concepts: {', '.join(concepts)}" if concepts else ""
    
    if df.empty:
        print(f"No entities with complete {year_str} data{concept_str} were found.")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Cleaned financial facts saved to: {output_path}")
    print(f"Years included: {', '.join(years)}")
    if concepts:
        print(f"Concepts included: {', '.join(concepts)}")
    print(f"Entities retained: {df['entity'].nunique()}")
    print(f"Rows written: {len(df)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter financial_facts.csv to keep only specified years and concepts "
            "for entities with complete values. If any specified concept is missing "
            "for an entity, all data from that entity is removed."
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
    parser.add_argument(
        "--years",
        nargs="+",
        default=DEFAULT_YEARS,
        help=f"Years to include (default: {' '.join(DEFAULT_YEARS)}). Example: --years 2023 2024 2025",
    )
    parser.add_argument(
        "--concepts",
        nargs="+",
        default=None,
        help="Concepts to include. If not specified, all concepts are included. If any concept is missing for an entity, all data from that entity is removed. Example: --concepts Equity Revenue Assets",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    clean_financial_facts(
        input_path, 
        output_path, 
        years=args.years,
        concepts=args.concepts
    )


if __name__ == "__main__":
    main()

