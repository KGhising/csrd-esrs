import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

DEFAULT_INPUT = Path("data/csv-data/financial_facts_cleaned.csv")
DEFAULT_OUTPUT = Path("data/csv-data/financial_ratios.csv")


def load_financial_facts(input_path: Path) -> pd.DataFrame:
    """Load financial facts CSV."""
    df = pd.read_csv(input_path)
    return df


def pivot_concepts_to_columns(df: pd.DataFrame, year_columns: list) -> pd.DataFrame:
    """
    Pivot the financial facts DataFrame so concepts become columns for each year.
    Expected input format: entity, company_name, concept, 2023, 2024, ...
    Output format: entity, company_name, year, ProfitLoss, Revenue, Assets, Equity, ...
    """
    # Get all year columns that exist in the dataframe
    available_years = [col for col in year_columns if col in df.columns]
    
    if not available_years:
        return pd.DataFrame()
    
    # Melt the dataframe to get year as a column
    id_cols = ["entity", "company_name", "concept"]
    value_cols = available_years
    
    # Melt to long format: entity, company_name, concept, year, value
    melted = df.melt(
        id_vars=id_cols,
        value_vars=value_cols,
        var_name="year",
        value_name="value"
    )
    
    # Pivot concepts to columns: entity, company_name, year, ProfitLoss, Revenue, Assets, ...
    pivoted = melted.pivot_table(
        index=["entity", "company_name", "year"],
        columns="concept",
        values="value",
        aggfunc="first"
    ).reset_index()
    
    return pivoted


def calculate_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate financial ratios from pivoted financial facts.
    Assumes cleaned data with standardized concept names.
    """
    # Initialize ratio columns
    ratios_df = df[["entity", "company_name", "year"]].copy()
    
    # Required concepts (must match exactly what's in the cleaned CSV)
    required_concepts = ["ProfitLoss", "Revenue", "Assets", "Equity", "CashFlowsFromUsedInOperatingActivities"]
    
    # Check for missing concepts
    missing_concepts = [concept for concept in required_concepts if concept not in df.columns]
    if missing_concepts:
        print(f"[warn] Missing required concepts in data: {', '.join(missing_concepts)}")
        # Add missing concepts as NaN columns
        for concept in missing_concepts:
            df[concept] = np.nan
    
    # Convert to numeric, handling non-numeric values
    for concept in required_concepts:
        df[concept] = pd.to_numeric(df[concept], errors="coerce")
    
    # 1. Profit margin = ProfitLoss / Revenue
    ratios_df["profit_margin"] = np.where(
        df["Revenue"] != 0,
        df["ProfitLoss"] / df["Revenue"],
        np.nan
    )
    
    # 2. Return on assets = ProfitLoss / Assets
    ratios_df["return_on_assets"] = np.where(
        df["Assets"] != 0,
        df["ProfitLoss"] / df["Assets"],
        np.nan
    )
    
    # 3. Return on Equity = ProfitLoss / Equity
    ratios_df["return_on_equity"] = np.where(
        df["Equity"] != 0,
        df["ProfitLoss"] / df["Equity"],
        np.nan
    )
    
    # 4. Assets Turnover = Revenue / Assets
    ratios_df["assets_turnover"] = np.where(
        df["Assets"] != 0,
        df["Revenue"] / df["Assets"],
        np.nan
    )
    
    # 5. Cash flow to assets = CashFlowsFromUsedInOperatingActivities / Assets
    ratios_df["cash_flow_to_assets"] = np.where(
        df["Assets"] != 0,
        df["CashFlowsFromUsedInOperatingActivities"] / df["Assets"],
        np.nan
    )
    
    # 6. Cash flow to Equity = CashFlowsFromUsedInOperatingActivities / Equity
    ratios_df["cash_flow_to_equity"] = np.where(
        df["Equity"] != 0,
        df["CashFlowsFromUsedInOperatingActivities"] / df["Equity"],
        np.nan
    )
    
    return ratios_df


def pivot_ratios_by_year(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot ratios DataFrame so years become columns.
    Input: entity, company_name, year, ratio1, ratio2, ...
    Output: entity, company_name, ratio1_2023, ratio1_2024, ratio2_2023, ...
    """
    # Get ratio columns (all except entity, company_name, year)
    ratio_cols = [col for col in df.columns if col not in ["entity", "company_name", "year"]]
    
    # Pivot each ratio
    pivot_list = []
    for ratio in ratio_cols:
        ratio_pivot = df.pivot_table(
            index=["entity", "company_name"],
            columns="year",
            values=ratio,
            aggfunc="first"
        )
        # Rename columns to include ratio name
        ratio_pivot.columns = [f"{ratio}_{col}" for col in ratio_pivot.columns]
        pivot_list.append(ratio_pivot)
    
    if not pivot_list:
        return pd.DataFrame()
    
    # Combine all ratios
    result = pd.concat(pivot_list, axis=1).reset_index()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate financial ratios from financial facts data."
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help=f"Path to the input financial facts CSV file (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help=f"Path to write the ratios CSV file (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--years",
        nargs="+",
        default=None,
        help="Specific years to include (default: all years in the input file). Example: --years 2023 2024",
    )
    parser.add_argument(
        "--pivot-years",
        action="store_true",
        help="Pivot ratios so years become columns (e.g., profit_margin_2023, profit_margin_2024)",
    )

    args = parser.parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Load financial facts
    print(f"Loading financial facts from {input_path}...")
    df = load_financial_facts(input_path)

    # Get year columns
    if args.years:
        year_columns = args.years
    else:
        # Extract year columns (numeric columns that are not entity, company_name, concept)
        base_cols = {"entity", "company_name", "concept"}
        year_columns = [col for col in df.columns if col not in base_cols and col.isdigit()]
        year_columns.sort()

    if not year_columns:
        print("No year columns found in the input file.")
        return

    print(f"Processing years: {', '.join(year_columns)}")

    # Pivot concepts to columns
    print("Reshaping data...")
    pivoted_df = pivot_concepts_to_columns(df, year_columns)

    if pivoted_df.empty:
        print("No data to process after reshaping.")
        return

    # Calculate ratios
    print("Calculating ratios...")
    ratios_df = calculate_ratios(pivoted_df)

    # Optionally pivot by year
    if args.pivot_years:
        print("Pivoting ratios by year...")
        final_df = pivot_ratios_by_year(ratios_df)
    else:
        # Keep year as a column (long format)
        final_df = ratios_df

    if final_df.empty:
        print("No ratios calculated.")
        return

    # Save to CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"\nFinancial ratios saved to: {output_path}")
    print(f"Entities processed: {final_df['entity'].nunique()}")
    print(f"Rows written: {len(final_df)}")
    print(f"\nRatios calculated:")
    print("  - Profit margin (ProfitLoss / Revenue)")
    print("  - Return on assets (ProfitLoss / Assets)")
    print("  - Return on Equity (ProfitLoss / Equity)")
    print("  - Assets Turnover (Revenue / Assets)")
    print("  - Cash flow to assets (CashFlowsFromUsedInOperatingActivities / Assets)")
    print("  - Cash flow to Equity (CashFlowsFromUsedInOperatingActivities / Equity)")


if __name__ == "__main__":
    main()

