import argparse
from pathlib import Path
from typing import List

import pandas as pd

DEFAULT_AI_KEYWORDS = Path("data/csv-data/ai_keywords_ixbrl_cleaned.csv")
DEFAULT_RATIOS = Path("data/csv-data/financial_ratios.csv")
DEFAULT_OUTPUT = Path("data/csv-data/final_data.csv")


def load_ai_keywords(input_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    return df


def load_financial_ratios(input_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    return df


def unpivot_ai_keywords(df: pd.DataFrame) -> pd.DataFrame:
    base_columns = ["entity", "company_name"]
    year_columns = [col for col in df.columns if col.isdigit()]
    
    if not year_columns:
        return pd.DataFrame()
    
    id_vars = base_columns
    value_vars = year_columns
    
    melted = df.melt(
        id_vars=id_vars,
        value_vars=value_vars,
        var_name="year",
        value_name="ai_keyword_total"
    )
    
    top_keywords_dict = {}
    for year in year_columns:
        top_keywords_col = f"top_keywords_{year}"
        if top_keywords_col in df.columns:
            for idx, row in df.iterrows():
                key = (row["entity"], year)
                top_keywords_dict[key] = row[top_keywords_col] if pd.notna(row[top_keywords_col]) else None
    
    melted["top_keywords"] = melted.apply(
        lambda row: top_keywords_dict.get((row["entity"], row["year"]), None),
        axis=1
    )
    
    melted = melted[melted["ai_keyword_total"].notna()]
    
    melted["year"] = melted["year"].astype(str)
    
    return melted


def combine_data(ai_keywords_df: pd.DataFrame, ratios_df: pd.DataFrame) -> pd.DataFrame:
    ai_keywords_df = ai_keywords_df.copy()
    ratios_df = ratios_df.copy()
    
    if ai_keywords_df.empty:
        print("[warn] AI keywords data is empty")
        return ratios_df
    
    if ratios_df.empty:
        print("[warn] Financial ratios file is empty")
        return ai_keywords_df
    
    entity_col = "entity"
    if entity_col not in ai_keywords_df.columns:
        raise ValueError(f"AI keywords data missing required column: {entity_col}")
    if entity_col not in ratios_df.columns:
        raise ValueError(f"Financial ratios file missing required column: {entity_col}")
    
    if "company_name" in ai_keywords_df.columns:
        ai_keywords_df = ai_keywords_df.rename(columns={"company_name": "ai_company_name"})
    
    if "year" in ai_keywords_df.columns:
        ratios_df["year"] = ratios_df["year"].astype(str)
        merge_on = [entity_col, "year"]
    else:
        merge_on = [entity_col]
    
    merged_df = ratios_df.merge(
        ai_keywords_df,
        on=merge_on,
        how="left",
        suffixes=("", "_ai")
    )
    
    if "ai_company_name" in merged_df.columns:
        if "company_name" in merged_df.columns:
            merged_df["company_name"] = merged_df["company_name"].fillna(merged_df["ai_company_name"])
        else:
            merged_df["company_name"] = merged_df["ai_company_name"]
        merged_df = merged_df.drop(columns=["ai_company_name"])
    
    column_order = ["entity", "company_name"]
    if "year" in merged_df.columns:
        column_order.append("year")
    
    other_columns = [col for col in merged_df.columns if col not in column_order]
    other_columns.sort()
    
    final_columns = column_order + other_columns
    merged_df = merged_df[final_columns]
    
    return merged_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine cleaned AI keywords and financial ratios data by entity and year."
    )
    parser.add_argument(
        "--ai-keywords",
        default=str(DEFAULT_AI_KEYWORDS),
        help=f"Path to cleaned AI keywords CSV file (default: {DEFAULT_AI_KEYWORDS})",
    )
    parser.add_argument(
        "--ratios",
        default=str(DEFAULT_RATIOS),
        help=f"Path to financial ratios CSV file (default: {DEFAULT_RATIOS})",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help=f"Path to write the combined CSV file (default: {DEFAULT_OUTPUT})",
    )

    args = parser.parse_args()
    ai_keywords_path = Path(args.ai_keywords)
    ratios_path = Path(args.ratios)
    output_path = Path(args.output)

    if not ai_keywords_path.exists():
        raise FileNotFoundError(f"AI keywords file not found: {ai_keywords_path}")
    if not ratios_path.exists():
        raise FileNotFoundError(f"Financial ratios file not found: {ratios_path}")

    print(f"Loading AI keywords from {ai_keywords_path}...")
    ai_keywords_wide = load_ai_keywords(ai_keywords_path)
    print(f"AI keywords (wide format): {len(ai_keywords_wide)} rows")

    print(f"Unpivoting AI keywords to long format...")
    ai_keywords_df = unpivot_ai_keywords(ai_keywords_wide)
    print(f"AI keywords (long format): {len(ai_keywords_df)} rows")

    print(f"Loading financial ratios from {ratios_path}...")
    ratios_df = load_financial_ratios(ratios_path)
    print(f"Financial ratios: {len(ratios_df)} rows")

    print("Combining data by entity and year...")
    combined_df = combine_data(ai_keywords_df, ratios_df)

    if combined_df.empty:
        print("No data after combining.")
        return

    print(f"Combined data: {len(combined_df)} rows, {combined_df['entity'].nunique()} unique entities")

    if "ai_keyword_total" in combined_df.columns:
        rows_with_ai = combined_df[combined_df["ai_keyword_total"].notna()]
        rows_without_ai = combined_df[combined_df["ai_keyword_total"].isna()]
        print(f"Rows with AI keywords: {len(rows_with_ai)}")
        print(f"Rows without AI keywords: {len(rows_without_ai)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"\nFinal data saved to: {output_path}")
    print(f"Columns: {', '.join(combined_df.columns)}")
    print(f"Rows written: {len(combined_df)}")


if __name__ == "__main__":
    main()
