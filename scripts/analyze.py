import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

DEFAULT_INPUT = Path("data/csv-data/final_data.csv")
DEFAULT_FACTS = Path("data/csv-data/financial_facts_cleaned.csv")
DEFAULT_OUTPUT_DIR = Path("data/analysis-results")


def pearsonr(x: pd.Series, y: pd.Series) -> tuple:
    """
    Calculate Pearson correlation coefficient and p-value.
    Replaces scipy.stats.pearsonr with pure pandas/numpy implementation.
    Uses approximate p-value calculation via t-distribution.
    """
    valid_mask = ~(pd.isna(x) | pd.isna(y))
    x_clean = x[valid_mask].values
    y_clean = y[valid_mask].values
    
    if len(x_clean) < 2:
        return (np.nan, np.nan)
    
    corr_matrix = np.corrcoef(x_clean, y_clean)
    corr = corr_matrix[0, 1]
    
    if np.isnan(corr):
        return (np.nan, np.nan)
    
    if abs(corr) >= 1.0 - 1e-10:
        return (corr, 0.0)
    
    n = len(x_clean)
    if n < 3:
        return (corr, np.nan)
    
    t_stat = corr * np.sqrt((n - 2) / (1 - corr ** 2 + 1e-10))
    df = n - 2
    
    p_value = 2 * (1 - _t_cdf_approx(abs(t_stat), df))
    
    return (corr, p_value)


def _t_cdf_approx(t: float, df: int) -> float:
    """
    Approximate t-distribution CDF using normal approximation for large df,
    or a simple approximation for small df. Pure numpy implementation.
    """
    if df > 30:
        return 0.5 * (1 + np.sign(t) * (1 - np.exp(-2 * t**2 / np.pi)))
    
    if df == 1:
        return 0.5 + np.arctan(t) / np.pi
    elif df == 2:
        return 0.5 + (t / (2 * np.sqrt(2 + t**2)))
    else:
        x = t / np.sqrt(df + t**2)
        beta_approx = 0.5 * (1 + np.sign(x) * np.sqrt(1 - np.exp(-2 * x**2 / np.pi)))
        return beta_approx


def load_final_data(input_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    return df


def load_financial_facts(facts_path: Path) -> pd.DataFrame:
    df = pd.read_csv(facts_path)
    return df


def prepare_controls(facts_df: pd.DataFrame) -> pd.DataFrame:
    controls = facts_df[facts_df['concept'].isin(['Assets', 'Revenue'])].copy()
    
    controls_list = []
    for _, row in controls.iterrows():
        entity = row['entity']
        company_name = row['company_name']
        concept = row['concept']
        
        for year in ['2023', '2024']:
            year_cols = [year, int(year) if year.isdigit() else None]
            year_cols = [y for y in year_cols if y is not None and y in row.index]
            
            for year_col in year_cols:
                if pd.notna(row[year_col]) and row[year_col] != '':
                    try:
                        value = float(row[year_col])
                        controls_list.append({
                            'entity': entity,
                            'company_name': company_name,
                            'year': year,
                            'concept': concept,
                            'value': value
                        })
                        break
                    except (ValueError, TypeError):
                        continue
    
    controls_df = pd.DataFrame(controls_list)
    
    if controls_df.empty:
        return pd.DataFrame(columns=['entity', 'company_name', 'year', 'Assets', 'Revenue'])
    
    controls_pivot = controls_df.pivot_table(
        index=['entity', 'company_name', 'year'],
        columns='concept',
        values='value',
        aggfunc='first'
    ).reset_index()
    
    return controls_pivot


def merge_data(final_df: pd.DataFrame, controls_df: pd.DataFrame) -> pd.DataFrame:
    final_df = final_df.copy()
    controls_df = controls_df.copy()
    
    final_df['year'] = final_df['year'].astype(str)
    if not controls_df.empty and 'year' in controls_df.columns:
        controls_df['year'] = controls_df['year'].astype(str)
    
    merged = final_df.merge(
        controls_df,
        on=['entity', 'company_name', 'year'],
        how='left'
    )
    
    merged['Assets'] = pd.to_numeric(merged['Assets'], errors='coerce')
    merged['Revenue'] = pd.to_numeric(merged['Revenue'], errors='coerce')
    
    merged = merged.dropna(subset=['ai_keyword_total', 'return_on_assets', 'return_on_equity', 'profit_margin'])
    
    return merged


def calculate_correlations(df: pd.DataFrame) -> pd.DataFrame:
    ratios = ['return_on_assets', 'return_on_equity', 'profit_margin']
    correlations = []
    
    for ratio in ratios:
        valid_data = df[[ratio, 'ai_keyword_total']].dropna()
        if len(valid_data) > 1:
            corr, p_value = pearsonr(valid_data[ratio], valid_data['ai_keyword_total'])
            correlations.append({
                'ratio': ratio,
                'correlation': corr,
                'p_value': p_value,
                'n_observations': len(valid_data)
            })
    
    return pd.DataFrame(correlations)


def run_regression(df: pd.DataFrame, dependent_var: str, output_dir: Path) -> dict:
    df_clean = df[[dependent_var, 'ai_keyword_total', 'Assets', 'Revenue']].dropna()
    
    if len(df_clean) < 4:
        return None
    
    y = df_clean[dependent_var]
    X = df_clean[['ai_keyword_total', 'Assets', 'Revenue']]
    
    X = sm.add_constant(X)
    
    model = sm.OLS(y, X).fit()
    
    results = {
        'dependent_var': dependent_var,
        'n_observations': len(df_clean),
        'rsquared': model.rsquared,
        'rsquared_adj': model.rsquared_adj,
        'f_statistic': model.fvalue,
        'f_pvalue': model.f_pvalue,
        'coefficients': model.params.to_dict(),
        'pvalues': model.pvalues.to_dict(),
        'conf_int': model.conf_int().to_dict(),
        'model_summary': str(model.summary())
    }
    
    summary_file = output_dir / f"regression_{dependent_var}_summary.txt"
    with open(summary_file, 'w') as f:
        f.write(f"Regression Results for {dependent_var}\n")
        f.write("=" * 80 + "\n\n")
        f.write(model.summary().as_text())
    
    return results


def visualize_data_overview(df: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    ax1 = axes[0, 0]
    ai_data = df[df['ai_keyword_total'] > 0]['ai_keyword_total']
    ax1.hist(ai_data, bins=30, color='#2E86AB', edgecolor='black', alpha=0.7)
    ax1.set_xlabel('AI Keyword Mentions', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.set_title('Distribution of AI Keyword Mentions', fontsize=12, fontweight='bold')
    ax1.axvline(ai_data.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {ai_data.mean():.1f}')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    ax2 = axes[0, 1]
    ratios = ['return_on_assets', 'return_on_equity', 'profit_margin']
    ratio_data = [df[r].dropna().values for r in ratios]
    bp = ax2.boxplot(ratio_data, labels=[r.replace('_', ' ').title() for r in ratios], 
                     patch_artist=True, widths=0.6)
    for patch, color in zip(bp['boxes'], ['#A23B72', '#F18F01', '#06A77D']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax2.set_ylabel('Ratio Value', fontsize=11)
    ax2.set_title('Distribution of Financial Ratios', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=15, ha='right')
    
    ax3 = axes[1, 0]
    year_counts = df.groupby('year')['entity'].count()
    colors = ['#2E86AB', '#A23B72']
    bars = ax3.bar(year_counts.index, year_counts.values, color=colors[:len(year_counts)], alpha=0.7)
    ax3.set_xlabel('Year', fontsize=11)
    ax3.set_ylabel('Number of Entities', fontsize=11)
    ax3.set_title('Data Coverage by Year', fontsize=12, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=10)
    
    ax4 = axes[1, 1]
    ai_by_year = df.groupby('year')['ai_keyword_total'].agg(['mean', 'median', 'std']).reset_index()
    x = np.arange(len(ai_by_year))
    width = 0.25
    ax4.bar(x - width, ai_by_year['mean'], width, label='Mean', color='#2E86AB', alpha=0.7)
    ax4.bar(x, ai_by_year['median'], width, label='Median', color='#A23B72', alpha=0.7)
    ax4.bar(x + width, ai_by_year['std'], width, label='Std Dev', color='#F18F01', alpha=0.7)
    ax4.set_xlabel('Year', fontsize=11)
    ax4.set_ylabel('AI Mentions', fontsize=11)
    ax4.set_title('AI Keyword Statistics by Year', fontsize=12, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(ai_by_year['year'])
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'data_overview.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Data overview visualization saved to: {output_dir / 'data_overview.png'}")


def visualize_correlations(corr_df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    bars = ax.bar(corr_df['ratio'], corr_df['correlation'], color=colors[:len(corr_df)])
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_ylabel('Pearson Correlation Coefficient', fontsize=12)
    ax.set_xlabel('Financial Ratio', fontsize=12)
    ax.set_title('Correlation between AI Mentions and Financial Ratios', fontsize=14, fontweight='bold')
    ax.set_ylim(-1, 1)
    ax.grid(axis='y', alpha=0.3)
    
    for i, (bar, row) in enumerate(zip(bars, corr_df.itertuples())):
        height = bar.get_height()
        p_text = f'p={row.p_value:.3f}' if row.p_value < 0.001 else f'p={row.p_value:.2f}'
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02 if height >= 0 else height - 0.05,
                f'r={row.correlation:.3f}\n{p_text}',
                ha='center', va='bottom' if height >= 0 else 'top', fontsize=10)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'correlations.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Correlation visualization saved to: {output_dir / 'correlations.png'}")


def visualize_scatter_plots(df: pd.DataFrame, output_dir: Path) -> None:
    ratios = ['return_on_assets', 'return_on_equity', 'profit_margin']
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for ax, ratio in zip(axes, ratios):
        data = df[[ratio, 'ai_keyword_total']].dropna()
        
        if len(data) > 0:
            ax.scatter(data['ai_keyword_total'], data[ratio], alpha=0.6, s=50, color='#2E86AB')
            
            z = np.polyfit(data['ai_keyword_total'], data[ratio], 1)
            p = np.poly1d(z)
            x_line = np.linspace(data['ai_keyword_total'].min(), data['ai_keyword_total'].max(), 100)
            ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, label='Trend line')
            
            corr, p_value = pearsonr(data[ratio], data['ai_keyword_total'])
            
            ax.set_xlabel('AI Keyword Mentions', fontsize=11)
            ax.set_ylabel(ratio.replace('_', ' ').title(), fontsize=11)
            title = f'{ratio.replace("_", " ").title()}\n(r={corr:.3f}, p={p_value:.3f})'
            ax.set_title(title, fontsize=12)
            ax.grid(alpha=0.3)
            ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'scatter_plots.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Scatter plots saved to: {output_dir / 'scatter_plots.png'}")


def visualize_regression_results(regression_results: list, output_dir: Path) -> None:
    if not regression_results or all(r is None for r in regression_results):
        print("No regression results to visualize.")
        return
    
    results_df = pd.DataFrame([
        {
            'ratio': r['dependent_var'].replace('_', ' ').title(),
            'R²': r['rsquared'],
            'R² Adj': r['rsquared_adj'],
            'AI Coefficient': r['coefficients'].get('ai_keyword_total', 0),
            'AI P-value': r['pvalues'].get('ai_keyword_total', 1)
        }
        for r in regression_results if r is not None
    ])
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    ax1 = axes[0, 0]
    bars1 = ax1.barh(results_df['ratio'], results_df['R²'], color='#2E86AB')
    ax1.set_xlabel('R²', fontsize=11)
    ax1.set_title('Model Fit (R²)', fontsize=12, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    for bar, val in zip(bars1, results_df['R²']):
        ax1.text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.3f}',
                va='center', fontsize=10)
    
    ax2 = axes[0, 1]
    bars2 = ax2.barh(results_df['ratio'], results_df['AI Coefficient'], 
                     color=['#A23B72' if p < 0.05 else '#D3D3D3' for p in results_df['AI P-value']])
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Coefficient', fontsize=11)
    ax2.set_title('AI Mentions Coefficient', fontsize=12, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    for bar, val, pval in zip(bars2, results_df['AI Coefficient'], results_df['AI P-value']):
        sig = '*' if pval < 0.05 else ''
        ax2.text(val + (0.01 if val >= 0 else -0.01), bar.get_y() + bar.get_height()/2, 
                f'{val:.4f}{sig}', va='center', ha='left' if val >= 0 else 'right', fontsize=10)
    
    ax3 = axes[1, 0]
    bars3 = ax3.barh(results_df['ratio'], -np.log10(results_df['AI P-value'] + 1e-10), color='#F18F01')
    ax3.axvline(x=-np.log10(0.05), color='red', linestyle='--', linewidth=1, label='p=0.05')
    ax3.set_xlabel('-log10(p-value)', fontsize=11)
    ax3.set_title('Statistical Significance', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(axis='x', alpha=0.3)
    
    ax4 = axes[1, 1]
    bars4 = ax4.barh(results_df['ratio'], results_df['R² Adj'], color='#06A77D')
    ax4.set_xlabel('Adjusted R²', fontsize=11)
    ax4.set_title('Adjusted R²', fontsize=12, fontweight='bold')
    ax4.grid(axis='x', alpha=0.3)
    for bar, val in zip(bars4, results_df['R² Adj']):
        ax4.text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.3f}',
                va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'regression_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Regression results visualization saved to: {output_dir / 'regression_results.png'}")


def visualize_comparison_by_ai_levels(df: pd.DataFrame, output_dir: Path) -> None:
    df_with_ai = df[df['ai_keyword_total'] > 0].copy()
    df_no_ai = df[df['ai_keyword_total'] == 0].copy()
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ratios = ['return_on_assets', 'return_on_equity', 'profit_margin']
    
    for ax, ratio in zip(axes, ratios):
        data_with = df_with_ai[ratio].dropna()
        data_without = df_no_ai[ratio].dropna()
        
        if len(data_with) > 0 and len(data_without) > 0:
            positions = [1, 2]
            bp = ax.boxplot([data_without, data_with], positions=positions, 
                           labels=['No AI Mentions', 'With AI Mentions'],
                           patch_artist=True, widths=0.6)
            
            bp['boxes'][0].set_facecolor('#D3D3D3')
            bp['boxes'][0].set_alpha(0.7)
            bp['boxes'][1].set_facecolor('#2E86AB')
            bp['boxes'][1].set_alpha(0.7)
            
            mean_without = data_without.mean()
            mean_with = data_with.mean()
            
            ax.set_ylabel(ratio.replace('_', ' ').title(), fontsize=11)
            ax.set_title(f'{ratio.replace("_", " ").title()}\n' + 
                        f'No AI: μ={mean_without:.3f}, With AI: μ={mean_with:.3f}', 
                        fontsize=12)
            ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_by_ai_levels.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison by AI levels saved to: {output_dir / 'comparison_by_ai_levels.png'}")


def save_results(corr_df: pd.DataFrame, regression_results: list, output_dir: Path) -> None:
    corr_df.to_csv(output_dir / 'correlations.csv', index=False)
    print(f"Correlation results saved to: {output_dir / 'correlations.csv'}")
    
    if regression_results and any(r is not None for r in regression_results):
        reg_summary = []
        for r in regression_results:
            if r is not None:
                reg_summary.append({
                    'dependent_variable': r['dependent_var'],
                    'n_observations': r['n_observations'],
                    'rsquared': r['rsquared'],
                    'rsquared_adj': r['rsquared_adj'],
                    'f_statistic': r['f_statistic'],
                    'f_pvalue': r['f_pvalue'],
                    'ai_coefficient': r['coefficients'].get('ai_keyword_total', None),
                    'ai_pvalue': r['pvalues'].get('ai_keyword_total', None),
                    'assets_coefficient': r['coefficients'].get('Assets', None),
                    'assets_pvalue': r['pvalues'].get('Assets', None),
                    'revenue_coefficient': r['coefficients'].get('Revenue', None),
                    'revenue_pvalue': r['pvalues'].get('Revenue', None),
                })
        
        reg_df = pd.DataFrame(reg_summary)
        reg_df.to_csv(output_dir / 'regression_summary.csv', index=False)
        print(f"Regression summary saved to: {output_dir / 'regression_summary.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze correlation and regression between AI mentions and financial ratios."
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help=f"Path to final data CSV (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--facts",
        default=str(DEFAULT_FACTS),
        help=f"Path to financial facts CSV for controls (default: {DEFAULT_FACTS})",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Directory to save analysis results (default: {DEFAULT_OUTPUT_DIR})",
    )
    
    args = parser.parse_args()
    input_path = Path(args.input)
    facts_path = Path(args.facts)
    output_dir = Path(args.output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if not facts_path.exists():
        raise FileNotFoundError(f"Financial facts file not found: {facts_path}")
    
    print(f"Loading data from {input_path}...")
    final_df = load_final_data(input_path)
    print(f"Final data: {len(final_df)} rows")
    print(f"Columns: {', '.join(final_df.columns)}")
    
    print(f"Loading financial facts from {facts_path}...")
    facts_df = load_financial_facts(facts_path)
    print(f"Financial facts: {len(facts_df)} rows")
    
    print("Preparing control variables (Assets, Revenue)...")
    controls_df = prepare_controls(facts_df)
    print(f"Controls: {len(controls_df)} entity-year combinations")
    
    print("Merging data...")
    merged_df = merge_data(final_df, controls_df)
    print(f"Merged data: {len(merged_df)} rows")
    print(f"Entities: {merged_df['entity'].nunique()}")
    print(f"Years: {sorted(merged_df['year'].unique())}")
    
    print("\n" + "="*80)
    print("GENERATING DATA OVERVIEW VISUALIZATIONS")
    print("="*80)
    visualize_data_overview(merged_df, output_dir)
    
    print("\n" + "="*80)
    print("1. CORRELATION ANALYSIS")
    print("="*80)
    corr_df = calculate_correlations(merged_df)
    print("\nPearson Correlations:")
    print(corr_df.to_string(index=False))
    
    print("\n" + "="*80)
    print("2. REGRESSION ANALYSIS")
    print("="*80)
    ratios = ['return_on_assets', 'return_on_equity', 'profit_margin']
    regression_results = []
    
    for ratio in ratios:
        print(f"\nRunning regression for {ratio}...")
        result = run_regression(merged_df, ratio, output_dir)
        if result:
            regression_results.append(result)
            print(f"  R² = {result['rsquared']:.4f}")
            print(f"  AI Coefficient = {result['coefficients'].get('ai_keyword_total', 'N/A'):.6f}")
            print(f"  AI P-value = {result['pvalues'].get('ai_keyword_total', 'N/A'):.6f}")
        else:
            regression_results.append(None)
            print(f"  Insufficient data for regression")
    
    print("\n" + "="*80)
    print("3. GENERATING VISUALIZATIONS")
    print("="*80)
    visualize_correlations(corr_df, output_dir)
    visualize_scatter_plots(merged_df, output_dir)
    visualize_regression_results(regression_results, output_dir)
    visualize_comparison_by_ai_levels(merged_df, output_dir)
    
    print("\n" + "="*80)
    print("4. SAVING RESULTS")
    print("="*80)
    save_results(corr_df, regression_results, output_dir)
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print(f"All results saved to: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
