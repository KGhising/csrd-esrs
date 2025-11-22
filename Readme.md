# CSRD-ESRS XBRL Data Tools

Tools for downloading and analyzing CSRD/ESRS iXBRL filings from the filings.xbrl.org API with support for financial fact extraction and AI keyword analysis.

## Features

- Download XBRL package files from the filings.xbrl.org API
- Extract iXBRL report files from downloaded packages
- Extract financial facts from iXBRL reports (Equity, Assets, Revenue, Profit/Loss, Cash Flows)
- Extract AI keyword counts from iXBRL reports
- Clean and filter extracted data with configurable concepts and years
- LEI-based entity identification (20-character alphanumeric codes)
- Company name extraction from filings
- Tabular reports with concepts as rows and periods as columns
- Progress bars for batch processing operations

## Prerequisites

- Python 3.10 or higher
- pip package manager

## Installation

### 1. Clone the repository

```bash
git clone "https://github.com/KGhising/csrd-esrs.git"
cd csrd-esrs
```

### 2. Set up a virtual environment

**On macOS/Linux:**
```bash
python -m venv .venv
source .venv/bin/activate
```

**On Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -U pip
pip install pandas requests tqdm
```

**Optional visualization packages:**
```bash
pip install matplotlib jupyter seaborn plotly
```

### 4. Create data directories

```bash
mkdir -p data/xbrl-packages data/ixbrl-reports data/csv-data
```

## Usage

### Workflow Overview

The typical workflow consists of these steps:

1. **Download XBRL packages** or **download iXBRL reports directly**
2. **Extract iXBRL reports** from packages (if using packages)
3. **Extract financial facts** from iXBRL reports
4. **Clean financial facts** (optional - filter by concepts and years)
5. **Extract AI keywords** from iXBRL reports (optional)

### Step 1: Download XBRL Packages or iXBRL Reports

#### Option A: Download XBRL Package Files

Download ZIP package files from the filings.xbrl.org API:

```bash
# Download 5 packages from Estonia
python scripts/download_packages.py --country EE --limit 5

# Download from a specific country
python scripts/download_packages.py --country FR --limit 10

# Download only filings processed after 2024
python scripts/download_packages.py --from-year 2024

# Download filings from a specific company
python scripts/download_packages.py --company "Apple" --limit 10

# Combine filters: country, date, and company
python scripts/download_packages.py --country EE --from-year 2024 --company "Silvano" --limit 20

# Overwrite existing files
python scripts/download_packages.py --overwrite
```

**Arguments for `download_packages.py`:**
- `--country`: Filter by country code (e.g., EE, FR, DE)
- `--company`: Filter by company/entity name (case-insensitive partial match)
- `--from-year`: Filter filings processed from this year onwards (e.g., `2024`)
- `--after-date`: Filter filings processed after this date (format: `YYYY-MM-DD` or `YYYY`)
- `--limit`: Maximum number of packages to download (default: 50)
- `--out-dir`: Output directory for packages (default: `data/xbrl-packages`)
- `--overwrite`: Overwrite existing package files

#### Option B: Download iXBRL Reports Directly

Download iXBRL report files (.xhtml) directly from the API:

```bash
# Download 5 iXBRL reports from Estonia
python scripts/fetch_filings.py --country EE --limit 5

# Download from a specific country
python scripts/fetch_filings.py --country FR --limit 10

# Download only filings processed after 2024
python scripts/fetch_filings.py --from-year 2024

# Download filings from a specific company
python scripts/fetch_filings.py --company "Apple" --limit 10

# Skip package fallback (faster, but may miss some reports)
python scripts/fetch_filings.py --skip-package-fallback
```

**Arguments for `fetch_filings.py`:**
- `--country`: Filter by country code (e.g., EE, FR, DE)
- `--company`: Filter by company/entity name (case-insensitive partial match)
- `--from-year`: Filter filings processed from this year onwards (e.g., `2024`)
- `--after-date`: Filter filings processed after this date (format: `YYYY-MM-DD` or `YYYY`)
- `--limit`: Maximum number of reports to download (default: 50)
- `--out-ixbrl`: Output directory for iXBRL reports (default: `data/ixbrl-reports`)
- `--skip-ixbrl`: Skip downloading iXBRL reports
- `--skip-package-fallback`: Skip downloading packages if direct download fails (faster but may miss reports)

### Step 2: Extract iXBRL Reports from Packages (if using packages)

If you downloaded XBRL packages, extract the iXBRL report files from them:

```bash
# Extract all iXBRL reports from packages
python scripts/extract_ixbrl_reports.py

# Extract with custom directories
python scripts/extract_ixbrl_reports.py --packages-dir data/xbrl-packages --output-dir data/ixbrl-reports

# Overwrite existing reports
python scripts/extract_ixbrl_reports.py --overwrite

# Limit number of packages to process
python scripts/extract_ixbrl_reports.py --limit 10
```

**Arguments for `extract_ixbrl_reports.py`:**
- `--packages-dir`: Directory containing XBRL package ZIP files (default: `data/xbrl-packages`)
- `--output-dir`: Directory to save extracted iXBRL reports (default: `data/ixbrl-reports`)
- `--overwrite`: Overwrite existing iXBRL report files
- `--limit`: Optional limit on the number of packages to process

### Step 3: Extract Financial Facts from iXBRL Reports

Extract financial facts from iXBRL reports and create a tabular CSV:

```bash
# Extract facts from all iXBRL reports
python scripts/extract_facts.py

# Extract with custom directories
python scripts/extract_facts.py --input-dir data/ixbrl-reports --output data/csv-data/financial_facts.csv

# Process a single file
python scripts/extract_facts.py --file data/ixbrl-reports/15608.xhtml
```

**Arguments for `extract_facts.py`:**
- `--input-dir`: Directory containing iXBRL report files (default: `data/ixbrl-reports`)
- `--output`: Output CSV file path (default: `data/csv-data/financial_facts.csv`)
- `--file`: Process a single file instead of directory

**Extracted Concepts:**
- `Equity` - Total equity
- `Assets` - Total assets
- `Revenue` - Revenue/income
- `ProfitLoss` - Profit or loss
- `CashFlowsFromUsedInOperatingActivities` - Operating cash flows

**Entity Identification:**
- Uses LEI (Legal Entity Identifier) - 20-character alphanumeric code
- Extracts company names from filings
- Automatically extracts period information (years)

### Step 4: Clean Financial Facts (Optional)

Filter and clean the extracted financial facts by concepts and years:

```bash
# Clean for default years (2023, 2024) with all concepts
python scripts/clean_financial_facts.py

# Clean for specific years
python scripts/clean_financial_facts.py --years 2023 2024 2025

# Clean for specific concepts
python scripts/clean_financial_facts.py --concepts Equity Revenue Assets

# Clean for specific years AND concepts
python scripts/clean_financial_facts.py --years 2023 2024 --concepts Equity Revenue

# Custom input/output
python scripts/clean_financial_facts.py --input data/csv-data/financial_facts.csv --output data/csv-data/financial_facts_cleaned.csv
```

**Arguments for `clean_financial_facts.py`:**
- `--input`: Input CSV file path (default: `data/csv-data/financial_facts.csv`)
- `--output`: Output CSV file path (default: `data/csv-data/financial_facts_cleaned.csv`)
- `--years`: Years to include (default: `2023 2024`). Example: `--years 2023 2024 2025`
- `--concepts`: Concepts to include (default: all). Example: `--concepts Equity Revenue Assets`

**Cleaning Logic:**
- Filters to keep only specified years and concepts
- Removes entities with incomplete data (any missing values for specified years)
- Ensures all entities have complete data across all specified concepts and years

### Step 5: Extract AI Keywords from iXBRL Reports (Optional)

Extract AI-related keyword counts from iXBRL reports, aggregated by entity (LEI):

```bash
# Extract AI keywords from all iXBRL reports
python scripts/export_ai_keywords_from_ixbrl.py

# Extract with custom directories
python scripts/export_ai_keywords_from_ixbrl.py --reports-dir data/ixbrl-reports --output data/csv-data/ai_keywords_ixbrl.csv
```

**Arguments for `export_ai_keywords_from_ixbrl.py`:**
- `--reports-dir`: Directory containing iXBRL report files (default: `data/ixbrl-reports`)
- `--output`: Output CSV file path (default: `data/csv-data/ai_keywords_ixbrl.csv`)

**Output Format:**
- Groups keyword counts by entity (LEI)
- Includes total AI keyword count per entity
- Lists top keywords with counts for each entity
- Extracts company names from filings

## Script Overview

| Script | Purpose | Default Output |
|--------|---------|----------------|
| `download_packages.py` | Download XBRL ZIP package files from API | `data/xbrl-packages/` |
| `fetch_filings.py` | Download iXBRL report files (.xhtml) directly from API | `data/ixbrl-reports/` |
| `extract_ixbrl_reports.py` | Extract iXBRL reports from downloaded ZIP packages | `data/ixbrl-reports/` |
| `extract_facts.py` | Extract financial facts from iXBRL reports | `data/csv-data/financial_facts.csv` |
| `clean_financial_facts.py` | Filter and clean financial facts by concepts and years | `data/csv-data/financial_facts_cleaned.csv` |
| `export_ai_keywords_from_ixbrl.py` | Extract AI keyword counts from iXBRL reports (by entity) | `data/csv-data/ai_keywords_ixbrl.csv` |

## CSV Output Format

### Financial Facts Tabular Format

The `extract_facts.py` script generates a tabular CSV with a pivot table structure:
- **Rows**: Entity (LEI), company name, and financial concepts
- **Columns**: Years extracted from reporting periods (2023, 2024, etc.)
- **Values**: Fact values for each entity-concept-year combination
- **Entity identification**: Uses LEI (Legal Entity Identifier) - 20-character alphanumeric code
- **Company names**: Extracted from filings where available

**Example output structure:**
```
entity,company_name,concept,2023,2024
213800I2B3OD5PUCLO62,Example Corp,Equity,15000000,16500000
213800I2B3OD5PUCLO62,Example Corp,Assets,50000000,55000000
213800I2B3OD5PUCLO62,Example Corp,Revenue,25000000,27500000
5493000FB0EA3L91L153,Another Company,Equity,8000000,8500000
5493000FB0EA3L91L153,Another Company,Assets,30000000,32000000
...
```

The tabular format is ideal for:
- Comparing values across periods and entities in spreadsheet applications
- Quick analysis and visualization
- Creating financial reports and dashboards
- Multi-entity analysis and benchmarking

### Cleaned Financial Facts Format

The `clean_financial_facts.py` script generates a filtered version of the financial facts CSV:
- **Filters**: Only includes specified years and concepts
- **Completeness**: Removes entities with missing data for any specified year or concept
- **Same structure**: Same columns as financial facts, but filtered

### AI Keywords Format

The `export_ai_keywords_from_ixbrl.py` script generates an entity-aggregated CSV:
- **Rows**: One row per entity (LEI)
- **Columns**: `entity`, `company_name`, `ai_keyword_total`, `top_keywords`
- **Aggregation**: Keyword counts aggregated across all documents for each entity

**Example output structure:**
```
entity,company_name,ai_keyword_total,top_keywords
213800I2B3OD5PUCLO62,Example Corp,42,artificial intelligence:15,machine learning:12,AI:10,automation:5
5493000FB0EA3L91L153,Another Company,28,AI:10,artificial intelligence:8,machine learning:6,robotics:4
...
```

## Project Structure

```
csrd-esrs/
├── data/
│   ├── xbrl-packages/      # Downloaded XBRL ZIP package files
│   ├── ixbrl-reports/      # Extracted iXBRL report files (.xhtml/.html)
│   └── csv-data/           # Exported tabular CSV files
├── scripts/
│   ├── download_packages.py            # Download XBRL ZIP packages from API
│   ├── fetch_filings.py                # Download iXBRL reports directly from API
│   ├── extract_ixbrl_reports.py        # Extract iXBRL reports from ZIP packages
│   ├── extract_facts.py                # Extract financial facts from iXBRL reports
│   ├── clean_financial_facts.py        # Filter and clean financial facts by concepts and years
│   └── export_ai_keywords_from_ixbrl.py # Extract AI keyword counts from iXBRL reports
├── .gitignore
└── Readme.md
```

## API Reference

The tools interact with the filings.xbrl.org API:
- **Base URL**: https://filings.xbrl.org/api/filings
- **Documentation**: https://filings.xbrl.org/docs/api

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.
