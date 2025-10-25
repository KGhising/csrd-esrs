## Setup step
Download python 3.10+

Setup python virtual env and activate
```bash
python -m venv .venv
```
Activete venv
```bash
source .venv/bin/activate
```
Install pip, dependency and packages
```bash
pip install -U pip
pip install pandas matplotlib jupyter requests
# Optional for visualization extras:
pip install seaborn plotly
```
Clone project
```bash
git clone "https://github.com/KGhising/csrd-esrs.git"
```
Create folder for data packages
```bash
mkdir -p data/samples/xbrl-json out/packages
```
Run test script to download data form API = "https://filings.xbrl.org/api/filings"
```bash
python scripts/fetch_filings.py --country EE --limit 5
```
