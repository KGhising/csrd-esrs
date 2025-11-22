import argparse
from urllib.parse import urlencode, urljoin
from pathlib import Path
from zipfile import ZipFile
import requests

API_BASE = "https://filings.xbrl.org/api/filings"
BASE_DOWNLOAD_URL = "https://filings.xbrl.org"

def api_get(url: str) -> dict:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()

def iter_filings(query: str, limit: int, min_date: str = None, company_filter: str = None):
    url = f"{API_BASE}?{query}" if query else API_BASE
    count = 0
    while url:
        data = api_get(url)
        for item in data.get("data", []):
            attrs = item.get('attributes', {})
            
            if min_date:
                processed = attrs.get('processed', '')
                if processed and processed < min_date:
                    continue
            
            if company_filter:
                relationships = item.get('relationships', {})
                entity_data = relationships.get('entity', {}).get('data', {})
                entity_attrs = entity_data.get('attributes', {}) if entity_data else {}
                
                entity_name = (
                    entity_attrs.get('name') or 
                    entity_attrs.get('entity_name') or
                    attrs.get('entity_name') or
                    attrs.get('name') or
                    ''
                )
                
                entity_id = (
                    entity_attrs.get('identifier') or
                    entity_attrs.get('entity_id') or
                    attrs.get('entity_id') or
                    attrs.get('identifier') or
                    ''
                )
                
                company_lower = company_filter.lower()
                if company_lower not in str(entity_name).lower() and company_lower not in str(entity_id).lower():
                    continue
            
            yield item
            count += 1
            if limit and count >= limit:
                return
        url = (data.get("links") or {}).get("next")

def first_attr(attrs: dict, *keys, default=None):
    for k in keys:
        if k in attrs and attrs[k]:
            return attrs[k]
    return default

def ensure_absolute_url(url: str) -> str:
    return urljoin(BASE_DOWNLOAD_URL, url)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--country')
    p.add_argument('--filter')
    p.add_argument('--after-date', help='Filter filings processed after this date (YYYY-MM-DD or YYYY)')
    p.add_argument('--from-year', type=int, help='Filter filings processed from this year onwards (e.g., 2024)')
    p.add_argument('--company', help='Filter by company/entity name (case-insensitive partial match)')
    p.add_argument('--include', default='entity')
    p.add_argument('--sort', default='-processed')
    p.add_argument('--page-size', type=int, default=200)
    p.add_argument('--limit', type=int, default=50)
    p.add_argument('--out-json', default='data/xbrl-json')
    p.add_argument('--out-package', default='data/xbrl-packages')
    p.add_argument('--skip-json', action='store_true')
    p.add_argument('--skip-package', action='store_true')
    p.add_argument('--out-ixbrl', default='data/ixbrl-reports')
    p.add_argument('--skip-ixbrl', action='store_true')
    args = p.parse_args()

    qp = {}
    if args.country:
        qp['filter[country]'] = args.country
    
    min_date = None
    if args.after_date:
        if len(args.after_date) == 4 and args.after_date.isdigit():
            min_date = f"{args.after_date}-01-01"
        else:
            min_date = args.after_date
    
    if args.from_year:
        min_date = f"{args.from_year}-01-01"
    
    if min_date:
        qp['filter[processed][gte]'] = min_date
    
    if args.company:
        qp['filter[entity.name]'] = args.company
    
    if args.include:
        qp['include'] = args.include
    if args.sort:
        qp['sort'] = args.sort
    qp['page[size]'] = min(args.page_size or 200, 200)
    query = args.filter if args.filter else urlencode(qp)
    
    if min_date or args.company:
        full_url = f"{API_BASE}?{query}"
        print(f"Query URL: {full_url}")
        if min_date:
            print(f"Filtering by processed date >= {min_date}")
        if args.company:
            print(f"Filtering by company/entity name containing: {args.company}")
        print(f"(Note: If API doesn't support these filters, will filter client-side)")

    out_json = Path(args.out_json); out_json.mkdir(parents=True, exist_ok=True)
    out_package = Path(args.out_package); out_package.mkdir(parents=True, exist_ok=True)
    out_ixbrl = Path(args.out_ixbrl); out_ixbrl.mkdir(parents=True, exist_ok=True)

    n = 0
    fetch_limit = args.limit * 10 if (min_date or args.company) and args.limit else None
    for filing in iter_filings(query, limit=fetch_limit, min_date=min_date, company_filter=args.company):
        attrs = filing.get('attributes', {})
        if args.limit and n >= args.limit:
            break
            
        filing_id = filing.get('id') or attrs.get('filing_index') or f'filing_{n+1}'
        json_url = first_attr(attrs, 'json_url', 'xbrl_json_url')
        package_url = first_attr(
            attrs,
            'package_url',
            'xbrl_package_url',
            'report_package_url',
            'filing_package_url',
            'package',
        )

        if json_url and not args.skip_json:
            json_url = ensure_absolute_url(json_url)
            pth = out_json / f"{filing_id}.json"
            try:
                r = requests.get(json_url, timeout=60)
                r.raise_for_status()
                pth.write_bytes(r.content)
                print('[JSON]', pth)
            except Exception as e:
                print('JSON failed:', filing_id, e)

        if package_url and not args.skip_package:
            package_url = ensure_absolute_url(package_url)
            pkg_path = out_package / f"{filing_id}.zip"
            try:
                r = requests.get(package_url, timeout=120)
                r.raise_for_status()
                pkg_path.write_bytes(r.content)
                print('[PKG]', pkg_path)

                if not args.skip_ixbrl:
                    try:
                        with ZipFile(pkg_path) as zf:
                            extracted = False
                            for member in zf.namelist():
                                member_lower = member.lower()
                                if member_lower.endswith('.xhtml') and '/reports/' in member_lower:
                                    data = zf.read(member)
                                    dest_dir = out_ixbrl
                                    dest_dir.mkdir(parents=True, exist_ok=True)
                                    dest_path = dest_dir / Path(member).name
                                    dest_path.write_bytes(data)
                                    extracted = True
                                    print('[IXBRL]', dest_path)
                            if not extracted:
                                print('[IXBRL] No .xhtml report found in', pkg_path)
                    except Exception as ix_err:
                        print('[IXBRL] Failed to extract from', pkg_path, ix_err)
            except Exception as e:
                print('Package failed:', filing_id, e)

        n += 1

    print('Done, files:', n)

if __name__ == '__main__':
    main()
