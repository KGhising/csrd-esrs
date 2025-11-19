"""
Download iXBRL package ZIP files from filings.xbrl.org API.

API Documentation: https://filings.xbrl.org/docs/api
The API follows the JSON API standard and provides pagination, filtering, and sorting.
"""
import argparse
import time
from pathlib import Path
from urllib.parse import urlencode, urljoin

import requests

# API endpoints as per https://filings.xbrl.org/docs/api
API_BASE = "https://filings.xbrl.org/api/filings"
BASE_DOWNLOAD_URL = "https://filings.xbrl.org"


def api_get(url: str) -> dict:
    """
    Make a GET request to the API and return JSON response.
    
    The API follows JSON API standard with pagination links in the 'links' object.
    """
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()


def ensure_absolute_url(url: str) -> str:
    return urljoin(BASE_DOWNLOAD_URL, url)


def first_attr(attrs: dict, *keys, default=None):
    for key in keys:
        value = attrs.get(key)
        if value:
            return value
    return default


def iter_filings(
    query: str,
    limit: int | None,
    min_date: str | None,
    company_filter: str | None,
    single_page: bool = False,
):
    """
    Iterate through filings from the API.
    
    According to the API docs (https://filings.xbrl.org/docs/api):
    - Pagination links are in the 'links' object within the response
    - Use page[size] and page[number] query parameters for pagination
    - Results follow JSON API standard format
    """
    url = f"{API_BASE}?{query}" if query else API_BASE
    count = 0
    while url:
        data = api_get(url)
        # JSON API standard: data is in the 'data' array
        for item in data.get("data", []):
            attrs = item.get("attributes", {})

            # Client-side date filtering (if server-side filter didn't work)
            if min_date:
                processed = attrs.get("processed", "")
                if processed and processed < min_date:
                    continue

            # Client-side company filtering (if server-side filter didn't work)
            if company_filter:
                # When include=entity is used, entity data is in relationships
                relationships = item.get("relationships", {})
                entity_data = relationships.get("entity", {}).get("data", {})
                entity_attrs = entity_data.get("attributes", {}) if entity_data else {}

                entity_name = (
                    entity_attrs.get("name")
                    or entity_attrs.get("entity_name")
                    or attrs.get("entity_name")
                    or attrs.get("name")
                    or ""
                )
                entity_id = (
                    entity_attrs.get("identifier")
                    or entity_attrs.get("entity_id")
                    or attrs.get("entity_id")
                    or attrs.get("identifier")
                    or ""
                )

                company_lower = company_filter.lower()
                if company_lower not in str(entity_name).lower() and company_lower not in str(entity_id).lower():
                    continue

            yield item
            count += 1
            if limit and count >= limit:
                return
        
        # JSON API standard: pagination links are in the 'links' object
        # If single_page is True, don't follow pagination links
        if single_page:
            break
        url = (data.get("links") or {}).get("next")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download iXBRL package ZIP files from filings.xbrl.org."
    )
    parser.add_argument("--country")
    parser.add_argument("--filter")
    parser.add_argument(
        "--after-date",
        help="Filter filings processed after this date (YYYY-MM-DD or YYYY).",
    )
    parser.add_argument(
        "--from-year",
        type=int,
        help="Filter filings processed from this year onward (e.g., 2024).",
    )
    parser.add_argument(
        "--company",
        help="Filter by company/entity name (case-insensitive substring match).",
    )
    parser.add_argument("--include", default="entity")
    parser.add_argument("--sort", default="-processed")
    parser.add_argument("--page-size", type=int, default=200)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument(
        "--out-dir",
        default="data/xbrl-packages",
        help="Directory where ZIP packages will be saved.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing ZIP files if they already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Build query parameters according to API documentation
    # See: https://filings.xbrl.org/docs/api
    query_params: dict[str, str] = {}
    
    # Filtering: filter[country]=GB format (per API docs)
    if args.country:
        query_params["filter[country]"] = args.country

    min_date: str | None = None
    if args.after_date:
        if len(args.after_date) == 4 and args.after_date.isdigit():
            min_date = f"{args.after_date}-01-01"
        else:
            min_date = args.after_date

    if args.from_year:
        min_date = f"{args.from_year}-01-01"

    # Date filtering (format may vary, using common JSON API filter syntax)
    if min_date:
        query_params["filter[processed][gte]"] = min_date

    # Company/entity filtering
    if args.company:
        query_params["filter[entity.name]"] = args.company

    # Include referenced resources: ?include=entity (per API docs)
    if args.include:
        query_params["include"] = args.include
    
    # Sorting: ?sort=-processed for most recent (per API docs)
    if args.sort:
        query_params["sort"] = args.sort
    
    # Pagination: page[size]=200 and page[number]=2 (per API docs)
    query_params["page[size]"] = str(min(args.page_size or 200, 200))

    query = args.filter if args.filter else urlencode(query_params)

    if min_date or args.company:
        full_url = f"{API_BASE}?{query}"
        print(f"Query URL: {full_url}")
        print(f"API Documentation: https://filings.xbrl.org/docs/api")
        if min_date:
            print(f"Filtering by processed date >= {min_date}")
        if args.company:
            print(f"Filtering by company/entity containing: {args.company}")
        print("(Note: Some filters may require client-side filtering if server-side support is limited.)")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total_start = time.time()
    total_processed = 0
    total_downloaded = 0

    for iteration in range(10):
        print(f"\n=== Iteration {iteration + 1}/10 (Page {iteration + 1}) ===")
        processed = 0
        downloaded = 0
        iteration_start = time.time()

        # Add page number to query for this iteration
        iteration_query_params = query_params.copy()
        iteration_query_params["page[number]"] = str(iteration + 1)
        iteration_query = args.filter if args.filter else urlencode(iteration_query_params)

        fetch_limit = args.limit * 10 if (min_date or args.company) and args.limit else None

        for filing in iter_filings(
            query=iteration_query,
            limit=fetch_limit,
            min_date=min_date,
            company_filter=args.company,
            single_page=True,  # Only fetch one page per iteration
        ):
            attrs = filing.get("attributes", {})
            if args.limit and processed >= args.limit:
                break

            filing_id = filing.get("id") or attrs.get("filing_index") or f"filing_{processed+1}"
            package_url = first_attr(
                attrs,
                "package_url",
                "xbrl_package_url",
                "report_package_url",
                "filing_package_url",
                "package",
            )

            processed += 1
            if not package_url:
                print(f"[warn] No package URL for {filing_id}")
                continue

            package_url = ensure_absolute_url(package_url)
            dest_path = out_dir / f"{filing_id}.zip"
            if dest_path.exists() and not args.overwrite:
                print(f"[skip] {dest_path} already exists (use --overwrite to replace).")
                downloaded += 1  # Count as downloaded since file already exists
                continue

            print(f"[download] {filing_id} -> {dest_path.name}")
            filing_start = time.time()
            try:
                response = requests.get(package_url, timeout=120)
                response.raise_for_status()
                dest_path.write_bytes(response.content)
                downloaded += 1
                elapsed = time.time() - filing_start
                print(f"[ok] Saved {dest_path} ({elapsed:.2f}s)")
            except Exception as exc:
                print(f"[error] Failed to download {filing_id}: {exc}")

        iteration_elapsed = time.time() - iteration_start
        total_processed += processed
        total_downloaded += downloaded
        print(f"Iteration {iteration + 1} done: processed {processed}, downloaded {downloaded} ({iteration_elapsed:.2f}s)")

    total_elapsed = time.time() - total_start
    avg = total_elapsed / total_downloaded if total_downloaded else 0.0
    print(f"\n=== Summary ===")
    print(f"Total processed filings: {total_processed}")
    print(f"Total packages downloaded: {total_downloaded}")
    print(f"Total time: {total_elapsed:.2f}s (avg {avg:.2f}s per download)")


if __name__ == "__main__":
    main()

