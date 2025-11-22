import argparse
from pathlib import Path
from zipfile import ZipFile


DEFAULT_PACKAGES_DIR = Path("data/xbrl-packages")
DEFAULT_OUTPUT_DIR = Path("data/ixbrl-reports")


def extract_reports_from_zip(zip_path: Path, output_dir: Path, overwrite: bool = False) -> tuple[int, int]:
    """
    Extract iXBRL reports from a ZIP package.
    
    Returns:
        tuple: (extracted_count, skipped_count)
    """
    extracted = 0
    skipped = 0
    with ZipFile(zip_path) as zf:
        for member in zf.namelist():
            member_lower = member.lower()
            if not member_lower.endswith((".xhtml", ".html")):
                continue
            file_name = Path(member).name
            dest_path = output_dir / file_name
            if dest_path.exists() and not overwrite:
                skipped += 1
                continue

            data = zf.read(member)
            dest_path.write_bytes(data)
            extracted += 1
    return extracted, skipped


def iter_package_files(packages_dir: Path):
    for path in sorted(packages_dir.glob("*.zip")):
        if path.is_file():
            yield path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract iXBRL (.xhtml/.html) reports from downloaded XBRL packages."
    )
    parser.add_argument(
        "--packages-dir",
        default=str(DEFAULT_PACKAGES_DIR),
        help="Directory containing downloaded XBRL package ZIP files.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Destination directory for extracted iXBRL reports.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing report files if they already exist.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional limit on the number of packages to process.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    packages_dir = Path(args.packages_dir)
    output_dir = Path(args.output_dir)

    if not packages_dir.exists():
        raise SystemExit(f"Packages directory not found: {packages_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    extracted_total = 0
    skipped_total = 0

    for zip_path in iter_package_files(packages_dir):
        if args.limit is not None and processed >= args.limit:
            break

        try:
            extracted, skipped = extract_reports_from_zip(zip_path, output_dir, overwrite=args.overwrite)
        except Exception as exc:
            print(f"[error] Failed to extract from {zip_path.name}: {exc}")
            continue

        if extracted > 0 or skipped > 0:
            msg_parts = []
            if extracted > 0:
                msg_parts.append(f"extracted {extracted}")
            if skipped > 0:
                msg_parts.append(f"skipped {skipped} (already exist)")
            print(f"[ok] {zip_path.name}: {', '.join(msg_parts)} report(s)")
            extracted_total += extracted
            skipped_total += skipped
        else:
            print(f"[warn] {zip_path.name}: no iXBRL reports found")

        processed += 1

    print(f"\nPackages processed: {processed}")
    print(f"Reports extracted: {extracted_total}")
    if skipped_total > 0:
        print(f"Reports skipped (already exist): {skipped_total}")


if __name__ == "__main__":
    main()

