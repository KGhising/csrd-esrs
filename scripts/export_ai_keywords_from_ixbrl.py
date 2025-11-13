import argparse
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from zipfile import ZipFile

import pandas as pd
import re
from tqdm import tqdm

DEFAULT_REPORTS_DIR = Path("data/ixbrl-reports")
DEFAULT_PACKAGES_DIR = Path("data/xbrl-packages")
DEFAULT_OUTPUT = Path("data/csv-data/ai_keywords_ixbrl.csv")

RAW_KEYWORDS: List[str] = [
    "artificial intelligence",
    "machine learning",
    "generative ai",
    "intelligence artificielle",
    "apprentissage automatique",
    "inteligencia artificial",
    "inteligência artificial",
    "intelligenza artificiale",
    "künstliche intelligenz",
    "tekoäly",
    "tehisintellekt",
]


def build_flexible_pattern(keyword: str) -> re.Pattern:
    tokens = keyword.split()
    token_patterns = []
    for token in tokens:
        letters = list(token)
        letter_pattern = "".join(f"{re.escape(letter)}[\\s\\u200b]*" for letter in letters)
        letter_pattern = letter_pattern.rstrip("[\\s\\u200b]*")
        token_patterns.append(letter_pattern)
    pattern_str = r"\b" + r"\s+".join(token_patterns) + r"\b"
    return re.compile(pattern_str, re.IGNORECASE)


KEYWORD_PATTERNS: Dict[str, re.Pattern] = {
    keyword: build_flexible_pattern(keyword) for keyword in RAW_KEYWORDS
}

SINGLE_WORD_KEYWORDS: Dict[str, re.Pattern] = {
    "ai": re.compile(r"\bai\b", re.IGNORECASE),
}


class SimpleHTMLExtractor(HTMLParser):
    """Extract visible text from iXBRL documents while skipping hidden content."""

    IGNORE_TAGS = {
        "script",
        "style",
        "noscript",
        "template",
        "svg",
        "math",
        "head",
        "meta",
        "link",
        "title",
        "object",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._ignore_stack: List[bool] = [False]
        self._chunks: List[str] = []

    def _should_ignore(self, tag: str, attrs) -> bool:
        tag_lower = tag.lower()
        if tag_lower in self.IGNORE_TAGS or tag_lower.startswith("ix:hidden"):
            return True

        attr_map = {name.lower(): (value or "") for name, value in attrs}

        if any(
            substring in attr_map.get("style", "").lower()
            for substring in ("display:none", "visibility:hidden", "opacity:0")
        ):
            return True

        if "hidden" in attr_map:
            return True

        if attr_map.get("aria-hidden", "").lower() in {"true", "1"}:
            return True

        return False

    def handle_starttag(self, tag: str, attrs) -> None:
        parent_ignored = self._ignore_stack[-1]
        ignore = parent_ignored or self._should_ignore(tag, attrs)
        self._ignore_stack.append(ignore)

    def handle_startendtag(self, tag: str, attrs) -> None:
        parent_ignored = self._ignore_stack[-1]
        ignore = parent_ignored or self._should_ignore(tag, attrs)
        if not ignore:
            pass

    def handle_endtag(self, tag: str) -> None:
        if self._ignore_stack:
            self._ignore_stack.pop()

    def handle_data(self, data: str) -> None:
        if not self._ignore_stack or self._ignore_stack[-1]:
            return
        if not data:
            return
        self._chunks.append(data)

    def handle_comment(self, data: str) -> None: 
        pass

    @property
    def text(self) -> str:
        raw = "".join(self._chunks)
        return re.sub(r"\s+", " ", raw).strip()


def decode_html(content: bytes) -> str:
    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    return content.decode("utf-8", errors="ignore")


def extract_text(html: str) -> str:
    parser = SimpleHTMLExtractor()
    try:
        parser.feed(html)
        parser.close()
    except Exception:
        return html
    return parser.text.lower()


def analyse_text(text: str) -> Counter:
    counter: Counter = Counter()
    for keyword, pattern in KEYWORD_PATTERNS.items():
        matches = pattern.findall(text)
        if matches:
            counter[keyword] += len(matches)

    for keyword, pattern in SINGLE_WORD_KEYWORDS.items():
        for match in pattern.finditer(text):
            start, end = match.span()
            prev_char = text[start - 1] if start > 0 else ""
            next_char = text[end] if end < len(text) else ""

            prev_alpha = prev_char.isalpha()
            next_alpha = next_char.isalpha()

            if prev_alpha and next_alpha:
                continue

            counter[keyword] += 1

    return counter


def process_ixbrl_document(name: str, content: bytes) -> Tuple[int, Counter]:
    html = decode_html(content)
    text = extract_text(html)
    keyword_counts = analyse_text(text)
    return sum(keyword_counts.values()), keyword_counts


def iter_ixbrl_files(zip_path: Path) -> List[Tuple[str, bytes]]:
    documents: List[Tuple[str, bytes]] = []
    with ZipFile(zip_path) as zf:
        for info in zf.infolist():
            filename = info.filename
            if info.is_dir():
                continue
            lower_name = filename.lower()
            if lower_name.endswith((".xhtml", ".html")):
                try:
                    with zf.open(info) as file_obj:
                        documents.append((filename, file_obj.read()))
                except Exception as exc:
                    print(f"[warn] Failed to read {filename} in {zip_path.name}: {exc}")
    return documents


def process_package(zip_path: Path) -> List[Dict[str, object]]:
    documents = iter_ixbrl_files(zip_path)
    if not documents:
        print(f"[warn] No iXBRL documents found in {zip_path.name}")
        return []

    package_counter: Counter = Counter()
    total_keywords = 0
    records: List[Dict[str, object]] = []

    for doc_name, content in documents:
        doc_total, doc_counter = process_ixbrl_document(doc_name, content)
        total_keywords += doc_total
        package_counter.update(doc_counter)

        records.append(
            {
                "package": zip_path.name,
                "document": doc_name,
                "ai_keyword_total": doc_total,
                "top_keywords": ", ".join(
                    f"{keyword}:{count}" for keyword, count in doc_counter.most_common() if count > 0
                )
                or "none",
            }
        )

    records.append(
        {
            "package": zip_path.name,
            "document": "__package_total__",
            "ai_keyword_total": total_keywords,
            "top_keywords": ", ".join(
                f"{keyword}:{count}" for keyword, count in package_counter.most_common() if count > 0
            )
            or "none",
        }
    )

    return records


def collect_results(packages_dir: Path, limit: Optional[int] = None) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    files = sorted(packages_dir.glob("*.zip"))
    if limit:
        files = files[:limit]
    for zip_path in tqdm(files, desc="Processing packages", unit="package"):
        results.extend(process_package(zip_path))

    return results


def collect_from_reports(reports_dir: Path) -> List[Dict[str, object]]:
    files = sorted(reports_dir.glob("*.xhtml")) + sorted(reports_dir.glob("*.html"))
    if not files:
        return []

    directory_name = reports_dir.name
    dir_counter: Counter = Counter()
    total_keywords = 0
    records: List[Dict[str, object]] = []

    for file_path in tqdm(files, desc=f"Processing {directory_name}", unit="file"):
        try:
            content = file_path.read_bytes()
        except Exception as exc:
            print(f"[warn] Failed to read {file_path}: {exc}")
            continue

        doc_total, doc_counter = process_ixbrl_document(file_path.name, content)
        total_keywords += doc_total
        dir_counter.update(doc_counter)

        records.append(
            {
                "package": directory_name,
                "document": file_path.name,
                "ai_keyword_total": doc_total,
                "top_keywords": ", ".join(
                    f"{keyword}:{count}" for keyword, count in doc_counter.most_common() if count > 0
                )
                or "none",
            }
        )

    if records:
        records.append(
            {
                "package": directory_name,
                "document": "__directory_total__",
                "ai_keyword_total": total_keywords,
                "top_keywords": ", ".join(
                    f"{keyword}:{count}" for keyword, count in dir_counter.most_common() if count > 0
                )
                or "none",
            }
        )

    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract AI keyword counts directly from iXBRL report packages."
    )
    parser.add_argument(
        "--reports-dir",
        default=str(DEFAULT_REPORTS_DIR),
        help="Directory containing extracted iXBRL report files (default: data/ixbrl-reports).",
    )
    parser.add_argument(
        "--packages-dir",
        help="Optional directory containing XBRL report packages (zip). "
             "If provided, packages are processed in addition to reports.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="CSV path to store AI keyword counts (default: data/csv-data/ai_keywords_ixbrl.csv).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional limit on the number of packages to process.",
    )

    args = parser.parse_args()
    reports_dir = Path(args.reports_dir)
    packages_dir = Path(args.packages_dir) if args.packages_dir else None

    results: List[Dict[str, object]] = []

    if reports_dir.exists():
        report_records = collect_from_reports(reports_dir)
        if report_records:
            results.extend(report_records)
        else:
            print(f"[warn] No iXBRL files found in {reports_dir}")
    else:
        print(f"[warn] Reports directory not found: {reports_dir}")

    if packages_dir:
        if packages_dir.exists():
            package_records = collect_results(packages_dir, limit=args.limit)
            if package_records:
                results.extend(package_records)
        else:
            print(f"[warn] Packages directory not found: {packages_dir}")

    if not results:
        print("No AI keyword results generated from the provided sources.")
        return

    df = pd.DataFrame(results)
    df = df.sort_values(by=["package", "document"]).reset_index(drop=True)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    packages_processed = df["package"].nunique()
    print(f"AI keyword counts (iXBRL) saved to: {output_path}")
    print(f"Packages processed: {packages_processed}")
    print(f"Rows written: {len(df)}")


if __name__ == "__main__":
    main()

