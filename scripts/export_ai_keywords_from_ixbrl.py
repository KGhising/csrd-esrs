import argparse
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlparse
from xml.etree import ElementTree as ET

import pandas as pd
import re
from tqdm import tqdm

DEFAULT_REPORTS_DIR = Path("data/ixbrl-reports")
DEFAULT_OUTPUT = Path("data/csv-data/ai_keywords_ixbrl.csv")

COMPANY_NAME_CONCEPTS: Sequence[str] = (
    "ifrs-full:NameOfReportingEntityOrOtherMeansOfIdentification",
    "ifrs-full:NameOfParentEntity",
    "esef_cor:NameOrOtherDesignationOfReportingEntity",
    "esef_cor:NameOfReportingEntityOrOtherMeansOfIdentification",
    "esef_cor:LegalName",
    "esef_cor:NameOfParentCompany",
    "esef_cor:NameOfIssuer",
    "esef_cor:EntityRegisteredName",
    "uk-core:EntityName",
    "core:EntityName",
)

IX_NS = "http://www.xbrl.org/2013/inlineXBRL"
XBRLI_NS = "http://www.xbrl.org/2003/instance"
LINK_NS = "http://www.xbrl.org/2003/linkbase"
XLINK_NS = "http://www.w3.org/1999/xlink"
LEI_PATTERN = re.compile(r"[A-Z0-9]{20}")

def iter_values(value: Any) -> Iterable[str]:
    if value is None:
        return
    if isinstance(value, str):
        yield value
        return
    if isinstance(value, (list, tuple, set)):
        for item in value:
            yield from iter_values(item)
        return
    if isinstance(value, dict):
        for item in value.values():
            yield from iter_values(item)
        return
    yield str(value)


def extract_lei(*candidates: Any) -> Optional[str]:
    for candidate in candidates:
        for text in iter_values(candidate):
            normalized = re.sub(r"[^A-Z0-9]", "", text.upper())
            match = LEI_PATTERN.search(normalized)
            if match:
                return match.group(0)
    return None


def split_slug_tokens(slug: str) -> List[str]:
    if not slug:
        return []
    raw_tokens = re.findall(r"[A-Za-z]+|\d+", slug)
    tokens: List[str] = []
    for token in raw_tokens:
        if token.isdigit():
            tokens.append(token)
        else:
            tokens.append(token.capitalize())
    if tokens and tokens[-1].lower() == "plc":
        tokens[-1] = "PLC"
    return tokens


def derive_company_name_from_href(href: Optional[str]) -> Optional[str]:
    if not href:
        return None
    parsed = urlparse(href)
    host = parsed.netloc or ""
    if not host:
        host = parsed.path.split("/")[0] if parsed.path else ""
    if not host:
        return None
    host = host.split(":")[0]
    if host.startswith("www."):
        host = host[4:]
    parts = host.split(".")
    core = parts[-2] if len(parts) >= 2 else parts[0]
    tokens = split_slug_tokens(core)
    if not tokens:
        return None
    return " ".join(tokens)


def build_continuation_map(root: ET.Element) -> Dict[str, ET.Element]:
    continuations: Dict[str, ET.Element] = {}
    for elem in root.findall(f".//{{{IX_NS}}}continuation"):
        elem_id = elem.get("id")
        if elem_id:
            continuations[elem_id] = elem
    return continuations


def get_fact_text(fact: ET.Element, continuation_map: Dict[str, ET.Element]) -> str:
    parts: List[str] = ["".join(fact.itertext())]
    continuation_ref = fact.get("continuedAt")
    while continuation_ref:
        continuation = continuation_map.get(continuation_ref)
        if continuation is None:
            break
        parts.append("".join(continuation.itertext()))
        continuation_ref = continuation.get("continuedAt")
    text = "".join(parts)
    return re.sub(r"\s+", " ", text).strip()


def extract_company_name_from_ixbrl(
    root: ET.Element,
    continuation_map: Dict[str, ET.Element],
) -> Optional[str]:
    for concept in COMPANY_NAME_CONCEPTS:
        for fact in root.findall(f".//{{{IX_NS}}}nonNumeric[@name='{concept}']"):
            text = get_fact_text(fact, continuation_map)
            if text:
                return text
    schema_ref = root.find(f".//{{{LINK_NS}}}schemaRef")
    if schema_ref is not None:
        href = schema_ref.get(f"{{{XLINK_NS}}}href")
        name = derive_company_name_from_href(href)
        if name:
            return name
    return None


def extract_metadata_from_html(html: str) -> Tuple[Optional[str], Optional[str]]:
    try:
        root = ET.fromstring(html)
    except ET.ParseError:
        return None, None

    lei = None
    for context in root.findall(f".//{{{XBRLI_NS}}}context"):
        identifier_elem = context.find(f".//{{{XBRLI_NS}}}identifier")
        if identifier_elem is not None and identifier_elem.text:
            lei = extract_lei(identifier_elem.text, identifier_elem.attrib)
            if lei:
                break

    if not lei:
        return None, None

    continuation_map = build_continuation_map(root)
    company_name = extract_company_name_from_ixbrl(root, continuation_map)
    return lei, company_name


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


def process_ixbrl_document(name: str, content: bytes) -> Optional[Tuple[str, Optional[str], int, Counter]]:
    html = decode_html(content)
    entity, company_name = extract_metadata_from_html(html)
    if not entity:
        return None
    text = extract_text(html)
    keyword_counts = analyse_text(text)
    return entity, company_name, sum(keyword_counts.values()), keyword_counts


def collect_from_reports(reports_dir: Path) -> List[Dict[str, object]]:
    files = sorted(reports_dir.glob("*.xhtml")) + sorted(reports_dir.glob("*.html"))
    if not files:
        return []

    entity_stats: Dict[str, Dict[str, Any]] = {}

    for file_path in tqdm(files, desc=f"Processing {reports_dir.name}", unit="file"):
        try:
            content = file_path.read_bytes()
        except Exception as exc:
            print(f"[warn] Failed to read {file_path}: {exc}")
            continue

        result = process_ixbrl_document(file_path.name, content)
        if result is None:
            continue
        entity, company_name, doc_total, doc_counter = result

        entry = entity_stats.setdefault(
            entity,
            {
                "entity": entity,
                "company_name": company_name or "",
                "ai_keyword_total": 0,
                "counter": Counter(),
            },
        )
        if company_name and not entry["company_name"]:
            entry["company_name"] = company_name
        entry["ai_keyword_total"] += doc_total
        entry["counter"].update(doc_counter)

    records: List[Dict[str, object]] = []
    for entity_id, stats in entity_stats.items():
        records.append(
            {
                "entity": entity_id,
                "company_name": stats["company_name"],
                "ai_keyword_total": stats["ai_keyword_total"],
                "top_keywords": ", ".join(
                    f"{keyword}:{count}" for keyword, count in stats["counter"].most_common() if count > 0
                )
                or "none",
            }
        )

    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract AI keyword counts from iXBRL reports, aggregated by entity (LEI)."
    )
    parser.add_argument(
        "--reports-dir",
        default=str(DEFAULT_REPORTS_DIR),
        help="Directory containing extracted iXBRL report files (default: data/ixbrl-reports).",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="CSV path to store AI keyword counts (default: data/csv-data/ai_keywords_ixbrl.csv).",
    )

    args = parser.parse_args()
    reports_dir = Path(args.reports_dir)

    results: List[Dict[str, object]] = []

    if reports_dir.exists():
        report_records = collect_from_reports(reports_dir)
        if report_records:
            results.extend(report_records)
        else:
            print(f"[warn] No iXBRL files found in {reports_dir}")
    else:
        print(f"[warn] Reports directory not found: {reports_dir}")

    if not results:
        print("No AI keyword results generated from the provided sources.")
        return

    df = pd.DataFrame(results)
    df = df.sort_values(by=["entity"]).reset_index(drop=True)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"AI keyword counts (iXBRL) saved to: {output_path}")
    print(f"Entities processed: {df['entity'].nunique()}")
    print(f"Rows written: {len(df)}")


if __name__ == "__main__":
    main()

