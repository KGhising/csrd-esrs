import argparse
import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlparse
from xml.etree import ElementTree as ET

import pandas as pd
from tqdm import tqdm

DEFAULT_INPUT_DIR = Path("data/ixbrl-reports")
DEFAULT_OUTPUT = Path("data/csv-data/financial_facts.csv")

COMPANY_NAME_CONCEPTS: Sequence[str] = (
    "ifrs-full:NameOfReportingEntityOrOtherMeansOfIdentification",
    "ifrs-full:NameOfParentEntity",
    "esef_cor:NameOfReportingEntityOrOtherMeansOfIdentification",
    "esef_cor:LegalName",
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


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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


def parse_decimal(value: str) -> Optional[Decimal]:
    if not value:
        return None

    cleaned = (
        value.replace("\u2212", "-")
        .replace("\u2013", "-")
        .replace("\xa0", "")
        .strip()
    )

    negative = False
    if cleaned.startswith("(") and cleaned.endswith(")"):
        negative = True
        cleaned = cleaned[1:-1]

    cleaned = cleaned.strip()
    if not cleaned:
        return None

    cleaned = cleaned.replace(" ", "")

    if cleaned.count(",") == 1 and cleaned.count(".") == 0:
        cleaned = cleaned.replace(",", ".")
    else:
        cleaned = cleaned.replace(",", "")

    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", cleaned)
    if not match:
        return None

    numeric = match.group(0)

    try:
        number = Decimal(numeric)
    except InvalidOperation:
        return None

    if negative:
        number = -number
    return number


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


def load_ixbrl(path: Path, target_concepts: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    try:
        tree = ET.parse(str(path))
    except ET.ParseError as exc:
        print(f"[warn] Failed to parse {path.name} as iXBRL: {exc}")
        return []

    root = tree.getroot()
    continuation_map = build_continuation_map(root)
    company_name = extract_company_name_from_ixbrl(root, continuation_map)

    contexts: Dict[str, Dict[str, str]] = {}
    for context in root.findall(f".//{{{XBRLI_NS}}}context"):
        context_id = context.get("id")
        if not context_id:
            continue

        identifier_elem = context.find(f".//{{{XBRLI_NS}}}identifier")
        if identifier_elem is None or not identifier_elem.text:
            continue

        lei = extract_lei(identifier_elem.text, identifier_elem.attrib)
        if not lei:
            continue

        period_elem = context.find(f".//{{{XBRLI_NS}}}period")
        period = "Unknown"
        if period_elem is not None:
            instant_elem = period_elem.find(f"./{{{XBRLI_NS}}}instant")
            if instant_elem is not None and instant_elem.text:
                period = instant_elem.text.strip()
            else:
                start_elem = period_elem.find(f"./{{{XBRLI_NS}}}startDate")
                end_elem = period_elem.find(f"./{{{XBRLI_NS}}}endDate")
                if (
                    start_elem is not None
                    and end_elem is not None
                    and start_elem.text
                    and end_elem.text
                ):
                    period = f"{start_elem.text.strip()}/{end_elem.text.strip()}"

        contexts[context_id] = {
            "entity": lei,
            "period": period or "Unknown",
        }

    rows: List[Dict[str, Any]] = []
    for fact in root.findall(f".//{{{IX_NS}}}nonFraction"):
        concept = fact.get("name", "")
        if not concept:
            continue

        if ":" in concept:
            namespace, name = concept.split(":", 1)
        else:
            namespace, name = "", concept

        # Filter by concepts only if target_concepts is specified and not empty
        if target_concepts and name not in target_concepts:
            continue

        context_ref = fact.get("contextRef")
        if not context_ref:
            continue

        context_info = contexts.get(context_ref)
        if not context_info:
            continue

        text_value = get_fact_text(fact, continuation_map)
        decimal_value = parse_decimal(text_value)
        if decimal_value is None:
            continue

        scale_attr = fact.get("scale")
        if scale_attr:
            try:
                scale = int(scale_attr)
                decimal_value = decimal_value * (Decimal(10) ** scale)
            except ValueError:
                pass

        if fact.get("sign") == "-":
            decimal_value = -decimal_value

        fact_id = fact.get("id") or f"{name}_{context_ref}"

        rows.append(
            {
                "fact_id": fact_id,
                "concept": name,
                "concept_namespace": namespace,
                "entity": context_info["entity"],
                "company_name": company_name,
                "period": context_info["period"],
                "value": float(decimal_value),
                "source_file": path.name,
            }
        )

    return rows


def extract_company_name_from_json(data: Dict[str, Any]) -> Optional[str]:
    for path in (
        ("documentInfo", "EntityRegistrantName"),
        ("documentInfo", "entityRegistrantName"),
        ("entityInformation", "EntityName"),
        ("entityInformation", "entityName"),
        ("entity", "name"),
        ("entity", "entityName"),
        ("metadata", "entityName"),
        ("companyName",),
    ):
        node: Any = data
        for key in path:
            if isinstance(node, dict) and key in node:
                node = node[key]
            else:
                node = None
                break
        if isinstance(node, str) and node.strip():
            return node.strip()
    return None


def coerce_numeric(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, str):
        decimal_value = parse_decimal(value)
        if decimal_value is not None:
            return float(decimal_value)
        try:
            return float(value)
        except ValueError:
            return None
    return None


def extract_row_from_json(
    fact_id: str,
    fact: Dict[str, Any],
    source_file: str,
    company_name: Optional[str],
    target_concepts: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    dimensions = fact.get("dimensions", {})
    concept: str = dimensions.get("concept", "")
    if not concept:
        return None

    if ":" in concept:
        namespace, name = concept.split(":", 1)
    else:
        namespace, name = "", concept

    # Filter by concepts only if target_concepts is specified and not empty
    if target_concepts and name not in target_concepts:
        return None

    lei = extract_lei(
        dimensions.get("entity"),
        dimensions.get("identifier"),
        fact.get("entity"),
        fact.get("identifier"),
        fact.get("entityIdentifier"),
    )
    if not lei:
        return None

    value = coerce_numeric(fact.get("value"))
    if value is None:
        return None

    period = dimensions.get("period")

    return {
        "fact_id": fact_id,
        "concept": name,
        "concept_namespace": namespace,
        "entity": lei,
        "company_name": company_name,
        "period": period,
        "value": value,
        "source_file": source_file,
    }


def get_processed_files(output_path: Path) -> set[str]:
    """Get set of source files that have already been processed (from existing CSV)."""
    if not output_path.exists():
        return set()
    try:
        df = pd.read_csv(output_path)
        if "source_file" in df.columns:
            return set(df["source_file"].unique())
    except Exception:
        pass
    return set()


def gather_concepts(
    files: List[Path],
    target_concepts: Optional[List[str]] = None,
    skip_processed: bool = False,
    processed_files: Optional[set[str]] = None,
) -> tuple[List[Dict[str, Any]], int]:
    """
    Gather concepts from files.
    
    Returns:
        tuple: (rows, skipped_count)
    """
    rows: List[Dict[str, Any]] = []
    skipped = 0
    processed_files = processed_files or set()
    
    for path in tqdm(files, desc="Processing filings", unit="file"):
        # Skip if file has already been processed
        if skip_processed and path.name in processed_files:
            skipped += 1
            continue
            
        suffix = path.suffix.lower()
        if suffix == ".json":
            try:
                data = load_json(path)
            except Exception as exc:
                print(f"[warn] Failed to read {path.name}: {exc}")
                continue
            company_name = extract_company_name_from_json(data)
            facts = data.get("facts", {})
            for fact_id, fact in facts.items():
                record = extract_row_from_json(fact_id, fact, path.name, company_name, target_concepts)
                if record:
                    rows.append(record)
        elif suffix in {".xhtml", ".html"}:
            rows.extend(load_ixbrl(path, target_concepts))
        else:
            print(f"[warn] Unsupported file type skipped: {path.name}")

    return rows, skipped


def clean_period(period: Optional[str]) -> str:
    if not period:
        return "Unknown"
    text = str(period)
    if "T" in text and "/" in text:
        text = text.split("/", 1)[0]
    if "T" in text:
        return text.split("T")[0][:4]
    if "/" in text:
        return text.split("/", 1)[0][:4]
    return text[:4]


def to_tabular(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    if "company_name" not in df.columns:
        df["company_name"] = None

    df["year"] = df["period"].apply(clean_period)
    df["entity"] = df["entity"].apply(str)
    df["company_name"] = df["company_name"].fillna(df["entity"])

    pivot = df.pivot_table(
        index=["entity", "company_name", "concept"],
        columns="year",
        values="value",
        aggfunc="first",
    )
    pivot = pivot.reset_index()
    pivot = pivot.sort_values(["entity", "concept"]).reset_index(drop=True)
    return pivot


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export financial facts from iXBRL or XBRL JSON filings. Extracts all facts by default."
    )
    parser.add_argument(
        "--input-dir",
        default=str(DEFAULT_INPUT_DIR),
        help="Directory containing iXBRL (.xhtml/.html) or XBRL JSON files (default: data/ixbrl-reports).",
    )
    parser.add_argument(
        "--file",
        help="Process a single iXBRL or JSON file instead of the entire directory.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Path to write the tabular CSV (default: data/csv-data/financial_facts.csv).",
    )
    parser.add_argument(
        "--concepts",
        nargs="+",
        default=None,
        help="Filter by specific concepts (e.g., --concepts Equity Revenue Assets). If not specified, extracts all facts.",
    )
    parser.add_argument(
        "--skip-processed",
        action="store_true",
        help="Skip files that have already been processed (based on source_file in existing output CSV).",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing CSV instead of overwriting (combines with existing data).",
    )

    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    if args.file:
        files = [Path(args.file)]
    else:
        files = sorted(input_dir.glob("*.json"))
        files += sorted(input_dir.glob("*.xhtml"))
        files += sorted(input_dir.glob("*.html"))

    if not files:
        target = input_dir if not args.file else Path(args.file).parent
        print(f"No supported files found in {target}")
        return

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get list of already processed files if skipping
    processed_files = set()
    existing_tabular = None
    if args.skip_processed or args.append:
        processed_files = get_processed_files(output_path)
        if args.append and output_path.exists():
            try:
                existing_tabular = pd.read_csv(output_path)
                if len(processed_files) > 0:
                    print(f"Found {len(processed_files)} already processed files in existing CSV")
            except Exception as exc:
                print(f"[warn] Could not read existing CSV: {exc}")

    target_concepts = args.concepts
    rows, skipped = gather_concepts(files, target_concepts, skip_processed=args.skip_processed, processed_files=processed_files)
    
    if skipped > 0:
        print(f"Skipped {skipped} already processed file(s)")
    
    if not rows:
        if existing_tabular is not None and args.append:
            print("No new facts found, keeping existing CSV.")
            return
        print("No key concept facts found in the provided files.")
        return

    tabular = to_tabular(rows)
    if tabular.empty:
        if existing_tabular is not None and args.append:
            print("No new tabular data produced, keeping existing CSV.")
            return
        print("No tabular data produced.")
        return

    # Combine with existing data if appending
    if args.append and existing_tabular is not None and not existing_tabular.empty:
        # Set index for merging, then combine (new values take precedence, old fills gaps)
        index_cols = ["entity", "company_name", "concept"]
        existing_tabular = existing_tabular.set_index(index_cols)
        tabular = tabular.set_index(index_cols)
        # Combine: new values override old ones, but keep old values where new ones don't exist
        # Start with new values, fill missing with old values
        tabular = tabular.combine_first(existing_tabular)
        tabular = tabular.reset_index()
        tabular = tabular.sort_values(["entity", "concept"]).reset_index(drop=True)

    tabular.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Key concept tabular data saved to: {output_path}")
    print(f"Shape: {tabular.shape[0]} rows × {tabular.shape[1]} columns")


if __name__ == "__main__":
    main()

