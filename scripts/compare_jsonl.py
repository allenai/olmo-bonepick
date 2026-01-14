#!/usr/bin/env python3
"""Compare two JSONL files and find disagreements on a specific field.

This script compares two compressed JSONL files (.jsonl.zst) and identifies
entries where they disagree on a specific field value (accessed via jq path).

Usage:
    python scripts/compare_jsonl.py file1.jsonl.zst file2.jsonl.zst \
        --value-path '.label' \
        --text-field 'text' \
        --output disagreements.jsonl
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import jq
import msgspec
import smart_open
from tqdm import tqdm


def compile_jq(jq_expr: str):
    """Compile a jq expression for extracting values."""
    compiled_jq = jq.compile(jq_expr)

    def extract(x: dict):
        try:
            result = compiled_jq.input_value(x).first()
            return result
        except StopIteration:
            return None

    return extract


def load_file_as_dict(file_path: Path, value_path: str, text_field: str) -> dict[str, tuple[Any, dict]]:
    """Load JSONL file into a dictionary keyed by text field.

    Args:
        file_path: Path to the JSONL file
        value_path: JQ expression to extract the comparison value
        text_field: Field name to use as the key

    Returns:
        Dictionary mapping text -> (extracted_value, full_row)
    """
    decoder = msgspec.json.Decoder()
    value_extractor = compile_jq(value_path)
    data = {}

    with smart_open.open(file_path, "rb") as f:
        for line_num, line in enumerate(f, 1):
            try:
                row = decoder.decode(line)
            except Exception as e:
                print(f"Error decoding line {line_num} in {file_path}: {e}", file=sys.stderr)
                continue

            if text_field not in row:
                print(f"Warning: '{text_field}' not found in line {line_num} of {file_path}", file=sys.stderr)
                continue

            text = str(row[text_field])
            value = value_extractor(row)

            # Store both the extracted value and the full row
            data[text] = (value, row)

    return data


def find_disagreements(
    file1_data: dict[str, tuple[Any, dict]],
    file2_data: dict[str, tuple[Any, dict]],
    file1_name: str,
    file2_name: str,
) -> list[dict]:
    """Find entries where the two files disagree on the value.

    Args:
        file1_data: Data from first file (text -> (value, row))
        file2_data: Data from second file (text -> (value, row))
        file1_name: Name of first file (for output)
        file2_name: Name of second file (for output)

    Returns:
        List of disagreement records with metadata
    """
    disagreements = []

    # Find common texts
    common_texts = set(file1_data.keys()) & set(file2_data.keys())

    for text in tqdm(common_texts, desc="Comparing entries", unit=" entries"):
        value1, row1 = file1_data[text]
        value2, row2 = file2_data[text]

        # Check if values disagree
        if value1 != value2:
            disagreements.append(
                {
                    "text": text,
                    "value_file1": value1,
                    "value_file2": value2,
                    "file1_name": file1_name,
                    "file2_name": file2_name,
                    "row_file1": row1,
                    "row_file2": row2,
                }
            )

    return disagreements


def main():
    parser = argparse.ArgumentParser(
        description="Compare two JSONL files and find disagreements on a specific field"
    )
    parser.add_argument("file1", type=Path, help="First JSONL file (compressed or uncompressed)")
    parser.add_argument("file2", type=Path, help="Second JSONL file (compressed or uncompressed)")
    parser.add_argument(
        "--value-path",
        type=str,
        required=True,
        help="JQ expression to extract the value to compare (e.g., '.label', '.metadata.score')",
    )
    parser.add_argument(
        "--text-field",
        type=str,
        default="text",
        help="Field name to use as the key for matching rows (default: 'text')",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output file for disagreements (JSONL format, optional zst compression)",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only print statistics, don't write output file",
    )

    args = parser.parse_args()

    # Validate input files
    if not args.file1.exists():
        print(f"Error: File not found: {args.file1}", file=sys.stderr)
        sys.exit(1)
    if not args.file2.exists():
        print(f"Error: File not found: {args.file2}", file=sys.stderr)
        sys.exit(1)

    # Load both files
    print(f"Loading {args.file1}...")
    file1_data = load_file_as_dict(args.file1, args.value_path, args.text_field)
    print(f"  Loaded {len(file1_data)} entries")

    print(f"Loading {args.file2}...")
    file2_data = load_file_as_dict(args.file2, args.value_path, args.text_field)
    print(f"  Loaded {len(file2_data)} entries")

    # Find disagreements
    print("\nFinding disagreements...")
    disagreements = find_disagreements(file1_data, file2_data, args.file1.name, args.file2.name)

    # Print statistics
    print("\n" + "=" * 60)
    print("COMPARISON STATISTICS")
    print("=" * 60)
    print(f"File 1: {args.file1.name} ({len(file1_data)} entries)")
    print(f"File 2: {args.file2.name} ({len(file2_data)} entries)")
    print(f"Common entries: {len(set(file1_data.keys()) & set(file2_data.keys()))}")
    print(f"Only in file 1: {len(set(file1_data.keys()) - set(file2_data.keys()))}")
    print(f"Only in file 2: {len(set(file2_data.keys()) - set(file1_data.keys()))}")
    print(f"Disagreements: {len(disagreements)}")

    if len(disagreements) > 0:
        common_count = len(set(file1_data.keys()) & set(file2_data.keys()))
        if common_count > 0:
            disagreement_rate = len(disagreements) / common_count * 100
            print(f"Disagreement rate: {disagreement_rate:.2f}%")

    # Write output if requested
    if args.output and not args.stats_only:
        print(f"\nWriting disagreements to {args.output}...")
        encoder = msgspec.json.Encoder()
        args.output.parent.mkdir(parents=True, exist_ok=True)

        with smart_open.open(args.output, "wb") as f:
            for disagreement in tqdm(disagreements, desc="Writing", unit=" entries"):
                f.write(encoder.encode(disagreement) + b"\n")

        print(f"Wrote {len(disagreements)} disagreements to {args.output}")
    elif not args.stats_only and len(disagreements) > 0:
        print("\nExample disagreements (first 5):")
        print("-" * 60)
        for i, d in enumerate(disagreements[:5], 1):
            print(f"\nExample {i}:")
            print(f"  Text: {d['text'][:100]}...")
            print(f"  {d['file1_name']}: {d['value_file1']}")
            print(f"  {d['file2_name']}: {d['value_file2']}")


if __name__ == "__main__":
    main()
