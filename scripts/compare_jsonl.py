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
from rich.console import Console
from rich.panel import Panel
from rich.table import Table


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


def show_disagreements_interactive(disagreements: list[dict], text_field: str, print_lines: int = 20):
    """Interactively display disagreements, one at a time.

    Args:
        disagreements: List of disagreement records
        text_field: Name of the text field being used for matching
    """
    console = Console()

    if not disagreements:
        console.print("\n[yellow]No disagreements to show.[/yellow]")
        return

    console.print(
        f"\n[bold cyan]{'=' * 80}[/bold cyan]\n"
        f"[bold cyan]INTERACTIVE DISAGREEMENT VIEWER[/bold cyan] [dim]({len(disagreements)} total)[/dim]\n"
        f"[bold cyan]{'=' * 80}[/bold cyan]"
    )
    console.print("[dim]Press ENTER to see next example, 'q' + ENTER to quit, 's' + ENTER to skip to end[/dim]\n")

    for i, d in enumerate(disagreements, 1):
        text = d["text"]
        if print_lines > 0:
            lines = d["text"].split("\n")[:print_lines]
            text_display = "\n".join(lines)
        else:
            text_display = d["text"]

        # Display the text (potentially truncated for readability)
        if len(text_display) < len(text_display):
            text_display = f"{text_display}\n[dim]... (truncated, {len(text)} chars total)[/dim]"

        # Create a panel for the disagreement
        content = (
            f"[bold white]{text_field}:[/bold white]\n{text_display}\n\n"
            f"[bold magenta]{d['file1_name']}:[/bold magenta] [magenta]{d['value_file1']}[/magenta]\n"
            f"[bold cyan]{d['file2_name']}:[/bold cyan] [cyan]{d['value_file2']}[/cyan]"
        )

        console.print(
            Panel(
                content,
                title=f"[bold yellow]Example {i} of {len(disagreements)}[/bold yellow]",
                border_style="yellow",
            )
        )

        # Wait for user input
        if i < len(disagreements):
            try:
                user_input = input("\n[ENTER=next, q=quit, s=skip to end] ").strip().lower()
                if user_input == "q":
                    console.print("\n[yellow]Exiting viewer.[/yellow]")
                    break
                elif user_input == "s":
                    console.print(
                        f"\n[yellow]Skipping to end. Showed {i} of {len(disagreements)} examples.[/yellow]"
                    )
                    break
                console.print()  # Add spacing between examples
            except (KeyboardInterrupt, EOFError):
                console.print("\n\n[yellow]Exiting viewer.[/yellow]")
                break
        else:
            console.print(
                f"\n[bold green]{'=' * 80}[/bold green]\n"
                f"[bold green]Reached end. Showed all {len(disagreements)} disagreements.[/bold green]\n"
                f"[bold green]{'=' * 80}[/bold green]"
            )


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
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Interactively browse disagreements (press ENTER to see each example)",
    )
    parser.add_argument(
        "--print-lines",
        default=20,
        help="Print the first N lines of the text field to compare",
        type=int,
    )
    args = parser.parse_args()

    console = Console()

    # Auto-enable interactive mode if no output file specified
    if not args.output and not args.stats_only and not args.interactive:
        args.interactive = True

    # Validate input files
    if not args.file1.exists():
        console.print(f"[bold red]Error:[/bold red] File not found: {args.file1}")
        sys.exit(1)
    if not args.file2.exists():
        console.print(f"[bold red]Error:[/bold red] File not found: {args.file2}")
        sys.exit(1)

    # Load both files
    console.print(f"[yellow]Loading {args.file1}...[/yellow]")
    file1_data = load_file_as_dict(args.file1, args.value_path, args.text_field)
    console.print(f"  Loaded [bold]{len(file1_data):,}[/bold] entries")

    console.print(f"[yellow]Loading {args.file2}...[/yellow]")
    file2_data = load_file_as_dict(args.file2, args.value_path, args.text_field)
    console.print(f"  Loaded [bold]{len(file2_data):,}[/bold] entries")

    # Find disagreements
    console.print("\n[yellow]Finding disagreements...[/yellow]")
    disagreements = find_disagreements(file1_data, file2_data, args.file1.name, args.file2.name)

    # Print statistics using rich table
    console.print()
    table = Table(title="[bold cyan]COMPARISON STATISTICS[/bold cyan]", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Count", justify="right", style="magenta")

    common_count = len(set(file1_data.keys()) & set(file2_data.keys()))
    only_in_1 = len(set(file1_data.keys()) - set(file2_data.keys()))
    only_in_2 = len(set(file2_data.keys()) - set(file1_data.keys()))

    table.add_row(f"File 1: {args.file1.name}", f"{len(file1_data):,}")
    table.add_row(f"File 2: {args.file2.name}", f"{len(file2_data):,}")
    table.add_row("Common entries", f"{common_count:,}")
    table.add_row("Only in file 1", f"{only_in_1:,}")
    table.add_row("Only in file 2", f"{only_in_2:,}")
    table.add_row("Disagreements", f"{len(disagreements):,}")

    if len(disagreements) > 0 and common_count > 0:
        disagreement_rate = len(disagreements) / common_count * 100
        table.add_row("Disagreement rate", f"{disagreement_rate:.2f}%")

    console.print(table)
    console.print()

    # Write output if requested
    if args.output and not args.stats_only:
        console.print(f"[yellow]Writing disagreements to {args.output}...[/yellow]")
        encoder = msgspec.json.Encoder()
        args.output.parent.mkdir(parents=True, exist_ok=True)

        with smart_open.open(args.output, "wb") as f:
            for disagreement in tqdm(disagreements, desc="Writing", unit=" entries"):
                f.write(encoder.encode(disagreement) + b"\n")

        console.print(f"[green]Wrote {len(disagreements):,} disagreements to {args.output}[/green]")

    # Interactive mode
    if args.interactive and len(disagreements) > 0:
        show_disagreements_interactive(disagreements, args.text_field, args.print_lines)
    elif not args.stats_only and not args.interactive and len(disagreements) > 0:
        console.print("\n[bold]Example disagreements (first 5):[/bold]")
        console.print("-" * 60)
        for i, d in enumerate(disagreements[:5], 1):
            console.print(f"\n[cyan]Example {i}:[/cyan]")
            console.print(f"  Text: {d['text'][:100]}...")
            console.print(f"  [magenta]{d['file1_name']}:[/magenta] {d['value_file1']}")
            console.print(f"  [cyan]{d['file2_name']}:[/cyan] {d['value_file2']}")
        if len(disagreements) > 5:
            console.print(f"\n[dim]... and {len(disagreements) - 5} more. Use --interactive to browse all.[/dim]")


if __name__ == "__main__":
    main()
