#!/usr/bin/env python3
"""
Fix leading spaces before <|im_end|> inside the text column of a CSV.

Example:
  python fix_im_end_spaces.py -p /home/user/data/in.csv
  # writes /home/user/data/in_fixed.csv

Notes:
- Uses csv module for safe quoting.
- Rich progress bar, batching, optional pre-count for ETA.
- Designed for 32GB RAM / good CPU but memory-light (streaming).
"""

import argparse
import csv
import os
import sys
import re
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Dict, List, Tuple

from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn

console = Console()

REPLACEMENT_FROM = " <|im_end|>"
REPLACEMENT_TO = "<|im_end|>"

def parse_args():
    p = argparse.ArgumentParser(description="Fix ' <|im_end|>' → '<|im_end|>' in a CSV's text column.")
    p.add_argument("-p", "--path", required=True, help="Input CSV path")
    p.add_argument("-o", "--out", default="", help="Output CSV path (default: <input>_fixed.csv)")
    p.add_argument("-tc", "--text-col", default="text", help="Name of the text column (default: text)")
    p.add_argument("-bs", "--batch-size", type=int, default=100_000, help="Rows per batch (default: 100000)")
    p.add_argument("-w", "--workers", type=int, default=0, help="Process workers (0=auto, 1=disable multiprocessing)")
    p.add_argument("--delimiter", default=",", help="CSV delimiter (default: ,)")
    p.add_argument("--quotechar", default='"', help='CSV quotechar (default: ")')
    p.add_argument("--no-count-first", action="store_true", help="Skip pre-count pass (progress shows rows only)")
    return p.parse_args()

FORBID_SET = set("0oOdDvVxXcC")

# Match any run of spaces/commas/colons right before the token
IM_END_PREFIX = re.compile(r'([,\s:]+)(<\|im_end\|>)')

def _should_block(prefix: str) -> bool:
    """
    Return True if the prefix contains a colon that is 'guarded' by a
    forbidden preceding char (symbol or one of 0/o/O/d/D/V/v/x/X/c/C),
    allowing whitespace between that char and the colon.
    """
    if ":" not in prefix:
        return False

    # check the last colon in the prefix (closest to the token)
    i = prefix.rfind(":")
    # scan backward to find the nearest non-space char before that colon
    j = i - 1
    while j >= 0 and prefix[j].isspace():
        j -= 1
    if j < 0:
        return False

    ch = prefix[j]
    # symbol = non-alphanumeric (underscore counts as alnum here)
    is_symbol = not ch.isalnum()
    if is_symbol:
        return True
    if ch in FORBID_SET:
        return True
    return False

def normalize_im_end(text: str) -> str:
    def repl(m: re.Match) -> str:
        prefix = m.group(1)
        token  = m.group(2)
        # Block if colon is guarded; otherwise normalize away the prefix
        return token if not _should_block(prefix) else m.group(0)
    # Collapse multiple occurrences in one pass
    return IM_END_PREFIX.sub(repl, text)


def count_rows(path: str) -> int:
    # Fast-ish header-aware count
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return 0
        return sum(1 for _ in reader)

def transform_row(row: Dict[str, str], text_col: str) -> Dict[str, str]:
    # Only touch the text field; leave everything else identical
    v = row.get(text_col)
    if v is not None:
        row[text_col] = normalize_im_end(v)
    return row

def transform_batch(rows: List[Dict[str, str]], text_col: str) -> List[Dict[str, str]]:
    # Pure-Python fast path
    for r in rows:
        v = r.get(text_col)
        if v is not None:
            r[text_col] = normalize_im_end(v)
    return rows

def main():
    args = parse_args()

    in_path = os.path.abspath(args.path)
    if not os.path.isfile(in_path):
        console.print(f"[bold red]Input not found:[/bold red] {in_path}")
        sys.exit(1)

    out_path = os.path.abspath(args.out or (os.path.splitext(in_path)[0] + "_fixed.csv"))
    tmp_path = out_path + ".tmp"

    total_rows = None
    if not args.no_count_first:
        console.print("[bold]Pre-counting rows for ETA...[/bold]")
        total_rows = count_rows(in_path)
        console.print(f"[bold green]Rows:[/bold green] {total_rows:,}")

    # Choose workers
    if args.workers < 0:
        workers = 1
    elif args.workers == 0:
        # Light CPU transform; IO-bound. Still allow procs if user insists.
        # We'll default to 1 (no multiprocessing) to keep ordering simple and overhead minimal.
        workers = 1
    else:
        workers = max(1, args.workers)

    # Open files and process streaming
    with open(in_path, "r", newline="") as fin, open(tmp_path, "w", newline="") as fout:
        reader = csv.DictReader(fin, delimiter=args.delimiter, quotechar=args.quotechar)
        if args.text_col not in reader.fieldnames:
            console.print(f"[bold red]Column not found:[/bold red] '{args.text_col}'. Available: {reader.fieldnames}")
            sys.exit(2)

        writer = csv.DictWriter(
            fout,
            fieldnames=reader.fieldnames,
            delimiter=args.delimiter,
            quotechar=args.quotechar,
            lineterminator="\n",
            quoting=csv.QUOTE_MINIMAL,
            extrasaction="ignore",
        )
        writer.writeheader()

        progress = Progress(
            TextColumn("[bold]Fixing[/bold]"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=console,
            transient=False,
        )

        batch_size = args.batch_size
        processed = 0

        with progress:
            task = progress.add_task("fix", total=total_rows)
            if workers == 1:
                # Single-process fast path
                batch: List[Dict[str, str]] = []
                for row in reader:
                    batch.append(row)
                    if len(batch) >= batch_size:
                        batch = transform_batch(batch, args.text_col)
                        writer.writerows(batch)
                        processed += len(batch)
                        progress.update(task, completed=processed)
                        batch.clear()
                if batch:
                    batch = transform_batch(batch, args.text_col)
                    writer.writerows(batch)
                    processed += len(batch)
                    progress.update(task, completed=processed)
            else:
                # Multiprocessing option (keeps input order by writing as batches return)
                transform = partial(transform_batch, text_col=args.text_col)
                with ProcessPoolExecutor(max_workers=workers) as ex:
                    futures = []
                    batch: List[Dict[str, str]] = []
                    for row in reader:
                        batch.append(row)
                        if len(batch) >= batch_size:
                            futures.append(ex.submit(transform, batch))
                            batch = []
                    if batch:
                        futures.append(ex.submit(transform, batch))

                    for fut in futures:
                        rows_out = fut.result()
                        writer.writerows(rows_out)
                        processed += len(rows_out)
                        progress.update(task, completed=processed)

    # Atomic replace
    os.replace(tmp_path, out_path)
    console.print(f"[bold green]Done.[/bold green] Wrote: [bold]{out_path}[/bold]")

if __name__ == "__main__":
    main()
