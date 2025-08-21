"""
Deduplicate ChatML-formatted conversations in a CSV.

This script scans a CSV containing a `text` column with ChatML samples,
parses messages between `<|im_start|>` and `<|im_end|>`, and removes
duplicate chains using a SHA-1 hash of message content. Deduplication is
done in streaming batches with multiprocessing and a Rich progress bar.

Key rules:
- Rows with identical message chains are collapsed.
- If duplicate chains differ only in their final assistant message,
  the version with the longest tokenized final message is kept.
- Output is a new CSV `<input>_deduped.csv` with only the kept rows.

Returns:
- Writes a deduplicated CSV file with row order preserved for survivors.
- Prints progress, total rows, kept, removed, and output path.
"""

import re
import polars as pl
from rich.progress import Progress
from rich.console import Console
from multiprocessing import Pool, cpu_count, Manager
from transformers import AutoTokenizer
import hashlib
import argparse
import os

parser = argparse.ArgumentParser(description="Token statistics generator")
parser.add_argument("-p", "--path", required=True, help="Path to CSV file")
args = parser.parse_args()
args.path = os.path.splitext(args.path)[0]

INPUT_PATH = f"{args.path}.csv"
OUTPUT_PATH = f"{args.path}_deduped.csv"
BATCH_SIZE = 10_000

console = Console()
tokenizer = AutoTokenizer.from_pretrained(
    "mlabonne/Hermes-3-Llama-3.1-8B-lorablated", use_fast=True
)

def parse_chatml(text: str):
    return re.findall(r"<\|im_start\|>.*?\n(.*?)<\|im_end\|>", text, flags=re.S)

def chain_hash(msgs):
    return hashlib.sha1("\n".join(msgs).encode("utf-8")).hexdigest()

def process_batch(batch_data):
    rows, kept_dict = batch_data
    new_kept = []
    for idx, text in rows:
        msgs = [m.strip() for m in parse_chatml(text)]
        if not msgs:
            continue
        h = chain_hash(msgs)
        if h in kept_dict:
            continue
        new_kept.append((idx, h, msgs))
    return new_kept

def count_rows_csv(path: str, batch_size: int = 100_000) -> int:
    reader = pl.read_csv_batched(path, batch_size=batch_size)
    total = 0
    while True:
        batches = reader.next_batches(1)
        if not batches:
            break
        total += len(batches[0])
    return total

console.print("[bold cyan]Starting streaming deduplication...[/bold cyan]")
manager = Manager()
kept_dict = manager.dict()
final_rows = []

total_rows = count_rows_csv(INPUT_PATH, BATCH_SIZE)
console.print(f"[bold yellow]Total rows: {total_rows}[/bold yellow]")

pool = Pool(cpu_count())

with Progress() as progress:
    task = progress.add_task("Processing", total=total_rows)

    reader = pl.read_csv_batched(INPUT_PATH, batch_size=BATCH_SIZE)
    row_offset = 0

    while True:
        batches = reader.next_batches(1)
        if not batches:
            break

        batch_df = batches[0]

        rows = list(zip(range(row_offset, row_offset + len(batch_df)), batch_df["text"].to_list()))
        row_offset += len(batch_df)

        results = process_batch((rows, kept_dict))

        for idx, h, msgs in results:
            kept_dict[h] = True
            final_rows.append((idx, h, msgs))

        progress.update(task, advance=len(rows))

pool.close()
pool.join()

console.print("[bold cyan]Resolving duplicate final assistant messages...[/bold cyan]")
best_by_final = {}
for idx, h, msgs in final_rows:
    final_msg = msgs[-1] if msgs else ""
    tok_len = len(tokenizer.encode(final_msg))
    if final_msg not in best_by_final or tok_len > best_by_final[final_msg][0]:
        best_by_final[final_msg] = (tok_len, idx)

final_indices = sorted(v[1] for v in best_by_final.values())

console.print("[bold cyan]Writing final deduped CSV...[/bold cyan]")
df_all = pl.read_csv(INPUT_PATH)
df_all = df_all.filter(pl.arange(0, df_all.height).is_in(final_indices))
df_all.write_csv(OUTPUT_PATH)

console.print(f"[bold green]Deduplication complete![/bold green]")
console.print(f"[bold yellow]Kept:[/bold yellow] {len(final_indices)} / {total_rows} rows")
console.print(f"[bold yellow]Removed:[/bold yellow] {total_rows - len(final_indices)} rows")
console.print(f"[bold cyan]Output saved to:[/bold cyan] {OUTPUT_PATH}")
