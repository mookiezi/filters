# Discord Dataset Cleaning Toolkit

<p align="center">
  <img src="https://raw.githubusercontent.com/mookiezi/site/refs/heads/main/filt_header_sm.png" alt="Dataset Cleaning Toolkit">
</p>

A comprehensive toolkit of five complementary cleaners for large-scale text datasets with Rich progress, multiprocessing, and 32 GB-RAM-friendly batching:

-   `filter.sql` — Postgres filter set (PII, commands, automation noise, ToS-risk).
-   `smartclean.py` — multi-stage smart clean (normalize → slang replace → resample → structural double-check).
-   `tos.py` — HF-ToS risk filter (drop or redact fuzzy/leet/diacritic matches).
-   `dedupe.py` — ChatML deduplication (drop duplicate chains, keep longest assistant ending).
-   `fixend.py` — normalize stray prefixes before `<|im_end|>` (collapse/guarded colon rules).

---

## Install

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

> Python **3.10+** recommended.

---

## `filter.sql` — SQL Filters

A Postgres-compatible SQL file containing regex/text filters for PII, bot/command patterns, and automation noise.

Usage:

```bash
# Run against a loaded table (example: messages)
psql -d mydb -f filter.sql

# Or include in a query
SELECT * FROM messages
WHERE NOT (content ~* ANY(ARRAY[
    -- patterns defined in filter.sql
]));
```

The script expects:

```
A Postgres table containing your dataset (e.g., messages).
A text column (content) that holds the message text.
Metadata columns (message_id, message_reference_message_id, guild_id, channel_id, author_id)
```

## `smartclean.py` — Multi-Stage Smart Clean

Pipeline:

1. **Trim & Strip**: normalize text, remove emojis/mentions/junk, drop structural spam (e.g., emoji floods, trade lists, code blocks) → `trimmed.csv`
2. **Slang Replacement**: map common shorthand/slang into normalized forms (e.g., `u → you`), with all substitutions logged → `slangremoved.csv` (+ `changes.csv`)
3. **Smart Resample**: group by token length, shuffle within buckets, and optionally trim to target size → `resampled.csv`
4. **Double-Check**: enforce ChatML role alternation and `<|end_of_text|>` termination, rejecting malformed samples → `done.csv` (+ `invalid.csv`)

**CLI**

```text
-f/--folder            Required folder name under /home/user/data/{folder}
```

The script expects:

```
/home/user/data/{folder}/dump.csv           # input
/home/user/data/{folder}/trimmed.csv        # stage 1 output
/home/user/data/{folder}/slangremoved.csv   # stage 2 output
/home/user/data/{folder}/resampled.csv      # stage 3 output
/home/user/data/{folder}/done.csv           # final valid
/home/user/data/{folder}/invalid.csv        # reasons for rejects
/home/user/data/{folder}/changes.csv        # slang replacements log
```

**Example**

```bash
python smartclean.py -f ALPHA
```

## `dedupe.py` — ChatML Deduplication

Stream-deduplicate ChatML-formatted rows in a CSV.
Parses messages between `<|im_start|>` and `<|im_end|>`, hashes full chains, and drops duplicates.
When duplicates differ only in their final assistant message, the version with the longest tokenized ending is kept.
Uses Polars streaming, multiprocessing, and Rich progress for efficient large-scale processing.

**CLI**

```text
-p/--path              Input CSV file (required, must contain `text` column)
```

**This script expects**

```
/path/to/input.csv               # input file with ChatML rows
/path/to/input_deduped.csv       # output deduplicated CSV
```

**Examples**

```bash
# Deduplicate a single CSV of ChatML conversations
python dedupe.py -p data/dump.csv

# Input: data/dump.csv
# Output: data/dump_deduped.csv
```

## `fixend.py` — Normalize `<|im_end|>` Prefixes

Stream-normalizes the `text` column of a CSV by collapsing any prefix of spaces, commas, or non-emoticon colons before `<|im_end|>` into the bare token → writes `*_fixed.csv`. Replacement is **blocked** if the closest colon is “guarded” by a preceding symbol or one of `0 o O d D V v x X c C` (whitespace between guard and colon allowed). Uses streaming, batching, and optional multiprocessing for large files.

**CLI**

```text
-p/--path              Input CSV path (required)
-o/--out               Output CSV path (default: <input>_fixed.csv)
-tc/--text-col         Text column name (default: text)
-bs/--batch-size       Rows per batch (default: 100000)
-w/--workers           Process workers (0=auto→single process; 1=force single)
--delimiter            CSV delimiter (default: ,)
--quotechar            CSV quote char (default: ")
--no-count-first       Skip pre-count pass for ETA
```

**This script expects**

```
/path/to/input.csv             # input file with ChatML rows
/path/to/input_fixed.csv        # output normalized CSV (default name)
```

**Examples**

```bash
# Normalize <|im_end|> tokens in a CSV
python fixend.py -p data/dump.csv

# Custom output path
python fixend.py -p data/dump.csv -o out/normalized.csv
```

## `tos.py` — HF-ToS Risk Filter

Drop or redact matches across one CSV/Parquet file **or a directory of shards**.  
Applies fuzzy/leet/diacritic-aware regex to remove or redact ToS-risk categories such as sexual violence, CSA, slurs, harassment, doxxing, self-harm, and extremism.  
Works with adaptive chunking for CSV and row-group batching for Parquet, with multiprocessing and Rich progress built in.

**CLI**

```text
-p/--in                Input CSV/Parquet file OR directory
-o/--out               Output path (.csv or .parquet)
--cols                 Optional list of text columns to scan (auto-detect if omitted)
--action               drop | redact (default: drop)
--workers              Processes (default: all cores)
--target-mem-gb        Approx total RAM budget (guides CSV chunking) [default: 32]
--chunksize            Override adaptive CSV rows/chunk
--parquet-rg-batch     Parquet row-groups per pool batch [default: 16]
```

**This script expects**

```
# If -p/--in is a single file
/path/to/dump.csv            # OR dump.parquet
/path/to/out/clean.csv       # OR clean.parquet (single consolidated output)
```

```
# If -p/--in is a directory
/path/to/shards/             # folder containing any mix of:
  *.parquet                  # processed by row-groups
  *.csv                      # processed by adaptive CSV chunks

# Output is ONE file at -o/--out:
out/clean.csv                # OR out/clean.parquet
```

**Examples**

```bash
# Drop risky rows from a CSV
python tos.py -p data/dump.csv -o out/clean.csv --action drop

# Redact risky spans into [REDACTED] and write Parquet
python tos.py -p data/dump.csv -o out/clean.parquet --action redact

# Process a folder of shards with explicit chunk size
python tos.py -p data/shards -o out/clean.csv --chunksize 120000
```

---

## Cross-Platform Notes

-   **Unix/macOS**: after `chmod +x script.py`, you can run `./script.py …` thanks to the shebang.
-   **Windows**: run as `python script.py …`; shebang is ignored by default shell, but code is fully supported.

---

## Outputs & Logs

-   `tos.py` prints kept/processed counts and rate.
-   `smartclean.py` emits stage artifacts and logs:
    -   `changes.csv`: slang replacements (row, column, original, cleaned)
    -   `invalid.csv`: structural reasons for rejection

---

## License

MIT

---

## Quick Start

```bash
# 1) Install
pip install -r requirements.txt

# 2) Run SmartClean pipeline (multi-stage normalize → slang replace → resample → double-check)
python smartclean.py -f ALPHA

# 3) Run ToS cleaner (drop or redact ToS-risk rows from CSV/Parquet)
python tos.py -p data/dump.csv -o out/clean.parquet --action drop

# 4) Run Deduper (remove duplicate ChatML chains, keep longest assistant ending)
python dedupe.py -p out/clean.parquet

# 5) Run FixEnd (normalize stray prefixes before <|im_end|>)
python fixend.py -p out/clean_deduped.csv -o out/clean_deduped_fixed.csv

# 6) Run SQL filter directly in Postgres (drop PII, commands, bot/automation noise)
psql -d mydb -f filter.sql

```
