#!/usr/bin/env python3
"""
HF-ToS cleaner for text datasets (with fuzzy detection) — MP/Chunked (32GB-ready).

- Scans CSV or Parquet (file or folder of shards)
- Multiprocessing over CSV chunks or Parquet row-groups
- Adaptive chunk size for CSV from target memory per worker
- Rich progress (bars, ETA, throughput)
- Argparse for paths/columns/action/batch size/workers/memory
- Broad ToS-risk filters (sexual violence, slurs, CSA, terrorism, doxxing, self-harm, etc.)
- Drop OR Redact (default drop)
- Unicode NFKC + leetspeak folding + fuzzy regex
- Outputs CSV or Parquet based on --out
"""

import os
import sys
import re
import argparse
import math
import signal
import unicodedata
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Iterator

import pandas as pd
from rich.console import Console
from rich.progress import (
    Progress, SpinnerColumn, BarColumn, TextColumn,
    TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn
)

console = Console()

# -------- Configurable defaults --------
DEFAULT_BATCH = 50_000
DEFAULT_TEXT_COLS = ["text", "content", "message"]
REDACTION_TOKEN = "[REDACTED]"

# --- Build risk patterns (fuzzy regex) ---
import re
from functools import lru_cache

# ---------- Master fuzzy classes (HF-like: leets, diacritics, homoglyphs) ----------
F = {
    "a": "aàáâãäåāăąæ@4",
    "b": "bƄḃƅ8ß",
    "c": "cçćĉč¢ḉɕ",
    "d": "dďđḍḋḏḑḍḏḍḏḍḏɖcl",   # 'cl' appears as fake 'd' in some spam; keep optional
    "e": "eèéêëēĕėęěæ3€",
    "f": "fƒḟ",
    "g": "gğġģǵɡ9",
    "h": "hĥħḣḧḥḫ",
    "i": "iìíîïĩīĭįı!|1lɪ",
    "j": "jĵǰј",
    "k": "kķĸκḱḳƙx",              # κ ~ k; sometimes x used as stylized k
    "l": "lļľŀł|!1iɫ",
    "m": "mṃṁɯṃ㎚",
    "n": "nñńņňŉŋ",
    "o": "oòóôõöōŏőøɵð0°¤",
    "p": "pṕṗρþ",
    "q": "qɋg9",
    "r": "rŕŗřʀ®",
    "s": "sśŝşšṡẛ$5z",
    "t": "tţťŧṫ†7+",
    "u": "uùúûüũūŭůųµ",
    "v": "vṽṿ∨",
    "w": "wŵẁẃẅvv",
    "x": "xẋẍ×✕",
    "y": "yýÿŷ¥",
    "z": "zźżžƶ2",
    "0": "0oòóôöōø°",
    "1": "1il|!ɪ",
    "2": "2z",
    "3": "3eéèë",
    "4": "4a@",
    "5": "5s$",
    "6": "6bɢg",
    "7": "7t+",
    "8": "8b",
    "9": "9gq",
}

# Build character classes like: [iìíîï...|!1l]
@lru_cache(maxsize=None)
def cls(ch: str) -> str:
    # Multi-char composites (e.g., "vv", "cl") should be a literal group, not a class
    if len(ch) > 1:
        return f"(?:{re.escape(ch)})"
    base = F.get(ch.lower(), ch)
    if len(base) == 1:
        return re.escape(base)
    uniq = "".join(sorted(set(base), key=base.index))
    return f"[{re.escape(uniq)}]"

# Convert a token -> fuzzy class sequence, allowing junk between letters
def fuzz_token(token: str, gap=r"[\W_]{0,3}") -> str:
    parts = []
    i = 0
    while i < len(token):
        # permit two-char composites defined in map (e.g., 'vv' fake 'w')
        two = token[i:i+2].lower()
        if two in F:
            parts.append(cls(two))
            i += 2
            if i < len(token): parts.append(gap)
            continue
        ch = token[i]
        parts.append(cls(ch))
        i += 1
        if i < len(token): parts.append(gap)
    return "".join(parts)

# Build alternatives from words/phrases
def any_of(words, gap=r"[\W_]{0,3}", wb=False):
    alts = []
    for w in words:
        if " " in w:
            toks = [fuzz_token(t, gap) for t in w.split()]
            pat = gap.join(toks)
        else:
            pat = fuzz_token(w, gap)
            # allow optional plural "s" (fuzzy class for s)
            pat = f"{pat}{cls('s')}?"
        alts.append(pat)
    body = "(?:" + "|".join(alts) + ")"
    return (r"\b" + body + r"\b") if wb else body

flags = re.IGNORECASE | re.UNICODE

# ---------------- Rebuilt categories (all via fuzzy builder) ----------------
SEXUAL_VIOLENCE = any_of(["rape", "raped", "raper", "rapes", "raping",
                          "revenge porn", "deepfake porn", "upskirt", "creepshot", "hidden cam sex"])  # suffixes explicit
MINORS_CSA = (
any_of(["child porn", "child abuse", "child molest", "child groom", "child diddler",
        "diddle", "molest", "molester", "child", "kid","children", "cp", "pedo", "paedo", "pedophile"], wb=False)
    + "|"
    + any_of(["cp", "loli", "shota", "minor sexual", "under age sex"], wb=False)
)
SEVERE_SLURS = any_of([
    "nigger", "faggot", "fag", "fagot", "kike", "spic", "chink", "tranny", "retard", "retarded", "niglet", "nigglet", "niger",  "jew"
])
EXTREME_HARASS = any_of([
    "kill yourself", "kill youreself", "kill your", "kills your", "kill youre", "kys", "gas the jews", "gas the gays", "genocide", "lynch", "i'll kill you", "i'm going to hurt you", "swat you", "ill kill you", "im going to hurt you", "swat you", "address", "phone number"
])
TERROR_PROMO = any_of([
    "isis", "al qaeda", "daesh", "hitler", "white power", "terrorist praise", "jihad", "nazi", "sieg", "heil", "1488", "14/88", "wpww", "kkk", "klan"
])
ILLEGAL_GUIDES = (
    any_of(["how to make bomb", "how to build bomb", "pipe bomb", "silencer",
            "meth", "fentanyl", "coke","crack","do speed", "take speed", "love speed", "smoke speed", "inject speed", "slam speed", "rail speed", "take ice", "love ice", "smoke ice", "inject ice", "slam ice", "rail ice","crystal","meth","molly","do x", "take x", "do bars", "take bars", "xanax","shrooms","kush","weed", "smoke bud", "selling bud", "ganja","dope","blow","oxy","perc","percs","lean","sizzurp","ecstasy","mdma","fentanyl","opioid","heroin","lsd","acid","dmt","ketamine","ghb", "ghost gun", "auto sear", "glock switch", "3d printed gun",
            "3d printed glock"], wb=False)
    + "|"
    + any_of(["credit card fraud", "credit card skimmer", "carding", "fullz", "cvv", "cvv dump", "buy logs", "credential dump", "skimmer", "checker", "cashout method", "money mule"], wb=False)
)
MALWARE_CRIME = "|".join([
    any_of(["ddos","botnet","ransomware","stealer","keylogger","rat for sale","dox","doxx"], wb=False),
    r"\b(remote\s+access\s+trojan|rat\s+builder|rat\s+stub|trojan\s+rat)\b",
    any_of(["njrat","nanorat","darkcomet","quasar","venomrat","redline","lumma","vidar","raccoon"], wb=False),
])
# PII/DOXX (keep numerics/emails, but you still benefit from fuzzy words like 'credit')
PII_DOXX = (
    r"(\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b)"
    r"|(\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b)"
    r"|([a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,})"
    r"|(\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)(?:\.(?!$)|$)){4}\b)"
    r"|(\b0x[a-fA-F0-9]{40}\b)"
    r"|(\b[13][a-km-zA-HJ-NP-Z1-9]{25,34}\b)"
    r"|\b(here('s| is)\s+(his|her|their)\s+address|post\s+(his|her|their)\s+address|i['’ ]?ll\s+dox\s+you|swat\s+you)\b"
)

SELF_HARM_INSTR = any_of([
    "how to suicide", "how to kill myself", "cut deep",
    "best way to die", "kym", "kill my", "cutting myself", "cut wrists",
    "slit wrists", "i want to die","i want to kill myself","i dont want to live",
    "end it all","wish i were dead","self harm","self-injury"
])

# Vector-friendly: single union pattern (still keeps flexibility)
UNION_RISK_PATTERN = re.compile(
    "|".join([
        SEXUAL_VIOLENCE, MINORS_CSA, SEVERE_SLURS, EXTREME_HARASS,
        TERROR_PROMO, ILLEGAL_GUIDES, PII_DOXX, SELF_HARM_INSTR,
        MALWARE_CRIME
    ]),
    flags
)

# Leetspeak folding
LEET_MAP = str.maketrans({
    "0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "$": "s", "@": "a"
})

def normalize_text(s: Optional[str]) -> str:
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKC", s).lower().translate(LEET_MAP)
    s = re.sub(r"\s+", " ", s)
    return s

def matches_risk(s: Optional[str]) -> bool:
    if not isinstance(s, str):
        return False
    raw = s
    # Pass 1: PII_DOXX against raw (digits preserved)
    if re.search(PII_DOXX, raw):
        return True
    # Pass 2: normalize and check union pattern
    text = normalize_text(raw)
    return UNION_RISK_PATTERN.search(text) is not None

def redact_text(s: Optional[str]) -> str:
    if not isinstance(s, str):
        return s
    raw = s
    # Pass 1: redact PII directly
    txt = re.sub(PII_DOXX, REDACTION_TOKEN, raw)
    # Pass 2: redact other risky patterns on normalized text
    norm = normalize_text(raw)
    if UNION_RISK_PATTERN.search(norm):
        txt = UNION_RISK_PATTERN.sub(REDACTION_TOKEN, txt)
    return txt

def detect_text_columns(df: pd.DataFrame, preferred: List[str]) -> List[str]:
    cols = [c for c in preferred if c in df.columns]
    if cols:
        return cols
    obj_cols = [c for c in df.columns if df[c].dtype == "object"]
    return obj_cols[:1] or [df.columns[0]]

def iter_input_files(path: Path) -> Iterable[Path]:
    if path.is_dir():
        # Prefer parquet first for speed
        for p in sorted(path.glob("*.parquet")):
            yield p
        for p in sorted(path.glob("*.csv")):
            yield p
    else:
        yield path

# ----------------- Multiprocessing helpers -----------------
# Globals set in worker via initializer (avoid pickling closures)
WORKER_ACTION = None
WORKER_COLS: List[str] = []
def _init_worker(action: str, cols: List[str]):
    # Make patterns available without re-pickling every call
    global WORKER_ACTION, WORKER_COLS
    WORKER_ACTION = action
    WORKER_COLS = cols

def _process_df(df: pd.DataFrame) -> Tuple[int, int, pd.DataFrame]:
    """Return (before_count, kept_count, out_df)."""
    if not WORKER_COLS:
        cols = detect_text_columns(df, DEFAULT_TEXT_COLS)
    else:
        cols = [c for c in WORKER_COLS if c in df.columns] or detect_text_columns(df, DEFAULT_TEXT_COLS)

    before = len(df)
    if before == 0:
        return 0, 0, df

    if WORKER_ACTION == "drop":
        # Vectorized mask across cols
        mask = df[cols].applymap(matches_risk).any(axis=1)
        out_df = df.loc[~mask].copy()
    else:
        out_df = df.copy()
        for c in cols:
            out_df[c] = out_df[c].apply(redact_text)
    kept = len(out_df)
    return before, kept, out_df

# ----------------- CSV handling -----------------
def _estimate_csv_chunksize(csv_path: Path, target_mem_bytes_per_worker: int, explicit_chunksize: Optional[int]) -> int:
    if explicit_chunksize and explicit_chunksize > 0:
        return explicit_chunksize
    # Sample first ~50k rows to estimate bytes/row
    sample_n = 50_000
    try:
        sample = pd.read_csv(csv_path, nrows=sample_n)
        # Rough bytes per row estimation
        bytes_est = sample.memory_usage(deep=True).sum()
        bpr = max(64, int(bytes_est / max(1, len(sample))))
        # Aim at ~target_mem per worker, capped
        est = max(10_000, min(1_000_000, target_mem_bytes_per_worker // bpr))
        return est
    except Exception:
        return DEFAULT_BATCH

def process_csv_parallel(
    in_path: Path, out_path: Path, cols: List[str], action: str,
    workers: int, chunksize_rows: int
):
    import multiprocessing as mp
    total_rows = 0
    try:
        with open(in_path, "r", encoding="utf-8", errors="ignore") as f:
            total_rows = sum(1 for _ in f) - 1
    except Exception:
        total_rows = 0

    first_write = True
    writer = None

    # Prepare iterator of chunks
    def chunk_iter() -> Iterator[pd.DataFrame]:
        for chunk in pd.read_csv(in_path, chunksize=chunksize_rows):
            yield chunk

    with Progress(
        SpinnerColumn(),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        MofNCompleteColumn(),  # completed/total
        TextColumn("{task.fields[rate]}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        TextColumn("{task.fields[status]}")
    ) as progress:
        task = progress.add_task(
            f"[bold cyan]Filtering {in_path.name}[/bold cyan]",
            total=total_rows if total_rows > 0 else None,
            rate="0 rows/s",
            status=""
        )

        processed = 0
        kept_total = 0
        start_time = progress.get_time()

        with mp.get_context("fork" if sys.platform != "win32" else "spawn").Pool(
            processes=workers, initializer=_init_worker, initargs=(action, cols)
        ) as pool:
            for before, kept, out_df in pool.imap_unordered(_process_df, chunk_iter(), chunksize=1):
                processed += before
                kept_total += kept

                # Write result
                if out_path.suffix.lower() == ".parquet":
                    import pyarrow as pa, pyarrow.parquet as pq
                    if writer is None:
                        writer = pq.ParquetWriter(
                            out_path,
                            pa.Table.from_pandas(out_df).schema,
                            compression="zstd"
                        )
                    writer.write_table(pa.Table.from_pandas(out_df))
                else:
                    out_df.to_csv(out_path, index=False, mode="w" if first_write else "a", header=first_write)
                first_write = False

                elapsed = max(1e-6, progress.get_time() - start_time)
                rate = f"{int(processed/elapsed):,} rows/s"
                progress.update(
                    task,
                    completed=processed if total_rows else None,
                    advance=before if total_rows else 0,
                    rate=rate,
                    status=f"kept {kept:,}/{before:,} (total kept {kept_total:,})"
                )

    if writer:
        writer.close()

# ----------------- Parquet handling -----------------
def process_parquet_parallel(
    in_path: Path, out_path: Path, cols: List[str], action: str,
    workers: int, rowgroups_per_batch: int
):
    import pyarrow.parquet as pq, pyarrow as pa
    import multiprocessing as mp

    pf = pq.ParquetFile(in_path)
    total_rows = pf.metadata.num_rows
    num_rgs = pf.num_row_groups

    def rowgroup_batches():
        for start in range(0, num_rgs, rowgroups_per_batch):
            yield range(start, min(start + rowgroups_per_batch, num_rgs))

    first_schema = None
    writer = None

    with Progress(
        SpinnerColumn(),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        MofNCompleteColumn(),
        TextColumn("{task.fields[rate]}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        TextColumn("{task.fields[status]}")
    ) as progress:
        task = progress.add_task(
            f"[bold cyan]Filtering {in_path.name}[/bold cyan]",
            total=total_rows,
            rate="0 rows/s",
            status=""
        )
        processed = 0
        kept_total = 0
        start_time = progress.get_time()

        ctx = mp.get_context("fork" if sys.platform != "win32" else "spawn")
        for rg_range in rowgroup_batches():
            # Prepare dataframes for this batch (in main proc; Arrow → pandas is fast)
            dfs = []
            rows_in_batch = 0
            for rg in rg_range:
                table = pf.read_row_group(rg)
                rows_in_batch += table.num_rows
                dfs.append(table.to_pandas(types_mapper=None))

            with ctx.Pool(processes=workers, initializer=_init_worker, initargs=(action, cols)) as pool:
                for before, kept, out_df in pool.imap_unordered(_process_df, dfs, chunksize=1):
                    processed += before
                    kept_total += kept
                    # Initialize writer lazily
                    if writer is None:
                        first_schema = pa.Table.from_pandas(out_df).schema
                        writer = pq.ParquetWriter(out_path, first_schema, compression="zstd")
                    writer.write_table(pa.Table.from_pandas(out_df))

                    elapsed = max(1e-6, progress.get_time() - start_time)
                    rate = f"{int(processed/elapsed):,} rows/s"
                    progress.update(
                        task,
                        advance=before,
                        rate=rate,
                        status=f"kept {kept:,}/{before:,} (total kept {kept_total:,})"
                    )

    if writer:
        writer.close()

# ----------------- File driver -----------------
def process_file(
    in_path: Path, out_path: Path, cols: List[str], action: str,
    workers: int, target_mem_gb: float, chunksize_rows: Optional[int],
    parquet_rg_batch: int
):
    if in_path.suffix.lower() == ".parquet":
        process_parquet_parallel(
            in_path, out_path, cols, action, workers, parquet_rg_batch
        )
    elif in_path.suffix.lower() == ".csv":
        # Compute per-worker mem budget
        target_bytes_per_worker = int(max(0.25, target_mem_gb) * (1024 ** 3) / max(1, workers))
        chunk_rows = _estimate_csv_chunksize(in_path, target_bytes_per_worker, chunksize_rows)
        process_csv_parallel(
            in_path, out_path, cols, action, workers, chunk_rows
        )
    else:
        console.print(f"[bold red]Unsupported file type:[/bold red] {in_path.suffix}")
        sys.exit(2)

# ----------------- CLI -----------------
def main():
    # Kill children on Ctrl+C
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    ap = argparse.ArgumentParser(description="Filter or redact HF-ToS-risk content from text datasets (MP/Chunked).")
    ap.add_argument("-p", "--in", dest="inp", required=True, help="Input CSV/Parquet file OR directory of shards")
    ap.add_argument("-o", "--out", dest="out", required=True, help="Output path (.csv or .parquet)")
    ap.add_argument("--cols", nargs="*", default=[], help="Columns to scan (default auto-detect)")
    ap.add_argument("--action", choices=["drop", "redact"], default="drop", help="Drop rows or redact matches")
    ap.add_argument("--batch", type=int, default=DEFAULT_BATCH, help="(Legacy) CSV rows per chunk (overridden by --chunksize)")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 8, help="Worker processes (default: all CPU cores)")
    ap.add_argument("--target-mem-gb", type=float, default=32.0, help="Approx total RAM to use (guides CSV chunking)")
    ap.add_argument("--chunksize", type=int, default=0, help="CSV rows per chunk (override adaptive)")
    ap.add_argument("--parquet-rg-batch", type=int, default=16, help="Parquet row-groups per pool batch")
    args = ap.parse_args()

    inp = Path(args.inp)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    console.print(
        f"[bold magenta]HF-ToS Cleaner[/bold magenta] "
        f"— action=[bold]{args.action}[/bold], out=[bold]{out}[/bold], "
        f"workers=[bold]{args.workers}[/bold], target-mem=[bold]{args.target_mem_gb}GB[/bold]"
    )

    files = list(iter_input_files(inp))
    if not files:
        console.print("[bold red]No input files found.[/bold red]")
        sys.exit(1)

    # If writing CSV, ensure fresh file on first pass
    if out.exists():
        console.print(f"[bold yellow]Overwriting existing output:[/bold yellow] {out}")
        out.unlink()

    first = True
    for f in files:
        mode_hint = "creating" if first else "appending to"
        console.print(f"[bold cyan]Processing[/bold cyan] {f} ({mode_hint} {out})")

        process_file(
            f, out, args.cols.copy(), args.action,
            args.workers, args.target_mem_gb,
            args.chunksize if args.chunksize > 0 else None,
            args.parquet_rg_batch
        )
        first = False

    console.print("[bold green]Done.[/bold green]")

if __name__ == "__main__":
    main()
