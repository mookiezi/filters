#!/usr/bin/env python3
"""
SmartClean — Advanced Dataset Cleaner (32GB-ready).

- Scans CSV/Parquet dumps with multiprocessing + chunking
- Applies regex/structural filters for bot-like or ToS-risk content
- Auto-detects text columns (slang, emojis, mentions, symbols, etc.)
- Drop vs Redact modes for flexible cleaning
- Unicode NFKC normalization, leetspeak folding, fuzzy regex
- Rich progress (bars, ETA, throughput)
- Argparse with folder-based input/output paths
- Multi-stage pipeline:
    1. Trim & Strip → normalize, reject, log
    2. Slang Replacement → maps casual/abbreviated text
    3. Smart Sampling → token-length balanced subsetting
    4. Double-Check → structural & alternation validation
- Outputs clean CSVs (and Parquet optional) with logs
- Optimized defaults for 32GB RAM + good CPU, configurable via args

Usage:
    python smartclean.py -f ALPHA

    # Expects:
    #   /home/user/data/ALPHA/dump.csv
    # Produces:
    #   /home/user/data/ALPHA/trimmed.csv
    #   /home/user/data/ALPHA/slangremoved.csv
    #   /home/user/data/ALPHA/resampled.csv
    #   /home/user/data/ALPHA/done.csv
    #   /home/user/data/ALPHA/invalid.csv
    #   /home/user/data/ALPHA/changes.csv
"""

import os
import csv
import psutil
import traceback
import polars as pl
import random
import re
import signal
import sys
import math
from collections import defaultdict
from transformers import AutoTokenizer
from multiprocessing import Pool, cpu_count
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn, TaskProgressColumn, MofNCompleteColumn, track
import pandas as pd
import multiprocessing as mp
from emoji import emoji_list
import unicodedata
import argparse

parser = argparse.ArgumentParser(description="Smart clean filtering system")
parser.add_argument("-f", "--folder", required=True, help="folder to CSV file")
args = parser.parse_args()

# === Config ===
tokenizer = AutoTokenizer.from_pretrained("NousResearch/Hermes-3-Llama-3.1-8B", use_fast=True)
INPUT_CSV = f"/home/user/data/{args.folder}/dump.csv"
TRIMMED_CSV = f"/home/user/data/{args.folder}/trimmed.csv"
BAD_ROWS_CSV = f"/home/user/data/{args.folder}/bad.csv"
FILTERED_CSV = f"/home/user/data/{args.folder}/trimmed.csv"
RESLANG_CSV = f"/home/user/data/{args.folder}/slangremoved.csv"
RESLANG_LOG_PATH = f"/home/user/data/{args.folder}/changes.csv"
RESAMPLED_CSV = f"/home/user/data/{args.folder}/resampled.csv"
DOUBLECHECK_OUTPUT_CSV = f"/home/user/data/{args.folder}/done.csv"
DOUBLECHECK_LOG_PATH = f"/home/user/data/{args.folder}/invalid.csv"
MAX_TOKENS = math.inf
CHUNK_SIZE = 630
TARGET_SIZE = 999_000_000
BATCH_SIZE = 20000
TEXT_COLUMN_NAME = "text"
NUM_PROCESSES = max(cpu_count() - 2, 2)
FAKE_TOKEN_RE = re.compile(r'<[a-zA-Z0-9_]+?>|<>')
LF_TRADE_HINT_RE = re.compile(r'\bLF[:\d]*\b', re.IGNORECASE)
DASHWORD_RE = re.compile(r"[—–-]\w", re.UNICODE)
PERCENT_ENCODE_RE = re.compile(r'%[0-9a-fA-F]{2,}', re.IGNORECASE)
BULLET_LINE_RE = re.compile(
    r'^\s*[-*`+>•‣⤷﹡◦▪▫‒–—·○●♦✓✔∘☑☐☒⬤◆◇■□🔹🔸➡➤➣➢➔➛➜➝➞➟⇨→⇢⇒«»‹›→⇒➡➤➥➦➧➨➩➪➫➬➭➮➯✦✧✩✪✫✬✭✮✯★☆✰✱✲✳✴✵✶✷✸✹☀☁☂☃☄✡❖❥❦❧❃❋⁂※『』「」【】〈〉《》🔘🔺🔻🔼🔽🔷🔶🔸🔹🞜🞛🞚🞙]'
)
bullet_symbol_string = (
    "-*+>•‣◦▪▫‒–—·○●♦✓✔∘☑☐☒⬤◆◇■□🔹🔸➡➤➣➢➔➛➜➝➞➟⇨→⇢⇒«»‹›→⇒➡➤➥➦➧➨➩➪➫➬➭➮➯"
    "✦✧✩✪✫✬✭✮✯★☆✰✱✲✳✴✵✶✷✸✹☀☁☂☃☄✡❖❥❦❧❃❋⁂※『』「」【】〈〉《》🔘🔺🔻🔼🔽🔷🔶🔸🔹🞜🞛🞚🞙"
)
additional_symbols = ":>-,.*"
all_symbols = bullet_symbol_string + additional_symbols
SERVICE_AD_RE = re.compile(
    r"\b(vouches?|payment first|dm me|prices?|accounts?|slots?|trials?|full gears?|lft|lf|1v1|1v2|2v1|2v2|3v3|4v4|5v5|6v6|7v7)\b",
    re.IGNORECASE
)
bullet_number_re = re.compile(
    r'^\s*'
    r'(?:'
        r'[-`*+>•‣◦▪▫‒–—·○●♦✓✔∘☑☐☒⬤◆◇■□🔹🔸➡➤➣➢➔➛➜➝➞➟⇨→⇢⇒«»‹›→⇒➡➤➥➦➧➨➩➪➫➬➭➮➯✦✧✩✪✫✬✭✮✯★☆✰✱✲✳✴✵✶✷✸✹☀☁☂☃☄✡❖❥❦❧❃❋⁂※『』「」【】〈〉《》🔘🔺🔻🔼🔽🔷🔶🔸🔹🞜🞛🞚🞙]|[xX]'
        r'\s*'
    r')?'
    r'\d+[\s).:-]?',
    re.UNICODE
)
DISCORD_TIMESTAMP_RE = re.compile(r"<t:\d{10}:[a-zA-Z]>", re.IGNORECASE)
MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
KEYWORD_LINE_RE = re.compile(
    r'^\s*(?<!\w)(?:'
    r'#include|c\+\+|c|#define|#ifdef|#ifndef|#endif|#elif|#else|#pragma|#undef|js|fn|Physics|transform|Translate|Raycast|Vector3|Time|deltaTime|Input|GetAxis|GetKey|Camera|GameObject|Rigidbody|Collider|MonoBehaviour|Start|Update|Awake|OnCollision|'
    r'std::|cin|cout|endl|scanf|printf|fprintf|sprintf|sscanf|'
    r'malloc|calloc|realloc|free|sizeof|alignof|offsetof|'
    r'auto|void|int|long|short|float|double|char|bool|true|false|signed|unsigned|struct|union|enum|class|template|typename|'
    r'virtual|override|static_cast|dynamic_cast|reinterpret_cast|const_cast|decltype|constexpr|mutable|explicit|inline|volatile|extern|register|'
    r'public:|private:|protected:|using namespace|namespace|'
    r'new|delete|this|nullptr|'
    r'if|else|for|while|do|switch|case|break|continue|return|goto|default|then|'
    r'def|lambda|async|await|yield|global|nonlocal|pass|raise|import|from|as|with|try|except|finally|args|kwargs|print|input|len|range|open|list|dict|set|tuple|int|float|str|bool|True|False|None|class|self|assert|is|in|not|or|and|del|'
    r'function|let|const|var|console\.log|document\.|window\.|=>|Promise|await|async|typeof|instanceof|this|new|try|catch|throw|import|export|require|return|'
    r'echo|read|cd|ls|pwd|cat|touch|mkdir|rm|cp|mv|grep|awk|sed|cut|find|xargs|chmod|chown|sudo|ifconfig|ip|netstat|ping|which|ps|kill|'
    r'spawn|exec|exit|trap|set|export|source|alias|'
    r'sh|bash|zsh|fish|dash|env|'
    r'BEGIN|END|fi|esac|then|elif|while|until|for|case|in|do|done|'
    r'make|cmake|g\+\+|gcc|clang|ld|objdump|strip|nm|cargo|rustc|go|javac|java|node|npm|npx|tsc|deno|python|python3|pip|pip3|conda|poetry|'
    r'docker|podman|kubectl|docker-compose|tmux|screen|ssh|scp|rsync|diff|patch|curl|wget|apt|yum|dnf|pacman|brew|nix|'
    r'\/usr\/|'
    r'(?:int|void|char|float|double|bool)\s+\w+\s*\(|def\s+\w+\s*\(|function\s+\w+\s*\('
    r')(?!\w)',
    re.IGNORECASE
)
NWORD_RE = re.compile(r"n[ìíîïĩīįıɨɩɪіί1|]g{1,3}[e3]r?[^\s]*s?", re.IGNORECASE)
CP_RE = re.compile(r"\bchild\s+(porn|pornography)\b", re.IGNORECASE)
MAX_TOKENS += 1
DOUBLECHECK_SOURCE_CSV = RESAMPLED_CSV
DOUBLECHECK_BATCH_SIZE = 4000
DOUBLECHECK_USER_RE = re.compile(r"<\|im_start\|>user\s+(.+?)<\|im_end\|>", re.DOTALL)
DOUBLECHECK_ASSISTANT_RE = re.compile(r"<\|im_start\|>assistant\s+(.+?)<\|im_end\|>", re.DOTALL)
DOUBLECHECK_EOT_RE = re.compile(r"<\|end_of_text\|>")
NONLATIN_REJECT_RE = re.compile(
    r'['
    r'\u0400-\u052F'  # Cyrillic (Russian, Ukrainian, etc.)
    r'\u0590-\u05FF'  # Hebrew
    r'\u0600-\u06FF'  # Arabic
    r'\u0750-\u077F'  # Arabic Supplement
    r'\u0900-\u097F'  # Devanagari (Hindi)
    r'\u0980-\u09FF'  # Bengali
    r'\u0E00-\u0E7F'  # Thai
    r'\u1100-\u11FF'  # Hangul Jamo (Korean)
    r'\u3040-\u30FF'  # Hiragana and Katakana (Japanese syllabaries)
    r'\u3400-\u4DBF'  # CJK Extension A
    r'\u4E00-\u9FFF'  # CJK Unified Ideographs (Chinese/Japanese)
    r'\uAC00-\uD7AF'  # Hangul Syllables (Korean)
    r'\uA640-\uA69F'  # Cyrillic Extended-B
    r'\u2DE0-\u2DFF'  # Cyrillic Extended-A
    r'\u2E80-\u2EFF'  # CJK Radicals Supplement
    r'\u3000-\u303F'  # CJK Symbols and Punctuation
    r'\u31A0-\u31BF'  # Bopomofo Extended
    r'\u31F0-\u31FF'  # Katakana Phonetic Extensions
    r'\u3100-\u312F'  # Bopomofo (Mandarin)
    r']'
)
def detect_nonlatin_script(text):
    # Returns a string like "Cyrillic", "Arabic", "CJK", etc. for the first major non-Latin char found.
    script_blocks = [
        (r'\u0400-\u052F', 'Cyrillic'),
        (r'\u0590-\u05FF', 'Hebrew'),
        (r'\u0600-\u06FF', 'Arabic'),
        (r'\u0750-\u077F', 'Arabic Supplement'),
        (r'\u0900-\u097F', 'Devanagari'),
        (r'\u0980-\u09FF', 'Bengali'),
        (r'\u0E00-\u0E7F', 'Thai'),
        (r'\u1100-\u11FF', 'Hangul Jamo'),
        (r'\u3040-\u30FF', 'Hiragana/Katakana'),
        (r'\u3400-\u4DBF', 'CJK Ext A'),
        (r'\u4E00-\u9FFF', 'CJK'),
        (r'\uAC00-\uD7AF', 'Hangul Syllables'),
        (r'\uA640-\uA69F', 'Cyrillic Ext B'),
        (r'\u2DE0-\u2DFF', 'Cyrillic Ext A'),
        (r'\u2E80-\u2EFF', 'CJK Radicals'),
        (r'\u3000-\u303F', 'CJK Symbols'),
        (r'\u31A0-\u31BF', 'Bopomofo Extended'),
        (r'\u31F0-\u31FF', 'Katakana Extensions'),
        (r'\u3100-\u312F', 'Bopomofo'),
    ]
    for block, name in script_blocks:
        block_re = re.compile(f'[{block}]')
        if block_re.search(text):
            return name
    return "Other"

def make_charclass(symbols):
    # Re-escape just in case some symbols are regex meta
    return "".join(re.escape(s) for s in symbols)

SYMBOL_CLASS = make_charclass(all_symbols)
CARD_TRADE_RE = re.compile(
    rf'(?:^|[\s{SYMBOL_CLASS}])'
    r'('
    r'id|hp|pr|evs?|ex|ga|tl|sts|ec|mi|gx|v(?:max|star)?|lf|ft|ss|cg|wtt?|iv'
    r'|have|want|trading|h|looking for|for trade'
    r')'
    rf'(?:[\s{SYMBOL_CLASS}]|$)',
    re.IGNORECASE
)

# Slang replacement logic
slang_map = {
    r'\bur\b': 'your',
    r'\byur\b': 'youre',
    r'\bu\b': 'you',
    r'\br\b': 'are',
    r'\bcuz\b': 'because',
    r'\bb4\b': 'before',
    r'\bthx\b': 'thanks',
    r'\bl8r\b': 'later',
    r'\b2nite\b': 'tonight',
    r'\bgr8\b': 'great',
    r'\bkewl\b': 'cool',
    r'\bkool\b': 'cool',
    r'\bttyl\b': 'bye',
    r'\blawl\b': 'lol',
    r'\bb\b': 'be',
}

def strip_emojis(text):
    if not text:
        return text
    for match in emoji_list(text):
        text = text.replace(match["emoji"], "")
    return text

# Sort replacements by descending pattern length
sorted_slang = sorted(slang_map.items(), key=lambda x: -len(x[0]))

def replace_slang(text):
    def smart_replace(match, replacement):
        word = match.group(0)
        return replacement.capitalize() if any(c.isupper() for c in word) else replacement

    for pattern, replacement in sorted_slang:
        text = re.sub(pattern, lambda m: smart_replace(m, replacement), text, flags=re.IGNORECASE)
    return text

def process_batch(batch_data, columns, batch_idx):
    changes = []
    for col in columns:
        originals = batch_data[col].astype(str)
        cleaned_col = originals.apply(replace_slang)

        diff_mask = originals != cleaned_col
        if diff_mask.any():
            for idx in batch_data.index[diff_mask]:
                changes.append((idx, col, originals[idx], cleaned_col[idx]))

        batch_data[col] = cleaned_col

    return batch_data, changes

def process_slang():
    df = pd.read_csv(FILTERED_CSV, dtype=str)
    text_columns = [col for col in df.columns if df[col].dtype == 'object' or df[col].dtype == 'string']

    print(f"Loaded {len(df)} rows, split into {((len(df)-1)//BATCH_SIZE)+1} batches of {BATCH_SIZE}.")
    print(f"Text columns detected: {text_columns}")
    print(f"Processing on {mp.cpu_count()} cores...")

    batches = [df.iloc[i:i+BATCH_SIZE].copy() for i in range(0, len(df), BATCH_SIZE)]

    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = [
            pool.apply_async(process_batch, args=(batch, text_columns, idx))
            for idx, batch in enumerate(batches)
        ]

        output = []
        for idx, result in enumerate(results):
            output.append(result.get())
            print(f"Batch {idx} done.")

    cleaned_batches = [item[0] for item in output]
    all_changes = [change for item in output for change in item[1]]

    final_df = pd.concat(cleaned_batches)
    final_df.to_csv(RESLANG_CSV, index=False)
    print(f"Done. Cleaned CSV saved to: {RESLANG_CSV}")

    if all_changes:
        pd.DataFrame(all_changes, columns=["row", "column", "original", "cleaned"]).to_csv(RESLANG_LOG_PATH, index=False)
        print(f"Changes logged to: {RESLANG_LOG_PATH}")
    else:
        print("No changes detected; no log file created.")

def strip_at_mentions(text):
    """
    Remove all @mentions (e.g. @someone) with optional whitespace after.
    Preserves newlines and normal text.
    """
    if not text:
        return text
    # Regex: '@' + non-whitespace, non-@ run + optional whitespace (but not across lines)
    return re.sub(r'@\S+\s*', '', text)

def clean_commas_and_periods(text):
    # Rule 1: Change "., " → ", "
    text = re.sub(r'\.\s*,', ',', text)
    text = re.sub(r'\.,', ',', text)

    # Rule 2: Change ",." → "."
    text = re.sub(r',\.', '.', text)

    # Rule 3: Collapse multiple commas → one comma
    text = re.sub(r',{2,}', ',', text)

    return text

def has_discord_timestamp(text):
    for _, msg in extract_chat_turns(text):
        if DISCORD_TIMESTAMP_RE.search(msg):
            return True
    return False

def has_nword(text):
    if not text:
        return False
    clean_text = strip_known_structural_tags(text)
    return bool(NWORD_RE.search(clean_text))

def has_cp_reference(text):
    if not text:
        return False
    clean_text = strip_known_structural_tags(text)
    return bool(CP_RE.search(clean_text))

def has_percent_encoding(text):
    if not text:
        return False
    return bool(PERCENT_ENCODE_RE.search(text))

def has_multi_newline_block(text):
    for _, msg in extract_chat_turns(text):
        if MULTI_NEWLINE_RE.search(msg):
            return True
    return False

def is_duplicate_lines_only(text):
    for _, msg in extract_chat_turns(text):
        lines = [l.strip() for l in msg.strip().splitlines() if l.strip()]
        if len(lines) < 2:
            continue
        if len(set(lines)) == 1:
            return True
    return False

def has_numbered_lines_block(text):
    x_bullet_re = re.compile(r'^\s*[xX]\s*\d+[\s).:-]?')
    number_re = re.compile(r'^\s*\d+[\s).:-]?')

    for _, msg in extract_chat_turns(text):
        lines = [l.strip() for l in msg.strip().splitlines() if l.strip()]
        if len(lines) <= 4:
            continue
        num_start = 0
        for l in lines:
            if BULLET_LINE_RE.match(l):
                tail = BULLET_LINE_RE.sub("", l, count=1)
                if number_re.match(tail):
                    num_start += 1
            elif x_bullet_re.match(l):
                num_start += 1
            elif number_re.match(l):
                num_start += 1
        if num_start >= 2:
            return True
    return False

def is_single_word_lines_block(text):
    for _, msg in extract_chat_turns(text):
        lines = [l.strip() for l in msg.strip().splitlines() if l.strip()]
        if len(lines) <= 4:
            continue
        short_lines = sum(1 for l in lines if len(l.split()) <= 2)
        if short_lines >= 2:
            return True
    return False

def doublecheck_is_valid(indexed_row):
    idx, text = indexed_row
    short = text[:120].replace("\n", " ")

    if not DOUBLECHECK_EOT_RE.search(text):
        return False, f"{idx}: missing <|end_of_text|> — {short}"

    turns = extract_chat_turns(text)

    if not turns:
        return False, f"{idx}: no valid chat turns found — {short}"

    # Allow either role to start, but require strict alternation from the first role
    first = turns[0][0]
    if first not in ("user", "assistant"):
        return False, f"{idx}: first speaker must be user or assistant — {short}"

    for i, (role, _) in enumerate(turns):
        expected = first if i % 2 == 0 else ("assistant" if first == "user" else "user")
        if role != expected:
            return False, f"{idx}: turn {i} expected {expected}, got {role} — {short}"

    if not text.strip().endswith("<|end_of_text|>"):
        return False, f"{idx}: <|end_of_text|> not at end — {short}"

    return True, None

def is_nonsensical_script_spam(text, threshold=0.3):
    """
    Returns the script name if ANY message block has over `threshold`
    non-Latin/diacritic chars. Otherwise returns False.
    """
    if not text:
        return False

    for _, msg in extract_chat_turns(text):
        clean_text = strip_known_structural_tags(msg)
        total_chars = len(clean_text)
        if total_chars == 0:
            continue

        weird_count = 0
        for ch in clean_text:
            cat = unicodedata.category(ch)
            if (
                cat.startswith("M") or
                cat in ("Cf", "Cn", "Cs") or
                NONLATIN_REJECT_RE.match(ch)
            ):
                weird_count += 1

        if (weird_count / total_chars) >= threshold:
            return detect_nonlatin_script(clean_text)

    return False

def strip_double_pipe(text):
    return text.replace("||", "") if text else text

def doublecheck_batch(batch):
    valid = []
    invalid = []
    for indexed_row in batch:
        ok, reason = doublecheck_is_valid(indexed_row)
        if ok:
            valid.append(indexed_row[1])
        else:
            invalid.append(reason)
    return valid, invalid

def doublecheck_batch_fn(batch):
    valid = []
    invalid = []
    for idx, row in batch:
        text = row[TEXT_COLUMN_NAME]
        ok, reason = doublecheck_is_valid((idx, text))
        if ok:
            valid.append(row)
        else:
            row_with_reason = dict(reason=reason, **row)
            invalid.append(row_with_reason)
    return valid, invalid

def doublecheck_final_pass():
    print(f"\nLoading CSV: {DOUBLECHECK_SOURCE_CSV}")
    if not os.path.exists(DOUBLECHECK_SOURCE_CSV):
        print("File not found.")
        return

    with open(DOUBLECHECK_SOURCE_CSV, newline='', encoding="utf-8") as infile:
        reader = csv.reader(infile)
        header = next(reader)
        rows = list(reader)

    text_index = header.index(TEXT_COLUMN_NAME)
    rows_check = [dict(zip(header, row)) for row in rows]

    print(f"🔎 Validating {len(rows_check):,} rows...")

    indexed_check = list(enumerate(rows_check))
    batches_check = [
        indexed_check[i:i + DOUBLECHECK_BATCH_SIZE]
        for i in range(0, len(indexed_check), DOUBLECHECK_BATCH_SIZE)
    ]

    valid_rows = []
    invalid_rows = []

    with Pool(NUM_PROCESSES) as pool:
        try:
            results = list(track(pool.imap(doublecheck_batch_fn, batches_check), total=len(batches_check), description="🔍 Double Check"))
        except KeyboardInterrupt:
            print("\n[!] Ctrl+C detected during multiprocessing. Cleaning up...")
            pool.terminate()
            pool.join()
            return

    for valid_batch, invalid_batch in results:
        valid_rows.extend(valid_batch)
        invalid_rows.extend(invalid_batch)

    print("\nDouble Check Complete.")
    print(f"Total rows:     {len(rows_check):,}")
    print(f"Valid rows:     {len(valid_rows):,}")
    print(f"Invalid rows:  {len(invalid_rows):,}")
    print(f"Valid percent:  {len(valid_rows) / len(rows_check) * 100:.2f}%")

    # Write valid rows
    with open(DOUBLECHECK_OUTPUT_CSV, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in valid_rows:
            writer.writerow([str(row.get(col, "")) for col in header])
    print(f"Cleaned CSV saved to: {DOUBLECHECK_OUTPUT_CSV}")

    # Write invalid rows with reason
    if invalid_rows:
        bad_header = ["reason"] + header
        with open(DOUBLECHECK_LOG_PATH, "w", newline='', encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(bad_header)
            for row in invalid_rows:
                writer.writerow([row.get("reason", "")] + [str(row.get(col, "")) for col in header])
        print(f"Invalid rows with reasons saved to: {DOUBLECHECK_LOG_PATH}")

def extract_chat_turns(text):
    """
    Returns a list of (role, message) pairs for all chat turns.
    Example: [('user', 'hello'), ('assistant', 'hi there'), ...]
    """
    pattern = re.compile(r"<\|im_start\|>(user|assistant)\s+(.+?)<\|im_end\|>", re.DOTALL)
    return [(m.group(1), m.group(2).strip()) for m in pattern.finditer(text or "")]

def strip_known_structural_tags(text):
    if not text:
        return ""
    return re.sub(r'<\|?(im_start\|user|im_start\|assistant|im_end|end_of_text)\|?>', '', text)

def has_fake_token(text):
    if not text:
        return False
    clean_text = strip_known_structural_tags(text)
    return bool(FAKE_TOKEN_RE.search(clean_text))

def has_consecutive_emojis_any_side(text, min_count=2):
    for _, msg in extract_chat_turns(text):
        if has_consecutive_emojis(msg, min_count):
            return True
    return False

def is_service_ad(text):
    if not text:
        return False
    lines = text.strip().splitlines()
    if len(lines) < 4:
        return False

    has_bullets = sum(1 for l in lines if BULLET_LINE_RE.match(l)) >= 2
    has_keywords = sum(1 for l in lines if SERVICE_AD_RE.search(l)) >= 2

    text_lc = text.lower()
    has_accepting_and_offering = "accepting" in text_lc and "offering" in text_lc

    return has_bullets and has_keywords and has_accepting_and_offering

def has_consecutive_emojis(text, min_count=2):
    if not text:
        return False
    matches = emoji_list(text)
    if len(matches) < min_count:
        return False

    matches.sort(key=lambda m: m["match_start"])
    streak = 1
    last_end = matches[0]["match_end"]

    for m in matches[1:]:
        gap = text[last_end:m["match_start"]]
        if gap.strip() == '':
            streak += 1
            if streak >= min_count:
                return True
        else:
            streak = 1
        last_end = m["match_end"]

    return False

def show_mem(label=""):
    mem = psutil.virtual_memory()
    print(f"[{label}] RAM used: {mem.used / (1024**3):.2f} GB / {mem.total / (1024**3):.2f} GB")

def is_lf_trade_hint(text):
    if not text:
        return False
    # match standalone 'lf', 'lf:', 'lf1', 'lf123' etc., but not 'alfredo', 'selfish', etc.
    return bool(LF_TRADE_HINT_RE.search(text))

def is_card_trade_post(text):
    text = text.strip()
    if len(text) < 10:
        return False
    if re.match(r"^\d{13,16}", text):
        return True
    if CARD_TRADE_RE.search(text):
        return True
    return False

def is_bulletpoint_list(text):
    if not text:
        return False
    return any(_is_bullety(msg) for _, msg in extract_chat_turns(text))

def _is_bullety(text):
    if not text:
        return False
    lines = text.strip().splitlines()
    if len(lines) <= 2:  # safe exemption for RP messages
        return False
    bullet_lines = sum(1 for l in lines if BULLET_LINE_RE.match(l))
    return bullet_lines / len(lines) >= 0.5

def has_numeric_only_line(text):
    for _, msg in extract_chat_turns(text):
        for line in msg.strip().splitlines():
            if line.strip().isdigit():
                return True
    return False

def has_dashword(text):
    return bool(DASHWORD_RE.search(text)) if text else False

def count_code_like_lines(text):
    if not text:
        return 0

    lines = text.strip().splitlines()
    count = 0
    for line in lines:
        line = line.strip()
        if re.search(r'[;{}=<>#\[\]()]', line):
            count += 1
        elif KEYWORD_LINE_RE.match(line):
            count += 1
        elif re.search(r'\+\+\w*|\w+\+\+', line):
            count += 1
    return count

def is_structurally_code_like_block(text):
    for _, msg in extract_chat_turns(text):
        lines = msg.strip().splitlines()
        code_like_lines = 0
        for line in lines:
            line = line.strip()
            if re.search(r'[;{}=<>#\[\]()]', line):
                code_like_lines += 1
            elif KEYWORD_LINE_RE.match(line):
                code_like_lines += 1
        if code_like_lines > 2:
            return True
    return False

def is_code_like_message(text):
    for _, msg in extract_chat_turns(text):
        if is_structurally_code_like_block(msg):
            return True
    return False

def strip_flower_and_trailing_colon(text):
    """
    Removes all 🥀 emojis, any ':' at the end of each line, and any starting '.' or ',' at the start of lines.
    """
    if not text:
        return text
    # Remove all '🥀'
    text = text.replace("🥀", "")
    # Remove all ':' at end of lines (including if space before colon)
    text = re.sub(r':[ \t]*$', '', text, flags=re.MULTILINE)
    # Remove all starting '.' or ',' on each line (optionally preceded by whitespace)
    text = re.sub(r'^[ \t]*[.,]+', '', text, flags=re.MULTILINE)

    return text

def normalize_spacing(text):
    # collapse multiple spaces into one (but preserve newlines/tabs/etc.)
    text = re.sub(r' {2,}', ' ', text)
    # remove space before punctuation (only , . ! ? ; :)
    text = re.sub(r' ([.,!?;])', r'\1', text)
    return text.strip()

def strip_hash_and_angle(text):
    """
    Removes all '#' at the beginning of lines (any count), and strips literal '<' and '>'
    only inside the text between <|im_start|>user/assistant and <|im_end|>.
    """
    if not text:
        return text

    def clean_block(msg):
        # Remove all # at start of each line
        msg = re.sub(r'^[#]+', '', msg, flags=re.MULTILINE)
        # Strip all < and > from this block
        return msg.replace('<', '').replace('>', '')

    cleaned_blocks = []
    for role, msg in extract_chat_turns(text):
        cleaned_blocks.append(f"<|im_start|>{role}\n{clean_block(msg)}<|im_end|>")

    # Keep <|end_of_text|> if present
    if text.strip().endswith("<|end_of_text|>"):
        return "\n".join(cleaned_blocks) + "<|end_of_text|>"
    return "".join(cleaned_blocks)

def filter_rows_batch(rows, row_offset, text_index):
    good_rows = []
    bad_rows = []

    for i, row in enumerate(rows):
        if len(row) <= text_index or not isinstance(row[text_index], str):
            continue
        text = row[text_index]
        
        text = strip_double_pipe(text)
        text = strip_hash_and_angle(text)
        text = strip_at_mentions(text)
        text = strip_flower_and_trailing_colon(text)
        text = strip_emojis(text)
        text = clean_commas_and_periods(text)
        text = normalize_spacing(text)

        if not text:
            continue

        row[text_index] = text  # update cleaned text

        # filtering
        rejected = None
        if is_card_trade_post(text):
            rejected = "CARD_TRADE"
        elif is_code_like_message(text):
            rejected = "CODE_LIKE"
        elif has_consecutive_emojis(text):
            rejected = "EMOJI_SPAM"
        elif is_bulletpoint_list(text):
            rejected = "BULLETPOINT_LIST"
        elif is_lf_trade_hint(text):
            rejected = "LF_HINT"
        elif has_fake_token(text):
            rejected = "FAKE_TOKEN"
        elif has_dashword(text):
            rejected = "DOUBLE_DASH"
        elif has_percent_encoding(text):
            rejected = "PERCENT_ENCODING"
        elif (script := is_nonsensical_script_spam(text)):
            rejected = f"NONLATIN_SPAM_{script.upper()}"
        elif is_single_word_lines_block(text):
            rejected = "SINGLE_WORD_LINES"
        elif is_service_ad(text):
            rejected = "SERVICE_AD"
        elif has_multi_newline_block(text):
            rejected = "MULTI_NEWLINES"
        elif has_discord_timestamp(text):
            rejected = "DISCORD_TIMESTAMP"
        elif is_duplicate_lines_only(text):
            rejected = "DUPLICATE_LINES_ONLY"
        elif has_numbered_lines_block(text):
            rejected = "NUMBERED_LINE_BLOCK"
        elif has_numeric_only_line(text):
            rejected = "NUMERIC_ONLY_LINE"
        elif has_nword(text):
            rejected = "NWORD"
        elif has_cp_reference(text):
            rejected = "CP_REFERENCE"

        if rejected:
            bad_rows.append([rejected] + row)
        else:
            good_rows.append(row)

    # token length cutoff
    accepted = []
    try:
        encodings = tokenizer([r[text_index] for r in good_rows], add_special_tokens=True, truncation=False)
        for r, enc in zip(good_rows, encodings["input_ids"]):
            if len(enc) <= MAX_TOKENS:
                accepted.append(r)
            else:
                bad_rows.append(["TOO_LONG"] + r)
    except:
        for r in good_rows:
            try:
                enc = tokenizer(r[text_index], add_special_tokens=True, truncation=False)["input_ids"]
                if len(enc) <= MAX_TOKENS:
                    accepted.append(r)
                else:
                    bad_rows.append(["TOO_LONG"] + r)
            except:
                bad_rows.append(["TOKEN_ERROR"] + r)

    return accepted, bad_rows

def filter_wrapper(batch_offset_and_index):
    (rows, offset, text_index) = batch_offset_and_index
    return filter_rows_batch(rows, offset, text_index)

def trim_and_strip():
    try:
        print(f"📂 Opening and writing to: {TRIMMED_CSV}")
        with open(INPUT_CSV, newline='', encoding='utf-8') as f:
            total = sum(1 for _ in csv.reader(f)) - 1

        with open(INPUT_CSV, newline='', encoding='utf-8') as infile:
            reader = csv.reader(infile)
            header = next(reader)
            rows = list(reader)
            text_col_index = header.index(TEXT_COLUMN_NAME)

        with open(TRIMMED_CSV, 'w', newline='', encoding='utf-8') as outfile, \
             open(BAD_ROWS_CSV, "w", newline="", encoding="utf-8") as badfile:

            writer = csv.writer(outfile)
            bad_writer = csv.writer(badfile)
            writer.writerow(header)
            bad_writer.writerow(["reason"] + header)

            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(bar_width=40),
                TaskProgressColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                refresh_per_second=5,
            ) as progress:
                task = progress.add_task("Trimming/Stripping", total=len(rows))

                with Pool(NUM_PROCESSES) as pool:
                    try:
                        for i in range(0, len(rows), CHUNK_SIZE * NUM_PROCESSES):
                            chunk_batches = [
                                (rows[j:j + CHUNK_SIZE], j, text_col_index)
                                for j in range(i, min(i + CHUNK_SIZE * NUM_PROCESSES, len(rows)), CHUNK_SIZE)
                            ]

                            show_mem(f"Batch {i + 1}–{min(i + CHUNK_SIZE * NUM_PROCESSES, len(rows))}")
                            # results = pool.imap_unordered(filter_wrapper, chunk_batches)
                            results = pool.map(filter_wrapper, chunk_batches)

                            accepted_buffer = []
                            bad_buffer = []

                            for accepted, bad in results:
                                accepted_buffer.extend(accepted)
                                bad_buffer.extend(bad)

                            writer.writerows(accepted_buffer)
                            bad_writer.writerows(bad_buffer)
                            progress.update(task, advance=len(accepted_buffer) + len(bad_buffer))
                    except KeyboardInterrupt:
                        print("\n[!] Ctrl+C detected during multiprocessing. Cleaning up...")
                        pool.terminate()
                        pool.join()
                        sys.exit(1)

        print(f"ROWS PROCESSED: {len(rows)}")
        with open(TRIMMED_CSV, newline='', encoding="utf-8") as f:
            nrows = sum(1 for _ in csv.reader(f)) - 1
            print("OUTPUT ROWS:", nrows)
    except KeyboardInterrupt:
        print("\n[!] Ctrl+C detected. Exiting trim_and_strip cleanly.")
        sys.exit(1)

def refilter_rows_batch(rows, row_offset, text_index):
    good_rows = []
    bad_rows = []

    for row in rows:
        if not row or len(row) <= text_index or not isinstance(row[text_index], str):
            continue

        text = row[text_index]
    
        text = strip_double_pipe(text)
        text = strip_hash_and_angle(text)
        text = strip_at_mentions(text)
        text = strip_flower_and_trailing_colon(text)
        text = strip_emojis(text)
        text = clean_commas_and_periods(text)
        text = normalize_spacing(text)

        row[text_index] = text

        if not text:
            continue

        rejected = None
        if is_card_trade_post(text):
            rejected = "CARD_TRADE"
        elif is_code_like_message(text):
            rejected = "CODE_LIKE"
        elif has_consecutive_emojis(text):
            rejected = "EMOJI_SPAM"
        elif is_bulletpoint_list(text):
            rejected = "BULLETPOINT_LIST"
        elif is_lf_trade_hint(text):
            rejected = "LF_HINT"
        elif has_fake_token(text):
            rejected = "FAKE_TOKEN"
        elif has_dashword(text):
            rejected = "DOUBLE_DASH"
        elif has_percent_encoding(text):
            rejected = "PERCENT_ENCODING"
        elif (script := is_nonsensical_script_spam(text)):
            rejected = f"NONLATIN_SPAM_{script.upper()}"
        elif is_single_word_lines_block(text):
            rejected = "SINGLE_WORD_LINES"
        elif is_service_ad(text):
            rejected = "SERVICE_AD"
        elif has_multi_newline_block(text):
            rejected = "MULTI_NEWLINES"
        elif has_discord_timestamp(text):
            rejected = "DISCORD_TIMESTAMP"
        elif is_duplicate_lines_only(text):
            rejected = "DUPLICATE_LINES_ONLY"
        elif has_numbered_lines_block(text):
            rejected = "NUMBERED_LINE_BLOCK"
        elif has_numeric_only_line(text):
            rejected = "NUMERIC_ONLY_LINE"
        elif has_nword(text):
            rejected = "NWORD"
        elif has_cp_reference(text):
            rejected = "CP_REFERENCE"

        if rejected:
            bad_rows.append([rejected] + row)
        else:
            good_rows.append(row)

    return good_rows, bad_rows

def refilter_trimmed():
    print(f"Opening: {TRIMMED_CSV}")
    with open(TRIMMED_CSV, newline='', encoding="utf-8") as infile:
        reader = csv.reader(infile)
        header = next(reader)
        rows = list(reader)
        text_col_index = header.index(TEXT_COLUMN_NAME)

    with open(RESLANG_CSV, "w", newline="", encoding="utf-8") as goodfile, \
         open(BAD_ROWS_CSV, "w", newline="", encoding="utf-8") as badfile:

        writer = csv.writer(goodfile)
        bad_writer = csv.writer(badfile)

        writer.writerow(header)
        bad_writer.writerow(["reason"] + header)

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=40),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            refresh_per_second=5,
        ) as progress:
            task = progress.add_task("🔍 Refiltering", total=len(rows))

            with Pool(NUM_PROCESSES) as pool:
                try:
                    for i in range(0, len(rows), CHUNK_SIZE * NUM_PROCESSES):
                        chunk_batches = [
                            (rows[j:j + CHUNK_SIZE], j, text_col_index)
                            for j in range(i, min(i + CHUNK_SIZE * NUM_PROCESSES, len(rows)), CHUNK_SIZE)
                        ]

                        show_mem(f"Batch {i + 1}–{min(i + CHUNK_SIZE * NUM_PROCESSES, len(rows))}")
                        results = pool.starmap(refilter_rows_batch, chunk_batches)

                        accepted_buffer = []
                        bad_buffer = []

                        for good, bad in results:
                            accepted_buffer.extend(good)
                            bad_buffer.extend(bad)

                        writer.writerows(accepted_buffer)
                        bad_writer.writerows(bad_buffer)
                        progress.update(task, advance=len(accepted_buffer) + len(bad_buffer))
                except KeyboardInterrupt:
                    print("\n[!] Ctrl+C detected during multiprocessing. Cleaning up...")
                    pool.terminate()
                    pool.join()

    print(f"Done: {len(rows)} processed")

def get_token_length(batch):
    return [len(tokenizer.encode(x, truncation=True, max_length=tokenizer.model_max_length)) for x in batch]

def smart_sample_ordered_random():
    try:
        print(f"Opening filtered CSV: {RESLANG_CSV}")
        df = pl.read_csv(RESLANG_CSV).drop_nulls(subset=[TEXT_COLUMN_NAME])
        texts = df[TEXT_COLUMN_NAME].to_list()

        print(f"Tokenizing {len(texts):,} rows and grouping by token length...")
        batches = [texts[i:i + BATCH_SIZE] for i in range(0, len(texts), BATCH_SIZE)]

        with Pool(NUM_PROCESSES) as pool:
            try:
                token_lens_batches = list(track(
                    pool.imap(get_token_length, batches),
                    total=len(batches),
                    description="Tokenizing"
                ))
            except KeyboardInterrupt:
                print("\n[!] Ctrl+C detected during tokenizing. Cleaning up...")
                pool.terminate()
                pool.join()
                return

        print("Tokenizing complete, flattening and bucketing...")

        flat_lens = [l for sub in token_lens_batches for l in sub]
        df = df.with_columns(pl.Series("tokens", flat_lens).cast(pl.Int32))

        # Shuffle all columns except the grouping key
        cols_to_shuffle = [c for c in df.columns if c != "tokens"]
        grouped = (
            df.group_by("tokens", maintain_order=True)
              .agg([pl.col(c).shuffle() for c in cols_to_shuffle])
              .explode(cols_to_shuffle)
              .sort("tokens", descending=True)
        )

        selected_df = grouped.head(TARGET_SIZE)

        print(f"\nSelected {selected_df.height:,} samples. Saving to: {RESAMPLED_CSV}")

        if selected_df.is_empty():
            print("No rows selected. Skipping resampled.csv write.")
            return

        # Reorder columns: text, tokens, then others
        column_order = ["text", "tokens"] + [c for c in selected_df.columns if c not in ("text", "tokens")]

        with open(RESAMPLED_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(column_order)
            for row in selected_df.iter_rows(named=True):
                writer.writerow([str(row.get(col, "")) for col in column_order])

    except KeyboardInterrupt:
        print("\n[!] Ctrl+C detected. Exiting smart_sample_ordered_random cleanly.")
        sys.exit(1)

if __name__ == "__main__":
    try:
        trim_and_strip()
        #refilter_trimmed()
        process_slang()
        smart_sample_ordered_random()
        doublecheck_final_pass()
    except KeyboardInterrupt:
        print("\n[!] Ctrl+C detected in main. Exiting now.")
        sys.exit(1)