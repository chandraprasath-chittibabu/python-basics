"""
Sample script demonstrating two useful pandas techniques:

  1. Converting Excel to Parquet
       - Parquet is a columnar binary format: much smaller on disk,
         much faster to read/write, and preserves data types (Excel
         loses them). Great for large datasets.

  2. Fuzzy deduplication
       - drop_duplicates() only catches exact matches. Real-world data
         has typos, casing differences, and formatting variations:
           "Robert Smith" vs "Bob Smith"
           "alice@corp.com" vs "Alice@Corp.Com "
         Fuzzy matching uses string similarity scores to catch these.

Required packages:
    pip install pandas openpyxl pyarrow rapidfuzz
"""

from pathlib import Path
import time

import pandas as pd
from rapidfuzz import fuzz, process


# ---------------------------------------------------------------------------
# PART 1: Excel -> Parquet
# ---------------------------------------------------------------------------

def excel_to_parquet(excel_path: Path, parquet_path: Path) -> pd.DataFrame:
    """
    Read an Excel file and save it as Parquet.

    Why bother?
      - A 100 MB Excel file often becomes a 10-20 MB Parquet file.
      - Reading Parquet is 10-50x faster than reading Excel.
      - Column data types (int, datetime, category) are preserved.
      - You can read just a few columns without loading the whole file.
    """
    print(f"Reading Excel: {excel_path}")
    t0 = time.time()
    df = pd.read_excel(excel_path)
    excel_read_time = time.time() - t0
    excel_size_mb = excel_path.stat().st_size / (1024 * 1024)

    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Excel file size: {excel_size_mb:.2f} MB")
    print(f"  Excel read time: {excel_read_time:.2f}s")

    # Write to Parquet. 'snappy' compression is the default and a good balance
    # of speed vs. size. Use 'gzip' for smaller files, 'zstd' for best ratio.
    df.to_parquet(parquet_path, engine="pyarrow", compression="snappy")

    parquet_size_mb = parquet_path.stat().st_size / (1024 * 1024)

    # Time a Parquet re-read to show the speedup
    t0 = time.time()
    _ = pd.read_parquet(parquet_path)
    parquet_read_time = time.time() - t0

    print(f"\n  Parquet file size: {parquet_size_mb:.2f} MB "
          f"({excel_size_mb / parquet_size_mb:.1f}x smaller)")
    print(f"  Parquet read time: {parquet_read_time:.2f}s "
          f"({excel_read_time / max(parquet_read_time, 0.001):.1f}x faster)")

    return df


# ---------------------------------------------------------------------------
# PART 2: Fuzzy deduplication
# ---------------------------------------------------------------------------

def normalize(text: str) -> str:
    """
    Lowercase, strip whitespace, collapse multiple spaces.

    Normalization catches trivial differences before we even reach the
    fuzzy comparison step. It also makes the fuzzy scores more meaningful
    (so "  Bob  " and "bob" score 100 instead of 85).
    """
    if pd.isna(text):
        return ""
    return " ".join(str(text).lower().strip().split())


def fuzzy_deduplicate(
    df: pd.DataFrame,
    key_columns: list[str],
    threshold: int = 90,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Remove near-duplicate rows based on fuzzy string matching.

    Approach (simple but effective for up to ~10k rows):
      1. Build a normalized 'match key' by joining the key columns.
      2. Walk the rows one by one. For each row, compare its key against
         all rows already kept. If similarity >= threshold, mark as dup.
      3. Otherwise, keep it.

    The similarity metric is rapidfuzz's token_sort_ratio, which handles
    word-order differences ("Smith Bob" vs "Bob Smith" -> 100).

    Args:
        df: input DataFrame
        key_columns: columns whose combined value identifies a "person"
        threshold: 0-100. Higher = stricter. 90 is a good default.
                   85 catches more but risks false positives.

    Returns:
        (kept_df, removed_df) — the deduped data and the rows that were
        flagged as duplicates (kept for auditing).
    """
    print(f"\nFuzzy dedup on {key_columns} (threshold={threshold})")

    # Build one normalized string per row that we'll compare against.
    df = df.copy()
    df["_match_key"] = df[key_columns].apply(
        lambda row: " ".join(normalize(v) for v in row),
        axis=1,
    )

    kept_indices: list[int] = []
    kept_keys: list[str] = []
    removed_rows: list[dict] = []

    for idx, row in df.iterrows():
        key = row["_match_key"]

        if not kept_keys:
            kept_indices.append(idx)
            kept_keys.append(key)
            continue

        # Find the best match among rows we've already decided to keep.
        # process.extractOne is much faster than looping ourselves because
        # rapidfuzz is implemented in C++.
        best = process.extractOne(
            key,
            kept_keys,
            scorer=fuzz.token_sort_ratio,
        )
        best_match_str, best_score, best_pos = best

        if best_score >= threshold:
            # Duplicate of an already-kept row
            removed_rows.append({
                **row.to_dict(),
                "_matched_against": best_match_str,
                "_similarity": best_score,
            })
        else:
            kept_indices.append(idx)
            kept_keys.append(key)

    kept_df = df.loc[kept_indices].drop(columns="_match_key")
    removed_df = pd.DataFrame(removed_rows)
    if "_match_key" in removed_df.columns:
        removed_df = removed_df.drop(columns="_match_key")

    print(f"  Kept: {len(kept_df)} rows")
    print(f"  Removed: {len(removed_df)} fuzzy duplicates")

    return kept_df, removed_df


# ---------------------------------------------------------------------------
# Demo / driver
# ---------------------------------------------------------------------------

def create_sample_excel(path: Path) -> None:
    """Generate a small Excel file with intentional fuzzy duplicates."""
    data = [
        {"id": 1, "name": "Robert Smith",  "email": "bob.smith@corp.com",   "city": "New York"},
        {"id": 2, "name": "Bob Smith",     "email": "Bob.Smith@corp.com ",  "city": "New York"},   # dup of 1
        {"id": 3, "name": "Alice Johnson", "email": "alice@corp.com",       "city": "Boston"},
        {"id": 4, "name": "alice johnson", "email": "alice@corp.com",       "city": "Boston"},     # dup of 3
        {"id": 5, "name": "Charlie Brown", "email": "charlie@corp.com",     "city": "Chicago"},
        {"id": 6, "name": "Chuck Brown",   "email": "chuck@corp.com",       "city": "Chicago"},    # NOT a dup - different email/nickname
        {"id": 7, "name": "Diana Prince",  "email": "diana@corp.com",       "city": "LA"},
        {"id": 8, "name": "Diana  Prince", "email": "diana@corp.com",       "city": "Los Angeles"},# dup of 7
    ]
    pd.DataFrame(data).to_excel(path, index=False)
    print(f"Created sample Excel: {path}\n")


def main() -> None:
    work_dir = Path(__file__).parent
    excel_path = work_dir / "sample_employees.xlsx"
    parquet_path = work_dir / "sample_employees.parquet"
    clean_path = work_dir / "sample_employees_clean.parquet"
    audit_path = work_dir / "sample_employees_removed.xlsx"

    # Step 1: create a demo Excel file so this script is self-contained
    create_sample_excel(excel_path)

    # Step 2: convert Excel -> Parquet
    df = excel_to_parquet(excel_path, parquet_path)

    # Step 3: fuzzy dedup on name + email
    kept, removed = fuzzy_deduplicate(
        df,
        key_columns=["name", "email"],
        threshold=88,
    )

    # Step 4: save results
    #   - Clean data as Parquet (fast, small, good for downstream processing)
    #   - Removed rows as Excel (easy for humans to review)
    kept.to_parquet(clean_path, engine="pyarrow", compression="snappy")
    if not removed.empty:
        removed.to_excel(audit_path, index=False)

    print(f"\nOutputs:")
    print(f"  Clean data:    {clean_path}")
    print(f"  Removed audit: {audit_path if not removed.empty else '(none)'}")


if __name__ == "__main__":
    main()
