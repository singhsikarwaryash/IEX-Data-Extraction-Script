#!/usr/bin/env python3
"""
Resilient & fast scraper for IEX → Green Day Ahead Market → Aggregate Demand–Supply.
- Parallel pass-1 across 96 blocks/day (persistent session).
- Detects missing time blocks; refetches only gaps in gentler passes.
- Ensures final CSV column order:
    delivery_date, time_block, price_range_rs_per_mwh, buy_mw, sell_mw, mcp_inr_per_mwh, mcv_mw
"""

import os
import re
import sys
import time
import random
import urllib.parse
from io import StringIO
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import pandas as pd
from bs4 import BeautifulSoup

# ======= USER CONFIGURABLE SECTION =======
START_DATE = "01-01-2025"   # DD-MM-YYYY
END_DATE   = "03-01-2025"   # DD-MM-YYYY
OUT_DIR    = "out"          # per-day CSVs
OUT_CSV    = f"gdam_aggregate_{START_DATE}_to_{END_DATE}.csv"

# Speed/robustness knobs
WORKERS_FIRST_PASS   = 10    # 8–12 is good; reduce if you see throttling
WORKERS_SECOND_PASS  = 5     # gentler for gaps
WORKERS_THIRD_PASS   = 1     # sequential final pass
PARSER               = "pandas"   # 'pandas' (robust) or 'bs4' (lighter)
INCLUDE_MCP          = True       # set False for a small CPU win
JITTER_RANGE         = (0.03, 0.10)  # polite jitter per request
RETRIES              = 3
TIMEOUT              = 30
NA_AS_BLANK          = False  # write blank cells instead of NaN
# ========================================

BASE_URL = "https://www.iexindia.com/market-data/green-day-ahead-market/aggregate-demand-supply"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/121.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

def daterange(start_date: datetime, end_date: datetime):
    d = start_date
    while d <= end_date:
        yield d
        d += timedelta(days=1)

def gen_time_blocks():
    """96 blocks: 00:00-00:15 ... 23:45-24:00 (last ends 24:00)."""
    blocks = []
    base = datetime(2000, 1, 1, 0, 0)
    for i in range(96):
        t1 = base + timedelta(minutes=15 * i)
        t2 = t1 + timedelta(minutes=15)
        s1 = t1.strftime("%H:%M")
        s2 = "24:00" if i == 95 else t2.strftime("%H:%M")
        blocks.append(f"{s1}-{s2}")
    return blocks

# ---------- Column normalization & sorting ----------

CANON_COLS = [
    "delivery_date",
    "time_block",
    "price_range_rs_per_mwh",
    "buy_mw",
    "sell_mw",
    "mcp_inr_per_mwh",
    "mcv_mw",
]

def canonicalize_and_sort(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy with canonical columns in order and rows sorted by
    (delivery_date, chronological time_block, price_range).
    """
    # Create a fresh copy to avoid SettingWithCopyWarning
    df = (df.copy() if df is not None else pd.DataFrame())
    if df.empty:
        return pd.DataFrame(columns=CANON_COLS)

    # Ensure columns exist & order (reindex is safe and avoids chained assignment)
    df = df.reindex(columns=CANON_COLS, fill_value=pd.NA)

    # Chronological sort by time_block using a mapping
    block_order = {tb: i for i, tb in enumerate(gen_time_blocks())}
    df.loc[:, "__tb_idx"] = df["time_block"].map(block_order)
    df.loc[:, "__tb_idx"] = df["__tb_idx"].fillna(1e9)
    df = df.sort_values(
        by=["delivery_date", "__tb_idx", "price_range_rs_per_mwh"],
        kind="stable"
    ).drop(columns="__tb_idx")
    return df

# ------------------- TABLE PARSERS -------------------

def extract_table_with_pandas(html_text: str):
    """Use pandas.read_html on table that has price/buy/sell columns."""
    try:
        tables = pd.read_html(StringIO(html_text))
    except ValueError:
        return None
    if not tables:
        return None

    for df in tables:
        cols = [str(c).strip().lower() for c in df.columns]
        df.columns = cols
        if any("price" in c for c in cols) and any("buy" in c for c in cols) and any("sell" in c for c in cols):
            price_col = next(c for c in cols if "price" in c)
            buy_col   = next(c for c in cols if "buy" in c)
            sell_col  = next(c for c in cols if "sell" in c)
            out = df[[price_col, buy_col, sell_col]].copy()
            out.columns = ["price_range_rs_per_mwh", "buy_mw", "sell_mw"]
            out["buy_mw"]  = pd.to_numeric(out["buy_mw"], errors="coerce")
            out["sell_mw"] = pd.to_numeric(out["sell_mw"], errors="coerce")
            return out

    # Fallback: largest table → best-effort rename
    df = max(tables, key=len).copy()
    cols = list(df.columns)
    while len(cols) < 3:
        cols.append(f"col_{len(cols)}")
    df.columns = ["price_range_rs_per_mwh", "buy_mw", "sell_mw"] + cols[3:]
    for c in ("buy_mw", "sell_mw"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[["price_range_rs_per_mwh", "buy_mw", "sell_mw"]]

def extract_table_with_bs4(html_text: str):
    """Lighter parser: parse first <table> and map headers heuristically."""
    soup = BeautifulSoup(html_text, "lxml")
    table = soup.find("table")
    if not table:
        return None

    headers = [th.get_text(strip=True).lower() for th in table.find_all("th")]
    if not headers:
        first_tr = table.find("tr")
        if first_tr:
            headers = [td.get_text(strip=True).lower() for td in first_tr.find_all("td")]

    def map_col(h):
        if "price" in h and ("mwh" in h or "range" in h): return "price_range_rs_per_mwh"
        if "price" in h and "rs" in h: return "price_range_rs_per_mwh"
        if "buy" in h or "demand" in h: return "buy_mw"
        if "sell" in h or "supply" in h: return "sell_mw"
        return h

    mapped = [map_col(h) for h in headers]
    rows = []
    for tr in table.find_all("tr"):
        tds = [td.get_text(strip=True) for td in tr.find_all("td")]
        if len(tds) >= 3:
            rows.append(tds[:3])

    if not rows:
        return None
    while len(mapped) < 3:
        mapped.append(f"col_{len(mapped)}")
    df = pd.DataFrame(rows, columns=mapped[:3])

    ren = {}
    for c in df.columns:
        cl = c.lower()
        if cl not in {"price_range_rs_per_mwh", "buy_mw", "sell_mw"}:
            if "price" in cl or "range" in cl: ren[c] = "price_range_rs_per_mwh"
            elif "buy" in cl or "demand" in cl: ren[c] = "buy_mw"
            elif "sell" in cl or "supply" in cl: ren[c] = "sell_mw"
    df = df.rename(columns=ren)

    for c in ("buy_mw", "sell_mw"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].str.replace(",", ""), errors="coerce")
    keep = [c for c in ["price_range_rs_per_mwh", "buy_mw", "sell_mw"] if c in df.columns]
    return df[keep] if keep else None

# ------------------- MCP/MCV -------------------

def extract_mcp_mcv_text(html_text: str):
    """Parse MCP/MCV if present near the chart (MCP : 3478.50 / MCV : 652.80)."""
    text = BeautifulSoup(html_text, "lxml").get_text(" ", strip=True)
    num = r"([\d,]*\.?\d+)"
    mcp = re.search(rf"MCP\s*:\s*{num}", text, flags=re.I)
    mcv = re.search(rf"MCV\s*:\s*{num}", text, flags=re.I)
    def to_f(s):
        try:
            return float(s.replace(",", ""))
        except Exception:
            return None
    return (to_f(mcp.group(1)) if mcp else None,
            to_f(mcv.group(1)) if mcv else None)

# ------------------- Fetch & Day Scrape -------------------

def fetch_block(session: requests.Session,
                parser: str,
                include_mcp: bool,
                date_ddmmyyyy: str,
                tb: str,
                retries: int = RETRIES,
                timeout: int = TIMEOUT):
    """Fetch one time block; return (time_block, DataFrame or None)."""
    time.sleep(random.uniform(*JITTER_RANGE))  # micro-jitter
    params = {"date": date_ddmmyyyy, "fromTime": tb, "toTime": tb}
    url = f"{BASE_URL}?{urllib.parse.urlencode(params)}"
    backoff = 0.5
    for attempt in range(retries):
        try:
            resp = session.get(url, timeout=timeout)
            resp.raise_for_status()
            df = (extract_table_with_bs4(resp.text) if parser == "bs4"
                  else extract_table_with_pandas(resp.text))
            if df is None or df.empty:
                raise RuntimeError("Empty table")
            if include_mcp:
                mcp, mcv = extract_mcp_mcv_text(resp.text)
                df["mcp_inr_per_mwh"] = mcp
                df["mcv_mw"] = mcv
            else:
                df["mcp_inr_per_mwh"] = None
                df["mcv_mw"] = None
            df["delivery_date"] = date_ddmmyyyy
            df["time_block"] = tb
            return tb, df
        except Exception:
            if attempt == retries - 1:
                return tb, None
            time.sleep(backoff)
            backoff *= 1.6

def scrape_day_pass(session, date_str, parser, include_mcp, workers):
    """One pass over all 96 blocks; returns dict{time_block: DataFrame or None}."""
    blocks = gen_time_blocks()
    random.shuffle(blocks)  # reduce patterned throttling
    results = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch_block, session, parser, include_mcp, date_str, tb): tb for tb in blocks}
        done = 0
        total = len(futs)
        for fut in as_completed(futs):
            tb, df = fut.result()
            results[tb] = df
            done += 1
            if done % 10 == 0 or done == total:
                print(f"    {date_str}: {done}/{total} blocks fetched", flush=True)
    return results

def combine_blocks(result_map):
    """Combine non-empty frames; return (combined_df, missing_blocks_set)."""
    dfs, missing = [], []
    for tb, df in result_map.items():
        if df is not None and not df.empty:
            dfs.append(df)
        else:
            missing.append(tb)
    combined = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    return combined, set(missing)

def scrape_day_with_retries(session: requests.Session, date_str: str):
    print(f"Scraping {date_str} (pass 1: workers={WORKERS_FIRST_PASS}, parser={PARSER}, MCP={INCLUDE_MCP}) ...", flush=True)
    pass1 = scrape_day_pass(session, date_str, PARSER, INCLUDE_MCP, WORKERS_FIRST_PASS)
    combined, missing = combine_blocks(pass1)

    if missing:
        print(f"  {date_str}: {len(missing)} missing blocks after pass 1 → pass 2 (workers={WORKERS_SECOND_PASS})", flush=True)
        pass2 = {}
        with ThreadPoolExecutor(max_workers=WORKERS_SECOND_PASS) as ex:
            futs = {ex.submit(fetch_block, session, PARSER, INCLUDE_MCP, date_str, tb): tb for tb in sorted(missing)}
            for fut in as_completed(futs):
                tb, df = fut.result()
                pass2[tb] = df
        pass1.update(pass2)
        combined, missing = combine_blocks(pass1)

    if missing:
        print(f"  {date_str}: {len(missing)} still missing → pass 3 (sequential, parser flip, MCP off)", flush=True)
        for tb in sorted(missing):
            _tb, df2 = fetch_block(session,
                                   parser=("bs4" if PARSER == "pandas" else PARSER),
                                   include_mcp=False,
                                   date_ddmmyyyy=date_str,
                                   tb=tb,
                                   retries=4,
                                   timeout=40)
            pass1[tb] = df2
        combined, missing = combine_blocks(pass1)

    # Canonicalize columns & sort
    combined = canonicalize_and_sort(combined)

    # Final report
    got_blocks = combined["time_block"].nunique() if not combined.empty else 0
    print(f"  {date_str}: rows={len(combined)} | unique time_blocks={got_blocks} | missing={len(missing)}", flush=True)
    if missing:
        print(f"  Missing blocks: {', '.join(sorted(missing))}", flush=True)
    return combined

# ------------------- Main -------------------

if __name__ == "__main__":
    t0 = time.perf_counter()
    fmt = "%d-%m-%Y"
    try:
        start_dt = datetime.strptime(START_DATE, fmt)
        end_dt = datetime.strptime(END_DATE, fmt)
    except ValueError:
        print("Dates must be in DD-MM-YYYY format.", file=sys.stderr, flush=True)
        sys.exit(1)

    os.makedirs(OUT_DIR, exist_ok=True)
    session = requests.Session()
    session.headers.update(HEADERS)

    all_days = []
    for d in daterange(start_dt, end_dt):
        ds = d.strftime("%d-%m-%Y")
        day_df = scrape_day_with_retries(session, ds)
        day_out = os.path.join(OUT_DIR, f"gdam_{ds}.csv")
        day_df.to_csv(day_out, index=False, encoding="utf-8", na_rep=("" if NA_AS_BLANK else None))
        print(f"  Wrote {len(day_df)} rows → {day_out}\n", flush=True)
        all_days.append(day_df)

    final = pd.concat(all_days, ignore_index=True) if all_days else pd.DataFrame(columns=CANON_COLS)
    final = canonicalize_and_sort(final)
    final.to_csv(OUT_CSV, index=False, encoding="utf-8", na_rep=("" if NA_AS_BLANK else None))
    t1 = time.perf_counter()
    print(f"Saved {len(final)} rows → {OUT_CSV} | Elapsed: {t1 - t0:0.1f}s", flush=True)