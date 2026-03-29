import re
import time
import random
import urllib.parse
from datetime import datetime, timedelta

import requests
import pandas as pd
from bs4 import BeautifulSoup

BASE_URL = "https://www.iexindia.com/market-data/green-day-ahead-market/aggregate-demand-supply"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/121.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}

def daterange(start_date: datetime, end_date: datetime):
    """Inclusive date range generator."""
    d = start_date
    while d <= end_date:
        yield d
        d += timedelta(days=1)

def gen_time_blocks():
    """96 blocks: 00:00-00:15 ... 23:45-24:00."""
    blocks = []
    base = datetime(2000, 1, 1, 0, 0)
    for i in range(96):
        t1 = base + timedelta(minutes=15*i)
        t2 = t1 + timedelta(minutes=15)
        s1 = t1.strftime("%H:%M")
        s2 = "24:00" if i == 95 else t2.strftime("%H:%M")
        blocks.append(f"{s1}-{s2}")
    return blocks

def extract_table_with_pandas(html_text: str):
    """Try pandas.read_html to pull the price-bucket table."""
    try:
        tables = pd.read_html(html_text)
    except ValueError:
        return None
    if not tables:
        return None

    # Choose the table that looks like: Price | Buy | Sell
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

    # Fallback: pick the largest table and best‑effort rename
    df = max(tables, key=len).copy()
    cols = list(df.columns)
    while len(cols) < 3:  # ensure at least 3
        cols.append(f"col_{len(cols)}")
    df.columns = ["price_range_rs_per_mwh", "buy_mw", "sell_mw"] + cols[3:]
    for c in ("buy_mw", "sell_mw"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[["price_range_rs_per_mwh", "buy_mw", "sell_mw"]]

def extract_mcp_mcv_text(html_text: str):
    """Pull MCP and MCV if they appear in text like 'MCP : 3478.50' / 'MCV : 652.80'."""
    text = BeautifulSoup(html_text, "lxml").get_text(" ", strip=True)
    num = r"([\d,]*\.?\d+)"
    mcp = re.search(rf"MCP\s*:\s*{num}", text, flags=re.I)
    mcv = re.search(rf"MCV\s*:\s*{num}", text, flags=re.I)
    to_f = lambda s: float(s.replace(",", "")) if s else None
    return to_f(mcp.group(1)) if mcp else None, to_f(mcv.group(1)) if mcv else None

def scrape_one_day(date_ddmmyyyy: str, throttle=(0.25, 0.75)):
    """Return a DataFrame for a single date by iterating its 96 time blocks."""
    all_df = []
    for tb in gen_time_blocks():
        params = {"date": date_ddmmyyyy, "fromTime": tb, "toTime": tb}
        url = f"{BASE_URL}?{urllib.parse.urlencode(params)}"
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()

        df = extract_table_with_pandas(resp.text)
        if df is None or df.empty:
            # If this happens systematically, switch to Selenium fallback.
            continue

        mcp, mcv = extract_mcp_mcv_text(resp.text)
        df["delivery_date"] = date_ddmmyyyy
        df["time_block"] = tb
        df["mcp_inr_per_mwh"] = mcp
        df["mcv_mw"] = mcv
        all_df.append(df)

        # Polite throttling
        time.sleep(random.uniform(*throttle))

    if not all_df:
        return pd.DataFrame(columns=[
            "delivery_date","time_block","price_range_rs_per_mwh",
            "buy_mw","sell_mw","mcp_inr_per_mwh","mcv_mw"
        ])

    out = pd.concat(all_df, ignore_index=True)
    cols = ["delivery_date","time_block","price_range_rs_per_mwh",
            "buy_mw","sell_mw","mcp_inr_per_mwh","mcv_mw"]
    return out[cols]

if __name__ == "__main__":
    # ======= SHORT DURATION (sample) =======
    START_DATE = "01-01-2025"  # DD-MM-YYYY
    END_DATE   = "03-01-2025"  # change later to "29-03-2026"

    fmt = "%d-%m-%Y"
    start_dt = datetime.strptime(START_DATE, fmt)
    end_dt   = datetime.strptime(END_DATE, fmt)

    daily_frames = []
    for d in daterange(start_dt, end_dt):
        ds = d.strftime("%d-%m-%Y")
        print(f"Scraping {ds} ...")
        day_df = scrape_one_day(ds)
        print(f"{ds}: {len(day_df)} rows")
        daily_frames.append(day_df)

    final = pd.concat(daily_frames, ignore_index=True)
    out_csv = f"gdam_aggregate_{START_DATE}_to_{END_DATE}.csv"
    final.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"Saved {len(final)} rows to {out_csv}")