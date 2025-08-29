from typing import List
from pydantic import TypeAdapter
import requests
import json
import os
import pandas as pd
import numpy as np
from datetime import timedelta
from bs4 import BeautifulSoup, Tag
import time

from pathlib import Path


from schemas import HourlyItemData  # type: ignore


# ---- Persistent position storage ----
POSITIONS_FILE = "gw2_positions.json"


def load_positions() -> dict:
    """Load per-item position state persisted across runs."""
    try:
        if os.path.exists(POSITIONS_FILE):
            with open(POSITIONS_FILE, "r") as f:
                return json.load(f)
    except Exception as e:
        print(f"Failed to load positions: {e}")
    return {}


def save_positions(positions: dict) -> None:
    """Persist per-item position state to disk."""
    try:
        with open(POSITIONS_FILE, "w") as f:
            json.dump(positions, f, indent=2)
    except Exception as e:
        print(f"Failed to save positions: {e}")


# ---- CSV holdings (manual input) ----

HOLDINGS_CSV = "gw2_holdings.csv"

# Feature flags for user-managed holdings
AUTO_UPDATE_FROM_SIGNALS = False  # Don't auto-enter/exit based on signals
AUTO_WRITE_HOLDINGS_CSV = False  # Do NOT rewrite CSV from positions
ENRICH_CSV_ON_LOAD = (
    True  # Only fill missing item_id/entry_time in-place when reading CSV
)

# --- Trading strategy constants ---
USE_ENTRY_FOR_SIGNALS = True
STOP_LOSS_MULT = 0.9  # Sell if price <= entry * 0.9


def holdings_dataframe_from_positions(positions: dict) -> pd.DataFrame:
    """Build a DataFrame of current holdings from the positions dict."""
    rows = []
    for key, pos in positions.items():
        if pos.get("in_position"):
            rows.append(
                {
                    "item_id": int(key) if str(key).isdigit() else key,
                    "item_name": pos.get("item_name", "Unknown"),
                    "entry_price": pos.get("entry_price"),
                    "entry_time": pos.get("entry_time"),
                }
            )
    if rows:
        return pd.DataFrame(
            rows, columns=["item_id", "item_name", "entry_price", "entry_time"]
        )
    else:
        return pd.DataFrame(
            columns=["item_id", "item_name", "entry_price", "entry_time"]
        )


def load_holdings_csv(path: str = HOLDINGS_CSV) -> list[dict]:
    """Load manual current holdings from a CSV file.
    Accepted columns:
      - item_name, entry_price  (preferred)
      - or item_id, entry_price
    Extra columns are ignored.
    """
    try:
        if os.path.exists(path):
            df = pd.read_csv(path)
            # Normalize column names
            cols = {c.strip().lower(): c for c in df.columns}
            have_name = "item_name" in cols
            have_id = "item_id" in cols
            have_price = "entry_price" in cols
            if not have_price or (not have_name and not have_id):
                print(
                    f"{path} missing required columns. Need 'entry_price' and one of 'item_name' or 'item_id'."
                )
                return []
            rows: list[dict] = []
            for _, r in df.iterrows():
                row = {}
                if have_name:
                    row["item_name"] = str(r[cols["item_name"]]).strip()
                if have_id and not pd.isna(r[cols["item_id"]]):
                    try:
                        row["item_id"] = int(r[cols["item_id"]])
                    except Exception:
                        pass
                try:
                    row["entry_price"] = float(r[cols["entry_price"]])
                except Exception:
                    continue
                rows.append(row)
            return rows
    except Exception as e:
        print(f"Failed to read {path}: {e}")
    return []


# --- Build positions dict from CSV as the sole source of truth ---
def positions_from_csv(rows: list[dict], item_id_source: dict) -> dict:
    """Construct a NEW positions dict using CSV rows as the sole source of truth.
    Requires entry_price AND (item_id OR item_name). Infers the other via mapping.
    Sets in_position=True and entry_time=now for all valid rows.
    """
    new_positions: dict = {}
    # Build reverse map id->name and name->id for inference
    name_to_id = item_id_source  # expected {name: id}
    id_to_name = {
        int(v): k
        for k, v in item_id_source.items()
        if isinstance(v, (int, float, str)) and str(v).isdigit()
    }

    now_iso = pd.Timestamp.now().isoformat()
    for row in rows:
        entry_price = row.get("entry_price")
        if entry_price is None:
            print(f"Skipping row without entry_price: {row}")
            continue
        item_id = row.get("item_id")
        item_name = row.get("item_name")

        # Infer missing piece from mapping
        if item_id is None and item_name:
            mapped = name_to_id.get(item_name)
            if mapped is not None:
                try:
                    item_id = int(mapped)
                except Exception:
                    item_id = None
        if item_name is None and item_id is not None:
            item_name = id_to_name.get(int(item_id), str(item_id))

        if item_id is None and item_name is None:
            print(f"Skipping row, cannot resolve item_id or item_name: {row}")
            continue
        if item_id is None:
            print(f"Skipping row, could not map item_name -> id: {row}")
            continue

        key = str(int(item_id))
        try:
            ep = float(entry_price)
        except Exception:
            print(f"Skipping row, invalid entry_price: {row}")
            continue
        new_positions[key] = {
            "in_position": True,
            "entry_price": ep,
            "entry_time": now_iso,
            "item_name": item_name if item_name is not None else str(item_id),
        }
    return new_positions


# --- Enrich holdings CSV in place: fill missing item_id and entry_time, keep extra columns ---
def enrich_holdings_csv_in_place(csv_path: str, item_id_source: dict) -> None:
    """Fill ONLY missing item_id and entry_time in the CSV, preserving all other columns as-is.
    - If item_id is missing but item_name exists and maps, fill item_id.
    - If entry_time column is missing or blank, fill with now ISO.
    - Writes back to the same CSV, preserving row order and extra columns.
    """
    if not os.path.exists(csv_path):
        return
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Failed to read {csv_path}: {e}")
        return

    if df.empty:
        return

    # Normalize known column names
    cols = {c.strip(): c for c in df.columns}
    # Support case-insensitive lookups
    lower_map = {c.lower(): c for c in df.columns}
    name_col = lower_map.get("item_name")
    id_col = lower_map.get("item_id")
    price_col = lower_map.get("entry_price")
    time_col = lower_map.get("entry_time")

    # Ensure required columns present minimally
    if price_col is None or (name_col is None and id_col is None):
        # Nothing to enrich safely
        return

    # Add entry_time column if missing
    if time_col is None:
        time_col = "entry_time"
        df[time_col] = pd.NA

    now_iso = pd.Timestamp.now().isoformat()

    # Fill entry_time where missing/NaN/empty
    mask_time_missing = df[time_col].isna() | (
        df[time_col].astype(str).str.strip() == ""
    )
    if mask_time_missing.any():
        df.loc[mask_time_missing, time_col] = now_iso

    # Fill item_id from item_name where missing
    if id_col is None:
        id_col = "item_id"
        df[id_col] = pd.NA
    if name_col is not None:
        # Build mapping once
        name_to_id = item_id_source
        mask_id_missing = df[id_col].isna() | (df[id_col].astype(str).str.strip() == "")
        if mask_id_missing.any():
            # Map names; only fill where mapping found
            mapped_ids = df.loc[mask_id_missing, name_col].map(name_to_id)
            # Coerce to Int64 (nullable) if possible
            try:
                df.loc[mask_id_missing, id_col] = pd.to_numeric(
                    mapped_ids, errors="coerce"
                ).astype("Int64")
            except Exception:
                df.loc[mask_id_missing, id_col] = mapped_ids

    # Write back preserving columns and order
    try:
        df.to_csv(csv_path, index=False)
    except Exception as e:
        print(f"Failed to write {csv_path}: {e}")


def apply_holdings_to_positions(
    rows: list[dict], item_id_source: dict, positions: dict
) -> None:
    """Apply CSV holdings into the in-memory positions dict (authoritative seed).
    Priority: item_id if present; else map item_name -> id via item_id_source.
    """
    name_to_id = {
        str(k): v for k, v in item_id_source.items()
    }  # incoming mapping looks like {name: id}
    for row in rows:
        entry_price = row.get("entry_price")
        if entry_price is None:
            continue
        item_id = row.get("item_id")
        item_name = row.get("item_name")
        if item_id is None and item_name:
            # Map name -> id using the provided source
            mapped = item_id_source.get(item_name)
            if mapped is not None:
                item_id = int(mapped)
        if item_id is None:
            print(f"Could not resolve item id for row: {row}")
            continue
        key = str(int(item_id))
        positions[key] = {
            "in_position": True,
            "entry_price": float(entry_price),
            "entry_time": positions.get(key, {}).get("entry_time")
            or pd.Timestamp.now().isoformat(),
            "item_name": item_name
            or next(
                (n for n, i in item_id_source.items() if i == int(item_id)),
                str(item_id),
            ),
        }


def export_positions_to_holdings_csv(positions: dict, path: str = HOLDINGS_CSV) -> None:
    """Write current in-position entries to CSV so the user can view/edit holdings easily."""
    try:
        rows = []
        for key, pos in positions.items():
            if pos.get("in_position"):
                rows.append(
                    {
                        "item_id": int(key) if key.isdigit() else key,
                        "item_name": pos.get("item_name", "Unknown"),
                        "entry_price": pos.get("entry_price"),
                        "entry_time": pos.get("entry_time"),
                    }
                )
        if rows:
            pd.DataFrame(rows).to_csv(path, index=False)
        else:
            # If no positions, still create an empty template with headers
            pd.DataFrame(columns=["item_name", "entry_price"]).to_csv(path, index=False)
    except Exception as e:
        print(f"Failed to write {path}: {e}")


def read_item_ids(file_path) -> dict | None:
    try:
        with open(file_path, "r") as file:
            data = json.load(file)
            if data:
                return data.get("item_ids", {})
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except json.JSONDecodeError as json_err:
        print(f"Error decoding JSON: {json_err}")
    return None


def make_api_request(url, params=None, headers=None):
    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        print(f"Request error: {e}")
    return None


def get_hourly_data(item_id: int) -> List[HourlyItemData]:
    API_URL = f"https://api.datawars2.ie/gw2/v2/history/hourly/json?itemID={item_id}"
    response = make_api_request(API_URL)
    if response:
        adapter = TypeAdapter(List[HourlyItemData])
        return adapter.validate_json(response)
    else:
        print("Failed to fetch data from API.")
    return []


def compute_current_signal(
    df: pd.DataFrame,
    short_ema_period: int = 6,
    long_sma_period: int = 21,
) -> pd.DataFrame:
    """
    Compute EMA/SMA crossover signal for the latest timestamp in the DataFrame.
    Buy signal: short EMA crosses above long SMA (bullish crossover).
    Sell signal: short EMA crosses below long SMA (bearish crossover).
    Returns a single-row DataFrame with item info and signal.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Calculate short EMA and long SMA for buy_price_avg and sell_price_min
    df["ema_buy_short"] = (
        df["buy_price_avg"].ewm(span=short_ema_period, adjust=False).mean()
    )
    df["sma_buy_long"] = df["buy_price_avg"].rolling(window=long_sma_period).mean()
    df["ema_sell_short"] = (
        df["sell_price_min"].ewm(span=short_ema_period, adjust=False).mean()
    )
    df["sma_sell_long"] = df["sell_price_min"].rolling(window=long_sma_period).mean()

    # For crossover, need current and previous value
    df_sorted = df.sort_values("date")
    # Use -2 as previous, -1 as current (if enough data)
    if len(df_sorted) < long_sma_period + 1:
        # Not enough data for crossover
        latest_row = df_sorted.iloc[-1].copy()
        signal = "Hold"
        # Fill output columns for compatibility
        latest_row["sma_buy"] = np.nan
        latest_row["ema_buy"] = np.nan
        latest_row["sma_sell"] = np.nan
        latest_row["ema_sell"] = np.nan
    else:
        prev = df_sorted.iloc[-2]
        curr = df_sorted.iloc[-1]
        # Buy: short EMA crosses above long SMA (bullish crossover)
        buy_crossover = (
            prev["ema_buy_short"] <= prev["sma_buy_long"]
            and curr["ema_buy_short"] > curr["sma_buy_long"]
        )
        # Sell: short EMA crosses below long SMA (bearish crossover)
        sell_crossover = (
            prev["ema_sell_short"] >= prev["sma_sell_long"]
            and curr["ema_sell_short"] < curr["sma_sell_long"]
        )
        if buy_crossover and sell_crossover:
            signal = "Buy and List Sell"
        elif buy_crossover:
            signal = "Buy Only"
        elif sell_crossover:
            signal = "Sell Only"
        else:
            signal = "Hold"
        # Prepare columns for output
        latest_row = curr.copy()
        latest_row["sma_buy"] = curr["sma_buy_long"]
        latest_row["ema_buy"] = curr["ema_buy_short"]
        latest_row["sma_sell"] = curr["sma_sell_long"]
        latest_row["ema_sell"] = curr["ema_sell_short"]

    latest_row["signal"] = signal
    latest_row["buy_price_avg"] = latest_row.get(
        "buy_price_avg", np.nan
    )  # include in output

    # Include item metadata if available
    if "item_name" in df.attrs:
        latest_row["item_name"] = df.attrs["item_name"]
    else:
        latest_row["item_name"] = "Unknown"

    if "itemID" in df.columns:
        latest_row["item_id"] = df["itemID"].iloc[0]
    else:
        latest_row["item_id"] = None

    # Select relevant columns and order them
    output_cols = [
        "item_name",
        "item_id",
        "date",
        "buy_price_avg",
        "sell_price_min",
        "sma_buy",
        "ema_buy",
        "sma_sell",
        "ema_sell",
        "signal",
    ]

    return pd.DataFrame([latest_row[output_cols]])


# --- Band Trading Strategy ---
def compute_band_strategy(
    df: pd.DataFrame,
    band_window: int = 48,
    band_threshold: float = 0.04,
    existing_entry: float | None = None,
    in_position: bool = False,
) -> pd.DataFrame:
    """
    Implements a band trading strategy:
    - Compute mean and standard deviation bands on sell_price_min
    - Signal 'Buy and List Sell' if price drops below lower band
    - Signal 'Sell Only' if price goes above upper band
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    # Ensure buy_price_max exists; fall back to buy_price_avg if needed
    if "buy_price_max" not in df.columns:
        if "buy_price_avg" in df.columns:
            df["buy_price_max"] = df["buy_price_avg"]
        else:
            df["buy_price_max"] = np.nan

    # Rolling stats with min_periods and safe defaults to avoid NaNs on short history
    df["mean_price"] = (
        df["sell_price_min"].rolling(window=band_window, min_periods=1).mean()
    )
    df["std_price"] = (
        df["sell_price_min"].rolling(window=band_window, min_periods=2).std()
    )
    df["std_price"] = df["std_price"].fillna(0.0)

    df["upper_band"] = df["mean_price"] + df["std_price"]
    df["deep_lower_band"] = df["mean_price"] - (1.75 * df["std_price"])

    latest = df.iloc[[-1]].copy()
    signal = "Hold"

    if latest["sell_price_min"].iloc[0] < latest["deep_lower_band"].iloc[0]:
        signal = "Buy and Hold"
    elif latest["sell_price_min"].iloc[0] > latest["upper_band"].iloc[0]:
        signal = "Sell Only"

    # --- Personalized signal logic with entry/stop-loss ---
    USE_ENTRY_FOR_SIGNALS = True
    if USE_ENTRY_FOR_SIGNALS:
        entry = existing_entry if in_position else np.nan
        tgt = float(latest["upper_band"].iloc[0])
        be = float(entry * 1.15) if pd.notna(entry) and entry != 0 else np.nan
        curr_price = float(latest["sell_price_min"].iloc[0])
        deep_lower = (
            float(latest["deep_lower_band"].iloc[0])
            if "deep_lower_band" in latest.columns
            else np.nan
        )

        if in_position and pd.notna(entry):
            # IN POSITION: allow Sell on take-profit or stop-loss
            threshold = max(tgt, be) if pd.notna(be) else tgt
            stop_loss_price = float(entry) * STOP_LOSS_MULT
            if curr_price >= threshold or curr_price <= stop_loss_price:
                latest.loc[:, "signal"] = "Sell Only"
            else:
                latest.loc[:, "signal"] = "Hold"
        else:
            # FLAT: never issue Sell; only Buy on deep value
            if not np.isnan(deep_lower) and curr_price < deep_lower:
                latest.loc[:, "signal"] = "Buy and Hold"
            else:
                latest.loc[:, "signal"] = "Hold"
    else:
        latest.loc[:, "signal"] = signal

    # Set item_name and item_id
    latest.loc[:, "item_name"] = df.attrs.get("item_name", "Unknown")
    latest.loc[:, "item_id"] = df["itemID"].iloc[0] if "itemID" in df.columns else None

    # Convert date to NY timezone naive (assume UTC if naive)
    latest.loc[:, "date"] = (
        pd.to_datetime(latest["date"], utc=True)
        .dt.tz_convert("America/New_York")
        .dt.tz_localize(None)
    )

    # Set entry/exit/break-even/roi columns
    # Determine entry price: if already in a position, KEEP the stored entry.
    # If flat and we just got a buy signal, set a fresh entry from current book.
    # Otherwise, leave entry as NaN so ROI fields stay blank.
    if in_position and existing_entry is not None:
        latest.loc[:, "entry_price"] = float(existing_entry)
    elif latest["signal"].iloc[0] == "Buy and Hold":
        latest.loc[:, "entry_price"] = latest["buy_price_max"]
    else:
        latest.loc[:, "entry_price"] = np.nan

    latest.loc[:, "target_exit_price"] = latest["upper_band"]

    # Only compute break-even and ROI when we actually have an entry price
    if pd.notna(latest["entry_price"].iloc[0]) and latest["entry_price"].iloc[0] != 0:
        latest.loc[:, "break_even_price"] = latest["entry_price"] * 1.15
        expected_roi = (
            (latest["target_exit_price"] / latest["entry_price"]) - 1 - 0.15
        ) * 100
        latest.loc[:, "expected_roi_pct"] = expected_roi.round(1)
    else:
        latest.loc[:, "break_even_price"] = np.nan
        latest.loc[:, "expected_roi_pct"] = np.nan

    output_cols = [
        "item_name",
        "item_id",
        "date",
        "sell_price_min",
        "entry_price",
        "target_exit_price",
        "break_even_price",
        "expected_roi_pct",
        "signal",
    ]

    return latest[output_cols]


def scrape_item_links_from_bltc(url: str, pages: int = 1) -> list:
    headers = {
        "User-Agent": "Mozilla/5.0",
    }
    hrefs = []

    for page in range(1, pages + 1):
        paged_url = f"{url}&page={page}"
        response = requests.get(paged_url, headers=headers)
        if response.status_code != 200:
            print(f"Failed to fetch page {page}: {response.status_code}")
            continue

        soup = BeautifulSoup(response.text, "html.parser")
        tds = soup.find_all("td", class_="td-name")
        if tds:
            for td in tds:
                href = td.find("a")["href"]  # type: ignore
                hrefs.append(href)

    return hrefs


# /en/item/42004-Catapult-Blueprint
def process_hrefs(hrefs: list) -> tuple[list[str], list[int]]:
    item_names = []
    item_ids = []
    for href in hrefs:
        # Extract the item name and ID from the href
        parts = href.split("/")[-1].split("-")
        item_id = parts[0]
        item_name = " ".join(parts[1:])
        item_names.append(item_name)
        item_ids.append(int(item_id))
    return item_names, item_ids


if __name__ == "__main__":
    os.system("clear")

    positions = load_positions()

    url = "https://www.gw2bltc.com/en/tp/search?buy-min=80&profit-pct-max=30&supply-min=10000&demand-min=10000&sold-day-min=10000&bought-day-min=10000&ipg=50&sort=profit-pct"

    items = []
    item_ids = []

    item_id_source = read_item_ids("processed_item_ids.json")
    if not item_id_source:
        print("No item IDs found. Exiting.")
        exit(1)

    # --- Apply manual CSV holdings (authoritative seed) ---
    csv_rows = load_holdings_csv()
    if csv_rows:
        if ENRICH_CSV_ON_LOAD:
            enrich_holdings_csv_in_place(HOLDINGS_CSV, item_id_source)
            # Reload rows after enrichment so item_id/entry_time are available
            csv_rows = load_holdings_csv()
        # OVERWRITE positions with CSV-derived data (CSV is source of truth)
        positions = positions_from_csv(csv_rows, item_id_source)
        save_positions(positions)
        print(
            f"Loaded {len(positions)} holdings from {HOLDINGS_CSV} (CSV is source of truth)\n"
        )
    else:
        print(
            f"No {HOLDINGS_CSV} found or it was empty. Create one with columns: item_name,entry_price (or item_id,entry_price).\n"
        )
    # positions now reflects ONLY the CSV holdings

    # --- Show holdings first ---
    holdings_df = holdings_dataframe_from_positions(positions)
    print("Current holdings:")
    if holdings_df.empty:
        print("(none)\n")
    else:
        # Pretty console print
        print(holdings_df.to_string(index=False))
        print()
    print("(Your holdings will be processed first.)\n")

    # custom_items = [
    #     "Pile of Shimmering Dust",
    #     "Fine Fish Fillet",
    #     "Black Peppercorn",
    #     "Mursaat Obsidian Chunk",
    # ]

    custom_items = []

    custom_item_ids = [
        item_id_source.get(item, None)
        for item in custom_items
        if item in item_id_source
    ]

    items.extend(custom_items)
    if custom_item_ids:
        item_ids.extend([i for i in custom_item_ids if i is not None])

    scraped_items, scraped_item_ids = process_hrefs(scrape_item_links_from_bltc(url))
    items.extend(scraped_items)
    item_ids.extend(scraped_item_ids)

    # --- Reorder and include held items at the top ---
    # Build a lookup from ids -> names using current lists
    scraped_lookup = {iid: nm for nm, iid in zip(items, item_ids)}

    # Collect held ids from positions (only those marked in_position)
    held_ids = []
    for k, v in positions.items():
        try:
            if v.get("in_position"):
                held_ids.append(int(k))
        except Exception:
            continue

    # Ensure all held ids are present in the processing lists; if missing, append them with their stored name
    for hid in held_ids:
        if hid not in scraped_lookup:
            held_name = positions[str(hid)].get("item_name", str(hid))
            items.insert(0, held_name)
            item_ids.insert(0, hid)
            scraped_lookup[hid] = held_name

    # Now stably reorder so that held ids come first, maintaining relative order within each group
    combined = list(zip(item_ids, items))
    held_set = set(held_ids)
    held_block = [pair for pair in combined if pair[0] in held_set]
    rest_block = [pair for pair in combined if pair[0] not in held_set]
    ordered = held_block + rest_block
    # De-duplicate by item_id while preserving order
    seen = set()
    unique_ordered = []
    for iid, nm in ordered:
        if iid in seen:
            continue
        seen.add(iid)
        unique_ordered.append((iid, nm))
    if unique_ordered:
        item_ids, items = [list(t) for t in zip(*unique_ordered)]
    else:
        item_ids, items = [], []

    print(f"Processing {len(item_ids)} items...\n")

    all_signals = []

    for i in range(len(item_ids)):
        item_id = item_ids[i]
        item = items[i]

        # Load existing position state for this item, if any (do this FIRST)
        pos_key = str(item_id)
        item_pos = positions.get(pos_key, {})
        in_position = bool(item_pos.get("in_position", False))
        existing_entry = item_pos.get("entry_price")

        # Try to fetch data
        hourly_data = get_hourly_data(item_id)
        if not hourly_data:
            print(
                f"No data for item ID {item_id}. Using placeholder so it still appears in output."
            )
            # Add placeholder row so holdings still appear in final output
            placeholder = pd.DataFrame(
                [
                    {
                        "item_name": item,
                        "item_id": item_id,
                        "date": pd.Timestamp.now(),
                        "sell_price_min": np.nan,
                        "entry_price": (
                            float(existing_entry)
                            if in_position and existing_entry is not None
                            else np.nan
                        ),
                        "target_exit_price": np.nan,
                        "break_even_price": (
                            (float(existing_entry) * 1.15)
                            if in_position and existing_entry is not None
                            else np.nan
                        ),
                        "expected_roi_pct": np.nan,
                        "signal": "Hold",
                        "in_position": in_position,
                    }
                ]
            )
            all_signals.append(placeholder)
            continue

        df = pd.DataFrame([entry.model_dump() for entry in hourly_data])
        df["date"] = pd.to_datetime(df["date"])
        df.attrs["item_name"] = item

        # Compute current signal with awareness of stored entry state
        # signals_df = compute_current_signal(df)
        signals_df = compute_band_strategy(
            df,
            existing_entry=existing_entry,
            in_position=in_position,
        )
        # Mark whether this row corresponds to an active holding
        if not signals_df.empty:
            signals_df.loc[:, "in_position"] = in_position
        # Ensure entry_price is populated from holdings if missing
        if in_position and not signals_df.empty:
            if "entry_price" in signals_df.columns and pd.isna(
                signals_df.iloc[-1]["entry_price"]
            ):
                signals_df.loc[:, "entry_price"] = float(existing_entry)
                # Also recompute dependent fields
                signals_df.loc[:, "break_even_price"] = float(existing_entry) * 1.15

        # Optional: quick debug print for tracing CSV linkage
        # print(f"{item} (id {item_id}) in_position={in_position}, entry={existing_entry}, signal={signals_df.iloc[-1]['signal']}")

        # Update persistent position state based on today's signal (disabled unless flag set)
        if AUTO_UPDATE_FROM_SIGNALS and not signals_df.empty:
            latest_signal = signals_df.iloc[-1]["signal"]
            current_buy_price = (
                float(df.iloc[-1]["buy_price_max"])
                if "buy_price_max" in df.columns
                else None
            )
            current_time = pd.to_datetime(df.iloc[-1]["date"]).isoformat()

            if (
                latest_signal == "Buy and Hold"
                and not in_position
                and current_buy_price is not None
            ):
                positions[pos_key] = {
                    "in_position": True,
                    "entry_price": current_buy_price,
                    "entry_time": current_time,
                    "item_name": item,
                }
            elif latest_signal == "Sell Only" and in_position:
                positions[pos_key] = {
                    "in_position": False,
                    "entry_price": None,
                    "last_exit_time": current_time,
                    "item_name": item,
                }
            else:
                if pos_key not in positions:
                    positions[pos_key] = {
                        "in_position": False,
                        "entry_price": None,
                        "item_name": item,
                    }

        if not signals_df.empty:
            all_signals.append(signals_df)

    # Persist updated positions so future runs keep the same entry prices
    save_positions(positions)

    if all_signals:
        final_df = pd.concat(all_signals, ignore_index=True)

        # Sort so holdings appear first in the output
        if "in_position" in final_df.columns:
            final_df = final_df.sort_values(
                by=["in_position", "item_name"], ascending=[False, True]
            ).reset_index(drop=True)

        # Convert 'date' to NY time zone naive for Excel compatibility
        final_df["date"] = pd.to_datetime(final_df["date"])

        # Write to Excel
        output_file = "gw2_current_signals.xlsx"
        final_df.to_excel(output_file, index=False)
        print(f"Signals saved to {output_file}")
    else:
        print("No signals generated.")


# Sample class for df that has hourly data
# class HourlyItemData(BaseModel):
# buy_delisted: int
# buy_listed: int
# buy_price_avg: int
# buy_price_max: int
# buy_price_min: int
# buy_price_stdev: float
# buy_quantity_avg: int
# buy_quantity_max: int
# buy_quantity_min: int
# buy_quantity_stdev: float
# buy_sold: int
# buy_value: int
# count: int
# date: datetime
# itemID: int
# sell_delisted: int
# sell_listed: int
# sell_price_avg: int
# sell_price_max: int
# sell_price_min: int
# sell_price_stdev: float
# sell_quantity_avg: int
# sell_quantity_max: int
# sell_quantity_min: int
# sell_quantity_stdev: float
# sell_sold: int
# sell_value: int
# type: Literal["hour"] = "hour"
