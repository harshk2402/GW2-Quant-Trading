#!/usr/bin/env python3
"""
Minimal GW2Trader fetcher:
- Calls the exact URL shared.
- Sends browser-like headers.
- Handles gzip content.
- Prints response['pageProps']['data'] as pretty JSON.
"""
from __future__ import annotations
import time
import pandas as pd
import math
import numpy as np
from gw2trader import GW2Trader
from gw2 import GW2


def calc_daily_avg_volumes(hourly_data: list) -> list:
    if not hourly_data:
        return []

    rows = []
    for rec in hourly_data:
        attrs = rec.get("attributes") or {}
        ts_ms = attrs.get("timestamp")
        v24 = attrs.get("volume_24h")
        if ts_ms is None or v24 is None:
            continue
        rows.append((pd.to_datetime(ts_ms, unit="ms", utc=True), float(v24)))

    if not rows:
        return []

    df = pd.DataFrame(rows, columns=["ts", "v24"]).sort_values("ts")

    # Hourly increment: positive diffs are new trades since last snapshot
    df["delta"] = df["v24"].diff().clip(lower=0)

    # Sum positive deltas within each UTC day → true daily flow
    df["day"] = df["ts"].dt.floor("D")
    daily_true = df.groupby("day", as_index=True)["delta"].sum()

    # Return as plain list, oldest -> newest
    return daily_true.tolist()


def poisson_exit_probability(T: float, rate: float) -> float:
    """
    Calculate the probabilities of H units exiting within time T1, T2, ...
    """
    if rate <= 0:
        return 0.0
    return 1.0 - math.exp(-rate * T)


def volume_analysis(
    daily_volume_data: list, stats: dict, lookback_days=14, conservative=True
) -> dict:
    hb = stats.get("highestBuyPrice", 0.0)
    ls = stats.get("lowestSellPrice", 0.0)

    if not daily_volume_data or hb <= 0 or ls <= 0:
        return {"status": "NO_DATA", "T_days": math.inf, "prob_exit_3d": 0.0}

    window = np.array(daily_volume_data[:lookback_days], dtype=float)

    ref = max(ls if conservative else (hb + ls) / 2.0, 1.0)

    units_per_day = window / ref

    # Dead-like check
    zero_streak = 0
    max_zero_streak = 0

    for daily_units in units_per_day[-7:]:
        if daily_units <= 0:
            zero_streak += 1
            max_zero_streak = max(max_zero_streak, zero_streak)
        else:
            zero_streak = 0

    if max_zero_streak >= 3:
        return {
            "status": "DEADLIKE",
            "T_days": math.inf,
            "avg_units_day_used": 0.0,
            "bursty": False,
            "zero_streak7": int(max_zero_streak),
            "prob_exit_3d": 0.0,
            "reason": f"{int(max_zero_streak)} consecutive zero-unit days in last 7",
        }

    # Robust daily rate estimates
    median_rate = float(np.median(units_per_day))
    percentile25_rate = (
        float(np.percentile(units_per_day, 25)) if units_per_day.size else 0.0
    )
    positive_days = units_per_day[units_per_day > 0]
    if positive_days.size:
        reciprocal = 1.0 / positive_days
        denom = float(reciprocal.sum())
        harmonic_mean_rate = float(positive_days.size) / denom if denom > 0 else 0.0
    else:
        harmonic_mean_rate = 0.0

    total_units = float(units_per_day.sum())

    if total_units <= 0:
        return {
            "status": "NO_FLOW",
            "T_days": math.inf,
            "avg_units_day_used": 0.0,
            "bursty": False,
            "zero_streak7": int(max_zero_streak),
            "prob_exit_3d": 0.0,
            "reason": "No executed trades detected in the lookback window (complete absence of observed sell-through)",
        }

    # Burstiness check
    burst_from_single_day = (np.sort(units_per_day)[-1] / total_units) > 0.70
    burst_from_top3_days = (
        float(np.sort(units_per_day)[-3:].sum()) / total_units
        if units_per_day.size >= 3
        else 1.0
    ) > 0.80
    bursty = burst_from_single_day or burst_from_top3_days

    # Choose rate: median if stable, else more conservative estimate
    effective_rate = (
        median_rate
        if not bursty
        else min(median_rate, percentile25_rate, harmonic_mean_rate)
    )
    if effective_rate <= 0:
        return {
            "status": "NO_FLOW",
            "T_days": math.inf,
            "avg_units_day_used": 0.0,
            "bursty": bursty,
            "zero_streak7": int(max_zero_streak),
            "prob_exit_3d": 0.0,
            "reason": "Trading observed, but effective daily sell-through rate collapsed to ~0 due to sparse or volatile recent activity",
        }

    # Days to exit calculation
    estimated_days_to_exit = int(math.ceil(1 / effective_rate))

    if estimated_days_to_exit <= 3:
        status = "OK_LIQUID"
    elif estimated_days_to_exit <= 7:
        status = "OK_SLOW"
    elif estimated_days_to_exit <= 14:
        status = "RISKY"
    else:
        status = "HOLD_RISK"

    prob_exit_3d = poisson_exit_probability(3, effective_rate)

    return {
        "status": status,
        "T_days": estimated_days_to_exit,
        "avg_units_day_used": round(effective_rate, 3),
        "bursty": bursty,
        "zero_streak7": int(max_zero_streak),
        "prob_exit_3d": prob_exit_3d,
        "reason": None,
    }


if __name__ == "__main__":
    min_buy_gold = 0
    max_buy_gold = 200
    min_sell_gold = 0
    max_sell_gold = 500
    min_vol_24h_gold = 1000
    max_vol_24h_gold = 0

    trader = GW2Trader()
    item_data = trader.get_flippable_items(
        min_buy_gold * 10000,
        max_buy_gold * 10000,
        min_sell_gold * 10000,
        max_sell_gold * 10000,
        min_vol_24h_gold * 10000,
        max_vol_24h_gold * 10000,
    )

    item_list = []
    item_list = item_data["items"]["data"]

    item_list = item_list[:10]  # Limit to first 20 items

    rows = []

    for item in item_list:
        time.sleep(0.3)  # Be polite to the server
        item_id = item["id"]
        item_name = item["name"]

        stats = item["stats"]
        hb = stats.get("highestBuyPrice", 0.0)
        ls = stats.get("lowestSellPrice", 0.0)

        expected_profit_gold = ((0.85 * ls) - hb) / 10000
        roi_pct = (((0.85 * ls) - hb) / hb * 100) if hb > 0 else None

        print(
            f"Item name: {item_name}  Item ID: {item_id}  Expected Profit: {expected_profit_gold:0.4f}g"
        )

        payload = trader.get_hourly_data(item_id)

        hourly_data = []

        try:
            hourly_data = payload["data"]["attributes"]["item_prices"]["data"]
        except (KeyError, TypeError):
            rows.append(
                {
                    "name": item_name,
                    "id": item_id,
                    "highestBuyPrice": hb,
                    "lowestSellPrice": ls,
                    "expected_profit_gold": expected_profit_gold,
                    "roi_pct": roi_pct,
                    "status": "NO_DATA",
                    "T_days": None,
                    "avg_units_day": None,
                    "bursty": None,
                    "zero_streak7": None,
                    "prob_exit_3d": None,
                    "reason": None,
                }
            )
            continue

        daily_vals = calc_daily_avg_volumes(
            hourly_data
        )  # list of value_24h per day (copper)
        res = volume_analysis(
            daily_vals, stats=stats, lookback_days=14, conservative=True
        )
        rows.append(
            {
                "name": item_name,
                "id": item_id,
                "highestBuyPrice": hb,
                "lowestSellPrice": ls,
                "expected_profit_gold": expected_profit_gold,
                "roi_pct": roi_pct,
                "status": res.get("status"),
                "T_days": res.get("T_days"),
                "avg_units_day": res.get("avg_units_day_used"),
                "bursty": res.get("bursty"),
                "zero_streak7": res.get("zero_streak7"),
                "prob_exit_3d": res.get("prob_exit_3d"),
                "reason": res.get("reason"),
            }
        )

    df = pd.DataFrame(
        rows,
        columns=[
            "name",
            "id",
            "highestBuyPrice",
            "lowestSellPrice",
            "expected_profit_gold",
            "roi_pct",
            "status",
            "T_days",
            "avg_units_day",
            "bursty",
            "zero_streak7",
            "prob_exit_3d",
            "reason",
        ],
    )

    df.to_excel("gw2_flip_volume_analysis.xlsx", index=False)

    print("Data written to excel file.")
