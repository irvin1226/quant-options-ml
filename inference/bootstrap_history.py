import os
import time
import numpy as np
import pandas as pd
from datetime import date, timedelta
from zoneinfo import ZoneInfo
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import httpx
import io

HISTORY_PATH  = "history.csv"
LOOKBACK_DAYS = 400
ET            = ZoneInfo('America/New_York')
BASE_URL      = "http://localhost:25503/v3"

_client = httpx.Client(timeout=60)


def _get_alpaca_client():
    apiKey    = os.environ["ALPACA_API_KEY"]
    secretKey = os.environ["ALPACA_SECRET_KEY"]
    return StockHistoricalDataClient(apiKey, secretKey)


# Fetches ATM IV for a single trading date. Returns float or None.
def _fetch_atm_iv_for_date(tradingDate):
    dateStr = pd.Timestamp(tradingDate).strftime("%Y%m%d")

    params = {
        "symbol":      "SPY",
        "expiration":  "*",
        "right":       "both",
        "start_date":  dateStr,
        "end_date":    dateStr,
        "max_dte":     35,
        "format":      "csv",
    }

    try:
        r = _client.get(f"{BASE_URL}/option/history/greeks/eod", params=params)

        if r.status_code != 200:
            return None

        responseIsEmpty = not r.text.strip()
        if responseIsEmpty:
            return None

        df = pd.read_csv(io.StringIO(r.text))

        if df.empty:
            return None

        df["right"] = df["right"].str.upper()
        df["underlying_price"] = df["underlying_price"].astype(float)
        df["strike"] = df["strike"].astype(float)
        df["moneyness"] = df["strike"] / df["underlying_price"]
        df["atm_distance"] = (df["moneyness"] - 1.0).abs()

        calls = df[df["right"] == "CALL"].copy()
        if calls.empty:
            return None

        calls = calls.sort_values("atm_distance")
        return float(calls.iloc[0]["implied_vol"])

    except Exception as e:
        print(f"IV fetch failed for {tradingDate}: {e}")
        return None


def _fetch_spy_daily_bars(alpacaClient, startDate, endDate):
    print(f"Fetching SPY 1-minute bars from {startDate} to {endDate}...")

    startDt = pd.Timestamp(startDate, tz=ET).replace(hour=9, minute=30)
    endDt   = pd.Timestamp(endDate, tz=ET).replace(hour=16, minute=0)

    request = StockBarsRequest(
        symbol_or_symbols="SPY",
        timeframe=TimeFrame.Minute,
        start=startDt,
        end=endDt,
    )

    barsResponse = alpacaClient.get_stock_bars(request)
    df = barsResponse.df

    if df.empty:
        print("No SPY bar data returned.")
        return {}

    df = df.reset_index()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date_only"] = df["timestamp"].dt.normalize().dt.date

    dailyBars = {}
    for tradingDate, group in df.groupby("date_only"):
        dailyBars[tradingDate] = group.sort_values("timestamp").reset_index(drop=True)

    print(f"Retrieved bars for {len(dailyBars)} trading days.")
    return dailyBars


def _compute_daily_stats(minuteBars):
    if minuteBars.empty:
        return 0.0, 0.0, 1e-8

    bars = minuteBars.sort_values("timestamp").copy()
    bars["return_5m"] = bars["close"].pct_change(5).fillna(0.0)

    dailyRv        = float((bars["return_5m"] ** 2).sum())
    realizedAbsSum = float(bars["return_5m"].abs().sum())
    firstPrice     = float(bars["close"].iloc[0])
    lastPrice      = float(bars["close"].iloc[-1])
    netMove        = abs(lastPrice - firstPrice) / max(firstPrice, 1e-8)

    return dailyRv, realizedAbsSum, netMove


def main():
    print("Starting history bootstrap...")

    endDate   = date.today() - timedelta(days=1)
    startDate = date.today() - timedelta(days=LOOKBACK_DAYS)

    print(f"Date range: {startDate} to {endDate}")

    # Fetch SPY minute bars from Alpaca
    alpacaClient = _get_alpaca_client()
    dailyBars    = _fetch_spy_daily_bars(alpacaClient, startDate, endDate)

    tradingDates = sorted(dailyBars.keys())

    if len(tradingDates) == 0:
        print("No trading days found. Aborting.")
        return

    print(f"Fetching ATM IV day-by-day for {len(tradingDates)} trading days...")
    print("(This will take a few minutes - one request per day)")

    rows = []
    for i, tradingDate in enumerate(tradingDates):
        atmIv      = _fetch_atm_iv_for_date(tradingDate)
        minuteBars = dailyBars.get(tradingDate, pd.DataFrame())

        dailyRv, realizedAbsSum, netMove = _compute_daily_stats(minuteBars)

        rows.append({
            "date":             pd.Timestamp(tradingDate),
            "atm_iv":           atmIv if atmIv is not None else np.nan,
            "daily_rv":         dailyRv,
            "realized_abs_sum": realizedAbsSum,
            "net_move":         netMove,
        })

        ivStr = f"{atmIv:.4f}" if atmIv is not None else "N/A"
        print(f"[{i+1}/{len(tradingDates)}] {tradingDate} | atm_iv={ivStr} | daily_rv={dailyRv:.6f}")

        # Be polite to the ThetaData Terminal (AKA don't call it all the damn time)
        time.sleep(0.2)

    history = pd.DataFrame(rows)
    history = history.sort_values("date").reset_index(drop=True)
    history.to_csv(HISTORY_PATH, index=False)

    ivCoverage  = history["atm_iv"].notna().sum()
    barCoverage = (history["daily_rv"] > 0).sum()

    print(f"\nhistory.csv written with {len(history)} rows.")
    print(f"IV coverage:  {ivCoverage} / {len(history)} days")
    print(f"Bar coverage: {barCoverage} / {len(history)} days")
    print("Bootstrap complete.")


if __name__ == "__main__":
    main()