import io
import os
import httpx
import pandas as pd
from datetime import date, datetime
from zoneinfo import ZoneInfo
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

BASE_URL = "http://localhost:25503/v3"
MIN_DTE = 21
MAX_DTE = 45
ET = ZoneInfo('America/New_York')

_client = httpx.Client(timeout=120)


def _get_alpaca_data_client():
    apiKey    = os.environ["ALPACA_API_KEY"]
    secretKey = os.environ["ALPACA_SECRET_KEY"]
    return StockHistoricalDataClient(apiKey, secretKey)


def _fetch_csv(endpointPath, queryParams):
    queryParams["format"] = "csv"
    response = _client.get(f"{BASE_URL}{endpointPath}", params=queryParams)
    response.raise_for_status()

    responseIsEmpty = not response.text.strip()
    if responseIsEmpty:
        return pd.DataFrame()

    return pd.read_csv(io.StringIO(response.text))


def _fetch_csv_no_data_ok(endpointPath, queryParams):
    queryParams["format"] = "csv"

    try:
        response = _client.get(f"{BASE_URL}{endpointPath}", params=queryParams)
        response.raise_for_status()

        responseIsEmpty = not response.text.strip()
        if responseIsEmpty:
            return pd.DataFrame()

        return pd.read_csv(io.StringIO(response.text))

    except httpx.HTTPStatusError as e:
        statusCode = e.response.status_code if e.response is not None else None
        noDataFound = statusCode == 472 or statusCode == 403 or statusCode == 500

        if noDataFound:
            return pd.DataFrame()

        raise


def _normalize_contract_identifiers(df):
    if df.empty:
        return df

    df["right"] = df["right"].str.upper()
    df["expiration"] = pd.to_datetime(df["expiration"]).dt.date
    df["strike"] = df["strike"].astype(float)

    return df


def _build_chain_query_params():
    return {
        "symbol": "SPY",
        "expiration": "*",
        "strike_range": 70,
        "max_dte": MAX_DTE,
    }


def _fetch_greeks_snapshot():
    queryParams = _build_chain_query_params()
    df = _fetch_csv_no_data_ok("/option/snapshot/greeks/all", queryParams)

    if df.empty:
        return df

    df = _normalize_contract_identifiers(df)

    columnsToKeep = [
        "expiration", "strike", "right",
        "timestamp", "bid", "ask",
        "delta", "gamma", "theta", "vega", "rho",
        "implied_vol", "underlying_price",
    ]
    presentColumns = [col for col in columnsToKeep if col in df.columns]
    df = df[presentColumns].copy()

    # mirrors training ETL bid/ask quality filter
    hasBid = df["bid"] > 0
    hasValidSpread = df["ask"] > df["bid"]
    df = df[hasBid & hasValidSpread]

    return df


def _fetch_ohlc_snapshot():
    queryParams = _build_chain_query_params()
    df = _fetch_csv_no_data_ok("/option/snapshot/ohlc", queryParams)

    if df.empty:
        return df

    df = _normalize_contract_identifiers(df)

    return df[["expiration", "strike", "right", "volume", "count"]].copy()


def _fetch_open_interest_snapshot():
    queryParams = _build_chain_query_params()
    df = _fetch_csv_no_data_ok("/option/snapshot/open_interest", queryParams)

    if df.empty:
        return df

    df = _normalize_contract_identifiers(df)

    return df[["expiration", "strike", "right", "open_interest"]].copy()


def _merge_chain_data(greeksData, ohlcData, openInterestData):
    joinColumns = ["expiration", "strike", "right"]

    ohlcIsMissing = ohlcData.empty
    if ohlcIsMissing:
        greeksData["volume"] = 0
        greeksData["count"] = 0
        chainData = greeksData
    else:
        chainData = greeksData.merge(ohlcData, on=joinColumns, how="inner")

    openInterestIsMissing = openInterestData.empty
    if openInterestIsMissing:
        chainData["open_interest"] = float("nan")
    else:
        chainData = chainData.merge(openInterestData, on=joinColumns, how="left")

    return chainData


def _add_derived_columns(chainData):
    today = date.today()

    chainData["mid"] = (chainData["bid"] + chainData["ask"]) / 2
    chainData["spread"] = chainData["ask"] - chainData["bid"]
    chainData["spread_pct"] = chainData["spread"] / chainData["mid"]
    chainData["date"] = today
    chainData["expiration"] = pd.to_datetime(chainData["expiration"])
    chainData["dte"] = (chainData["expiration"] - pd.Timestamp(today)).dt.days
    chainData["moneyness"] = chainData["strike"] / chainData["underlying_price"]
    chainData["symbol"] = "SPY"

    return chainData


def _filter_by_dte(chainData):
    # min_dte is not a documented snapshot param - must filter post-fetch
    meetsMinDte = chainData["dte"] >= MIN_DTE
    meetsMaxDte = chainData["dte"] <= MAX_DTE

    return chainData[meetsMinDte & meetsMaxDte]


# Fetches the full SPY options chain snapshot with greeks, volume, and open interest merged.
def get_chain():
    greeksData = _fetch_greeks_snapshot()

    if greeksData.empty:
        return pd.DataFrame()

    ohlcData = _fetch_ohlc_snapshot()
    openInterestData = _fetch_open_interest_snapshot()

    chainData = _merge_chain_data(greeksData, ohlcData, openInterestData)
    chainData = _add_derived_columns(chainData)
    chainData = _filter_by_dte(chainData)

    return chainData.reset_index(drop=True)


# Fetches SPY 1-minute OHLC bars for the current trading day via Alpaca.
def get_spy_minute_bars(tradingDate=None):
    if tradingDate is None:
        tradingDate = date.today()

    marketOpen  = datetime(tradingDate.year, tradingDate.month, tradingDate.day, 9, 30, tzinfo=ET)
    marketClose = datetime(tradingDate.year, tradingDate.month, tradingDate.day, 16, 0, tzinfo=ET)

    try:
        alpacaClient = _get_alpaca_data_client()

        request = StockBarsRequest(
            symbol_or_symbols="SPY",
            timeframe=TimeFrame.Minute,
            start=marketOpen,
            end=marketClose,
            feed='iex'
        )

        barsResponse = alpacaClient.get_stock_bars(request)
        df = barsResponse.df

        if df.empty:
            return pd.DataFrame()

        # get_stock_bars returns a MultiIndex (symbol, timestamp) - drop symbol level
        df = df.reset_index()
        df = df.rename(columns={"timestamp": "timestamp"})
        df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)

        return df

    except Exception as e:
        print(f"[fetch] SPY minute bars unavailable: {e}")
        return pd.DataFrame()


# Extracts today's ATM implied vol from the already-fetched chain.
# Uses the call contract with moneyness closest to 1.0.
def get_atm_implied_vol(chainData):
    if chainData.empty:
        return None

    callContracts = chainData[chainData["right"] == "CALL"].copy()

    if callContracts.empty:
        return None

    callContracts["atm_distance"] = (callContracts["moneyness"] - 1.0).abs()
    closestAtmIndex = callContracts["atm_distance"].idxmin()

    return float(callContracts.loc[closestAtmIndex, "implied_vol"])


# One-time bootstrap: fetches historical daily ATM IV to seed the history file.
# Uses the greeks/eod history endpoint, matching the training ETL.
# Not called during live inference.
def get_historical_atm_iv(startDate, endDate):
    queryParams = {
        "symbol": "SPY",
        "expiration": "*",
        "right": "both",
        "start_date": pd.Timestamp(startDate).strftime("%Y%m%d"),
        "end_date": pd.Timestamp(endDate).strftime("%Y%m%d"),
        "max_dte": 35,
    }

    df = _fetch_csv_no_data_ok("/option/history/greeks/eod", queryParams)

    if df.empty:
        return pd.DataFrame(columns=["date", "atm_iv"])

    df["right"] = df["right"].str.upper()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["underlying_price"] = df["underlying_price"].astype(float)
    df["strike"] = df["strike"].astype(float)
    df["moneyness"] = df["strike"] / df["underlying_price"]
    df["atm_distance"] = (df["moneyness"] - 1.0).abs()

    callContracts = df[df["right"] == "CALL"].copy()
    callContracts = callContracts.sort_values("atm_distance")

    atmPerDay = callContracts.groupby("date").first().reset_index()
    atmHistory = atmPerDay[["date", "implied_vol"]].rename(columns={"implied_vol": "atm_iv"})

    return atmHistory


def close():
    _client.close()