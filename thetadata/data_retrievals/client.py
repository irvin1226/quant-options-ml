import os
import httpx

BASE_URL = 'http://localhost:25503/v3'

_client = httpx.Client(timeout=120)
ERROR_DIR = os.path.join("theta_data_responses", "errors")
ERROR_LOG = os.path.join(ERROR_DIR, "errors.log")

def get_eod(date):
    url = f"{BASE_URL}/option/history/eod"
    params = {
        "start_date": date,
        "end_date": date,
        "symbol": "SPY",
        "expiration": "*",
        "strike": "*",
        "right": "both",
        "max_dte": 60,
        "format": "csv"
    }

    response = _client.get(url, params=params)
    response.raise_for_status()
    return response.text

def get_ohlc(date, expiration):
    url = f"{BASE_URL}/option/history/ohlc"
    params = {
        "date": date,
        "symbol": "SPY",
        "expiration": expiration,
        "right": "both",
        "interval": "1m",
        "strike_range": 70,
        "format": "csv"
    }
    try:
        response = _client.get(url, params=params)
        response.raise_for_status()
        return response.text

    except httpx.HTTPStatusError as error:
        status = error.response.status_code if error.response is not None else None

        # 472: No data found for the request
        if status == 472:
            os.makedirs(ERROR_DIR, exist_ok=True)
            with open(ERROR_LOG, "a", encoding="utf-8") as file:
                file.write(
                    f"url={error.request.url} params={params}\n"
                )
            return ""

def get_greeks_all(date, expiration):
    url = f"{BASE_URL}/option/history/greeks/all"
    params = {
        "date": date,
        "symbol": "SPY",
        "expiration": expiration,
        "right": "both",
        "interval": "1m",
        "strike_range": 70,
        "format": "csv"
    }
    try:
        response = _client.get(url, params=params)
        response.raise_for_status()
        return response.text

    except httpx.HTTPStatusError as error:
        status = error.response.status_code if error.response is not None else None

        # 472: No data found for the request
        if status == 472:
            os.makedirs(ERROR_DIR, exist_ok=True)
            with open(ERROR_LOG, "a", encoding="utf-8") as file:
                file.write(
                    f"url={error.request.url} params={params}\n"
                )
            return ""

def get_iv(date, expiration):
    url = f"{BASE_URL}/option/history/greeks/implied_volatility"
    params = {
        "date": date,
        "symbol": "SPY",
        "expiration": expiration,
        "right": "both",
        "interval": "1m",
        "strike_range": 70,
        "format": "csv"
    }
    try:
        response = _client.get(url, params=params)
        response.raise_for_status()
        return response.text

    except httpx.HTTPStatusError as error:
        status = error.response.status_code if error.response is not None else None

        # 472: No data found for the request
        if status == 472:
            os.makedirs(ERROR_DIR, exist_ok=True)
            with open(ERROR_LOG, "a", encoding="utf-8") as file:
                file.write(
                    f"url={error.request.url} params={params}\n"
                )
            return ""

def get_open_interest(date):
    url = f"{BASE_URL}/option/history/open_interest"
    params = {
        "date": date,
        "symbol": "SPY",
        "expiration": "*",
        "format": "csv"
    }
    try:
        response = _client.get(url, params=params)
        response.raise_for_status()
        return response.text

    except httpx.HTTPStatusError as error:
        status = error.response.status_code if error.response is not None else None

        # 472: No data found for the request
        if status == 472:
            os.makedirs(ERROR_DIR, exist_ok=True)
            with open(ERROR_LOG, "a", encoding="utf-8") as file:
                file.write(
                    f"url={error.request.url} params={params}\n"
                )
            return ""

def get_quote(date, expiration):
    url = f"{BASE_URL}/option/history/quote"
    params = {
        "date": date,
        "symbol": "SPY",
        "expiration": expiration,
        "right": "both",
        "interval": "1m",
        "strike_range": 70,
        "format": "csv"
    }
    try:
        response = _client.get(url, params=params)
        response.raise_for_status()
        return response.text

    except httpx.HTTPStatusError as error:
        status = error.response.status_code if error.response is not None else None

        # 472: No data found for the request
        if status == 472:
            os.makedirs(ERROR_DIR, exist_ok=True)
            with open(ERROR_LOG, "a", encoding="utf-8") as file:
                file.write(
                    f"url={error.request.url} params={params}\n"
                )
            return ""

def get_trade(date, expiration):
    url = f"{BASE_URL}/option/history/trade"
    params = {
        "date": date,
        "symbol": "SPY",
        "expiration": expiration,
        "right": "both",
        "interval": "1m",
        "strike_range": 70,
        "format": "csv"
    }
    try:
        response = _client.get(url, params=params)
        response.raise_for_status()
        return response.text

    except httpx.HTTPStatusError as error:
        status = error.response.status_code if error.response is not None else None

        # 472: No data found for the request
        if status == 472:
            os.makedirs(ERROR_DIR, exist_ok=True)
            with open(ERROR_LOG, "a", encoding="utf-8") as file:
                file.write(
                    f"url={error.request.url} params={params}\n"
                )
            return ""

def get_greeks_eod(date):
    url = f"{BASE_URL}/option/history/greeks/eod"
    params = {
        "symbol": "SPY",
        "expiration": "*",
        "right": "both",
        "start_date": date,
        "end_date": date,
        "max_dte": 60,
        "format": "csv"
    }
    try:
        response = _client.get(url, params=params)
        response.raise_for_status()
        return response.text

    except httpx.HTTPStatusError as error:
        status = error.response.status_code if error.response is not None else None

        # 472: No data found for the request
        if status == 472:
            os.makedirs(ERROR_DIR, exist_ok=True)
            with open(ERROR_LOG, "a", encoding="utf-8") as file:
                file.write(
                    f"url={error.request.url} params={params}\n"
                )
            return ""

def close():
    _client.close()
