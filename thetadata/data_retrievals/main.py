import pandas as pd
import client
import os
from io import StringIO

path = os.path.join("spy_yearly_opening", "SPY_2026.csv")
output_folder = "theta_data_responses"

EOD_KEY_COLS = ["symbol", "expiration", "strike", "right"]

def save_eod(date: str):
    year, month, day = date.split("-")
    day_dir = os.path.join(output_folder, year, month, day)
    os.makedirs(day_dir, exist_ok=True)

    original_response = os.path.join(day_dir, "eod_response.csv")

    #response without duplicates
    cleaned_response = os.path.join(day_dir, "eod.csv")

    if os.path.exists(original_response) and os.path.exists(cleaned_response):
        return

    # API Call
    response_text = client.get_eod(date)
    with open(original_response, "w", encoding="utf-8") as f:
        f.write(response_text)

    df = pd.read_csv(StringIO(response_text))

    if "created" in df.columns:
        df["created"] = pd.to_datetime(df["created"], errors="coerce")
        df = df.sort_values("created")

    df_dedup = df.drop_duplicates(subset=EOD_KEY_COLS, keep="last")
    df_dedup.to_csv(cleaned_response, index=False)


def save_ohlc(date: str):
    year, month, day = date.split("-")
    day_dir = os.path.join(output_folder, year, month, day)
    os.makedirs(day_dir, exist_ok=True)

    eod_path = os.path.join(day_dir, "eod.csv")
    eod = pd.read_csv(eod_path, usecols=["expiration"])

    expirations = (
        eod["expiration"]
        .drop_duplicates()
        .tolist()
    )

    ohlc_dir = os.path.join(day_dir, "ohlc")
    os.makedirs(ohlc_dir, exist_ok=True)

    for expiration in expirations:
        out_path = os.path.join(ohlc_dir, f"ohlc_{expiration}.csv")
        if os.path.exists(out_path):
            continue
        
        # API Call
        response_text = client.get_ohlc(date, expiration)
        if response_text == "":
            continue
        with open(out_path, "w", encoding="utf-8") as file:
            file.write(response_text)

def save_greeks_all(date: str):
    year, month, day = date.split("-")
    day_dir = os.path.join(output_folder, year, month, day)
    os.makedirs(day_dir, exist_ok=True)

    eod_path = os.path.join(day_dir, "eod.csv")
    eod = pd.read_csv(eod_path, usecols=["expiration"])

    expirations = (
        eod["expiration"]
        .drop_duplicates()
        .tolist()
    )

    greeks_all_dir = os.path.join(day_dir, "greeks_all")
    os.makedirs(greeks_all_dir, exist_ok=True)

    for expiration in expirations:
        out_path = os.path.join(greeks_all_dir, f"greeks_all_{expiration}.csv")
        if os.path.exists(out_path):
            continue
        
        # API Call
        response_text = client.get_greeks_all(date, expiration)
        if response_text == "":
            continue
        with open(out_path, "w", encoding="utf-8") as file:
            file.write(response_text)

def main():
    df = pd.read_csv(path, usecols=[0])
    os.makedirs(output_folder, exist_ok=True)

    for date in df.iloc[:, 0]:
        # save_eod(date)
        save_ohlc(date)
        # save_greeks_all(date)

if __name__ == "__main__":
    try:
        main()
    finally:
        client.close()

