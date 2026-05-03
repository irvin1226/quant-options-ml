import pandas as pd
import client
import os
from io import StringIO

path = os.path.join("spy_yearly_opening", "SPY_2026.csv")
output_folder = "theta_data_responses"

EOD_KEY_COLS = ["symbol", "expiration", "strike", "right"]

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

def main():
    df = pd.read_csv(path, usecols=[0])
    os.makedirs(output_folder, exist_ok=True)

    months = df.iloc[:, 0].str[5:7]
    q1 = ['01', '02', '03']
    q1_dates = df[months.isin(q1)]

    for date in q1_dates.iloc[:, 0]:
        save_ohlc(date)

if __name__ == "__main__":
    try:
        main()
    finally:
        client.close()