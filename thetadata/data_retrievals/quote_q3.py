import pandas as pd
import client
import os
from io import StringIO

path = os.path.join("spy_yearly_opening", "SPY_2026.csv")
output_folder = "theta_data_responses"

EOD_KEY_COLS = ["symbol", "expiration", "strike", "right"]

def save_quote(date: str):
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

    quote_dir = os.path.join(day_dir, "quote")
    os.makedirs(quote_dir, exist_ok=True)

    for expiration in expirations:
        out_path = os.path.join(quote_dir, f"quote_{expiration}.csv")
        if os.path.exists(out_path):
            continue
        
        # API Call
        response_text = client.get_quote(date, expiration)
        if response_text == "":
            continue
        with open(out_path, "w", encoding="utf-8") as file:
            file.write(response_text)

def main():
    df = pd.read_csv(path, usecols=[0])
    os.makedirs(output_folder, exist_ok=True)

    months = df.iloc[:, 0].str[5:7]
    q3 = ['07', '08', '09']
    q3_dates = df[months.isin(q3)]

    for date in q3_dates.iloc[:, 0]:
        save_quote(date)

if __name__ == "__main__":
    try:
        main()
    finally:
        client.close()