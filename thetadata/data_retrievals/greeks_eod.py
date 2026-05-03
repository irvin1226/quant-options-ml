import pandas as pd
import client
import os
from io import StringIO

path = os.path.join("spy_yearly_opening", "SPY_2026.csv")
output_folder = "theta_data_responses"

EOD_KEY_COLS = ["symbol", "expiration", "strike", "right"]

def save_greeks_eod(date: str):
    year, month, day = date.split("-")
    day_dir = os.path.join(output_folder, year, month, day)
    os.makedirs(day_dir, exist_ok=True)

    out_path = os.path.join(day_dir, "greeks_eod.csv")
    if os.path.exists(out_path):
        return

    response_text = client.get_greeks_eod(date)
    if response_text == "":
        return

    with open(out_path, "w", encoding="utf-8") as file:
        file.write(response_text)


def main():
    df = pd.read_csv(path, usecols=[0])
    os.makedirs(output_folder, exist_ok=True)

    for date in df.iloc[:, 0]:
        save_greeks_eod(date)

if __name__ == "__main__":
    try:
        main()
    finally:
        client.close()