import yfinance as yf
import pandas as pd

spy = yf.download("SPY", start="2026-01-01", end="2026-04-05", progress=False)
trading_days = spy.index.strftime("%Y-%m-%d").tolist()

df = pd.DataFrame(trading_days, columns=["date"])
df.to_csv("spy_yearly_opening/SPY_2026.csv", index=False)

print(f"Saved {len(trading_days)} trading days")
for d in trading_days:
    print(d)
